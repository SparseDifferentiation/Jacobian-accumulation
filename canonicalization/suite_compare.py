"""Full-suite problem-data extraction comparison for JuMP/MOI and AMPL/ASL,
driven by the tool-neutral lowered blocks of _lowered_blocks.py.

The CVXPY reference side (models, seeds, parameter draws) is reused verbatim
from the CasADi harness Specs (casadi_compare.MAKERS), so every problem is
verified against the identical ground truth as the CasADi comparison:

    python canonicalization/suite_compare.py --verify      --tool jump|asl
    python canonicalization/suite_compare.py --verify-full --tool jump|asl
    python canonicalization/suite_compare.py --time        --tool jump|asl

Timing columns per problem: build_s (modeling layer: block assembly + model
objects -- reported, excluded from the headline, like casadi_build_s),
single_s (the extraction: copy_to for JuMP; nl write + ASL read + Jacobian/
Hessian/gradient evals for ASL), and for parametric problems resolve_s
(JuMP: fresh copy_to; ASL: reeval lower bound + full rewrite).

Cone rows (SOC/PSD) are pure metadata: JuMP gets the real MOI cone sets
(grouped Zeros -> Nonnegatives -> SOC -> PSD = CVXPY's CLARABEL row order);
ASL gets the affine bodies with >=0 tags -- ASL differentiates bodies and
never sees cone semantics, exactly like the CasADi side's g vector.

Exclusions: kron trio (follow-up), CVaR at full size (100M-nnz dense block;
CVaRSlice in julia_compare.py/asl_compare.py is the compared instance), and
for ASL SimpleLP at full size (2e7 Pyomo constraint objects exceed the
modeling layer, not ASL; the E4 diag probe covers ASL's linear regime).

Env: CASADI_ITERS (3), CASADI_RESOLVES (20), BENCH_TIMEOUT (s, default 3600),
JULIA (julia binary), SUITE_REWRITES (3).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

import numpy as np
import scipy.sparse as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _lowered_blocks import SUITE, build_blocks  # noqa: E402

ITERS = int(os.environ.get("CASADI_ITERS", "3"))
RESOLVES = int(os.environ.get("CASADI_RESOLVES", "20"))
REWRITES = int(os.environ.get("SUITE_REWRITES", "3"))
TIMEOUT = float(os.environ.get("BENCH_TIMEOUT", "3600"))
JULIA = os.environ.get("JULIA", "julia")

KIND_CODE = {"zero": 0, "nonneg": 1, "soc": 2, "psd": 3}
ASL_SKIP_FULL = {"SimpleLP"}   # 2e7 Pyomo constraint objects; modeling-layer limit

ENV1 = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def _spec(name: str, full: bool):
    from casadi_compare import MAKERS
    return MAKERS[name](full=full)


def _theta(spec, size_rng=None):
    rng = size_rng or np.random.default_rng(42)
    return spec.draw(rng) if spec.parametric else []


def reference_data(name: str, full: bool, theta):
    import cvxpy as cp
    spec = _spec(name, full)
    prob, params = spec.cvxpy_build()
    for p, v in zip(params, theta):
        p.value = v
    data, _, _ = prob.get_problem_data(solver=cp.CLARABEL, ignore_dpp=True)
    return data


# --------------------------------------------------------------------------- #
# blocks -> npz (JuMP worker input)
# --------------------------------------------------------------------------- #
def save_blocks_npz(bl: dict, path: Path):
    nz = sum(l for _, l in bl["segments"])
    out = {"nz": np.int64(nz), "nblocks": np.int64(len(bl["rows"])),
           "c": np.asarray(bl["c"], float)}
    if bl["P"] is not None and bl["P"].nnz:
        Pc = sps.coo_matrix(bl["P"])
        out.update(P_rows=Pc.row.astype(np.int64), P_cols=Pc.col.astype(np.int64),
                   P_vals=Pc.data.astype(float))
    for k, blk in enumerate(bl["rows"]):
        A = sps.coo_matrix(blk["A"])
        out[f"rb{k}_rows"] = A.row.astype(np.int64)
        out[f"rb{k}_cols"] = A.col.astype(np.int64)
        out[f"rb{k}_vals"] = A.data.astype(float)
        out[f"rb{k}_m"] = np.int64(A.shape[0])
        out[f"rb{k}_kind"] = np.int64(KIND_CODE[blk["kind"]])
        out[f"rb{k}_b"] = np.asarray(blk["b"], float)
        if blk["kind"] == "soc":
            out[f"rb{k}_dims"] = np.asarray(blk["dims"], np.int64)
        if blk["kind"] == "psd":
            out[f"rb{k}_side"] = np.int64(blk["side"])
    np.savez(path, **out)


# --------------------------------------------------------------------------- #
# blocks -> Pyomo model (ASL side)
# --------------------------------------------------------------------------- #
def build_pyomo_model(bl: dict):
    import pyomo.environ as pyo
    from pyomo.core.expr.numeric_expr import LinearExpression

    nz = sum(l for _, l in bl["segments"])
    m = pyo.ConcreteModel()
    m.z = pyo.Var(range(nz))
    zs = [m.z[j] for j in range(nz)]

    c = np.asarray(bl["c"], float)
    idx = np.nonzero(c)[0]
    lin = LinearExpression(constant=0.0, linear_coefs=c[idx].tolist(),
                           linear_vars=[zs[j] for j in idx]) if idx.size else 0.0
    if bl["P"] is not None and bl["P"].nnz:
        Pc = sps.csr_matrix(bl["P"])
        terms = []
        for i in range(nz):
            lo, hi = Pc.indptr[i], Pc.indptr[i + 1]
            if hi > lo:
                terms.append(zs[i] * LinearExpression(
                    constant=0.0,
                    linear_coefs=(0.5 * Pc.data[lo:hi]).tolist(),
                    linear_vars=[zs[j] for j in Pc.indices[lo:hi]]))
        m.obj = pyo.Objective(expr=sum(terms) + lin)
    else:
        m.obj = pyo.Objective(expr=lin if idx.size else 0.0)

    for k, blk in enumerate(bl["rows"]):
        A = sps.csr_matrix(blk["A"])
        b = np.asarray(blk["b"], float)
        eq = blk["kind"] == "zero"
        exprs = []
        for i in range(A.shape[0]):
            lo, hi = A.indptr[i], A.indptr[i + 1]
            exprs.append(LinearExpression(
                constant=float(b[i]),
                linear_coefs=(-A.data[lo:hi]).tolist(),
                linear_vars=[zs[j] for j in A.indices[lo:hi]]))
        con = pyo.Constraint(
            range(A.shape[0]),
            rule=(lambda mo, i, exprs=exprs, eq=eq:
                  exprs[i] == 0.0 if eq else exprs[i] >= 0.0))
        setattr(m, f"g{k}", con)
    return m


# --------------------------------------------------------------------------- #
# workers
# --------------------------------------------------------------------------- #
def run_jump_worker(name: str, full: bool, theta, td: Path, dump: bool,
                    resolves: int, iters: int) -> dict:
    blocks = build_blocks(name, full, theta)
    data_npz = td / f"{name}_blocks.npz"
    save_blocks_npz(blocks, data_npz)
    spec_small = _spec(name, False)
    theta_small = _theta(spec_small)
    warm_npz = td / f"{name}_blocks_small.npz"
    save_blocks_npz(build_blocks(name, False, theta_small), warm_npz)
    out_json = td / f"{name}_result.json"
    cmd = [JULIA, "--project=" + str(HERE.parent / "julia"), "--threads=1",
           str(HERE / "julia" / "jump_generic.jl"),
           "--data", str(data_npz), "--warmup-data", str(warm_npz),
           "--out", str(out_json), "--iters", str(iters),
           "--resolves", str(resolves)]
    dump_npz = td / f"{name}_mats.npz"
    if dump:
        cmd += ["--dump-npz", str(dump_npz)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=TIMEOUT, env={**os.environ, **ENV1})
        err = (None if proc.returncode == 0
               else f"jump worker failed for {name}:\n{proc.stderr[-4000:]}")
    except subprocess.TimeoutExpired:
        err = f"timeout after {TIMEOUT}s"
    result = json.loads(out_json.read_text()) if out_json.exists() else {}
    if err:
        if dump:
            raise RuntimeError(err)
        result["error"] = err
        print(f"  {name}: worker died ({err[:100]}); partial phases kept")
    if dump and dump_npz.exists():
        result["_mats"] = dict(np.load(dump_npz))
    return result


def asl_worker(name: str, size_key: str, mode: str, out_path: str):
    from asl_compare import asl_extract

    full = size_key == "full"
    spec = _spec(name, full)
    theta = _theta(spec)
    result = {"problem": name, "size": size_key}

    def flush():
        Path(out_path).write_text(json.dumps(result))

    def build_model_timed():
        gc.collect()
        t0 = time.perf_counter()
        bl = build_blocks(name, full, theta)
        t_blocks = time.perf_counter() - t0
        t0 = time.perf_counter()
        model = build_pyomo_model(bl)
        return model, t_blocks, time.perf_counter() - t0

    if mode == "verify":
        model, t_blocks, t_model = build_model_timed()
        with tempfile.TemporaryDirectory() as td:
            P, c, A, b, _, stages = asl_extract(model, Path(td), with_matrices=True)
        result.update(stages)
        result["t_blocks_s"], result["t_model_s"] = t_blocks, t_model
        A, P = sps.coo_matrix(A), sps.coo_matrix(P)
        np.savez(out_path + ".npz",
                 A_data=A.data, A_row=A.row, A_col=A.col,
                 A_shape=np.asarray(A.shape),
                 P_data=P.data, P_row=P.row, P_col=P.col,
                 P_shape=np.asarray(P.shape), b=b, c=c)
        result["mats"] = out_path + ".npz"
        flush()
        return

    singles, builds = [], []
    stages0 = None
    for it in range(ITERS):
        model, t_blocks, t_model = build_model_timed()
        builds.append(t_blocks + t_model)
        with tempfile.TemporaryDirectory() as td:
            _, _, _, _, nlp, stages = asl_extract(model, Path(td),
                                                  with_matrices=False)
            singles.append(stages["single_s"])
            if it == 0:
                stages0 = stages
                if spec.parametric:
                    warm = []
                    for _ in range(RESOLVES):
                        gc.collect()
                        t0 = time.perf_counter()
                        nlp.evaluate_constraints()
                        nlp.evaluate_jacobian()
                        nlp.evaluate_hessian_lag()
                        nlp.evaluate_grad_objective()
                        warm.append(time.perf_counter() - t0)
                    result["asl_resolve_reeval_s"] = median(warm)
        del model
        result["asl_single_s"] = median(singles)
        result["asl_build_s"] = median(builds)
        result["stages"] = stages0
        flush()

    if spec.parametric:
        rng = np.random.default_rng(43)
        rewrites = []
        for _ in range(REWRITES):
            theta_new = spec.draw(rng)
            bl = build_blocks(name, full, theta_new)
            model = build_pyomo_model(bl)
            with tempfile.TemporaryDirectory() as td:
                _, _, _, _, _, stages = asl_extract(model, Path(td),
                                                    with_matrices=False)
            rewrites.append(stages["single_s"])
            del model
        result["asl_resolve_rewrite_s"] = median(rewrites)
        flush()


def run_asl_worker(name: str, full: bool, mode: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "result.json")
        cmd = [sys.executable, __file__, "--worker", name,
               "full" if full else "small", mode, out]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=TIMEOUT, env={**os.environ, **ENV1})
            err = (None if proc.returncode == 0
                   else f"asl worker failed for {name}:\n{proc.stderr[-4000:]}")
        except subprocess.TimeoutExpired:
            err = f"timeout after {TIMEOUT}s"
        if err is not None and mode == "verify":
            raise RuntimeError(err)
        result = json.loads(Path(out).read_text()) if Path(out).exists() else {}
        if err is not None:
            result["error"] = err
            print(f"  {name}: worker died ({err[:100]}); partial phases kept")
        if "mats" in result:
            result["_mats"] = dict(np.load(result["mats"]))
    return result


# --------------------------------------------------------------------------- #
# verify / time
# --------------------------------------------------------------------------- #
def _coo(mats, key, style):
    if style == "jump":
        return sps.coo_matrix(
            (mats[f"{key}_vals"], (mats[f"{key}_rows"], mats[f"{key}_cols"])),
            shape=tuple(mats[f"{key}_shape"])).tocsc()
    return sps.coo_matrix(
        (mats[f"{key}_data"], (mats[f"{key}_row"], mats[f"{key}_col"])),
        shape=tuple(mats[f"{key}_shape"])).tocsc()


def verify(name: str, full: bool, tool: str) -> bool:
    from casadi_compare import _sparse_close

    spec = _spec(name, full)
    theta = _theta(spec)
    data = reference_data(name, full, theta)
    if tool == "jump":
        with tempfile.TemporaryDirectory() as td:
            res = run_jump_worker(name, full, theta, Path(td), dump=True,
                                  resolves=0, iters=1)
    else:
        res = run_asl_worker(name, full, "verify")
    mats = res["_mats"]

    ok, msgs = True, []
    good, msg = _sparse_close(_coo(mats, "A", tool), sps.csc_matrix(data["A"]))
    ok &= good
    msgs.append(f"A[{'OK' if good else 'FAIL'} {msg}]")
    for vec_name in ("b", "c"):
        ref = np.asarray(data[vec_name]).ravel()
        got = np.asarray(mats[vec_name]).ravel()
        good = got.shape == ref.shape and bool(
            np.isclose(got, ref, atol=1e-9, rtol=1e-9, equal_nan=True).all())
        ok &= good
        msgs.append(f"{vec_name}[{'OK' if good else 'FAIL'}]")
    P_ref = data.get("P")
    P_got = _coo(mats, "P", tool)
    if P_ref is not None:
        good, msg = _sparse_close(P_got, sps.csc_matrix(P_ref))
        msgs.append(f"P[{'OK' if good else 'FAIL'} {msg}]")
    else:
        good = P_got.nnz == 0
        msgs.append(f"P[{'OK (both absent)' if good else 'FAIL (P nonzero)'}]")
    ok &= good
    size = "full" if full else "small"
    print(f"  {name:<28s} ({size}): {'MATCH' if ok else 'MISMATCH'}  "
          + " ".join(msgs), flush=True)
    return bool(ok)


def time_problem(name: str, tool: str) -> dict:
    spec = _spec(name, True)
    if tool == "jump":
        theta = _theta(spec)
        with tempfile.TemporaryDirectory() as td:
            res = run_jump_worker(
                name, True, theta, Path(td), dump=False, iters=ITERS,
                resolves=RESOLVES if spec.parametric else 0)
        row = {"name": name, "parametric": spec.parametric, "tool": tool,
               "single_s": res.get("single_s"),
               "single_iter1_s": res.get("single_iter1_s"),
               "build_s": res.get("build_s"),
               "resolve_s": res.get("resolve_s"),
               "versions": res.get("versions", {})}
    else:
        res = run_asl_worker(name, True, "time")
        row = {"name": name, "parametric": spec.parametric, "tool": tool,
               "single_s": res.get("asl_single_s"),
               "build_s": res.get("asl_build_s"),
               "resolve_reeval_s": res.get("asl_resolve_reeval_s"),
               "resolve_rewrite_s": res.get("asl_resolve_rewrite_s"),
               "stages": res.get("stages")}
    if "error" in res:
        row["error"] = res["error"]

    def ms(v):
        return f"{v * 1e3:9.1f}ms" if isinstance(v, (int, float)) else "     --  "

    print(f"{name:<28s} single: {ms(row.get('single_s'))} "
          f"(build excl. {ms(row.get('build_s'))})"
          + (f"  resolve {ms(row.get('resolve_s') or row.get('resolve_reeval_s'))}"
             if spec.parametric else ""), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-full", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--tool", default="jump", choices=("jump", "asl"))
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--worker", nargs=4, metavar=("NAME", "SIZE", "MODE", "OUT"))
    args = ap.parse_args()

    if args.worker:
        asl_worker(*args.worker)
        return

    names = [n for n in args.only.split(",") if n] or list(SUITE)
    for n in names:
        if n not in SUITE:
            raise SystemExit(f"unknown problem {n!r}; choose from {list(SUITE)}")

    out = args.out or f"results_suite_{args.tool}_compare.json"

    if args.verify or args.verify_full:
        full = args.verify_full
        use = [n for n in names
               if not (args.tool == "asl" and full and n in ASL_SKIP_FULL)]
        skipped = [n for n in names if n not in use]
        if skipped:
            print(f"skipping for {args.tool} at full size: {skipped}")
        print(f"Verifying {args.tool}-extracted (P, c, A, b) == cvxpy data:")
        oks = {}
        for n in use:
            try:
                oks[n] = verify(n, full, args.tool)
            except Exception as exc:  # keep going; report at the end
                oks[n] = False
                print(f"  {n:<28s}: ERROR {str(exc)[:200]}", flush=True)
        print("ALL MATCH" if all(oks.values()) else
              f"MISMATCHES: {[n for n, v in oks.items() if not v]}")
        (HERE / f"suite_verify_{args.tool}_{'full' if full else 'small'}.json"
         ).write_text(json.dumps(oks, indent=2))
        if not all(oks.values()):
            raise SystemExit(1)

    if args.time:
        gate = HERE / f"suite_verify_{args.tool}_full.json"
        green = None
        if gate.exists():
            green = {n for n, v in json.loads(gate.read_text()).items() if v}
        rows = []
        for n in names:
            if args.tool == "asl" and n in ASL_SKIP_FULL:
                continue
            if green is not None and n not in green:
                print(f"{n:<28s} skipped (not verified green)", flush=True)
                continue
            rows.append(time_problem(n, args.tool))
            (HERE / out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {HERE / out}")


if __name__ == "__main__":
    main()
