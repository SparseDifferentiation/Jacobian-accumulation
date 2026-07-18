"""Compare problem-data extraction (P, c, A, b) between the CVXPY diff engine
and the generic Julia sparse-AD stack (SparseConnectivityTracer +
SparseMatrixColorings + DifferentiationInterface over ForwardDiff/ReverseDiff),
on the LOWERED problem formulations.

Same methodology as casadi_compare.py, on the 4 representative problems of
_lowered_data.py.  The Julia side runs as a subprocess worker
(canonicalization/julia/main.jl) with a wall-clock timeout; it receives the
problem constants as npz, extracts

    A = -sparse_jacobian(g)|_{z=0}       b = g(0)
    c = gradient(f)|_{z=0}               P = sparse_hessian(f)

with per-stage timers (pattern detection / coloring / preparation / eval), and
returns timings as JSON plus, under --verify, the matrices as npz COO triplets
for the bit-identical check against CVXPY's CLARABEL data (same _sparse_close,
tol 1e-9).

Boundary notes (mirrors the CasADi fairness rules; see README):
  - closure construction (build_s) is excluded from the headline, like
    casadi_build_s; it is trivial here (a few array captures).
  - the worker warms up on the small instance first so Julia JIT compilation is
    excluded; the target-size first iteration (single_iter1_s) still contains
    ForwardDiff chunk-size specialization and is reported separately.
  - P is extracted via DI's native sparse Hessian AND, where SCT tracers can
    flow through the ReverseDiff tape, via jacobian-of-gradient; the faster
    route is reported (single_best_s), mirroring the CasADi hessian rule.
  - warm re-solves re-run the compressed Jacobian sweeps with cached
    preparation: the generic stack has no CasADi-style parametric tape, and
    this is its honest parameter-refresh path.

Usage:
    python canonicalization/julia_compare.py --verify [--only SimpleQP,...]
    python canonicalization/julia_compare.py --verify-full
    python canonicalization/julia_compare.py --time
Env: CASADI_ITERS (default 3), CASADI_RESOLVES (default 20), BENCH_TIMEOUT
(seconds per problem, default 3600), JULIA (julia binary, default "julia").
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _lowered_data import PROBLEMS, reference_data  # noqa: E402
from casadi_compare import _sparse_close  # noqa: E402

ITERS = int(os.environ.get("CASADI_ITERS", "3"))
RESOLVES = int(os.environ.get("CASADI_RESOLVES", "20"))
TIMEOUT = float(os.environ.get("BENCH_TIMEOUT", "3600"))
JULIA = os.environ.get("JULIA", "julia")

ENV = {**os.environ,
       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def _save_npz(path: Path, d: dict):
    flat = {}
    for k, v in d.items():
        if k == "dims":
            flat.update({dk: np.int64(dv) for dk, dv in v.items()})
        else:
            flat[k] = v
    np.savez(path, **flat)


def run_worker(name: str, size: str, warmup_size: str, td: Path,
               dump: bool, theta=None, resolves: int = 0,
               tool: str = "sct", iters: int = ITERS) -> dict:
    entry = PROBLEMS[name]
    data_npz = td / f"{name}_{size}.npz"
    warm_npz = td / f"{name}_{warmup_size}_warm.npz"
    out_json = td / f"{name}_{size}_result.json"
    _save_npz(data_npz, entry.data(size))
    _save_npz(warm_npz, entry.data(warmup_size))
    script = {"sct": "main.jl", "jump": "jump_main.jl"}[tool]
    cmd = [JULIA, "--project=" + str(HERE.parent / "julia"), "--threads=1",
           str(HERE / "julia" / script),
           "--problem", name, "--size", size,
           "--data", str(data_npz), "--warmup-data", str(warm_npz),
           "--out", str(out_json),
           "--iters", str(iters), "--resolves", str(resolves)]
    dump_npz = td / f"{name}_{size}_mats.npz"
    if dump:
        cmd += ["--dump-npz", str(dump_npz)]
    if theta is not None:
        theta_npz = td / f"{name}_{size}_theta.npz"
        np.savez(theta_npz, **{f"theta_{i}": v for i, v in enumerate(theta)})
        cmd += ["--theta", str(theta_npz)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=TIMEOUT, env=ENV)
    if proc.returncode != 0:
        raise RuntimeError(f"julia worker failed for {name}:\n{proc.stderr[-4000:]}")
    result = json.loads(out_json.read_text())
    if dump:
        result["_mats"] = dict(np.load(dump_npz))
    return result


def _coo(mats: dict, key: str) -> sps.csc_matrix:
    return sps.coo_matrix(
        (mats[f"{key}_vals"], (mats[f"{key}_rows"], mats[f"{key}_cols"])),
        shape=tuple(mats[f"{key}_shape"]),
    ).tocsc()


def verify(name: str, full: bool, tool: str) -> bool:
    entry = PROBLEMS[name]
    size = entry.sizes[1] if full else entry.sizes[0]
    rng = np.random.default_rng(42)
    theta = entry.draw(rng, entry.data(size)) if entry.parametric else None
    data = reference_data(name, size, theta)

    with tempfile.TemporaryDirectory() as td:
        res = run_worker(name, size, entry.sizes[0], Path(td), dump=True,
                         theta=theta, tool=tool, iters=1)
    mats = res["_mats"]

    ok, msgs = True, []
    good, msg = _sparse_close(_coo(mats, "A"), sps.csc_matrix(data["A"]))
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
    P_jl = _coo(mats, "P")
    if P_ref is not None:
        good, msg = _sparse_close(P_jl, sps.csc_matrix(P_ref))
        msgs.append(f"P[{'OK' if good else 'FAIL'} {msg}]")
    else:
        good = P_jl.nnz == 0
        msgs.append(f"P[{'OK (both absent)' if good else 'FAIL (julia P nonzero)'}]")
    ok &= good
    if "jacgrad_error" in res.get("stages_iter1", {}):
        msgs.append("jacgrad[unavailable]")
    print(f"  {name:<16s} ({size}): {'MATCH' if ok else 'MISMATCH'}  " + " ".join(msgs))
    return bool(ok)


def time_problem(name: str, tool: str) -> dict:
    entry = PROBLEMS[name]
    size = entry.sizes[1]
    # jump coefficients are plain data, so parametric timing needs concrete θ;
    # the sct worker draws its own θ internally when none is given
    theta = None
    if entry.parametric and tool == "jump":
        theta = entry.draw(np.random.default_rng(42), entry.data(size))
    with tempfile.TemporaryDirectory() as td:
        t0 = time.perf_counter()
        try:
            res = run_worker(name, size, entry.sizes[0], Path(td), dump=False,
                             theta=theta, tool=tool,
                             resolves=RESOLVES if entry.parametric else 0)
        except (subprocess.TimeoutExpired, RuntimeError) as exc:
            # salvage the phases the worker managed to flush before dying
            out_json = Path(td) / f"{name}_{size}_result.json"
            res = json.loads(out_json.read_text()) if out_json.exists() else {}
            res["error"] = (f"timeout after {TIMEOUT}s"
                            if isinstance(exc, subprocess.TimeoutExpired)
                            else str(exc)[:2000])
            print(f"  {name}: worker died ({res['error'][:100]}); "
                  f"partial phases kept")
        wall = time.perf_counter() - t0
    st = res.get("stages_iter1", {})
    single_native = res.get("single_s", float("nan"))
    single_jacgrad = st.get("single_jacgrad_s", float("nan"))
    row = {
        "name": name, "size": size, "parametric": entry.parametric,
        "tool": tool,
        "julia_single_s": float(np.nanmin([single_native, single_jacgrad])),
        "julia_single_native_s": single_native,
        "julia_single_jacgrad_s": single_jacgrad,
        "julia_single_iter1_s": res.get("single_iter1_s", float("nan")),
        "julia_resolve_s": res.get("resolve_s", float("nan")),
        "julia_build_s": res.get("build_s", float("nan")),
        "julia_worker_wall_s": wall,
        "stages_iter1": {k: v for k, v in st.items() if not k.startswith("_")},
        "versions": res.get("versions", {}),
    }
    if "error" in res:
        row["error"] = res["error"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-full", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--tool", default="sct", choices=("sct", "jump"),
                    help="sct = SparseConnectivityTracer+SMC+DI stack; "
                         "jump = JuMP/MOI copy_to (no AD)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = args.out or f"results_{'julia' if args.tool == 'sct' else 'jump'}_compare.json"

    names = [n for n in args.only.split(",") if n] or list(PROBLEMS)
    for n in names:
        if n not in PROBLEMS:
            raise SystemExit(f"unknown problem {n!r}; choose from {list(PROBLEMS)}")

    if args.verify or args.verify_full:
        print(f"Verifying {args.tool}-extracted (P, c, A, b) == cvxpy ignore_dpp data:")
        all_ok = all([verify(n, args.verify_full, args.tool) for n in names])
        print("ALL MATCH" if all_ok else "MISMATCHES FOUND")
        if not all_ok:
            raise SystemExit(1)

    if args.time:
        rows = []
        for n in names:
            row = time_problem(n, args.tool)
            rows.append(row)
            print(f"{n:<16s} single: julia {row['julia_single_s']*1e3:9.1f}ms "
                  f"(native {row['julia_single_native_s']*1e3:9.1f}ms, "
                  f"jacgrad {row['julia_single_jacgrad_s']*1e3:9.1f}ms, "
                  f"iter1 {row['julia_single_iter1_s']*1e3:9.1f}ms)"
                  + (f"   resolve {row['julia_resolve_s']*1e3:9.2f}ms"
                     if row["parametric"] else ""))
            (HERE / out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {HERE / out}")


if __name__ == "__main__":
    main()
