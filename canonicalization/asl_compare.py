"""Compare problem-data extraction (P, c, A, b) between the CVXPY diff engine
and the AMPL Solver Library (ASL), on the LOWERED problem formulations.

Same methodology as casadi_compare.py / julia_compare.py, on the 4
representative problems of _lowered_data.py.  The ASL side builds the lowered
model in Pyomo (variables in CVXPY column order, constraints in CVXPY row
order, coefficient blocks taken from the user's data -- never a pre-assembled
coefficient matrix), writes an AMPL .nl file, and loads it through PyNumero's
AslNLP (the real compiled ASL).  Extraction:

    J = evaluate_jacobian()          ->  A = -J        (g(z) = b - A z bodies)
    evaluate_hessian_lag()|duals=0   ->  P             (Hessian of objective)
    evaluate_grad_objective()|z=0    ->  c
    evaluate_constraints()|z=0 - lb  ->  b   (nl moves affine constants to bounds)

The headline "single" is  t_nl_write + t_asl_read + t_cons + t_jac + t_hess +
t_grad: the .nl round trip IS ASL's interface -- there is no way to hand ASL an
in-memory expression graph.  Pyomo model construction (t_model_build) is the
graph-construction analog: excluded from the headline like casadi_build_s, but
reported -- it is far larger than CasADi's, and AMPL's own translator would do
that job in C.

There is NO sparsity detection and NO coloring anywhere in this pipeline:
linear constraint coefficients are explicit sparse J-segments in the .nl file,
and Hessian structure is found once at read time from partial separability of
the objective expression (Gay 1996).

Warm re-solves (ParametrizedQP): ASL has no parameter concept, so two bounds
are reported:  asl_resolve_reeval_s re-evaluates Jacobian+Hessian+gradient on
the loaded NLP (a lower bound -- the numbers cannot actually change);
asl_resolve_rewrite_s rebuilds the model with fresh data and repeats
write+read+extract (the true refresh path; model rebuild again excluded).

Row/column order: the nl writer permutes rows/columns (a deterministic
permutation, recovered from the .row/.col files); verification compares after
un-permuting.  ASL returns the Hessian's lower triangle; it is symmetrized
before comparison against CVXPY's full symmetric P.

Usage:
    python canonicalization/asl_compare.py --verify [--only SimpleQP,...]
    python canonicalization/asl_compare.py --verify-full
    python canonicalization/asl_compare.py --time
Env: CASADI_ITERS (default 3), CASADI_RESOLVES (default 20; rewrite rounds are
capped at 3), BENCH_TIMEOUT (seconds per problem worker, default 3600).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
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

from _lowered_data import PROBLEMS, reference_data  # noqa: E402

ITERS = int(os.environ.get("CASADI_ITERS", "3"))
RESOLVES = int(os.environ.get("CASADI_RESOLVES", "20"))
REWRITES = 3
TIMEOUT = float(os.environ.get("BENCH_TIMEOUT", "3600"))


# --------------------------------------------------------------------------- #
# Pyomo models of the lowered problems (constraint bodies are g(z) = b - A z,
# so every row is "g_i == 0" (zero cone) or "g_i >= 0" (nonneg cone), in CVXPY
# row order; z is declared in CVXPY column order).
# --------------------------------------------------------------------------- #
def _le(const, coefs, vars_):
    from pyomo.core.expr.numeric_expr import LinearExpression
    return LinearExpression(constant=float(const), linear_coefs=coefs,
                            linear_vars=vars_)


def build_simple_qp_model(d, theta=None):
    import pyomo.environ as pyo
    n = d["dims"]["n"]
    P0, q0, G0, h0, Aeq, beq = (d["P0"], d["q0"], d["G0"], d["h0"],
                                d["Aeq"], d["beq"])
    m = pyo.ConcreteModel()
    m.z = pyo.Var(range(n))
    zs = [m.z[j] for j in range(n)]
    quad = sum(zs[i] * _le(0.0, P0[i].tolist(), zs) for i in range(n))
    m.obj = pyo.Objective(expr=0.5 * quad + _le(0.0, q0.tolist(), zs))
    ne = len(beq)
    m.g = pyo.Constraint(
        range(ne + len(h0)),
        rule=lambda mo, i: (
            _le(beq[i], (-Aeq[i]).tolist(), zs) == 0.0 if i < ne
            else _le(h0[i - ne], (-G0[i - ne]).tolist(), zs) >= 0.0
        ),
    )
    return m


def build_least_squares_model(d, theta=None):
    import pyomo.environ as pyo
    mm, n = d["dims"]["m"], d["dims"]["n"]
    A0, b0 = d["A0"], d["b0"]
    m = pyo.ConcreteModel()
    m.z = pyo.Var(range(mm + n))          # z = [t, x]
    ts = [m.z[i] for i in range(mm)]
    xs = [m.z[mm + j] for j in range(n)]
    m.obj = pyo.Objective(expr=sum(t * t for t in ts))
    m.g = pyo.Constraint(
        range(mm),
        rule=lambda mo, i: _le(b0[i], [1.0] + (-A0[i]).tolist(), [ts[i]] + xs) == 0.0,
    )
    return m


def build_parametrized_qp_model(d, theta):
    import pyomo.environ as pyo
    mm, n = d["dims"]["m"], d["dims"]["n"]
    Ap, bp = theta
    m = pyo.ConcreteModel()
    m.z = pyo.Var(range(mm + n))          # z = [t, x]
    ts = [m.z[i] for i in range(mm)]
    xs = [m.z[mm + j] for j in range(n)]
    m.obj = pyo.Objective(expr=sum(t * t for t in ts))
    # rows: zero (b + t - A x), then nonneg (x, 1 - x)
    m.g = pyo.Constraint(
        range(mm),
        rule=lambda mo, i: _le(bp[i], [1.0] + (-Ap[i]).tolist(), [ts[i]] + xs) == 0.0,
    )
    m.g_lo = pyo.Constraint(range(n), rule=lambda mo, j: _le(0.0, [1.0], [xs[j]]) >= 0.0)
    m.g_hi = pyo.Constraint(range(n), rule=lambda mo, j: _le(1.0, [-1.0], [xs[j]]) >= 0.0)
    return m


def build_cvar_model(d, theta=None):
    import pyomo.environ as pyo
    n_scen, nx = d["dims"]["n_scen"], d["dims"]["nx"]
    A0, c0 = d["A0"], d["c0"]
    x_min, x_max = d["x_min"], d["x_max"]
    gamma, kappa = float(d["gamma"]), float(d["kappa"])
    m = pyo.ConcreteModel()
    m.z = pyo.Var(range(nx + 1 + n_scen))     # z = [x, alpha, u]
    xs = [m.z[j] for j in range(nx)]
    alpha = m.z[nx]
    us = [m.z[nx + 1 + s] for s in range(n_scen)]
    m.obj = pyo.Objective(expr=_le(0.0, c0.tolist(), xs))
    m.g_scen = pyo.Constraint(
        range(n_scen),
        rule=lambda mo, s: _le(0.0, [1.0, 1.0] + (-A0[s]).tolist(),
                               [us[s], alpha] + xs) >= 0.0,
    )
    m.g_upos = pyo.Constraint(range(n_scen),
                              rule=lambda mo, s: _le(0.0, [1.0], [us[s]]) >= 0.0)
    m.g_cvar = pyo.Constraint(
        expr=_le(kappa, [-1.0] + [-gamma] * n_scen, [alpha] + us) >= 0.0)
    m.g_lo = pyo.Constraint(range(nx),
                            rule=lambda mo, j: _le(-x_min[j], [1.0], [xs[j]]) >= 0.0)
    m.g_hi = pyo.Constraint(range(nx),
                            rule=lambda mo, j: _le(x_max[j], [-1.0], [xs[j]]) >= 0.0)
    return m


MODEL_BUILDERS = {
    "SimpleQP": build_simple_qp_model,
    "LeastSquares": build_least_squares_model,
    "ParametrizedQP": build_parametrized_qp_model,
    "CVaRSlice": build_cvar_model,
}


# --------------------------------------------------------------------------- #
# ASL extraction
# --------------------------------------------------------------------------- #
def _scatter_from_names(path: Path, model_names: list,
                        n_nl: int | None = None) -> sps.csr_matrix:
    """S with S[i, j] = 1 iff nl position j holds the i-th model row/column.

    The nl writer drops variables that appear nowhere with a nonzero
    coefficient and (potentially) constraints whose body is constant; a
    missing name yields an all-zero row/column in S, so S @ X @ S.T scatters
    the nl-order matrix into model order with zero fill -- matching CVXPY,
    which keeps those rows/columns as structural zeros."""
    nl_names = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if n_nl is not None:
        nl_names = nl_names[:n_nl]   # .row appends objective names after rows
    pos = {nm: j for j, nm in enumerate(nl_names)}
    ij = [(i, pos[nm]) for i, nm in enumerate(model_names) if nm in pos]
    rows = np.array([i for i, _ in ij], dtype=np.int64)
    cols = np.array([j for _, j in ij], dtype=np.int64)
    return sps.csr_matrix((np.ones(len(ij)), (rows, cols)),
                          shape=(len(model_names), len(nl_names)))


def asl_extract(model, td: Path, with_matrices: bool = True):
    """write .nl, load through ASL, extract (P, c, A, b) with per-stage timers.

    Matrices are returned in MODEL order (nl permutation already undone).
    With with_matrices=False (timing mode) the write carries no symbolic
    labels (.row/.col are harness bookkeeping AMPL users never pay for) and
    the permutation/matrix recovery is skipped -- only timers are returned."""
    from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP

    nl = td / "m.nl"
    io_options = {"symbolic_solver_labels": True} if with_matrices else {}
    gc.collect()
    t0 = time.perf_counter()
    model.write(str(nl), io_options=io_options)
    t_write = time.perf_counter() - t0

    gc.collect()
    t0 = time.perf_counter()
    nlp = AslNLP(str(nl))
    t_read = time.perf_counter() - t0

    nz = nlp.n_primals()
    nlp.set_primals(np.zeros(nz))
    nlp.set_duals(np.zeros(nlp.n_constraints()))
    nlp.set_obj_factor(1.0)

    gc.collect()
    t0 = time.perf_counter()
    cons0 = nlp.evaluate_constraints()
    t_cons = time.perf_counter() - t0
    t0 = time.perf_counter()
    J = nlp.evaluate_jacobian()
    t_jac = time.perf_counter() - t0
    t0 = time.perf_counter()
    H = nlp.evaluate_hessian_lag()
    t_hess = time.perf_counter() - t0
    t0 = time.perf_counter()
    grad0 = nlp.evaluate_grad_objective()
    t_grad = time.perf_counter() - t0

    stages = {"t_nl_write_s": t_write, "t_asl_read_s": t_read,
              "t_cons_s": t_cons, "t_jac_s": t_jac, "t_hess_s": t_hess,
              "t_grad_s": t_grad,
              "single_s": t_write + t_read + t_cons + t_jac + t_hess + t_grad,
              "single_aslread_s": t_read + t_cons + t_jac + t_hess + t_grad,
              "nl_bytes": nl.stat().st_size}
    if not with_matrices:
        return None, None, None, None, nlp, stages

    # undo the nl writer's (deterministic) row/col permutation; dropped
    # rows/columns (all-zero coefficients) scatter back as structural zeros
    import pyomo.environ as pyo
    con_names = [con.name for con in
                 model.component_data_objects(pyo.Constraint, active=True)]
    var_names = [v.name for v in
                 model.component_data_objects(pyo.Var, active=True)]
    J = sps.csr_matrix(J)
    Sr = _scatter_from_names(td / "m.row", con_names, J.shape[0])
    Sc = _scatter_from_names(td / "m.col", var_names, J.shape[1])
    J = Sr @ J @ Sc.T
    A = -J
    # affine constants live in the constraint bounds: g(0) = body(0) - lb
    # (a dropped row had an all-zero body; its constant is 0 by construction)
    b = Sr @ (cons0 - nlp.constraints_lb())
    c = Sc @ np.asarray(grad0)
    H = Sc @ sps.csr_matrix(H) @ Sc.T
    # ASL returns one triangle of the Hessian; symmetrize unless already full
    if H.nnz and abs(H - H.T).max() > 1e-12:
        H = H + H.T - sps.diags(H.diagonal())
    P = H
    return P, c, A, b, nlp, stages


# --------------------------------------------------------------------------- #
# Worker (one problem per process; parent applies the timeout)
# --------------------------------------------------------------------------- #
def worker(name: str, size: str, mode: str, out_path: str):
    entry = PROBLEMS[name]
    d = entry.data(size)
    rng = np.random.default_rng(42)
    theta = entry.draw(rng, d) if entry.parametric else None
    result = {"problem": name, "size": size}

    def flush():
        Path(out_path).write_text(json.dumps(result))

    if mode == "verify":
        with tempfile.TemporaryDirectory() as td:
            t0 = time.perf_counter()
            model = MODEL_BUILDERS[name](d, theta)
            result["t_model_build_s"] = time.perf_counter() - t0
            P, c, A, b, _, stages = asl_extract(model, Path(td))
        result.update(stages)
        np.savez(out_path + ".npz",
                 A_data=A.tocoo().data, A_row=A.tocoo().row, A_col=A.tocoo().col,
                 A_shape=np.asarray(A.shape),
                 P_data=P.tocoo().data, P_row=P.tocoo().row, P_col=P.tocoo().col,
                 P_shape=np.asarray(P.shape), b=b, c=c)
        result["mats"] = out_path + ".npz"
        flush()
        return

    # timing mode
    singles, builds = [], []
    stages0 = None
    for it in range(ITERS):
        gc.collect()
        t0 = time.perf_counter()
        model = MODEL_BUILDERS[name](d, theta)
        builds.append(time.perf_counter() - t0)
        with tempfile.TemporaryDirectory() as td:
            _, _, _, _, nlp, stages = asl_extract(model, Path(td))
            singles.append(stages["single_s"])
            if it == 0:
                stages0 = stages
                # in-place re-evals on the loaded NLP (warm lower bound)
                if entry.parametric:
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

    # true refresh path: fresh data -> rebuild -> write+read+extract
    if entry.parametric:
        rewrites = []
        for _ in range(REWRITES):
            theta_new = entry.draw(rng, d)
            gc.collect()
            t0 = time.perf_counter()
            model = MODEL_BUILDERS[name](d, theta_new)
            t_build = time.perf_counter() - t0
            with tempfile.TemporaryDirectory() as td:
                _, _, _, _, _, stages = asl_extract(model, Path(td))
            rewrites.append(stages["single_s"])  # build excluded, like cold
            del model
        result["asl_resolve_rewrite_s"] = median(rewrites)
        flush()


# --------------------------------------------------------------------------- #
# Parent
# --------------------------------------------------------------------------- #
def _run_worker(name: str, size: str, mode: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "result.json")
        env = {**os.environ,
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
        try:
            proc = subprocess.run(
                [sys.executable, __file__, "--worker", name, size, mode, out],
                capture_output=True, text=True, timeout=TIMEOUT, env=env)
            err = (None if proc.returncode == 0
                   else f"worker failed for {name}:\n{proc.stderr[-4000:]}")
        except subprocess.TimeoutExpired:
            err = f"timeout after {TIMEOUT}s"
        if err is not None and mode == "verify":
            raise RuntimeError(err)
        # timing mode: keep whatever phases the worker flushed before dying
        result = json.loads(Path(out).read_text()) if Path(out).exists() else {}
        if err is not None:
            result["error"] = err
            print(f"  {name}: worker died ({err[:100]}); partial phases kept")
        if "mats" in result:
            result["_mats"] = dict(np.load(result["mats"]))
    return result


def verify(name: str, full: bool) -> bool:
    from casadi_compare import _sparse_close

    entry = PROBLEMS[name]
    size = entry.sizes[1] if full else entry.sizes[0]
    rng = np.random.default_rng(42)
    theta = entry.draw(rng, entry.data(size)) if entry.parametric else None
    data = reference_data(name, size, theta)
    res = _run_worker(name, size, "verify")
    mats = res["_mats"]

    def coo(key):
        return sps.coo_matrix(
            (mats[f"{key}_data"], (mats[f"{key}_row"], mats[f"{key}_col"])),
            shape=tuple(mats[f"{key}_shape"])).tocsc()

    ok, msgs = True, []
    good, msg = _sparse_close(coo("A"), sps.csc_matrix(data["A"]))
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
    P_asl = coo("P")
    if P_ref is not None:
        good, msg = _sparse_close(P_asl, sps.csc_matrix(P_ref))
        msgs.append(f"P[{'OK' if good else 'FAIL'} {msg}]")
    else:
        good = P_asl.nnz == 0
        msgs.append(f"P[{'OK (both absent)' if good else 'FAIL (asl P nonzero)'}]")
    ok &= good
    print(f"  {name:<16s} ({size}): {'MATCH' if ok else 'MISMATCH'}  " + " ".join(msgs))
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-full", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="results_asl_compare.json")
    ap.add_argument("--worker", nargs=4, metavar=("NAME", "SIZE", "MODE", "OUT"))
    args = ap.parse_args()

    if args.worker:
        worker(*args.worker)
        return

    names = [n for n in args.only.split(",") if n] or list(PROBLEMS)
    for n in names:
        if n not in PROBLEMS:
            raise SystemExit(f"unknown problem {n!r}; choose from {list(PROBLEMS)}")

    if args.verify or args.verify_full:
        print("Verifying ASL-extracted (P, c, A, b) == cvxpy ignore_dpp data:")
        all_ok = all([verify(n, args.verify_full) for n in names])
        print("ALL MATCH" if all_ok else "MISMATCHES FOUND")
        if not all_ok:
            raise SystemExit(1)

    if args.time:
        rows = []
        for n in names:
            entry = PROBLEMS[n]
            res = _run_worker(n, entry.sizes[1], "time")
            row = {
                "name": n, "size": entry.sizes[1], "parametric": entry.parametric,
                "asl_single_s": res.get("asl_single_s"),
                "asl_build_s": res.get("asl_build_s"),
                "asl_resolve_reeval_s": res.get("asl_resolve_reeval_s"),
                "asl_resolve_rewrite_s": res.get("asl_resolve_rewrite_s"),
                "stages": res.get("stages"),
            }
            if "error" in res:
                row["error"] = res["error"]
            rows.append(row)

            def ms(v, fmt="9.1f"):
                return f"{v*1e3:{fmt}}ms" if v is not None else "     --  "

            extra = ""
            if entry.parametric:
                extra = (f"   resolve: reeval {ms(row['asl_resolve_reeval_s'], '9.2f')}"
                         f"  rewrite {ms(row['asl_resolve_rewrite_s'])}")
            print(f"{n:<16s} single: asl {ms(row['asl_single_s'])} "
                  f"(build excl. {ms(row['asl_build_s'])})" + extra)
            (HERE / args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {HERE / args.out}")


if __name__ == "__main__":
    main()
