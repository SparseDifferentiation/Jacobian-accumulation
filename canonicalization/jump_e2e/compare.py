"""End-to-end MODEL -> SOLVER-HANDOFF comparison: CVXPY user model vs
idiomatic JuMP user model, identical raw data (problems.py), same target
solver form (SCS).  SCS itself is NEVER RUN by --time.

Each side is charged its full user pipeline up to the moment the solver could
start:
    cvxpy: fresh cp.Problem construction  +  get_problem_data(solver=SCS)
    jump:  model build (macro layer)      +  MOI.copy_to into the bridged
           SCS.Optimizer cache (SCS.jl's own zero-based-CSC input form)
Both stop with SCS's input materialized and no solver call -- "model to
solver time", one shot.

Correctness gate (--verify, OPT-IN, the only mode that actually calls SCS):
JuMP never materializes CVXPY's (P, c, A, b) -- what it hands SCS comes from
its own bridging path -- so problem-data equality is not checkable by
construction; instead the optimal values of the two pipelines are compared
(both sides SCS at EPS_VERIFY, default 1e-6; small instances are
milliseconds):

    |obj_jump - obj_cvxpy| / max(1, |obj_cvxpy|) <= tol (1e-3)

Hand-verification of the models themselves: every models/<Name>.jl docstring
states the CVXPY original and the epigraph rewriting used; compare with the
cvxpy_problem builder in problems.py, which consumes the same arrays.

Timing (--time): median of E2E_ITERS fresh runs per side.  Per side:
build_s (model construction), handoff_s (canonicalization/translation),
e2e_s = build + handoff.  The JuMP worker warms up on the small instance
first (JIT excluded); CVXPY runs in-process, fresh Problem objects per
iteration, parametric problems lowered with ignore_dpp=True (the suite's
fairness setting).

Usage:
    python canonicalization/jump_e2e/compare.py --time [--only A,B] [--out f]
    python canonicalization/jump_e2e/compare.py --verify [--full] [--only A,B]
Env: E2E_ITERS (3), EPS_VERIFY (1e-6), BENCH_TIMEOUT (3600), JULIA (julia
binary).  BLAS should be pinned to 1 thread for timing runs.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from problems import PROBLEMS  # noqa: E402

ITERS = int(os.environ.get("E2E_ITERS", "3"))
EPS_VERIFY = float(os.environ.get("EPS_VERIFY", "1e-6"))
TIMEOUT = float(os.environ.get("BENCH_TIMEOUT", "3600"))
JULIA = os.environ.get("JULIA", "julia")

ENV1 = {**os.environ,
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def _theta(entry, d):
    rng = np.random.default_rng(42)
    return entry.draw(rng, d) if entry.parametric else None


def run_jump(name: str, full: bool, mode: str, iters: int,
             eps: float = EPS_VERIFY) -> dict:
    entry = PROBLEMS[name]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        d = entry.data(full)
        np.savez(td / "data.npz", **d)
        dw = entry.data(False)
        np.savez(td / "warm.npz", **dw)
        cmd = [JULIA, "--project=" + str(HERE.parents[1] / "julia"),
               "--threads=1", str(HERE / "runner.jl"),
               "--problem", name, "--data", str(td / "data.npz"),
               "--warmup-data", str(td / "warm.npz"),
               "--out", str(td / "result.json"),
               "--iters", str(iters), "--mode", mode, "--eps", repr(eps)]
        theta = _theta(entry, d)
        if theta is not None:
            np.savez(td / "theta.npz",
                     **{f"theta_{i}": v for i, v in enumerate(theta)})
            cmd += ["--theta", str(td / "theta.npz")]
            theta_w = _theta(entry, dw)
            np.savez(td / "theta_w.npz",
                     **{f"theta_{i}": v for i, v in enumerate(theta_w)})
            cmd += ["--warmup-theta", str(td / "theta_w.npz")]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=TIMEOUT, env=ENV1)
        out = td / "result.json"
        if not out.exists() or not out.read_text().strip():
            raise RuntimeError(
                f"jump e2e worker failed for {name} (rc={proc.returncode}):\n"
                f"{proc.stderr[-4000:]}")
        result = json.loads(out.read_text())
        if proc.returncode != 0:
            result["error"] = proc.stderr[-4000:]
        return result


def run_cvxpy_handoff(name: str, full: bool, iters: int) -> dict:
    """Fresh Problem build + get_problem_data(solver=SCS); SCS is never run."""
    import cvxpy as cp

    entry = PROBLEMS[name]
    d = entry.data(full)
    theta = _theta(entry, d)
    builds, handoffs = [], []
    for _ in range(iters):
        gc.collect()
        t0 = time.perf_counter()
        prob, params = entry.cvxpy_problem(d)
        for p, v in zip(params, theta or []):
            p.value = v
        t_build = time.perf_counter() - t0

        kw = {"ignore_dpp": True} if entry.parametric else {}
        t0 = time.perf_counter()
        prob.get_problem_data(solver=cp.SCS, **kw)
        t_handoff = time.perf_counter() - t0
        builds.append(t_build)
        handoffs.append(t_handoff)
    return {"problem": name,
            "build_s": median(builds), "handoff_s": median(handoffs),
            "e2e_s": median(b + h for b, h in zip(builds, handoffs)),
            "versions": {"cvxpy": cp.__version__}}


def run_cvxpy_solve(name: str, full: bool, eps: float) -> dict:
    """One full CVXPY+SCS solve; used only by the opt-in --verify gate."""
    import cvxpy as cp

    entry = PROBLEMS[name]
    d = entry.data(full)
    theta = _theta(entry, d)
    prob, params = entry.cvxpy_problem(d)
    for p, v in zip(params, theta or []):
        p.value = v
    kw = dict(solver=cp.SCS, eps_abs=eps, eps_rel=eps, verbose=False)
    if entry.parametric:
        kw["ignore_dpp"] = True
    prob.solve(**kw)
    return {"problem": name, "objective": prob.value, "status": prob.status,
            "versions": {"cvxpy": cp.__version__}}


def verify(name: str, full: bool, tol: float = 1e-3) -> bool:
    jr = run_jump(name, full, "solve", iters=1, eps=EPS_VERIFY)
    cr = run_cvxpy_solve(name, full, EPS_VERIFY)
    oj, oc = jr.get("objective"), cr.get("objective")
    rel = abs(oj - oc) / max(1.0, abs(oc)) if oj is not None and oc is not None \
        else float("inf")
    ok = np.isfinite(rel) and rel <= tol
    size = "full" if full else "small"
    print(f"  {name:<26s} ({size}): {'MATCH' if ok else 'MISMATCH'}  "
          f"obj jump {oj:.8g} vs cvxpy {oc:.8g}  rel {rel:.2e}  "
          f"[{jr.get('status')}/{cr.get('status')}]", flush=True)
    return bool(ok)


def time_problem(name: str) -> dict:
    entry = PROBLEMS[name]
    jr = run_jump(name, True, "handoff", iters=ITERS)
    cr = run_cvxpy_handoff(name, True, iters=ITERS)
    keys = ("build_s", "handoff_s", "e2e_s", "versions")
    row = {"name": name, "parametric": entry.parametric,
           "jump": {k: jr.get(k) for k in keys},
           "cvxpy": {k: cr.get(k) for k in keys}}
    if "error" in jr:
        row["jump"]["error"] = jr["error"]

    def ms(v):
        return f"{v * 1e3:10.1f}ms" if isinstance(v, (int, float)) else "      --  "

    print(f"{name:<26s} e2e  jump {ms(jr.get('e2e_s'))}  "
          f"cvxpy {ms(cr.get('e2e_s'))}   "
          f"(build {ms(jr.get('build_s'))} / {ms(cr.get('build_s'))}, "
          f"handoff {ms(jr.get('handoff_s'))} / {ms(cr.get('handoff_s'))})",
          flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="verify at full size (slow: full SCS solves)")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="results_jump_e2e_compare.json")
    args = ap.parse_args()

    names = [n for n in args.only.split(",") if n] or list(PROBLEMS)
    for n in names:
        if n not in PROBLEMS:
            raise SystemExit(f"unknown problem {n!r}; choose from {list(PROBLEMS)}")

    if args.verify:
        print(f"Verifying optimal values, jump vs cvxpy, both SCS eps={EPS_VERIFY}:")
        oks = {n: verify(n, args.full) for n in names}
        print("ALL MATCH" if all(oks.values()) else
              f"MISMATCHES: {[n for n, v in oks.items() if not v]}")
        if not all(oks.values()):
            raise SystemExit(1)

    if args.time:
        rows = []
        for n in names:
            rows.append(time_problem(n))
            (HERE / args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {HERE / args.out}")


if __name__ == "__main__":
    main()
