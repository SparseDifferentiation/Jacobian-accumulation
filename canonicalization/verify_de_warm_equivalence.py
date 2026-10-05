"""Artefact equivalence for the warm (parametric re-solve) diff-engine path.

A timing table over matrices that differ is meaningless. Before any warm
number from `run_backend_benchmarks.py` is trusted, this checks that the
DIFFENGINE canon backend and CVXPY's tensor backends produce the *same*
(P, c, A, b) -- not on the cold compile, but on a re-solve after the
parameter values have changed, which is the regime the Table 2 columns
time.

Per problem: two fresh instances (the chain cache key excludes
canon_backend, so one shared Problem would silently reuse the first
strategy's cached program), the same seeded parameter stream on both, one
warm-up compile, then a second assignment and the compile that is compared.

Backends compared, in the same discipline as `casadi_compare.py --verify`
(`_sparse_close`, MATCH / MISMATCH with max|diff|):

    de_cached  canon_backend="DIFFENGINE"   vs   dpp  (no canon_backend)
    de_cached  canon_backend="DIFFENGINE"   vs   dpp_coo / dpp_scipy

Non-DPP problems have no tensor reference on this cvxpy: the non-DPP route
is itself DIFFENGINE and an explicit canon_backend there raises. Those rows
report SKIP with the reason rather than a false pass.

Usage:
    <venv>/bin/python canonicalization/verify_de_warm_equivalence.py [--only A,B]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from casadi_compare import _sparse_close  # noqa: E402
from run_backend_benchmarks import (  # noqa: E402
    BENCH_DIR,
    PARAMETRIC_PROBLEMS,
    _assign_param,
    _chain_is_dpp,
    _find_problem,
    discover,
)

SOLVER = "CLARABEL"
# The reference is the chain's own default tensor path -- the `dpp` column of
# Table 2 and the baseline the engine has to beat. COO/SCIPY are the same
# mathematics through a different assembly kernel, and their tensor build is
# pathological on the two big-parameter problems (upstream reference records
# TIMEOUT(>1200s) / crashed-killed for them), so they are opt-in:
#   --refs dpp,dpp_coo,dpp_scipy
ALL_REFERENCES = {"dpp": {}, "dpp_coo": {"canon_backend": "COO"},
                  "dpp_scipy": {"canon_backend": "SCIPY"}}
REFERENCES = [("dpp", {})]


def _data_after_reassign(Cls, kwargs, seed=0):
    """(P, c, A, b) from a WARM re-compile: warm-up, reassign, compile."""
    inst = Cls()
    inst.setup()
    prob = _find_problem(inst)
    rng = np.random.default_rng(seed)
    params = prob.parameters()
    for p in params:
        _assign_param(p, rng)
    prob.get_problem_data(solver=SOLVER, **kwargs)  # warm-up, populates caches
    for p in params:
        _assign_param(p, rng)
    data, _, _ = prob.get_problem_data(solver=SOLVER, **kwargs)
    return data, prob


def _compare(d_de, d_ref):
    """[(label, ok, message)] over the four problem-data artefacts."""
    out = []
    for key in ("A", "P"):
        M_de, M_ref = d_de.get(key), d_ref.get(key)
        if M_de is None and M_ref is None:
            continue
        if (M_de is None) != (M_ref is None):
            out.append((key, False, f"present={M_de is not None} vs {M_ref is not None}"))
            continue
        out.append((key, *_sparse_close(sps.csc_matrix(M_de), sps.csc_matrix(M_ref))))
    for key in ("c", "b"):
        v_de, v_ref = d_de.get(key), d_ref.get(key)
        if v_de is None and v_ref is None:
            continue
        v_de, v_ref = np.asarray(v_de).ravel(), np.asarray(v_ref).ravel()
        if v_de.shape != v_ref.shape:
            out.append((key, False, f"shape {v_de.shape} vs {v_ref.shape}"))
            continue
        err = float(np.nanmax(np.abs(v_de - v_ref))) if v_de.size else 0.0
        ok = bool(np.allclose(v_de, v_ref, rtol=1e-9, atol=1e-9, equal_nan=True))
        out.append((key, ok, f"max|diff|={err:.3g}"))
    return out


def main(only):
    targets = [(m, c) for m, c, err in discover()
               if c and c in PARAMETRIC_PROBLEMS and (not only or c in only)]
    print(f"Warm-path artefact equivalence: DIFFENGINE vs the tensor backends")
    print(f"benchmark dir: {BENCH_DIR}")
    import cvxpy as cp
    import importlib.metadata as md
    print(f"cvxpy {cp.__version__}   sparsediffpy {md.version('sparsediffpy')}\n")

    all_ok = True
    for module_name, class_name in sorted(targets, key=lambda t: t[1]):
        import importlib
        Cls = getattr(importlib.import_module(module_name), class_name)
        try:
            d_de, prob = _data_after_reassign(Cls, {"canon_backend": "DIFFENGINE"})
        except Exception as exc:  # noqa: BLE001
            print(f"  {class_name:<36s} DIFFENGINE ERR: {type(exc).__name__}: {exc}")
            all_ok = False
            continue
        if not _chain_is_dpp(prob):
            print(f"  {class_name:<36s} SKIP  non-DPP: the tensor path is unreachable "
                  f"(this cvxpy routes the non-DPP branch to DIFFENGINE)")
            continue
        for ref_name, ref_kwargs in REFERENCES:
            try:
                d_ref, _ = _data_after_reassign(Cls, ref_kwargs)
            except Exception as exc:  # noqa: BLE001
                print(f"  {class_name:<36s} vs {ref_name:<10s} REF ERR: "
                      f"{type(exc).__name__}: {exc}")
                continue
            res = _compare(d_de, d_ref)
            ok = all(r[1] for r in res)
            all_ok &= ok
            msgs = " ".join(f"{k}:{m}" for k, ok_k, m in res)
            print(f"  {class_name:<36s} vs {ref_name:<10s} "
                  f"{'MATCH   ' if ok else 'MISMATCH'} {msgs}")
    print("\n" + ("ALL MATCH" if all_ok else "MISMATCHES FOUND"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    only = set()
    if "--only" in sys.argv:
        only = {s.strip() for s in sys.argv[sys.argv.index("--only") + 1].split(",")}
    if "--refs" in sys.argv:
        names = [s.strip() for s in sys.argv[sys.argv.index("--refs") + 1].split(",")]
        REFERENCES = [(n, ALL_REFERENCES[n]) for n in names]
    raise SystemExit(main(only))
