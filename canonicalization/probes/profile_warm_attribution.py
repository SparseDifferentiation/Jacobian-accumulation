"""Where does a warm diff-engine re-solve actually spend its time?

A `de_cached` column that is still slower than `dpp` can lose for two very
different reasons, and the ratio alone does not say which:

  engine   the C engine refreshing and re-evaluating the Jacobian/Hessian --
           what SparseDiffEngine #125 (parameter-free subtree pruning)
           attacks;
  scipy    `extractor.py` handing scipy raw (vals, rows, cols) triplets on
           every `extract()`, so scipy re-derives a sort order that cannot
           change between calls -- Finding 1 of
           `../diffengine_issue_draft_warm_extraction.md`, still unfixed;
  glue     cvxpy's reduction chain and solver.apply around both.

cProfiles one warm `get_problem_data` per problem and splits total time
three ways by function name, so a remaining loss can be attributed instead
of guessed at.

Usage:
    <venv>/bin/python canonicalization/probes/profile_warm_attribution.py \
        [--only A,B] [--backend DIFFENGINE]
"""
from __future__ import annotations

import cProfile
import importlib
import pstats
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from run_backend_benchmarks import (  # noqa: E402
    PARAMETRIC_PROBLEMS,
    _assign_param,
    _find_problem,
    discover,
)

SOLVER = "CLARABEL"

# Matched against "<file>:<line>(<func>)" as pstats renders it.
ENGINE = ("update_params", "eval_jacobian_vals", "eval_hessian_vals_coo",
          "problem_gradient", "objective_forward", "constraint_forward",
          "problem_jacobian", "problem_hessian")
SCIPY = ("coo_tocsr", "csr_sort_indices", "sort_indices", "sum_duplicates",
         "coo_todense", "tocsc", "tocsr", "csr_sum_duplicates",
         "coo_matrix", "csc_matrix", "csr_matrix", "_coo_", "compressed")


def bucket(entry: str) -> str:
    low = entry.lower()
    if any(k.lower() in low for k in ENGINE):
        return "engine"
    if "scipy" in low or any(k.lower() in low for k in SCIPY):
        return "scipy"
    return "glue"


def profile_one(Cls, kwargs):
    inst = Cls()
    inst.setup()
    prob = _find_problem(inst)
    rng = np.random.default_rng(0)
    params = prob.parameters()
    for p in params:
        _assign_param(p, rng)
    prob.get_problem_data(solver=SOLVER, **kwargs)  # warm-up
    for p in params:
        _assign_param(p, rng)
    pr = cProfile.Profile()
    pr.enable()
    prob.get_problem_data(solver=SOLVER, **kwargs)
    pr.disable()
    st = pstats.Stats(pr)
    totals = {"engine": 0.0, "scipy": 0.0, "glue": 0.0}
    # tottime (self time) partitions the wall clock exactly once per frame.
    for func, (_, _, tottime, _, _) in st.stats.items():
        totals[bucket(f"{func[0]}:{func[1]}({func[2]})")] += tottime
    return st.total_tt, totals, st


def main(only, backend):
    import cvxpy as cp
    import importlib.metadata as md
    kwargs = {} if backend == "AUTO" else {"canon_backend": backend}
    print(f"warm re-solve attribution  (canon_backend={backend})")
    print(f"cvxpy {cp.__version__}   sparsediffpy {md.version('sparsediffpy')}\n")
    hdr = f"{'benchmark':<38s}{'total_s':>9s}{'engine':>9s}{'scipy':>9s}{'glue':>9s}   shares"
    print(hdr)
    print("-" * len(hdr))
    for module_name, class_name in sorted(
            [(m, c) for m, c, e in discover()
             if c and c in PARAMETRIC_PROBLEMS and (not only or c in only)],
            key=lambda t: t[1]):
        Cls = getattr(importlib.import_module(module_name), class_name)
        try:
            total, t, st = profile_one(Cls, kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"{class_name:<38s} ERR {type(exc).__name__}: {exc}")
            continue
        s = total or 1.0
        print(f"{class_name:<38s}{total:9.4f}{t['engine']:9.4f}"
              f"{t['scipy']:9.4f}{t['glue']:9.4f}   "
              f"eng {t['engine']/s:4.0%} / scipy {t['scipy']/s:4.0%} "
              f"/ glue {t['glue']/s:4.0%}")
        if "--top" in sys.argv:
            print("      top self-time frames:")
            rows = sorted(st.stats.items(), key=lambda kv: -kv[1][2])[:6]
            for func, (_, _, tt, _, _) in rows:
                print(f"        {tt:8.4f}  {func[2]}  ({Path(func[0]).name}:{func[1]})")


if __name__ == "__main__":
    only = set()
    if "--only" in sys.argv:
        only = {s.strip() for s in sys.argv[sys.argv.index("--only") + 1].split(",")}
    backend = "DIFFENGINE"
    if "--backend" in sys.argv:
        backend = sys.argv[sys.argv.index("--backend") + 1]
    main(only, backend)
