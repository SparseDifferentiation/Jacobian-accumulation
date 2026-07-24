# End-to-end model→solver handoff: CVXPY vs idiomatic JuMP

The copy_to comparison (`suite_compare.py --tool jump`) measures JuMP's
canonicalization analog in isolation, on pre-lowered blocks. This folder
measures what a *user* pays end to end to get a problem in front of a solver:
write the model in the tool's native language, stop at the moment the solver
could start. **SCS is never run by the timed comparison** — both sides stop
with SCS's input form materialized:

- **CVXPY**: fresh `cp.Problem` construction + `get_problem_data(solver=SCS)`
  (canonicalization to SCS standard form, no solver call);
- **JuMP**: model build (macro layer) + `MOI.copy_to` into the bridged
  `SCS.Optimizer` cache — SCS.jl's own input form (zero-based-CSC
  `MatrixOfConstraints` in SCS cone order). This is byte-for-byte the copy
  SCS.jl performs at the start of `optimize!`; only the solver call itself is
  omitted.

## Layout — one file per model, hand-verifiable

- `problems.py` — per problem: raw-data generator (seeds byte-identical to
  `casadi_compare.py`'s Specs) and the CVXPY user model built **from those
  arrays**.
- `models/<Name>.jl` — the JuMP user model for the same problem, one file
  each. The docstring states the CVXPY original and, where JuMP lacks the
  atom (`huber`, `norm1`, `pos`, `sum_squares`), the epigraph rewriting used —
  the same reformulation CVXPY's canonicalization applies, written out so it
  can be checked by hand against `problems.py`.
- `runner.jl` — Julia worker; default `--mode handoff` never runs SCS and
  reports `build_s` (macro layer) + `handoff_s` (bridging + SCS-cache
  assembly). `--mode solve` (opt-in) additionally calls `optimize!` for the
  objective gate. Warms up on the small instance so JIT is excluded.
- `compare.py` — driver.

## Commands

```sh
# timed model->solver-handoff, full size, median of E2E_ITERS (no SCS run):
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 python canonicalization/jump_e2e/compare.py --time

# OPT-IN correctness gate: small instances, actually solves with SCS on both
# sides (eps 1e-6, millisecond solves) and compares optimal values:
python canonicalization/jump_e2e/compare.py --verify
python canonicalization/jump_e2e/compare.py --verify --full   # full-size solves
```

Env knobs: `E2E_ITERS` (3), `EPS_VERIFY` (1e-6), `BENCH_TIMEOUT` (3600 s),
`JULIA`.

## Fairness rules

- Identical raw arrays on both sides (`problems.py` is the single source).
- Same target form (SCS input), single-threaded BLAS, no two timing jobs at
  once.
- JuMP models follow the JuMP performance tips (macro-built expressions,
  `set_string_names_on_creation(false)`, epigraph variables instead of
  expanding dense quadratics in the macro layer).
- Parametric problems: JuMP has no parameter objects, so θ is baked in at
  build time (fresh build per draw is its honest one-shot path;
  ParametricOptInterface.jl is the ecosystem's warm-path answer). The CVXPY
  side keeps its `cp.Parameter` form, lowered with `ignore_dpp=True` — the
  suite's fairness setting.
- Both sides are charged their full pipeline; neither side's canonicalization
  output is precomputed.

## Caveats

- The two sides do NOT hand SCS bit-identical data: CVXPY and MOI's bridges
  make different (equivalent) reformulation choices, so matrix-level identity
  checks do not apply here — that is exactly what the lowered-form
  comparisons elsewhere in `canonicalization/` are for. Model equivalence is
  instead gated on optimal values via the opt-in `--verify` (the only mode
  that runs SCS; small instances are millisecond solves).
- JuMP's `handoff_s` excludes SCS.jl's final cache→C-array pass (it happens
  inside `optimize!` and cannot be timed separately without calling the
  solver); the cache built here is already SCS-ordered CSC, so the omitted
  pass is a copy, not a canonicalization step.
