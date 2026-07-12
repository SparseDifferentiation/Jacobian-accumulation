# Canonicalization benchmarks

Everything here measures one thing: the cost of **constructing solver-ready problem
data** `(P, c, A, b)` for a convex program — CVXPY's canonicalization, i.e. the
*constant* part of the diff engine's job. No problem is ever solved. Two regimes:

- **single** — one cold construction from a freshly built expression tree;
- **re-solve** — re-evaluating the data after parameter values change, on a warm
  cached program/function.

Two comparisons share this folder:

| Script | Compares | Output |
|---|---|---|
| `casadi_compare.py` (+ `casadi_problems_ext_*.py`) | diff engine vs **CasADi** symbolic AD | `results_casadi_compare*.json` |
| `run_backend_benchmarks.py` | diff engine vs CVXPY's **cvxcore** backends | `results_*.txt` |

Both run the 25 problems of the public [CVXPY benchmark
suite](https://github.com/cvxpy/benchmarks) (the CasADi comparison covers 22: the three
`kron` problems need complex variables / partial-trace operators CasADi cannot express).

## The CasADi comparison, and why it is fair

CasADi has no notion of disciplined convex programming, so it receives the
**already-lowered** formulation: the same epigraph variables and the same conic rows
CVXPY's canonicalization produces, hand-written as `MX` expressions `f(z, θ)` and
`g(z, θ) = b − Az` per problem (that is what the `casadi_problems_ext_*` modules are).
Extraction then follows CasADi's own documented QP path — differentiate the graph, no
solver involved:

```
grad = casadi.gradient(f, z);  P = casadi.jacobian(grad, z)
c = grad|_{z=0};               A = -casadi.jacobian(g, z);   b = g|_{z=0}
```

wrapped in a `casadi.Function(θ → (P, c, A, b))`. Fairness rules, all deliberate:

- **Jacobian-of-gradient, not `casadi.hessian`**: the symmetry-exploiting star coloring
  in `hessian()` degenerates on dense Hessian patterns (SimpleQP's dense 1600² quadratic
  did not finish in 6.5 h; the identical matrix takes < 1 min the other way). CasADi
  always gets its faster path. `probes/profile_simpleqp_stages.py` demonstrates this.
- **Graph construction excluded on both sides** — both timers start from a built
  expression tree/graph.
- **Verified identical outputs**: `--verify` (small instances) and `--verify-full`
  (benchmark sizes) check the two systems produce bit-identical `(P, c, A, b)` (up to
  explicitly stored zeros; two kron-structured problems agree to 1 ulp). The timing
  columns therefore measure construction of the same matrices.
- **The boundary asymmetry is reported, not hidden**: the engine column includes CVXPY's
  Python-side reduction chain, the CasADi column starts from the hand-lowered model. On
  tiny problems (~200 variables) that fixed overhead dominates and CasADi wins the
  single column — see the results discussion below.

## How to run

```bash
uv sync                                # installs casadi + the cvxpy fork (see caveat below)
git clone --depth 1 https://github.com/cvxpy/benchmarks canonicalization/cvxpy_benchmarks

# CasADi comparison
python canonicalization/casadi_compare.py --verify        # matrix-identity checks, small
python canonicalization/casadi_compare.py --verify-full   # same at benchmark sizes (slow)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python canonicalization/casadi_compare.py --time        # the timing table

# Backend comparison (CPP vs diff engine, cold + parametric re-solve)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  BENCH_TIMEOUT=1200 BENCH_BACKENDS=CPP,DIFFENGINE,DE_DPP \
  python canonicalization/run_backend_benchmarks.py
```

Knobs: `--only Name1,Name2` restricts problems; `CASADI_ITERS`/`CASADI_RESOLVES` and
`BENCH_ITERS`/`BENCH_ONLY`/`BENCH_OUT` control repetition and output. Pin BLAS to one
thread (as above) for comparable numbers.

**Engine version caveat**: warm re-solve numbers exercise the cached-program refresh
path and require an engine with the composite `param_source` refresh fix
([SparseDiffEngine #107](https://github.com/SparseDifferentiation/SparseDiffEngine/pull/107)).
PyPI `sparsediffpy 0.6.0` predates it — cold/single extraction numbers are valid there,
but for re-solves build SparseDiffPy from source with the engine at
`param-source-mark-refresh` (or any release containing it).

## Reference results

Measured 2026-07-11/12 on a MacBook Pro (15-inch, 2018), 6-core Intel i7-8850H,
16 GB RAM, macOS 15.7, Python 3.14, CVXPY 1.10-dev (`pr-c-resolve-caching`),
NumPy 2.4, SciPy 1.17, CasADi 3.7.2, engine = main + #107 fix. Raw files in
`results/`. Geometric means of per-problem ratios (engine/baseline; < 1 favors the
engine):

| Comparison | single | re-solve |
|---|---|---|
| vs cvxcore (25 problems) | **0.24** | **0.78** (8 parametric) |
| vs CasADi (22 problems) | **0.24** | **2.19** |

The CasADi single column decomposes into three regimes:

1. **Dense blocks** (regression/scenario matrices, dense quadratic forms, parametric
   matrices): engine 20–300× faster. CasADi must *discover* the Jacobian sparsity by
   bitvector AD sweeps and *reconstruct* values via one seeded sweep per color — and a
   dense block forces the sweep count to the full dimension; the engine reads each
   atom's Jacobian block off structurally. Extremes: ParametrizedQP 306×
   (dense *symbolic* block), CVaR DNF > 10 h (100 M-nnz dense block).
   `probes/probe_why_slow.py` isolates the mechanism (same expression, dense vs sparse
   constant: 18.4 s vs 0.04 s); `probes/probe_simpleqp_scaling.py` shows the ≈O(n³·⁷)
   growth.
2. **Structured patterns** (diagonals, differences, kron/svec): both systems scale
   linearly; constant-factor gap of 4–8×, shrinking to parity on the svec-structured
   SDPs.
3. **Tiny problems**: CasADi wins 14–40× — there the engine column is CVXPY chain
   overhead, not extraction (the boundary asymmetry above).

On **warm re-solves** CasADi's fixed-tape VM wins ~2× overall (dramatically where
parameters only touch `b`); the engine keeps the advantage where a parameter reshapes
`P` itself (FactorCovarianceModel, 7×).

Incidental find: CasADi 3.7.2 mis-evaluates Jacobian *values* (sparsity correct) for
scalar-broadcast expressions of the form `u + alpha - mtimes(dense, x)`; see the
work-around and repro notes in the CVaR builder in `casadi_problems_ext_lp.py`.

## Files

- `casadi_compare.py` — harness: Spec contract, extraction, verification, timing.
- `casadi_problems_ext_lp.py` / `_cones.py` / `_kron.py` — hand-lowered models for all
  25 suite problems (LP/QP, SOC/PSD, and the kron trio), each mirroring CVXPY's exact
  row/column order; auto-discovered by the harness.
- `run_backend_benchmarks.py` — cvxcore-backend comparison; subprocess-isolated per
  problem, streams partial results, self-contained.
- `probes/` — the mechanism experiments quoted above.
- `extras/` — an early draft (parametric Newton re-solves via the derivative oracle),
  kept for reference.

This folder backs chapter 4 of the associated master's thesis and the benchmark tables
of [cvxpy#3348](https://github.com/cvxpy/cvxpy/pull/3348).
