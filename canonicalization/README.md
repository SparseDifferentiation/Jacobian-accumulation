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
scalar-broadcast operands whenever `jacobian()` runs in reverse mode with batched
adjoint directions — all directions after the first are dropped. Dense blocks (as in
CVaR's `u + alpha - mtimes(dense, x)`) trigger it indirectly by flipping the ad-mode
heuristic to reverse; `repmat`, `max_num_dir=1`, forward mode, plain `F.reverse(n)`,
and SX are all unaffected. Work-around in the CVaR builder in
`casadi_problems_ext_lp.py`; upstream report drafted in
`casadi_issue_draft_broadcast_bug.md`.

## Beyond CasADi: the Julia sparse-AD stack, JuMP/MOI, and AMPL/ASL

Three further comparisons on 4 representative problems (`_lowered_data.py`:
SimpleQP, LeastSquares, ParametrizedQP, and CVaRSlice = the benchmark CVaR
generator at 8192 scenarios — the full 131072 DNFs in CasADi and exceeds this
machine through any modeling layer). Same lowered forms, same seeds, verified
bit-identical against CVXPY's CLARABEL data before timing (`--verify`,
`--verify-full`: ALL MATCH):

| Runner | Represents | Mechanism |
|---|---|---|
| `julia_compare.py --tool sct` | 2025 state-of-the-art *generic* sparse AD: SparseConnectivityTracer + SparseMatrixColorings + DifferentiationInterface | trace pattern → greedy color → compressed ForwardDiff sweeps |
| `julia_compare.py --tool jump` | JuMP/MOI `copy_to` into a `MatrixOfConstraints` cache (what Clarabel.jl et al. run in `attach_optimizer`) | **no AD**: coefficient assembly from declared structure |
| `asl_compare.py` | the real AMPL Solver Library (Pyomo nl writer → PyNumero `AslNLP`) | **no detection, no coloring**: linear parts are explicit J-segments in the .nl; Hessian structure via partial separability at read time (Gay 1996) |

Measured 2026-07-17/18, same machine/setup as above (Julia 1.10.5 single-threaded,
pyomo 6.10.1, ASL built from source). Cold single extraction, seconds:

| Problem | engine | CasADi | SCT stack | JuMP `copy_to` | ASL (.nl round trip) |
|---|---|---|---|---|---|
| SimpleQP | 1.70 | 53.7 | 85.5 | **0.74** | 79.3 (30.9 write + 47.8 read; evals 0.63) |
| LeastSquares | 1.62 | 36.6 | 90.5 | **0.76** | 58.5 (evals 0.35) |
| ParametrizedQP | 2.52 | 771.5 | 111.5 | **0.83** | 61.0 (evals 0.46) |
| CVaRSlice | — | ~28.8¹ | 51.3 | **0.075** | 9.7 (evals 0.04) |

¹ CasADi at the slice size from `probes/probe_coloring_sweeps.py` (pattern
11.8 s + jacobian 17.0 s); the suite-size CVaR is the DNF > 10 h case.

Warm re-solve, ParametrizedQP (fresh parameter values): engine 0.94 s,
CasADi 0.94 s, SCT stack 17.9 s (re-runs its compressed sweeps with cached
preparation — the generic stack has no parametric tape), JuMP 0.63 s (fresh
`copy_to`; base JuMP has no vectorized parameter refresh — see
ParametricOptInterface.jl), ASL 0.076 s in-memory re-eval (lower bound; the
numbers cannot actually change) / 84.7 s true .nl rewrite.

Take-aways, backed by the dense-block probes
(`probes/probe_dense_block_julia.jl`, `probes/probe_dense_block_asl.py`,
mirroring `probe_why_slow.py`; results in `results/`):

1. **The dense-block pathology is paradigm-wide, not CasADi-specific.** On the
   Jacobian of `x ↦ Ax` with dense 4000×2000 `A`, the Julia stack needs 53 s
   (40 s of it greedy distance-2 *coloring*; 2000 forward colors) vs CasADi's
   16.3 s; the sparse control collapses to ~0.03 s for both. Per-stage: SCT
   detection is much faster than CasADi's bitvector sweeps, SMC coloring much
   slower, compressed accumulation comparable. On SCT's Hessian side, star
   coloring on SimpleQP's dense 1600² pattern costs 26.2 s prep — the same
   degeneration `casadi.hessian` shows.
2. **Declared-structure systems never pay it.** ASL is linear in nnz in every
   probe experiment — its only real cost is the ASCII .nl round trip (~4 s per
   M-nnz to write + ~4 s per M-nnz to parse; CPU-bound formatting, not disk —
   and AMPL's own C translator emitting *binary* .nl would cut both), after
   which Jacobian+Hessian evaluation is 0.2–0.6 s where the AD tools pay tens
   of seconds. JuMP's `copy_to` is pure data-structure assembly and beats
   everything, including the engine, on these 4 problems — though it starts
   from explicit coefficient arrays rather than a DCP atom tree, so it does
   strictly less work than CVXPY+engine canonicalization (see fairness notes).
3. **CVaR's epigraph-sum row defeats coloring entirely** (all 8193 u/α columns
   pairwise conflict → 8961 colors), which is why generic AD cannot compress
   it at any implementation quality; JuMP/ASL are indifferent.

Fairness / boundary notes (mirroring the CasADi rules above):

- All tools receive the already-lowered formulation; model/graph construction
  is excluded from the headline and reported (`build_s`). The excluded share
  differs by tool and is documented rather than hidden: CasADi's MX graph
  build ~0.1–3 s, SCT closures ~0, JuMP macro build 1.7–5.5 s, ASL's Pyomo
  build 5–10 s (a Python-loop overestimate of AMPL's C translator).
- Declared-structure tools (JuMP, ASL) consume explicit coefficient blocks —
  that is their input contract, assembled at model-build time from the user's
  data arrays; the AD tools must *recover* those coefficients from the
  expression graph, which is precisely the mechanism under study.
- SCT numbers exclude Julia JIT (small-instance warmup; iteration 1 reported
  separately); P uses the faster of DI's native sparse Hessian vs
  jacobian-of-gradient per problem, mirroring the CasADi hessian rule.
- ASL's headline includes the .nl disk round trip because the file *is* ASL's
  interface (no in-memory API exists); nl file sizes are reported. Timed
  writes carry no symbolic labels; verification (which needs the .row/.col
  permutation files) runs separately.
- Cone rows are metadata everywhere: ASL sees affine bodies with ==/>= tags,
  JuMP the genuine MOI cone sets; extraction cost is set-agnostic.

### Full-suite results (JuMP and ASL, 2026-07-18/19)

`_lowered_blocks.py` (tool-neutral lowered form per problem) +
`suite_compare.py --tool jump|asl` extend both comparisons to the whole
CasADi-comparison problem set. Coverage: JuMP 20/21 (SlowPruning's 8.4M-nnz
kron block exceeds this 16 GB machine through the harness path; small-size
MATCH), ASL 19/21 (SimpleLP and SlowPruning exceed the scalar `pyomo.environ`
modeling layer; the `ASL_MODEL_LAYER=kernel` matrix_constraint layer removes
that ceiling and is small-size verified — full-size rerun pending). Every
timed problem is full-size verified bit-identical first (`max|diff|=0`
throughout; the one interesting incident: OptimalAdvertising's generator
produces a NaN coefficient at full size, and CVXPY's b carries `0·NaN = NaN`
— the blocks encode the same IEEE `b := g(0)` convention). Cold single
extraction, benchmark sizes:

| Problem | engine | CasADi | JuMP `copy_to` | ASL nl+read+eval | ASL derivative evals only |
|---|---|---|---|---|---|
| SimpleLP (10⁷) | 5.0 s | 38.1 s | 42.0 s | — | — |
| ScalarParamLP | 1.0 s | 4.6 s | 1.2 s | 81.6 s | 0.49 s |
| FullParamLP | 0.48 s | 2.4 s | 0.53 s | 39.0 s | 0.16 s |
| LeastSquares | 1.6 s | 36.6 s | 0.79 s | 58.1 s | 0.43 s |
| SimpleQP | 1.7 s | 53.7 s | 0.76 s | 77.1 s | 0.68 s |
| ParametrizedQP | 2.5 s | 771.5 s | 0.79 s | 59.9 s | 0.51 s |
| HuberRegression | 1.9 s | 42.3 s | 0.71 s | 57.5 s | 0.55 s |
| SVM + L1 | 1.8 s | 145.5 s | 0.59 s | 52.3 s | 0.35 s |
| ConeMatrixStuffing | 3.9 s | 6.1 s | 2 ms | 114 ms | 1 ms |
| SmallMatrixStuffing | 2.7 s | 2.8 s | 1 ms | 97 ms | 1 ms |
| ParamConeStuffing | 143 ms | 4 ms | 0.2 ms | 8 ms | 0.5 ms |
| ParamSmallStuffing | 210 ms | 7 ms | 0.2 ms | 10 ms | 0.5 ms |
| Yitzhaki | 0.47 s | 8.7 s | 162 ms | 20.4 s | 123 ms |
| Murray | 3.5 s | 25.9 s | 171 ms | 11.0 s | 62 ms |
| Cajas | 1.1 s | 4.9 s | 112 ms | 11.8 s | 87 ms |
| OptimalAdvertising | 6.2 s | 9.5 s | 163 ms | 7.0 s | 41 ms |
| FactorCovarianceModel | 2.0 s | 250.4 s | 371 ms | 29.4 s | 226 ms |
| ConvexPlasticity | 2.3 s | 133 ms | 17 ms | 1.1 s | 14 ms |
| TvInpainting | 1.1 s | 8.2 s | 854 ms | 45.1 s | 219 ms |
| SDP | 2.1 s | 66.4 s | 120 ms | 10.6 s | 78 ms |

The last column is the point about ASL: its *derivative* work (constraint
eval + Jacobian + Hessian-of-Lagrangian + gradient) is 0.5 ms–0.7 s on every
problem — ParametrizedQP's derivatives cost ASL 0.51 s where CasADi pays
771 s. Everything else in ASL's pipeline is ASCII .nl serialization. JuMP's
`copy_to` is the assembly floor (its macro layer, `build_s` in the results
files, is where its coefficient gathering actually happens — 7–9 s typical,
29 min at n=10⁷); the engine performs full canonicalization from the DCP atom
tree while staying within a few seconds of that floor. CasADi wins where its
paradigm wins (structured-sparse ConvexPlasticity, tiny problems); the
dense-data problems are where discovery collapses. Raw files:
`results/results_suite_{jump,asl}_compare_2026071[89].json`.

## Files

- `casadi_compare.py` — harness: Spec contract, extraction, verification, timing.
- `casadi_problems_ext_lp.py` / `_cones.py` / `_kron.py` — hand-lowered models for all
  25 suite problems (LP/QP, SOC/PSD, and the kron trio), each mirroring CVXPY's exact
  row/column order; auto-discovered by the harness.
- `run_backend_benchmarks.py` — cvxcore-backend comparison; subprocess-isolated per
  problem, streams partial results, self-contained.
- `_lowered_data.py` / `julia_compare.py` / `asl_compare.py` — the 4-problem
  SCT/JuMP/ASL comparisons (workers in `julia/`); `_lowered_blocks.py` /
  `suite_compare.py` — the tool-neutral full-suite versions (JuMP + ASL).
- `probes/` — the mechanism experiments quoted above, plus `probe_qpsol_path.py`
  (CasADi's own `qpsol`/`hessian`/`quadratic_coeff` routes vs the harness path:
  all slower, so the harness understates CasADi's cost on its own QP interface)
  and `probe_coloring_sweeps.py` (makes the sweep counts visible:
  `uni_coloring` sizes vs measured jacobian time, dense/sparse/tridiagonal),
  and `probe_sx_vs_mx.py` (SX scalar-expansion is 200–300× slower than MX for
  extraction at every size — nodes ~ nnz — and cannot reach benchmark sizes;
  MX is CasADi's best representation for this workload).
- `extras/` — an early draft (parametric Newton re-solves via the derivative oracle),
  kept for reference.

This folder backs chapter 4 of the associated master's thesis and the benchmark tables
of [cvxpy#3348](https://github.com/cvxpy/cvxpy/pull/3348).
