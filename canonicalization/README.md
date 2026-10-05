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

# Stock backends on the ignore_dpp path (upstream cvxpy only -- see below)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  BENCH_TIMEOUT=600 \
  BENCH_BACKENDS=CPP,SCIPY,COO,CPP_ND,SCIPY_ND,COO_ND,CPP_DEFAULT \
  BENCH_STRATEGIES=dpp,dpp_coo,dpp_scipy,nodpp_cpp,nodpp_scipy,nodpp_coo \
  BENCH_RATIO=CPP_ND/CPP BENCH_RATIO_WARM=nodpp_cpp/dpp \
  canonicalization/.venv-upstream/bin/python canonicalization/run_backend_benchmarks.py

# Memory comparison (peak-RSS delta per backend/strategy, one subprocess per
# measurement; also reads the diff engine's own allocation counters). Not
# timing-sensitive, no BLAS pinning needed.
BENCH_MEMORY=1 python canonicalization/run_backend_benchmarks.py
```

Knobs: `--only Name1,Name2` restricts problems; `CASADI_ITERS`/`CASADI_RESOLVES` and
`BENCH_ITERS`/`BENCH_ONLY`/`BENCH_OUT` control repetition and output. Pin BLAS to one
thread (as above) for comparable numbers.

### `ignore_dpp` fairness: the `*_ND` targets and the upstream env

The default backend table is not a like-for-like comparison. On the DPP path
`CPP`/`SCIPY`/`COO` build a **parameter → data tensor** (strictly more work, and
it only pays off across re-solves), while the diff-engine columns run with
`ignore_dpp=True` and produce concrete `(P, c, A, b)` for one parameter value.
Comparing a tensor build against a tree evaluation flatters whichever side is
doing less work.

The `*_ND` cold targets (and `nodpp_*` warm strategies) fix this: same backend,
`ignore_dpp=True`, so `EvalParams` bakes the parameters into constants and the
backend builds a plain non-parametric matrix — the same artefact the engine
produces. **They only run on upstream cvxpy.** The diff-engine fork raises
`ValueError` for an explicit `canon_backend` on the `ignore_dpp` path
(`solving_chain.py`: parametrized ≤2-D problems are force-routed to
`DIFFENGINE`), so a second, isolated environment is required:

```bash
uv venv canonicalization/.venv-upstream --python 3.14
uv pip install --python canonicalization/.venv-upstream/bin/python cvxpy==1.9.2 numpy scipy
```

It must be isolated: cvxpy 1.9.2 pins `sparsediffpy<0.4.0`, which would clobber
the local 0.7.0 dev build in `.venv`. Upstream's backends are `CPP`, `SCIPY`,
`COO` (there is no `NUMPY` backend; `RUST` needs a separate package).

Two related corrections landed with these targets:

- **`CPP` is now pinned** with an explicit `canon_backend="CPP"`. Passing nothing
  lets cvxpy switch silently to `COO` once total parameter size reaches
  `DPP_PARAM_THRESHOLD` (1000), which made the `CPP` and `COO` columns the same
  code on 4 of the 8 parametric problems in
  `results/results_backends_all_20260724_repl.txt`. The gap is large: on
  `ParamConeMatrixStuffing`, true `CPP` is 3.31 s where the unpinned column read
  0.27 s. `CPP_DEFAULT` reproduces the old behaviour when a published table
  needs replicating.
- **The `dpp?` column now uses the chain's predicate**,
  `is_dpp('dcp', quad_form_dpp='qp')`, rather than the bare `problem.is_dpp()`.
  They disagree on `ConvexPlasticity`, which older tables reported as DPP even
  though the chain sends it down the non-DPP branch — so that row's "CPP
  baseline" was itself the diff engine.

Artefact equivalence (the thing that makes the timings comparable at all) is
checked by `verify_nodpp_equivalence.py`, which dumps `(P, c, A, b)` from each
interpreter and compares them with the same `_sparse_close` MATCH discipline
`casadi_compare.py` uses.

**Engine version caveat**: all timing runs need `sparsediffpy` ≥ 0.6.1. PyPI
0.6.0 lacks the composite `param_source` refresh fix
([SparseDiffEngine #107](https://github.com/SparseDifferentiation/SparseDiffEngine/pull/107)),
invalidating warm re-solves, **and** the "Swedish" sparsity-fill gather (424ddde),
whose absence makes even cold extraction quadratic on large PSD blocks
(SemidefiniteProgramming 433.8 s vs 2.0 s, QuantumHilbertMatrix 22.2 s vs 2.1 s —
`results/results_backends_scipy_coo_20260724.txt`). `uv sync` silently downgrades
the venv to PyPI 0.6.0: check the engine version after every sync and reinstall
the local build if needed (see CLAUDE.md).

### The #125 / #3449 A/B (2026-09-19, different machine — read before comparing)

`results/results_backends_pr125_summary_20260919.md` re-measures the **warm**
half of Table 2 on two newer branches: the DIFFENGINE canon backend of
[cvxpy #3449](https://github.com/cvxpy/cvxpy/pull/3449)
(`Transurgeon/cvxpy@pr-b-ignoredpp-default`), with the engine A/B'd across
[SparseDiffEngine #125](https://github.com/SparseDifferentiation/SparseDiffEngine/pull/125)
(parameter-free subtree pruning in the refresh walk). Two venvs,
`.venv-de-base` and `.venv-de-pr125`, identical but for the sparsediffpy wheel.

**Those numbers cannot be compared to the reference table below.** They were
taken on an Apple M2 / macOS 26.5 / Python 3.13, not the 2018 Intel machine, and
against a different cvxpy baseline — the pinned `pr-c-resolve-caching` branch has
since been **deleted** from the fork, so `uv sync` no longer resolves and the pin
needs a commit SHA. Both sides of every ratio in that file were re-measured
in-run.

Headline: #125 speeds up the engine's own share of a warm re-solve by up to
**14.9×** (SVM) but the end-to-end geomean only moves 1.302× → 1.172×
(engine/tensor), because after the PR the C engine is 3–20 % of a warm
`get_problem_data` while 57–84 % is the scipy CSC rebuild of
**Finding 1** in `diffengine_issue_draft_warm_extraction.md`, still unfixed.
The write-up bounds what fixing that would buy.

Two tools added with it, both reusable:

- `verify_de_warm_equivalence.py` — the artefact gate the warm table needs:
  DIFFENGINE vs the tensor path on `(P, c, A, b)` after a parameter update,
  same `_sparse_close` MATCH discipline as `casadi_compare.py --verify`. It
  reported ALL MATCH, max|diff| = 0 in both builds.
- `probes/profile_warm_attribution.py` — splits one warm `get_problem_data`
  into engine / scipy / glue, so a remaining loss can be attributed instead of
  guessed at.

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

### The AMPL-native pipeline (binary .nl via the real translator)

With an AMPL CE/academic license (`amplpy`; key gitignored in
`.ampl_ce_license.txt`), `ASL_MODEL_LAYER=ampl` replaces Pyomo's Python-ASCII
writer with the genuine article: models built in the AMPL engine, the C
translator writing **binary** .nl (`option presolve 0` keeps the instance
verbatim), then the identical AslNLP extraction. All 19 covered problems
verify `max|diff|=0`; the timed region is pure AMPL/ASL C code (translate +
binary parse + evals — verification and aux name files are excluded; timed
writes carry no labels). Selected numbers, seconds:

| Problem | Pyomo-ASCII pipeline | AMPL-binary pipeline | of which translate / read | derivative evals |
|---|---|---|---|---|
| SimpleLP (10⁷) | — (modeling layer OOM) | 120.6 | 58.7 / 49.1 | 9.7 |
| SimpleQP | 77.1 | 35.6 | 16.2 / 16.5 | 1.98 |
| ParametrizedQP | 59.9 | 18.6 | 9.3 / 8.8 | 0.53 |
| HuberRegression | 57.5 | 17.5 | 8.5 / 8.1 | 0.48 |
| TvInpainting | 45.1 | 14.9 | 8.2 / 6.4 | 0.22 |
| FactorCovarianceModel | 29.4 | 26.1 | 11.6 / 13.3 | 0.72 |
| Yitzhaki | 20.4 | 18.3 | 3.5 / 7.4 | 0.37 |
| SDP | 10.6 | 3.1 | 1.5 / 1.5 | 0.11 |

Conclusions the two ASL pipelines support jointly:

1. **Serialization is 95–99 % of ASL's cost in every regime**; the derivative
   system itself is milliseconds everywhere (≤ 2 s even at 2·10⁷ rows,
   linear). The .nl architecture — serialize, parse, build tapes — has a
   data-volume floor no format cleverness removes: C+binary buys ~2–3× over
   Python+ASCII on dense-block problems and roughly nothing on many-row
   models (genmod's per-row instantiation dominates there).
2. **The file interface is also why AMPL scales**: it is the only modeling
   layer here that survives SimpleLP at n=10⁷ on 16 GB (it streams; the
   others materialize object graphs). Decoupling and scale are what the 1990
   design bought; per-instance latency is the price, invisible when one
   instance is solved once, decisive when canonicalization repeats.
3. **AMPL is the only compared system that rejects the NaN-poisoned
   OptimalAdvertising instance** ("can't multiply z[278] by NaN") where
   CVXPY/CasADi/JuMP/ASL-via-Pyomo propagate it — arguably the soundest
   policy; documented alongside the upstream non-finite-data reports.
   SlowPruning's 8.4M-nnz kron block OOMs a fourth modeling layer (the AMPL
   engine) on this machine; only the expression-graph systems (CasADi,
   engine) represent it compactly.
4. Interop footnote: AMPL's `.row`/`.col` name files %g-shorten round indices
   (`z[100000]` → `z[1e+05]`); verification canonicalizes numeric bracket
   indices (`asl_compare._canon_name`).

Raw file: `results/results_suite_asl_ampl_20260719.json`. Readable
per-problem `.mod`/`.dat` artifacts (models formulated algebraically, data
read timed as AMPL work) are the planned v2 of this pipeline; see
`ampl_models/SimpleQP.mod` for the exemplar format.

### Declared-structure systems, finalized: JuMP vs AMPL with the .nl split

The thesis table for the two declared-structure systems, full suite. The AMPL
pipeline is decomposed into its three stages so the .nl serialization cost and
the derivative work are separate columns: **write** = the C translator
emitting binary .nl (genmod + write), **read** = ASL's binary parse + tape
build, **derivative evals** = constraint eval + Jacobian + Hessian-of-Lagrangian
+ gradient. JuMP's `copy_to` has no such split — it is a single in-memory
assembly step. Seconds; same runs as above
(`results/results_suite_jump_compare_20260719.json`,
`results/results_suite_asl_ampl_20260719.json`).

| Problem | JuMP `copy_to` | AMPL .nl write | AMPL .nl read | AMPL derivative evals | AMPL total |
|---|---|---|---|---|---|
| SimpleLP (10⁷) | 42.0 | 58.7 | 49.1 | 9.72 | 120.6 |
| ScalarParamLP | 1.16 | 11.1 | 8.9 | 0.32 | 21.1 |
| FullParamLP | 0.53 | 4.5 | 4.5 | 0.17 | 9.1 |
| LeastSquares | 0.79 | 8.7 | 8.3 | 0.42 | 18.9 |
| SimpleQP | 0.76 | 16.2 | 16.5 | 1.98 | 35.6 |
| ParametrizedQP | 0.79 | 9.3 | 8.8 | 0.53 | 18.6 |
| HuberRegression | 0.71 | 8.6 | 8.1 | 0.48 | 17.5 |
| SVM + L1 | 0.59 | 25.8 | 24.0 | 1.06 | 50.0 |
| ConeMatrixStuffing | 2 ms | 44 ms | 96 ms | 3 ms | 31 ms |
| SmallMatrixStuffing | 1 ms | 11 ms | 27 ms | 1 ms | 39 ms |
| ParamConeStuffing | 0.2 ms | 5 ms | 54 ms | 1 ms | 12 ms |
| ParamSmallStuffing | 0.2 ms | 6 ms | 54 ms | 1 ms | 13 ms |
| Yitzhaki | 0.16 | 3.5 | 7.4 | 0.37 | 18.3 |
| Murray | 0.17 | 6.0 | 5.5 | 0.20 | 11.7 |
| Cajas | 0.11 | 6.5 | 7.0 | 0.25 | 13.8 |
| OptimalAdvertising | 0.16 | —¹ | — | — | — |
| FactorCovarianceModel | 0.37 | 11.6 | 13.3 | 0.72 | 26.1 |
| ConvexPlasticity | 17 ms | 0.20 | 0.22 | 17 ms | 0.41 |
| TvInpainting | 0.85 | 8.2 | 6.4 | 0.22 | 14.9 |
| SDP | 0.12 | 1.5 | 1.5 | 0.11 | 3.1 |
| SlowPruning | —² | — | — | — | — |

¹ AMPL's translator rejects the NaN-poisoned instance (see conclusion 3).
² Both stacks OOM on the 8.4M-nnz kron block through this 16 GB machine's
harness path (small-size verified MATCH for JuMP).

Reading notes: the stage splits come from the instrumented draw while totals
are suite medians, so rows need not sum exactly — visible on Yitzhaki (an
instrumented draw faster than the median) and inverted on the millisecond
stuffing rows, where the instrumented draw is cold (file-cache) and the
median of repeats is warmer than its own decomposition. Excluded build stages
per the fairness rules: JuMP's macro build (7–9 s typical, 29 min at n=10⁷)
is where its coefficient gathering happens; AMPL's `build_s` blends harness
prep with amplpy data transfer. The headline contrast: JuMP's in-memory
assembly is the floor everywhere it fits in memory, AMPL pays a serialization
tax of ~1 s per 10–20 MB of .nl in each direction, and on every single
problem the actual *derivative* work (last-but-one column) is a rounding
error next to that tax.

### End-to-end model→solver handoff: `jump_e2e/` (CVXPY vs idiomatic JuMP)

`jump_e2e/` charges each stack the full user pipeline up to the moment SCS
could start — native user model → canonicalization → SCS input form
materialized; **the solver is never run by the timed comparison** — on
identical raw arrays (`jump_e2e/problems.py`, seeds byte-identical to the
Specs). CVXPY side: fresh Problem + `get_problem_data(solver=SCS)`; JuMP
side: macro build + `MOI.copy_to` into the bridged `SCS.Optimizer` cache
(SCS.jl's own zero-based-CSC input form). One JuMP model per file under
`jump_e2e/models/`, each docstring stating the CVXPY original and the
epigraph rewriting used, so the mathematical equivalence is checkable by
hand. The two sides' SCS inputs are equivalent but not bit-identical
(different bridge/reformulation choices), so correctness is gated on optimal
values via the opt-in `--verify`, the only mode that actually solves. See
`jump_e2e/README.md` for commands and fairness rules.

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
  all slower, so the harness understates CasADi's cost on its own QP interface;
  its path E measures the other commonly-suggested route, "pass DM matrices
  through the low-level `conic` interface": no extraction happens there —
  `conic` takes the numeric `(H, A)` the other paths exist to compute, so the
  canonicalization has simply moved to the user (0.6 s of numpy assembly at
  n=800, declared-structure extraction by hand) — and even then qrqp's
  structural setup on the dense declared pattern costs 30.8 s at n=800, 7×
  the harness's entire extraction path; the harness already uses `DM` for
  every constant in the user's data anyway. Results in
  `results/results_probe_qpsol_path_20260724.txt`)
  and `probe_coloring_sweeps.py` (makes the sweep counts visible:
  `uni_coloring` sizes vs measured jacobian time, dense/sparse/tridiagonal),
  and `probe_sx_vs_mx.py` (SX scalar-expansion is 200–300× slower than MX for
  extraction at every size — nodes ~ nnz — and cannot reach benchmark sizes;
  MX is CasADi's best representation for this workload), and
  `probe_hierarchical_trace.py` (stage split detection / coloring /
  accumulation on the exact ParametrizedQP rows: the paper's Example-1
  tridiagonal control reproduces (3 colors), detection+coloring stay minor,
  and the minutes sit in accumulation = one AD sweep per color with
  colors = the dense block's full column dimension; constant-DM and
  parameter-MX A cost the same. Corollary for the commonly-suggested
  mitigations: hand-supplying the pattern à la `Sparsity.dense` could at
  best remove detection — <1 % of the bill at suite size (8.2 s of ~892 s
  on ParametrizedQP) — because coloring a dense block still yields n colors
  and n sweeps; the only true bypass is supplying the Jacobian coefficients
  themselves, which is declared-structure extraction, i.e. the other
  paradigm in this comparison).
- `jump_e2e/` — end-to-end user-model→SCS-solution comparison vs idiomatic
  JuMP; one model file per problem (see `jump_e2e/README.md`).
- `extras/` — an early draft (parametric Newton re-solves via the derivative oracle),
  kept for reference.

This folder backs chapter 4 of the associated master's thesis and the benchmark tables
of [cvxpy#3348](https://github.com/cvxpy/cvxpy/pull/3348).

## Final-deposit experiments of the thesis (October 2026)

The jury and the examiner asked for replicates and p-values, an ablation of the two
contributions, non-DPP problems, sensitivity to size and density, instance dimensions,
a phase breakdown, and a stated correctness check. All of it is produced by:

- `thesis_final_benchmarks.py`: the runner (timing, memory and correctness modes).
  It reuses the helpers of `run_backend_benchmarks.py`. Each replicate runs in a fresh
  subprocess, with up to `BENCH_REPS=20` samples per measurement, or 5 when the first
  sample exceeds 30 s. Every output file records the machine, the versions and the
  engine's BLAS.
- `thesis_extra_problems.py`: families beyond the suite. These are the
  DPP / non-DPP / parameter-free variants (`lasso`, `factor_cov`) and scalable copies of
  HuberRegression, OptimalAdvertising and FactorCovarianceModel, all with seeds.
- `thesis_stats.py`: the summary. It reports mean ± std, Mann–Whitney U per problem
  with Holm correction, a Wilcoxon signed-rank test over problems, bootstrap CIs of the
  geometric mean, ablation contrasts and phase shares.
- `run_thesis_final.sh`: the whole sweep, writing into `results/thesis/`.

The ablation switches and the phase timers live in the CVXPY fork, on branch
`Transurgeon/cvxpy@thesis-ablation` (based on `b7bb765`, PR #3449):

| variable | effect |
|---|---|
| `DIFFENGINE_NO_DENSE=1` | constant matrices go to the sparse CSR bindings, so no permuted dense blocks are built from constant data |
| `DIFFENGINE_REBUILD=1` | conversion and symbolic pass repeated on every parameter update (no symbolic/numeric split) |
| `DIFFENGINE_PROFILE=1` | per-phase times in `extractor.PHASE_TIMES` |

Environments (both gitignored):

```bash
uv venv canonicalization/.venv-thesis --python 3.13
uv pip install --python canonicalization/.venv-thesis/bin/python \
  numpy==2.5.3 scipy==1.18.1 clarabel==0.11.1 threadpoolctl
# SparseDiffPy checkout whose SparseDiffEngine submodule is at 64f7432 (#125 merged)
uv pip install --python canonicalization/.venv-thesis/bin/python --no-deps <SparseDiffPy>
uv pip install --python canonicalization/.venv-thesis/bin/python -e <cvxpy@thesis-ablation>
uv venv canonicalization/.venv-upstream --python 3.13
uv pip install --python canonicalization/.venv-upstream/bin/python \
  cvxpy==1.9.2 numpy==2.5.3 scipy==1.18.1 clarabel==0.11.1 threadpoolctl
```
