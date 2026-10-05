# SparseDiffEngine #125 on the parametric re-solve table

**Question.** Does [SparseDiffEngine #125](https://github.com/SparseDifferentiation/SparseDiffEngine/pull/125)
(*prune parameter-free subtrees; reach `param_source` through the children hook*),
measured on the new DIFFENGINE canon backend from
[cvxpy #3449](https://github.com/cvxpy/cvxpy/pull/3449), make the diff engine's
warm re-solve faster than CVXPY's parameter-tensor path on **all** parametric
problems of the suite?

**Answer: no.** #125 is a real and sometimes large win — up to **14.9×** on the
engine's own share of a warm re-solve — but it moves the end-to-end geometric
mean only from **1.302× to 1.172×** (engine / tensor, <1 favours the engine).
The engine still loses on 5 of the 7 problems that have a tensor baseline. The
reason is measured, not guessed: after #125 the C engine is **3–20 %** of a warm
`get_problem_data`, and **57–84 %** is scipy re-deriving a CSC sort order that
cannot change between calls — Finding 1 of
`../diffengine_issue_draft_warm_extraction.md`, still unfixed.

## Environment

| | |
|---|---|
| machine | Apple M2, macOS 26.5.1, 24 GB |
| python | 3.13.5 |
| cvxpy | `0.1.0.dev4043+gb7bb76591` = `Transurgeon/cvxpy@pr-b-ignoredpp-default` (**PR #3449**, `b7bb765`) |
| sparsediffpy | 0.7.0 dev, built from `SparseDiffPy@9e0246e` |
| engine `engbase` | `SparseDiffEngine@4dbb53b` (`main`, the merge-base of #125) |
| engine `pr125` | `SparseDiffEngine@b9b5771` (`prune-param-free-refresh`, #125 head) |
| numpy / scipy / clarabel | 2.5.3 / 1.18.1 / 0.11.1 |
| benchmark suite | `cvxpy/benchmarks@77c7fa4` |
| BLAS | pinned to 1 thread; runs strictly sequential |
| repetition | `BENCH_ITERS=15`, 3 independent replicate sweeps per build, median of the per-replicate means |

**These numbers are not comparable to `results_thesis_final_20260711_pr102.txt`.**
That table was measured on an Intel i7-8850H / macOS 15.7 / Python 3.14 against
the `pr-c-resolve-caching` fork (a branch that has since been deleted from
`Transurgeon/cvxpy`). Both the machine and the cvxpy baseline changed here, so
every column below was re-measured in-run.

## Gates passed before timing

- `verify_de_warm_equivalence.py` — **ALL MATCH, max|diff| = 0** on every
  artefact `(P, c, A, b)` of all 7 DPP problems, in *both* builds, on a warm
  re-compile after the parameter values changed. The timing table is over
  bit-identical matrices.
- cvxpy `test_diffengine_backend.py` + `test_ignore_dpp.py`: 50 passed, 1
  skipped, in both builds. Engine `ctest` green at `b9b5771`.
- The `dpp` / `dpp_coo` / `dpp_scipy` columns are identical across the two
  builds (e.g. SVM 0.0389 / 0.0389). They share one cvxpy and must not move
  when only the engine wheel changes — so the A/B is not contaminated.
- Replicate spread `(max−min)/median` stayed under 10 % for every
  (build, problem, strategy). No conclusion below rests on a difference
  inside that spread.

## Table 2 — warm re-solve (s), median of 3 replicates × 15 iterations

`dpp` = cached DPP parameter→data tensor (the bar). `de_cached` =
`canon_backend="DIFFENGINE"` on the DPP path. Ratio < 1 favours the engine.

| benchmark | `dpp` | `de_cached` base | `de_cached` #125 | ratio base | ratio #125 | #125 speedup |
|---|---:|---:|---:|---:|---:|---:|
| ParamSmallMatrixStuffing | 0.0018 | 0.0005 | 0.0005 | **0.25×** | **0.25×** | 1.00× |
| ParamConeMatrixStuffing | 0.0010 | 0.0003 | 0.0003 | **0.30×** | **0.30×** | 1.00× |
| SimpleFullyParametrizedLP | 0.0472 | 0.0583 | 0.0530 | 1.24× | 1.12× | 1.10× |
| ParametrizedQPBenchmark | 0.3028 | 0.4272 | 0.4260 | 1.41× | 1.41× | 1.00× |
| SimpleScalarParametrizedLP | 0.0648 | 0.1156 | 0.1046 | 1.80× | 1.62× | 1.10× |
| FactorCovarianceModel | 0.0209 | 0.0990 | 0.0813 | 4.73× | 3.90× | 1.22× |
| SVMWithL1Regularization | 0.0389 | 0.2176 | 0.1539 | 5.60× | 3.95× | **1.41×** |
| **geometric mean** | | | | **1.302×** | **1.172×** | |

`ConvexPlasticity` has **no tensor baseline in this environment** and is not in
the mean. It is non-DPP under the chain's own predicate, and #3449 routes the
non-DPP branch to DIFFENGINE, where an explicit `canon_backend` now raises. Its
`dpp` column *is* the diff engine (0.2057 s vs `diffengine` 0.2052 s), so the
row is a #125 A/B only: 0.2047 → 0.2052 s, no change.

### The bar moved

The engine's own times track the CPU change from the Intel reference
(SVM 0.530 → 0.217 s, ≈2.4×; FactorCov 0.277 → 0.099 s, ≈2.8×). The tensor path
improved far more than hardware explains (SVM 0.292 → 0.039 s, ≈7.5×;
FactorCov 0.144 → 0.021 s, ≈6.9×). Whatever landed in cvxpy master between
`pr-c-resolve-caching` and #3449 made the DPP tensor apply substantially faster.
That, not an engine regression, is why the thesis table's 0.78× geomean reads
1.30× here before #125 is applied at all.

## Where the warm second actually goes (cProfile, one warm `get_problem_data`)

`engine` = `update_params` + `eval_jacobian_vals` + `eval_hessian_vals_coo` +
forward/gradient. `scipy` = `coo_tocsr`, `csr_sort_indices`, `csr_tocsc`,
`csr_row_index`, `sum_duplicates` — the conversion `extractor.py` re-runs from
raw triplets on every `extract()`. `glue` = cvxpy's chain and `solver.apply`.

| benchmark | total (s) | engine base | engine #125 | **engine speedup** | scipy | share after #125 |
|---|---:|---:|---:|---:|---:|---|
| SVMWithL1Regularization | 0.1643 | 0.0685 | 0.0046 | **14.9×** | 0.1093 | eng 3 % / scipy 67 % / glue 31 % |
| FactorCovarianceModel | 0.0846 | 0.0211 | 0.0030 | **7.0×** | 0.0573 | eng 4 % / scipy 68 % / glue 29 % |
| SimpleFullyParametrizedLP | 0.0570 | 0.0123 | 0.0062 | 2.0× | 0.0323 | eng 11 % / scipy 57 % / glue 32 % |
| SimpleScalarParametrizedLP | 0.1069 | 0.0221 | 0.0121 | 1.8× | 0.0621 | eng 11 % / scipy 58 % / glue 31 % |
| ParametrizedQPBenchmark | 0.4335 | 0.0838 | 0.0850 | 1.0× | 0.2492 | eng 20 % / scipy 57 % / glue 23 % |
| ConvexPlasticity | 0.2104 | 0.0242 | 0.0239 | 1.0× | 0.1770 | eng 11 % / scipy 84 % / glue 5 % |

The `scipy` column is **unchanged between the two builds** (SVM 0.1090 →
0.1093; ConvexPlasticity 0.1767 → 0.1770) — an independent check that the only
thing that moved is the engine.

Two problems get nothing from #125, for different reasons:

- **ConvexPlasticity** — the engine bucket is dominated by
  `problem_eval_hessian_vals_coo` (0.0211 of 0.0239 s) over the padded dense
  quadratic block of Finding 2 in the same draft. Those entries *do* depend on
  parameters, so there is no parameter-free subtree to prune.
- **ParametrizedQPBenchmark** — the bucket is `problem_constraint_forward`
  (0.0442 s) plus `eval_jacobian_vals` (0.0378 s), both on parameter-reachable
  nodes. #125 prunes the refresh walk, not the evaluation of nodes a parameter
  genuinely reaches.

## What would close the gap

Taking the scipy bucket to zero (the optimistic bound for Finding 1: cache
`indptr`/`indices`/`perm` at `build()` and gather into them, so no conversion,
sort or dedup runs per call):

| benchmark | `dpp` | #125 total | minus scipy | implied ratio |
|---|---:|---:|---:|---:|
| SimpleFullyParametrizedLP | 0.0472 | 0.0570 | 0.0247 | **0.52×** |
| ParametrizedQPBenchmark | 0.3028 | 0.4335 | 0.1843 | **0.61×** |
| SimpleScalarParametrizedLP | 0.0648 | 0.1069 | 0.0448 | **0.69×** |
| FactorCovarianceModel | 0.0209 | 0.0846 | 0.0273 | 1.31× |
| SVMWithL1Regularization | 0.0389 | 0.1643 | 0.0550 | 1.41× |

That flips three more problems to wins and brings the remaining two from ≈4×
to ≈1.4×, where the residue is `glue` (29–31 % on both) rather than anything
the engine does. It is an upper bound — a cached-permutation gather is not
free — but it is the order of the remaining opportunity, and it is roughly
**ten times** what #125 had available to win.

## Reproducing

```bash
# two venvs, identical cvxpy, one wheel apart
uv venv canonicalization/.venv-de-base  --python 3.13
uv venv canonicalization/.venv-de-pr125 --python 3.13
uv pip install --python <venv>/bin/python numpy scipy clarabel casadi pytest \
  "cvxpy @ git+https://github.com/Transurgeon/cvxpy.git@b7bb76591be1a585784fe18b2deeee32b3f33bcb"
# sparsediffpy built from SparseDiffPy with SparseDiffEngine at 4dbb53b / b9b5771
uv pip install --python <venv>/bin/python --force-reinstall --no-deps <wheel>

# gate
<venv>/bin/python canonicalization/verify_de_warm_equivalence.py

# one replicate (repeat for rep1..3 and both builds; never two at once)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
BENCH_COLD=0 BENCH_ITERS=15 BENCH_TIMEOUT=900 BENCH_BACKENDS=DE_DPP \
BENCH_STRATEGIES=dpp,dpp_coo,dpp_scipy,diffengine,de_cached \
BENCH_RATIO_WARM=de_cached/dpp \
BENCH_ONLY=FactorCovarianceModel,ConvexPlasticity,ParamConeMatrixStuffing,ParamSmallMatrixStuffing,SimpleScalarParametrizedLPBenchmark,SVMWithL1Regularization \
BENCH_OUT=results/results_backends_pr125_<build>_rep<k>_20260919.txt \
  <venv>/bin/python canonicalization/run_backend_benchmarks.py

# attribution
<venv>/bin/python canonicalization/probes/profile_warm_attribution.py --top
```

`SimpleFullyParametrizedLPBenchmark` and `ParametrizedQPBenchmark` run in a
second pass with `BENCH_STRATEGIES=dpp,diffengine,de_cached` (files `*_rep<k>big_*`).
Their explicit COO/SCIPY tensor builds are pathological — the committed upstream
reference records `TIMEOUT(>1200s)` and `crashed/killed` for them — which is the
same split `results_backends_fork_bigparam_20260808.txt` used.

## Raw files

`results_backends_pr125_{engbase,pr125}_rep{1,2,3}[big]_20260919.{txt,json}`.
Every table above is re-derivable from the `.json` siblings with
`python canonicalization/summarize_pr125_replicates.py canonicalization/results`.

## Note on the runner's legend

`run_backend_benchmarks.py`'s `TARGET_LEGEND` still says `diffengine` means "the
capsule is rebuilt per solve" and `nodpp_*` means "every re-solve is a FULL
recompile". Under #3449 that is no longer true — keeping parameters symbolic and
caching the compiled program on the non-DPP route is precisely what the PR does,
and the `diffengine` column here is a cached warm path, within 1 % of
`de_cached` on every row. The runner's timing code was deliberately left
untouched so these tables stay comparable in shape to the committed ones.

## Unrelated breakage found on the way

`pyproject.toml` pins `cvxpy @ git+…/Transurgeon/cvxpy.git@pr-c-resolve-caching`.
That branch no longer exists on the fork (`git ls-remote` shows only
`pr-a-diffengine-backend`, `pr-a-full-parametric`, `pr-b-ignoredpp-default`), so
`uv sync` cannot resolve. Left as-is here; it wants a commit SHA rather than a
branch name.
