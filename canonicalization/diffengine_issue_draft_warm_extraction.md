# Draft issue: warm extraction is 79 % scipy CSC re-canonicalization, and
# ConvexPlasticity's engine-path P carries 274× explicit-zero padding

Target: the diff-engine cvxpy fork (`Transurgeon/cvxpy@pr-c-resolve-caching`),
`cvxpy/reductions/solvers/nlp_solvers/diff_engine/extractor.py` (Finding 1) and
`.../diff_engine/converters.py` (Finding 2). Finding 1 found 2026-07-24 while
analyzing the benchmark suite's engine-loss cases; measured on commit
`ca08a26d1` + local sparsediffpy 0.7.0.dev (engine main), Python 3.14.4,
numpy 2.4.6, scipy 1.17.1, BLAS pinned to 1 thread. Finding 2 root-caused
2026-08-13 (this rewrite supersedes the 2026-07-24 version of Finding 2, which
wrongly blamed a shared upstream lowering — see the correction note below).

Reproducer: `cvxpy_benchmarks` `high_dim_convex_plasticity.ConvexPlasticity`
(N = 3000; 33,000 vars after lowering), `get_problem_data(solver=CLARABEL,
ignore_dpp=True)`, warm re-solve after changing parameter values.

## Finding 1 — the warm path rebuilds and re-sorts the CSC every call

`DiffEngineExtractor.build()` already captures the COO structure exactly once
(`jac_structure`, `hess_structure` — the docstrings say so), but
`_build_hessian_csc()` / `_build_jacobian_csc()` still hand scipy raw
`(vals, (rows, cols))` triplets on **every** `extract()`. scipy then re-runs
its full canonicalization — `coo_tocsr`, `csr_sort_indices`,
`sum_duplicates` — on a fixed pattern each warm solve.

cProfile of one warm `get_problem_data` on ConvexPlasticity (0.640 s total):

| stage | seconds | share |
|---|---:|---:|
| `csr_sort_indices` | 0.324 | 51 % |
| `coo_tocsr` (×2) | 0.158 | 25 % |
| `csr_sum_duplicates` + misc scipy | ~0.02 | 3 % |
| **engine `problem_eval_hessian_vals_coo`** | **0.073** | **11 %** |
| other (gradient, forward, glue) | ~0.06 | 10 % |

The C engine's actual work is 11 % of the warm bill; ~79 % is scipy
re-deriving a sort order that cannot change between calls.

**Proposed fix**: at `build()` time, canonicalize the pattern once — compute
`indptr`, `indices`, and the permutation `perm` mapping the engine's COO value
order to canonical CSC order (plus `np.add.reduceat` segment boundaries if
duplicate coordinates exist; on this problem there are none — COO count ==
CSC nnz). Each `extract()` then constructs the CSC directly:
`csc_matrix((full_vals[perm], cached_indices, cached_indptr), shape=...)`
with no conversion, sort, or dedup. Same applies to the Jacobian path (minor
here — 90,000 entries, 3 ms — but free to do identically). Expected effect on
this problem: warm 0.74 s → ~0.15 s. Every warm re-solve in the suite pays
some version of this tax; it is just largest where nnz is large.

## Finding 2 — the quad-form converter trusts the container: a dense-stored
## diagonal becomes a dense N×N Hessian block

**Correction (2026-08-13).** The 2026-07-24 version of this finding claimed the
padded P was "NOT engine-specific" because "the default CPP backend's
`data["P"]` has the identical 9,048,000 stored entries". That measurement was
contaminated by the mislabeled baseline documented in the README: on this
fork, ConvexPlasticity's old "CPP" rows were themselves the diff engine (the
chain's `is_dpp` predicate routed the problem down the non-DPP branch).
Re-measured on genuinely upstream cvxpy 1.9.2 (`.venv-upstream`), `data["P"]`
is **clean**: stored == true nonzeros == 11N, zero explicit zeros, at every
size probed (N = 25/50/100). The padding is engine-path-only.

**Mechanism, traced.** The model builds its hardening term as
`D = H * np.eye(N)` — a *dense* ndarray whose content is a diagonal — and
passes it to `cp.quad_form(p - p_old, D)`. In
`converters.py::convert_symbolic_quad_form`, the constant-P handling branches
on the **container**: scipy-sparse P → the sparse quad_form binding; dense
ndarray P → `make_quad_form(..., "dense", P_dense.flatten(order='F'), n)`
unconditionally. So the engine receives a fully dense N×N quadratic form and
derives a dense N² Hessian block over `p`. The elastic term's `S_sparsed`
(scipy `block_diag`) takes the sparse branch and contributes 16 stored
entries per 4×4 block (10 true + 6 explicit zeros).

This decomposes the measured pattern exactly, verified by a scaling probe at
N = 25/50/100 (fork env vs `.venv-upstream`):

- stored = N² (dense-path D) + 16N (S blocks) → 9,048,000 at N = 3000 ✓
- explicit zeros = (N² − N) (off-diagonal of D) + 6N (S block zeros)
  = N² + 5N → 9,015,000 at N = 3000 ✓
- true nonzeros = 10N (S upper structure) + N (D diagonal) = 11N = 33,000 ✓

**Proposed fix — the same density test `convert_matmul` already has.** The
matmul converter routes a constant dense-but-mostly-zero matrix to the sparse
CSR binding (`density < s.SPARSE_DENSITY_THRESHOLD`, = 0.05, restricted to
parameter-free constants so no pattern is frozen). `convert_symbolic_quad_form`
lacks the identical test in its dense-constant branch; adding it routes
`D` (density 1/N ≈ 0.0003) to the existing sparse quad_form binding. No engine
(C) changes: the sparse binding is already exercised by `S_sparsed`. A
genuinely dense P (density ≥ 0.05) keeps the dense path and its BLAS kernels,
so the permuted-dense design is untouched — this is the same
content-not-container classification the PD detector applies in the other
direction. Optional companion: `eliminate_zeros()` on the sparse branch's CSR
(drops S's 6N stored zeros; harmless but free).

User-side workaround (footnote, not the fix): the model could pass
`scipy.sparse.eye(N)`; the suite measures backends on problems as users write
them, and users write `np.eye`.

## Benchmark context (why this matters)

ConvexPlasticity is the engine's single worst loss in the 25-problem suite:
cold 2.07 s vs 0.35 s for every upstream backend (ratio 5.97, the only ratio
above 1.4 in `tab:oneshot-compile`), and the engine's memory delta is +1.2 GB
where upstream's backends sit at the 3–4 MB measurement floor. Both findings
together account for essentially all of it: the warm bill is 79 % scipy
re-canonicalization of a pattern 274× larger than the mathematics requires,
11 % engine evaluation of that same padded pattern. Neither cost is Jacobian
accumulation. With Finding 2 fixed the pattern collapses from 9,048,000 to
~33k–50k stored entries, and cold time, warm time, and the 1.2 GB all follow;
Finding 1 then removes the per-call CSC tax that remains for every problem in
the suite.
