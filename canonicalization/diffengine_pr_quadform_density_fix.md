# PR plan: density-route constant dense P in the quad-form converter

Target repo/branch: `Transurgeon/cvxpy@pr-c-resolve-caching` (the diff-engine
fork behind cvxpy PR #3448). One file:
`cvxpy/reductions/solvers/nlp_solvers/diff_engine/converters.py`. No engine
(C) or binding changes — the sparse quad_form binding this routes to is
already exercised in production (e.g. by ConvexPlasticity's `S_sparsed`).

## Problem

`convert_symbolic_quad_form` picks the engine binding for a constant P by its
**container**: scipy-sparse → `"sparse"` binding, dense ndarray → `"dense"`
binding, unconditionally. A dense-stored but sparse-in-content P (the
canonical case: `D = H * np.eye(N)` in the ConvexPlasticity benchmark,
density 1/N) becomes a fully dense N×N quadratic form, and the engine derives
a dense N² Hessian block: 9,048,000 stored entries at N = 3000 where the true
pattern is 33,000. Measured consequences on that problem: cold 2.07 s vs
0.35 s for every upstream backend, +1.2 GB peak RSS vs the measurement floor.
Full analysis: `diffengine_issue_draft_warm_extraction.md` (Finding 2,
2026-08-13 revision) in the Jacobian-accumulation repo.

`convert_matmul` in the same file already solved the identical problem for
constant matmul operands (the `density < s.SPARSE_DENSITY_THRESHOLD` route,
threshold 0.05 in `cvxpy/settings.py`); the quad-form converter simply never
got the same test.

## Change

In `convert_symbolic_quad_form`, constant-P handling (the code after the
`if P.parameters():` branch, currently):

```python
        P_val = P.value
        if sparse.issparse(P_val):
            P_csr = P_val.tocsr()
            return _diffengine.make_quad_form(
                None, x_c, "sparse",
                P_csr.data.astype(np.float64),
                P_csr.indices.astype(np.int32),
                P_csr.indptr.astype(np.int32),
                P_csr.shape[0], P_csr.shape[1])
        P_dense = to_dense_float(P_val)
        return _diffengine.make_quad_form(
            None, x_c, "dense", P_dense.flatten(order='F'), n)
```

becomes:

```python
        P_val = P.value
        if not sparse.issparse(P_val):
            P_dense = to_dense_float(P_val)
            # A constant dense P that is mostly zeros: route it to the sparse
            # quad_form binding to avoid building a dense Hessian block.
            # Mirrors convert_matmul; constants only, so no parametric
            # sparsity pattern is ever frozen (parametric P took the branch
            # above).
            density = np.count_nonzero(P_dense) / P_dense.size if P_dense.size else 1.0
            if density < s.SPARSE_DENSITY_THRESHOLD:
                P_val = sp.csr_array(P_dense)
        if sparse.issparse(P_val):
            P_csr = P_val.tocsr()
            P_csr.eliminate_zeros()
            return _diffengine.make_quad_form(
                None, x_c, "sparse",
                P_csr.data.astype(np.float64),
                P_csr.indices.astype(np.int32),
                P_csr.indptr.astype(np.int32),
                P_csr.shape[0], P_csr.shape[1])
        return _diffengine.make_quad_form(
            None, x_c, "dense", P_dense.flatten(order='F'), n)
```

Notes:
- `s` (cvxpy.settings) is already imported in this module (used by
  `convert_matmul`); match whatever the module's scipy.sparse alias is
  (`sparse` above per current imports).
- `eliminate_zeros()` also drops explicit zeros arriving in *already-sparse*
  P (ConvexPlasticity's `S_sparsed` carries 6 stored zeros per 4×4 block from
  `scipy.sparse.block_diag` of dense blocks). Cheap, and the engine then sees
  the true pattern. If bit-identical P output against the pre-change fork is
  a review concern, split this into its own commit so the density route and
  the zero-drop are separately revertable.
- Genuinely dense P (density ≥ 0.05) is untouched — same dense binding, same
  BLAS path. The permuted-dense design is not affected; this is the same
  content-over-container classification the PD detector applies in the
  opposite direction.

## Tests to add (fork test suite)

1. Unit, converter-level: `quad_form(x, H * np.eye(n))` for n large enough
   that density < 0.05 (n ≥ 21) → `get_problem_data(CLARABEL,
   ignore_dpp=True)` returns `data["P"]` with stored nnz == n (diagonal), not
   n². Companion case with a dense random PSD P (density ≥ 0.05) still
   returns the dense-path result, stored nnz == expected full pattern.
2. Values regression: same problems, `data["P"].toarray()` equal (tolerance)
   between the two routes — force the dense path by monkeypatching the
   threshold to 0 if a direct comparison hook is awkward.
3. Parametric guard: quad_form with parametric P still takes the
   `P.parameters()` branch (no sparsification of parametric data) — assert
   route, not just values.

## Verification against the benchmark harness

(Repo `/Users/trevorhegarty/Documents/Jacobian-accumulation`; rebuild/install
per its CLAUDE.md engine-version gotcha, then:)

1. Correctness gate: `verify_nodpp_equivalence.py` on ConvexPlasticity plus a
   few quad-objective problems (SimpleQP, HuberRegression,
   FactorCovarianceModel) — `_sparse_close` MATCH against
   `canon_backend="CPP"`. Upstream's P for ConvexPlasticity is clean (11N
   stored), so the fixed engine handoff should now *agree more*, not less.
2. Targeted timing: `BENCH_ONLY=ConvexPlasticity BENCH_BACKENDS=CPP,DIFFENGINE`
   — expect cold to collapse from 2.07 s toward the shared-reductions floor
   (upstream backends sit at 0.35 s); measure, don't assume the exact figure.
3. Targeted memory: `BENCH_MEMORY=1 BENCH_ONLY=ConvexPlasticity ...` — expect
   the +1.2 GB engine delta to drop to tens of MB.
4. Full-suite non-regression (timing, all 25 problems): only problems passing
   a constant dense low-density P through quad_form should change. Watch the
   other quad-objective rows for noise-level movement only.

## Thesis note (do not fold into the PR)

The thesis benchmark chapter reports the engine as measured (ratio 5.97 row,
censored memory point); if this fix lands and the suite is re-run before
submission, `tab:oneshot-compile`, the memory-profile plateau (24/25), and
the §4.1 loss-shape sentence all need refreshing together.
