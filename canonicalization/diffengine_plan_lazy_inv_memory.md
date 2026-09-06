# SparseDiffEngine: eliminate O(blocks × n_vars) inverse-map memory in permuted_dense

Target repo: `/Users/trevorhegarty/Documents/SparseDiffEngine` (branch `main`, HEAD 4172c5e, clean).
This plan is self-contained — hand it to a session working in that repo.

## Context

Every `permuted_dense` (PD) unconditionally allocates two dense inverse-permutation
arrays sized by its **global** dimensions: `col_inv` (length `n` = global column count)
and `row_inv` (length `m`), at `src/utils/permuted_dense.c:605-606`, filled mostly
with `-1`. For Jacobians of dense-vector matmuls over a large variable (`D @ ones(n)`,
`D.T @ ones(m)`, `vec @ D[i,:]`), the kron path (`left_matmul.c:131` →
`BA_dense_kron_matrices_alloc` → `kron_alloc_blockwise` → `BA_pd_csc_alloc` at
`permuted_dense_linalg.c:274`) creates **p independent PD blocks** (p = m or n), each
carrying its own full-length `col_inv` AND a `row_inv` of length `m0·p` — total
`O(p·n_vars + p²·m0)` ints per node, replicated again by every
`copy_sparsity`/`transpose`/`reshape` chain node. Measured on the CVXPY
OptimalAdvertising benchmark (m=250, n=1000): **5.9 GB engine-internal peak where
true nnz needs ~5 MB** (law: ~20 B × (m+n) × mn, verified by 6-point scaling probe
in the Jacobian-accumulation repo). Values and the final CSR pattern are exact-nnz
throughout — the entire cost is this index metadata, retained live after
`problem_init_jacobian`. Same family plausibly explains the CVaR (8.9 GB) and SDP
multi-GB engine peaks in `results_backends_memory_20260802.txt`.

Fix strategy (grounded in a full consumer audit, below): make the inv arrays
**lazy**, convert init-path consumers to **sorted merge/binary-search scans** (the
perms are asserted strictly increasing — `permuted_dense.c:573-588`), and precompute
the one fill-path lookup whose indices are fixed. The kron output blocks' inv arrays
are then never materialized at all: the kron kernels read only the **shared mutable
scratch PD** (`stacked_pd_kron_linalg.c:240`, `state->col_inv`), which keeps its
arrays as today.

## Consumer audit (all read sites; the ground truth for the changes)

EVAL-path (need O(1) or precompute; cannot just delete):
- E1 `permuted_dense_linalg.c:247` `sparse_dot_dense`: `inv[idxs[e]]` per CSC nonzero — the only genuinely hot dense-inverse consumer. Reached via:
  - E2 `matmul_dispatchers.c:101` (single-PD left operand, `B->col_inv`)
  - E3 `stacked_pd_linalg.c:571` (spd left operand, per-block `Bq->col_inv`)
  - E4 `permuted_dense_linalg.c:441` (`B->row_inv`), E5 `:522` (`A->row_inv`)
  - E6 `stacked_pd_kron_linalg.c:240` — **shared kron scratch only, not the p output blocks**
- E7 `stacked_pd_linalg.c:337`: `C->col_inv[Ak->col_perm[j]]` — both sorted → merge-scan
- E8 `stacked_pd_coalesce.c:345` (`row_inv`), E9 `:358` (`col_inv`) — both sorted → merge-scan; E9 is loop-invariant in `i` (hoistable)
- E10 `permuted_dense.c:138` `index_pd_fill_values`: `A->row_inv[indices[i]]` — `indices` fixed between alloc and fill (`index.c:70/:80`, `transpose.c:53/:63`) → precompute per-output-row source offsets at alloc time into the existing `kernel_iwork` slot

INIT-only (membership tests over sorted data → replace with scans, no inv needed):
- I1 `permuted_dense_linalg.c:268`, I2 `:402`, I3 `:456` — all via `idxs_hits_set` (`:226-236`)
- I4 `permuted_dense.c:118` `index_pd_alloc`

Writes: only constructors (`permuted_dense.c:619-632`) and the kron scratch
(`stacked_pd_kron_linalg.c:110,115` — the sole mutable PD; must keep private arrays).
Everything else is immutable after construction. No refcounting exists on matrices
(`permuted_dense_free` unconditionally frees at `:35-36`) — do NOT introduce it;
laziness makes it unnecessary.

Structure is never rebuilt after init (`expr.c:99-103` early-return;
`problem_update_params` only refreshes values), so lazy-once semantics are safe.

## Changes

### 1. Helpers (new, in `permuted_dense.c` / `.h` and a small shared header)

- `int pd_ensure_col_inv(permuted_dense *pd)` / `pd_ensure_row_inv(...)`: allocate
  (`sp_malloc`) and fill on first call; return -1/NULL-pattern on alloc failure per
  house style (NULL-return unwind, no errno). Fill loop = the current
  `permuted_dense.c:617-633` body, factored out.
- `bool sorted_hits(const int *idxs, int len, const int *perm, int n0)`: does the
  (sorted-within-CSC-column) index list intersect the strictly-increasing `perm`?
  Merge scan, O(len + n0) — replaces `idxs_hits_set`. Note this REPLACES an
  O(len) loop with O(len + n0); for the kron path it also deletes the O(p·n_vars)
  scratch rebuild per block, so init time should improve, but verify (step V5).
- `int sorted_pos(const int *perm, int n0, int g)`: binary search global→local, for
  the cold conversions (E7-E9 fallback and assert paths).

### 2. Lazy construction — SCOPED to the kron path (blast-radius control)

The default `new_permuted_dense` stays EAGER — every existing PD in every other
problem keeps today's arrays, allocation profile, and O(1) code paths, bit-identical.
Laziness is opt-in and propagates only through the kron-origin family:

- Add `new_permuted_dense_lazy(...)` (same signature; sets `col_inv = row_inv = NULL`,
  skips the `:605-606` allocations and `:617-633` fills). Alternatively an
  `inv_mode` flag on an internal constructor both wrappers call — implementer's pick.
- Call it from exactly ONE production site: `BA_pd_csc_alloc`
  (`permuted_dense_linalg.c:274`) — the kron-path block allocation. (`n_blocks==1`
  callers also route here via `BA_pd_matrices_alloc`; that single-block case is
  equally safe lazy since its inv consumers are gated in §3.)
- **Laziness propagates through copies**: `copy_sparsity_pd_alloc`
  (`permuted_dense_linalg.c:30-34`), `transpose_pd_alloc` (`:36-41`), and
  `index_pd_alloc` (`permuted_dense.c:111-128`) construct their result lazy iff the
  SOURCE pd has NULL inv arrays. Copies of eager PDs stay eager. This confines the
  new representation to kron blocks and their chain descendants
  (transpose/reshape/neg/scalar_mult of matmul Jacobians) — precisely the objects
  whose inv arrays blow up.
- `permuted_dense_free` (`:30-52`): NULL-guard the two `sp_free`s (match the existing
  `csr_cache`/`pre_coalesce` guard style).
- Kron scratch (`kron_scratch_init`, `stacked_pd_kron_linalg.c:65`): unchanged and
  explicitly eager — it is the mutable exception; its kernels (E6) require the arrays.

### 3. Consumers: fast path preserved, scan path only when inv is NULL

Every converted site keeps the existing O(1) dense-inverse code as the primary
branch and falls back to a sorted scan ONLY for lazy PDs — non-kron problems
execute byte-for-byte the same loops as today:

- I1-I3 (`idxs_hits_set` callers): `if (B->col_inv) idxs_hits_set(...) else
  sorted_hits(idxs, len, B->col_perm, B->n0)`.
- I4 (`index_pd_alloc:118`): same gate on `A->row_inv`, else binary search in
  `A->row_perm`.
- E7 (`stacked_pd_linalg.c:337`): gate; lazy branch = merge-scan over
  `Ak->col_perm` × `C->col_perm` (both strictly increasing).
- E8/E9 (coalesce `:345/:358`): gate; lazy branch = merge-scan (E9's lookup is
  loop-invariant in `i` — hoist it in the lazy branch only, leave the eager branch
  untouched).
- E10 (`index_pd_fill_values:138`): for a lazy source, precompute
  `old_row_of_new_row[]` at `index_pd_alloc` time via binary search (indices are
  fixed between alloc and fill — `index.c:70/:80`, `transpose.c:53/:63`) into
  `kernel_iwork`; eager sources keep the current `row_inv` read.
- E2/E4/E5 (single-PD operands feeding `sparse_dot_dense`): call the ensure helper
  once in the corresponding `*_alloc` so the hot fill kernel is UNTOUCHED either way.
  These arrays index the operand's own row/column space (child-output dimension),
  not n_vars — ensuring them is cheap; the n_vars-sized arrays on result PDs simply
  never materialize.
- E3 (`BA_spd_csc` per-block `Bq->col_inv`): `pd_ensure_col_inv(Bq)` per block in
  the alloc twin — only spds that actually reach this dispatch pay; document as the
  one remaining per-block-inv site (shared-scratch rewrite = follow-up).
- E1 (`sparse_dot_dense`) and E6 (kron kernels): **zero changes** — their operands
  are guaranteed ensured/eager by the above.

### 4. Tests

- Update `tests/utils/test_permuted_dense.h:326-338` (`test_permuted_dense_col_inv`,
  registered `all_tests.c:427`): call the ensure helper first, or convert to test
  lazy semantics (NULL after construction, correct contents after ensure).
- Add the repo's FIRST peak-memory regression test (hook: `g_peak_bytes` from
  `tracked_alloc.h`, reset semantics per `problem.c:35-38`): build the row-sum +
  col-sum Jacobian shape (matrix variable X (a×b), constraints `X @ ones(b)` and
  `X.T @ ones(a)` through the public expr API at e.g. a=b=64), run
  `problem_init_jacobian`, assert `g_peak_bytes < C_small · a·b` for a generous
  constant (e.g. 200 bytes/var) that the old code exceeds by ~30× ((a+b)·ab·20B vs
  ab·200B at 64: 164 MB vs 0.8 MB). Place next to `tests/utils/test_alloc_overflow.h`
  style; register in `all_tests.c`.
- Grep sweep: no remaining reads of `->col_inv`/`->row_inv` outside the ensure
  helpers, E1-kernel signatures, kron scratch, and gated frees.

### 5. Explicitly OUT of scope (documented follow-ups, separate PRs)

- `csc_iwork`/CSC-cache per sparse matrix (`sparse_matrix.c:71-72`): eval-path live
  (`sparse_refresh_csc_values` `:326-331`), ~12 B × n_vars per epigraph branch
  (~0.75 GB of OptAdv's 5.9). Fix = problem-level shared scratch; invasive plumbing.
- Eval-path `copy_sparsity` churn in `BTDA_*_fill_values` (`stacked_pd_linalg.c:368-396`
  TODOs) — laziness already shrinks each churned copy; the structural fix is separate.
- `permuted_dense_ensure_transpose_cache` doubling (`permuted_dense_linalg.c:48-61`)
  — becomes cheap automatically once constructors are lazy.

## House rules (from repo conventions)

- All allocations via `sp_malloc`/`sp_free` (`tracked_alloc.h:40-42`) — bypassing
  corrupts the counters this fix is measured by.
- clang-format: LLVM/Allman, 4-space, 85 col, `(double *) x` casts — run
  `clang-format -i` on touched files (CI `formatting.yml` gates it).
- Asserts document invariants (compiled out in Release); keep the sorted-perm asserts.
- Read `docs/thesis/permuted_dense.md` (untracked, in-tree) before starting — it
  defines the A = Srᵀ X Sc selector algebra and the alloc-then-fill two-phase protocol
  the changes must respect.

## Verification

V1. Unit: `cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug && cmake --build build --target all_tests -j && ./build/all_tests` — all pass, including the updated col_inv test and the new peak-memory test.
V2. Sanitizers (CI-equivalent): ASan+UBSan build (`-fsanitize=address,undefined -fno-omit-frame-pointer -g`, dir `build-asan/` already configured) → `./build-asan/all_tests`; then valgrind gate as in `.github/workflows/valgrind.yml`.
V3. Memory: `PROFILE_ONLY=ON` build; also check `tests/profiling/profile_memory.h` is actually registered (`all_tests.c:572-580` — it appears included but unregistered; register it while there).
V4. End-to-end vs the benchmark harness (repo `/Users/trevorhegarty/Documents/Jacobian-accumulation`):
   - Rebuild bindings: SparseDiffPy vendors the engine as a **git submodule**; check the branch out inside `/Users/trevorhegarty/Documents/SparseDiffPy/SparseDiffEngine`, then `uv pip install --python <bench-repo>/.venv/bin/python --force-reinstall --no-deps /Users/trevorhegarty/Documents/SparseDiffPy` (per bench-repo CLAUDE.md engine-version gotcha).
   - Correctness: for ~5 problems (incl. OptimalAdvertising, CVaRBenchmark), compare `get_problem_data(CLARABEL, ignore_dpp=True)` P/c/A/b against `canon_backend="CPP"` with the `_sparse_close` MATCH discipline.
   - Memory win: `BENCH_MEMORY=1 BENCH_ONLY=OptimalAdvertising,CVaRBenchmark,SemidefiniteProgramming BENCH_BACKENDS=CPP,DIFFENGINE python canonicalization/run_backend_benchmarks.py` — expect OptAdv engine peak to drop from 5901 MB to low tens of MB (values 8mn ≈ 4 MB + per-branch CSC ~0.75 GB remains until the follow-up); Table 3M is the readout.
   - No time regression: timing mode `BENCH_ONLY=OptimalAdvertising` cold DIFFENG vs the 5.12 s reference (init should get FASTER — the per-block O(n_vars) scans disappear).
V5. Submodule pointer: after the engine PR lands, bump SparseDiffPy's submodule (currently at fa911a7, one commit behind engine main) and commit.
V6. **Full-suite timing non-regression (mandatory gate before merging).** With the rebuilt engine in the bench venv (verify version first — the uv-sync-downgrades-to-0.6.0 gotcha):
   - `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BENCH_TIMEOUT=1200 BENCH_OUT=results_timing_lazyinv.txt python canonicalization/run_backend_benchmarks.py` — all 25 problems, all backends/strategies.
   - Compare per-problem cold DIFFENG/DE_DPP and warm diffeng/de_cached against the reference `results/results_backends_all_20260724_repl.txt`: no problem may regress beyond run-to-run noise (>15% on any cold ≥0.5 s, or >20% on any warm ≥10 ms → investigate before merging; problems the change shouldn't touch at all — no dense-vector matmuls, e.g. the pure LPs — should be within a few %).
   - Engine micro-benchmarks before/after on the same machine: `PROFILE_ONLY=ON` binary (`profile_left_matmul`, `profile_BTA_pd_csr_vs_csc`) — these exercise the gated dispatch sites directly.
   - Full memory suite re-run (`BENCH_MEMORY=1`, all problems) to confirm no OTHER problem's memory grew (the eager default should make every non-kron row identical) and to quantify the wins on OptAdv/CVaR/SDP/Quantum/TvInpainting for the thesis tables.
V7. Blast-radius audit (code-level confirmation of the scoping): grep the final diff — `new_permuted_dense_lazy` must be called only from `BA_pd_csc_alloc` and the three laziness-propagating copy constructors; every other diff hunk must be inside an `inv == NULL` branch, a NULL-guard, or the ensure helpers. Any hunk touching an unconditional eager path is out of scope and must be reverted.
