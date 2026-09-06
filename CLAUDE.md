# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Benchmarking `SparseDiffEngine` against AD tools (CasADi, JAX, ADOL-C) and CVXPY's
`cvxcore` backends. The focus: what do sparsity-pattern detection, graph coloring, and
Jacobian accumulation cost compared to knowing the derivative structure per atom a priori.

## Repository Structure

- `canonicalization/` — the main, validated experiments: convex-program problem-data
  extraction `(P, c, A, b)` (no solving), diff engine vs CasADi and vs cvxcore, over the
  25 problems of the public CVXPY benchmark suite. See `canonicalization/README.md` for
  methodology (fairness rules!), commands, env knobs, and reference results in
  `canonicalization/results/`. The `casadi_problems_ext_*.py` modules hand-mirror
  CVXPY's lowered forms **exactly** (row/column order); `casadi_compare.py --verify`
  must stay green (bit-identical matrices) after any change to them.
- `jacobian/` — small per-tool sparse-Jacobian scripts on a shared tridiagonal system.
- Additional canonicalization comparisons beyond CasADi/cvxcore, on 4 representative
  problems (`canonicalization/_lowered_data.py`): `julia_compare.py --tool sct` (the
  generic Julia sparse-AD stack, worker in `canonicalization/julia/main.jl`),
  `julia_compare.py --tool jump` (JuMP/MOI `copy_to` matrix assembly, no AD,
  `julia/jump_main.jl`), and `asl_compare.py` (AMPL Solver Library via Pyomo +
  PyNumero). All three verify against CVXPY's CLARABEL data with the same
  `_sparse_close` MATCH discipline as `casadi_compare.py`.

## Environment

- `uv sync` creates `.venv/` with casadi, numpy, scipy, the CVXPY diff-engine fork
  (`Transurgeon/cvxpy@pr-c-resolve-caching`), and `sparsediffpy`.
- IMPORTANT: benchmarks need sparsediffpy ≥ 0.6.1 (engine with the #107
  composite-`param_source` refresh fix AND the "Swedish" sparsity-fill gather,
  424ddde). PyPI 0.6.0 lacks both: its warm re-solves are invalid, and its cold
  extraction has a quadratic Jacobian-init scan that explodes on large PSD blocks
  (SemidefiniteProgramming 433.8 s vs 2.0 s, QuantumHilbertMatrix 22.2 s vs 2.1 s
  — see results_backends_scipy_coo_20260724.txt). `uv sync` silently downgrades
  the venv back to PyPI 0.6.0 — after any sync, check
  `python -c "import importlib.metadata as m; print(m.version('sparsediffpy'))"`
  and reinstall the local dev build via
  `uv pip install --python .venv/bin/python --force-reinstall --no-deps <path-to-SparseDiffPy>`.
- The backend runner needs `git clone --depth 1 https://github.com/cvxpy/benchmarks
  canonicalization/cvxpy_benchmarks` (gitignored).
- The `*_ND` / `nodpp_*` (ignore_dpp) backend targets need a SECOND, isolated env
  with stock upstream cvxpy: `uv venv canonicalization/.venv-upstream --python 3.14`
  then `uv pip install --python canonicalization/.venv-upstream/bin/python
  cvxpy==1.9.2 numpy scipy` (gitignored). It cannot share `.venv`: cvxpy 1.9.2 pins
  `sparsediffpy<0.4.0`. The fork rejects an explicit `canon_backend` on the
  `ignore_dpp` path, which is why the sweep needs upstream at all — see the
  "`ignore_dpp` fairness" section of `canonicalization/README.md`.
- Julia comparisons: juliaup Julia + `julia --project=julia -e 'using Pkg;
  Pkg.instantiate()'` (Project/Manifest checked in under `julia/`). Workers are
  invoked with `--project=julia --threads=1`.
- ASL comparison: `uv sync --extra asl` then `pyomo build-extensions` (needs
  cmake+clang; `download-extensions` does not ship `libpynumero_ASL` on macOS).
  Check with `AmplInterface.available()`.

## Benchmark hygiene

- Pin BLAS to one thread for any timing run: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`. Julia timing runs additionally use
  `--threads=1` and call `BLAS.set_num_threads(1)` in-script.
- Never run two timing jobs concurrently; verification (`--verify*`) is not
  timing-sensitive.
- New results go under `canonicalization/results/` with machine + versions + engine
  build noted; do not overwrite the reference files.
- When adding a CasADi problem model: decode CVXPY's lowering empirically at a tiny
  size first (`get_problem_data(solver=cp.CLARABEL, ignore_dpp=True)`), mirror it, and
  iterate `--verify --only <Name>` until MATCH with max|diff|=0. Constants that exist in
  the user's data may be `DM`s; never precompute the lowered coefficient matrix
  numerically (that would do the extraction outside the timer).
