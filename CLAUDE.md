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

## Environment

- `uv sync` creates `.venv/` with casadi, numpy, scipy, the CVXPY diff-engine fork
  (`Transurgeon/cvxpy@pr-c-resolve-caching`), and `sparsediffpy`.
- IMPORTANT: warm re-solve benchmarks need an engine build containing the
  composite-`param_source` refresh fix (SparseDiffEngine #107). PyPI 0.6.0 lacks it;
  cold-extraction numbers are still valid there. Local dev builds install via
  `uv pip install --python .venv/bin/python --force-reinstall --no-deps <path-to-SparseDiffPy>`.
- The backend runner needs `git clone --depth 1 https://github.com/cvxpy/benchmarks
  canonicalization/cvxpy_benchmarks` (gitignored).

## Benchmark hygiene

- Pin BLAS to one thread for any timing run: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`.
- Never run two timing jobs concurrently; verification (`--verify*`) is not
  timing-sensitive.
- New results go under `canonicalization/results/` with machine + versions + engine
  build noted; do not overwrite the reference files.
- When adding a CasADi problem model: decode CVXPY's lowering empirically at a tiny
  size first (`get_problem_data(solver=cp.CLARABEL, ignore_dpp=True)`), mirror it, and
  iterate `--verify --only <Name>` until MATCH with max|diff|=0. Constants that exist in
  the user's data may be `DM`s; never precompute the lowered coefficient matrix
  numerically (that would do the extraction outside the timer).
