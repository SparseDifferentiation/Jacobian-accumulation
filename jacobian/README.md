# Sparse Jacobian mini-benchmarks

One script per tool, all on the same tridiagonal system (each `f_i` depends on
`x[i-1], x[i], x[i+1]`, so the Jacobian is sparse):

- `casadi_tridiag_jacobian.py` — CasADi detects the pattern and colors it.
- `jax_tridiag_jacobian.py` — JAX has no sparsity machinery: `jacfwd`/`jacrev` cost n
  (resp. m) passes regardless of structure.
- `diffengine_tridiag_jacobian.py` — SparseDiffEngine reads the pattern off the
  expression graph structurally (via the CVXPY/DNLP interface).
- `adolc_tridiag_jacobian.cpp` (+ `Makefile`) — ADOL-C's taped sparse drivers.
- `sct_tridiag_jacobian.jl` — the Julia sparse-AD stack (SparseConnectivityTracer
  detection + SparseMatrixColorings coloring + DifferentiationInterface/ForwardDiff
  accumulation; run with `julia --project=julia --threads=1`).
- `asl_tridiag_jacobian.py` — the AMPL Solver Library via Pyomo + PyNumero: the .nl
  file encodes the pattern, so there is no detection and no coloring stage at all
  (needs `uv sync --extra asl` and `pyomo build-extensions`).

The full-scale, thesis-grade comparison lives in `../canonicalization/`.
