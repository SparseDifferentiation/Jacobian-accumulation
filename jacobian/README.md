# Sparse Jacobian mini-benchmarks

One script per tool, all on the same tridiagonal system (each `f_i` depends on
`x[i-1], x[i], x[i+1]`, so the Jacobian is sparse):

- `casadi_tridiag_jacobian.py` — CasADi detects the pattern and colors it.
- `jax_tridiag_jacobian.py` — JAX has no sparsity machinery: `jacfwd`/`jacrev` cost n
  (resp. m) passes regardless of structure.
- `diffengine_tridiag_jacobian.py` — SparseDiffEngine reads the pattern off the
  expression graph structurally (via the CVXPY/DNLP interface).
- `adolc_tridiag_jacobian.cpp` (+ `Makefile`) — ADOL-C's taped sparse drivers.

The full-scale, thesis-grade comparison lives in `../canonicalization/`.
