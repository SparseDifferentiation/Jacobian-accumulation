"""Make CasADi's sweep count visible: why extracting a Jacobian whose entries
are already constants in the graph costs sweeps, and how many.

casadi.jacobian on an MX graph is generic AD; it never pattern-matches
"g is affine with constant coefficient A, so J = A".  It runs three stages:

  1. sparsity detection -- propagate 64-bit bitvectors through the graph,
     ceil(n/64) forward (or ceil(m/64) reverse) sweeps, just for the PATTERN;
  2. coloring -- greedy unidirectional coloring of the pattern (columns for
     forward mode, rows for reverse); columns may share a color only if no row
     touches both;
  3. accumulation -- ONE AD sweep of the whole graph per color, seeded with
     that color's indicator; results are scattered into the Jacobian nonzeros.

A dense m x n block admits no compression: every pair of columns collides, so
uni-coloring needs n colors (m in reverse) and stage 3 degenerates to
min(m, n) full-graph sweeps -- the matrix is rebuilt column by column even
though it sits verbatim in the graph as a DM constant.  A banded/diagonal
pattern colors in O(1), which is why the same-shape sparse case collapses.

This probe reports, per pattern: the coloring sizes casadi's own
Sparsity::uni_coloring produces (= the sweep counts it will use), the
pattern-detection time, and the actual jacobian construction + evaluation
times, for g = A @ x with A constant:

  dense tall (m=2n)   -> n forward colors, cost ~ n * nnz
  same shape, sparse  -> O(1) colors, cost ~ nnz
  tridiagonal         -> 3 colors regardless of n
  CVaR-shaped slice   -> dense 8192 x 769 block (1/16 of the benchmark's
                         131072 rows); 769 colors predicts the full-size DNF

Usage: python canonicalization/probes/probe_coloring_sweeps.py
"""
import time

import numpy as np
import scipy.sparse as sps

import casadi as ca


def report(label, A_const):
    m, n = A_const.shape
    D = ca.DM(A_const.tocsc()) if sps.issparse(A_const) else ca.DM(np.asarray(A_const))
    x = ca.MX.sym("x", n)
    g = ca.mtimes(D, x)

    t0 = time.perf_counter()
    sp = ca.jacobian_sparsity(g, x)           # stage 1 only
    t_pattern = time.perf_counter() - t0

    fwd_colors = sp.uni_coloring().size2()    # stage 2, forward (column) mode
    adj_colors = sp.T.uni_coloring().size2()  # stage 2, reverse (row) mode

    t0 = time.perf_counter()
    J = ca.jacobian(g, x)                     # stages 1-3, symbolic
    t_sym = time.perf_counter() - t0
    t0 = time.perf_counter()
    ca.Function("j", [], [J]).call([])
    t_eval = time.perf_counter() - t0

    print(f"  {label:<26s} nnz {sp.nnz():>9d}  pattern {t_pattern:7.3f}s  "
          f"colors fwd/adj {fwd_colors:>5d}/{adj_colors:>5d}  "
          f"jac(sym) {t_sym:8.2f}s  ctor+eval {t_eval:6.2f}s", flush=True)


rng = np.random.RandomState(0)

print("dense tall blocks (m=2n): forward colors = n, no compression possible")
for n in (250, 500, 1000, 2000):
    report(f"dense {2*n}x{n}", rng.randn(2 * n, n))

print("same shapes, ~10 nnz/row: coloring compresses, sweeps collapse")
for n in (250, 500, 1000, 2000):
    report(f"sparse {2*n}x{n}", sps.random(2 * n, n, density=10.0 / n,
                                           random_state=0, format="csc"))

print("tridiagonal: 3 colors at any size")
for n in (10**5, 10**6):
    report(f"tridiag {n}", sps.diags([np.ones(n - 1), np.ones(n), np.ones(n - 1)],
                                     [-1, 0, 1], format="csc"))

print("CVaR-shaped slice (full benchmark block is 131072 x 769 dense):")
report("dense 8192x769", rng.randn(8192, 769))
