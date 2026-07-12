"""Why is CasADi single-extraction slow? Isolate the mechanism.

Hypothesis: for affine graphs the Jacobian entries already sit in the graph as
constants; CasADi still *reconstructs* them via (a) bitvector sparsity pattern
detection (O(nnz * cols/64)) and (b) coloring + one seed sweep per color (dense
block => full-dimension colors). The engine reads each atom's Jacobian block off
structurally in O(nnz).

Experiments (all MX, all affine g = DM @ x):
  E1 dense m x n block, m = 2n, n doubling: superlinear growth expected.
  E2 SAME shape but sparse (~10 nnz/row): cost should collapse => it's not
     graph size, it's density-driven sweeps.
  E3 dense but WIDE (n = 2m): reverse-mode row coloring should kick in
     (alpha=2 heuristic) - cost tracks the smaller dimension's sweeps.
  E4 diagonal at LP scale: linear regime, constant-factor gap only.
"""
import time

import numpy as np
import scipy.sparse as sps

import casadi as ca


def timed_jac(A_const, label):
    m, n = A_const.shape
    x = ca.MX.sym("x", n)
    D = ca.DM(A_const) if not sps.issparse(A_const) else ca.DM(A_const.tocsc())
    g = ca.mtimes(D, x)
    t0 = time.perf_counter()
    J = ca.jacobian(g, x)
    t_sym = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = ca.Function("j", [], [J]).call([])
    t_eval = time.perf_counter() - t0
    nnz = out[0].nnz()
    print(f"  {label:<34s} jac(sym) {t_sym:8.2f}s  ctor+eval {t_eval:6.2f}s  nnz {nnz}",
          flush=True)


rng = np.random.RandomState(0)

print("E1: dense tall block (m=2n), n doubling")
for n in (250, 500, 1000, 2000):
    timed_jac(rng.randn(2 * n, n), f"dense {2*n}x{n}")

print("E2: same shapes, sparse ~10 nnz/row")
for n in (250, 500, 1000, 2000):
    timed_jac(sps.random(2 * n, n, density=10.0 / n, random_state=0, format="csc"),
              f"sparse {2*n}x{n} (10/row)")

print("E3: dense wide block (n=2m), m doubling")
for m in (250, 500, 1000, 2000):
    timed_jac(rng.randn(m, 2 * m), f"dense {m}x{2*m}")

print("E4: diagonal, LP scale (linear regime)")
for n in (10**5, 10**6, 4 * 10**6):
    timed_jac(sps.eye(n, format="csc"), f"diag {n}")
