import casadi as ca
import numpy as np

# =============================================================================
# Sparse Jacobian computation with CasADi
# =============================================================================
# We define a tridiagonal system: each f_i depends only on x[i-1], x[i], x[i+1].
# This makes the Jacobian sparse — most entries are structurally zero.

n = 10
x = ca.MX.sym("x", n)

f_list = []
for i in range(n):
    expr = ca.sin(x[i])
    if i > 0:
        expr += x[i - 1] * x[i]
    if i < n - 1:
        expr += ca.exp(-x[i] * x[i + 1])
    f_list.append(expr)

f = ca.vertcat(*f_list)

# =============================================================================
# Compute the symbolic Jacobian
# =============================================================================
# CasADi walks the expression DAG and applies the chain rule symbolically.
# It automatically detects structural zeros — entries that are always zero
# regardless of the input values — and never computes them.

J = ca.jacobian(f, x)

sp = J.sparsity()
print(f"Jacobian size: {sp.size1()} x {sp.size2()}")
print(f"Non-zeros: {sp.nnz()} out of {sp.size1() * sp.size2()} elements")
print(f"\nSparsity pattern (1 = nonzero):\n{ca.DM(sp, 1)}")

# =============================================================================
# Graph coloring — how CasADi reduces the cost of Jacobian evaluation
# =============================================================================
#
# PROBLEM: Computing a Jacobian column-by-column requires n forward-mode AD
# passes (one directional derivative per column). For large n this is expensive.
#
# KEY INSIGHT: If columns j and k have no row where BOTH are nonzero (i.e. they
# are "structurally orthogonal"), we can compute them in a SINGLE forward pass
# by seeding both directions at once — the results won't interfere.
#
# ALGORITHM (unidirectional graph coloring):
#   1. Build a graph: vertices = columns, edge between j and k if they share
#      a nonzero row (i.e. they would interfere).
#   2. Color the graph so no two adjacent vertices share a color.
#   3. Each color = one forward-mode pass computing all columns of that color.
#
# For a tridiagonal matrix the bandwidth is 3, so only 3 colors suffice
# regardless of n. That's 3 forward passes instead of n.

coloring = sp.uni_coloring()
n_colors = coloring.size2()
print(f"\nGraph coloring: {n_colors} colors (forward passes) vs {n} without coloring")
print(f"Coloring matrix (rows=columns of J, cols=colors, 1=assigned):\n{ca.DM(coloring, 1)}")

# =============================================================================
# Numerical evaluation at a test point
# =============================================================================

func = ca.Function("f", [x], [J])
x0 = np.ones(n)
print(f"\nNumerical Jacobian at x = ones(10):\n{func(x0)}")
