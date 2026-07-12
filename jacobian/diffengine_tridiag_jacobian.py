import cvxpy as cp
import numpy as np
from scipy.sparse import coo_matrix
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.c_problem import C_problem

# =============================================================================
# Same tridiagonal system as casadi/ and jax/ examples, using SparseDiffEngine
# via the CVXPY/DNLP interface.
# =============================================================================
# SparseDiffEngine builds a computational graph from the CVXPY expressions,
# automatically detects the sparsity pattern, and computes the Jacobian
# using only the structurally nonzero entries.

n = 10
x = cp.Variable(n)

# Build constraints: each f_i = sin(x_i) + x_{i-1}*x_i + exp(-x_i * x_{i+1})
constraints = []
for i in range(n):
    expr = cp.sin(x[i])
    if i > 0:
        expr = expr + x[i - 1] * x[i]
    if i < n - 1:
        expr = expr + cp.exp(-x[i] * x[i + 1])
    constraints.append(expr == 0)

prob = cp.Problem(cp.Minimize(0), constraints)

# Convert to C problem (SparseDiffEngine)
c_prob = C_problem(prob, verbose=False)
c_prob.init_jacobian_coo()

# Get sparsity pattern
rows, cols = c_prob.get_jacobian_sparsity_coo()
print(f"Jacobian sparsity: {len(rows)} non-zeros out of {n * n}")
print("\nSparsity pattern (row, col):")
for r, c in zip(rows, cols):
    print(f"  ({r}, {c})")

# Evaluate Jacobian at x = ones(n)
x0 = np.ones(n)
c_prob.constraint_forward(x0)
vals = c_prob.eval_jacobian_vals()

print("\nSparse Jacobian values:")
for r, c, v in zip(rows, cols, vals):
    print(f"  J[{r},{c}] = {v:.6f}")

# Build full sparse matrix for display
J = coo_matrix((vals, (rows, cols)), shape=(n, n)).toarray()
print(f"\nFull Jacobian:\n{J}")
