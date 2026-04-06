import jax
import jax.numpy as jnp

# =============================================================================
# Same tridiagonal system as casadi/casadi_sparse_jacobian.py
# =============================================================================
# Each f_i depends only on x[i-1], x[i], x[i+1], so the Jacobian is sparse.
# However, JAX has no built-in sparsity detection or graph coloring.
# jacfwd always runs n forward passes; jacrev always runs m reverse passes —
# regardless of sparsity structure.


def f(x):
    n = x.shape[0]
    result = []
    for i in range(n):
        expr = jnp.sin(x[i])
        if i > 0:
            expr += x[i - 1] * x[i]
        if i < n - 1:
            expr += jnp.exp(-x[i] * x[i + 1])
        result.append(expr)
    return jnp.array(result)


n = 10
x0 = jnp.ones(n)

# JAX computes the full dense 10x10 Jacobian — no sparsity exploitation.
# Compare with CasADi which detects the tridiagonal pattern and uses only
# 3 forward passes via graph coloring instead of 10.
J = jax.jacfwd(f)(x0)

print(f"Jacobian shape: {J.shape}")
print(f"Non-zeros: {jnp.count_nonzero(J).item()} out of {J.size}")
print(f"\nFull (dense) Jacobian:\n{J}")
