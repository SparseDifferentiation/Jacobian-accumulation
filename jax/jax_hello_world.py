import jax
import jax.numpy as jnp

x = jnp.array(1.0)

# Three ways to differentiate sin(x) at x=1
print("grad:  ", jax.grad(jnp.sin)(x))       # reverse-mode scalar gradient
print("jacfwd:", jax.jacfwd(jnp.sin)(x))      # forward-mode Jacobian
print("jacrev:", jax.jacrev(jnp.sin)(x))      # reverse-mode Jacobian
