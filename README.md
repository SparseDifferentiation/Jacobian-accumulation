### Jacobian accumulation

The goal of this repository is to benchmark ``SparseDiffEngine`` against other methods based on automatic or algorithmic
differentiation for computing derivatives.

We will be comparing against the following tools:
- [ADOL-C](https://github.com/coin-or/ADOL-C)
- [JAX](https://github.com/jax-ml/jax)
- [CasADi](https://github.com/casadi/casadi)

Our goal is to understand the benefits of different approaches, since they seem to add a lot of complexity
with terms like ``graph coloring``, ``jacobian accumulation`` and ``vertex elimination`` to compute the sparsity
pattern of the jacobian.
