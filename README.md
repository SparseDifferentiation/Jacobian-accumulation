# Jacobian accumulation

Benchmarks of [`SparseDiffEngine`](https://github.com/SparseDifferentiation/SparseDiffEngine)
— a C engine that assembles sparse derivatives from per-atom closed-form Jacobian blocks —
against tools based on automatic/algorithmic differentiation
([CasADi](https://github.com/casadi/casadi), [JAX](https://github.com/jax-ml/jax),
[ADOL-C](https://github.com/coin-or/ADOL-C)) and against CVXPY's `cvxcore` backends.
The recurring question: what do graph coloring, sparsity-pattern detection, and
Jacobian accumulation cost, compared to knowing the derivative structure a priori?

## Contents

- **[`canonicalization/`](canonicalization/)** — the main experiments: constructing
  convex-program problem data `(P, c, A, b)` with the diff engine vs CasADi's symbolic
  AD, and vs CVXPY's `cvxcore` backends, over the 25 problems of the public CVXPY
  benchmark suite. Cold construction and warm parametric re-solves; outputs verified
  bit-identical across systems before timing. Includes reference results and the
  mechanism probes (dense-block coloring blow-up, `casadi.hessian` star-coloring
  pathology). Backs chapter 4 of the associated thesis and
  [cvxpy#3348](https://github.com/cvxpy/cvxpy/pull/3348).
- **[`jacobian/`](jacobian/)** — small self-contained sparse-Jacobian comparisons on a
  tridiagonal system: one script per tool (CasADi, JAX, ADOL-C, SparseDiffEngine),
  illustrating how each recovers — or fails to exploit — the sparsity pattern.

## Setup

```bash
uv sync            # casadi, numpy, scipy, the CVXPY diff-engine fork, sparsediffpy
```

The CVXPY dependency is the diff-engine fork (`Transurgeon/cvxpy` branch
`pr-c-resolve-caching`, the [cvxpy#3348](https://github.com/cvxpy/cvxpy/pull/3348)
stack). For warm re-solve benchmarks the engine needs the composite-source refresh fix
([SparseDiffEngine #107](https://github.com/SparseDifferentiation/SparseDiffEngine/pull/107));
see `canonicalization/README.md` for the version caveat.
