# Draft for a GitHub Discussion on casadi/casadi (Q&A category)

Post the section below verbatim; fill in the repo link. The broadcast
value-bug (see CVaR builder) is deliberately NOT mixed in here — file it as a
separate issue with its own minimal repro.

---

**Title:** Fastest way to extract QP problem data (H, c, A, b) from an MX
graph? Dense constant blocks cost one AD sweep per column

We are benchmarking the cost of *constructing* convex-program problem data —
extraction of (H, c, A, b) from an expression graph, no solving — for a
master's thesis comparing generic AD against per-atom structural
differentiation (CVXPY's canonicalization backends). We want to double-check
with you that we are using CasADi the fastest supported way before we publish
numbers, and to understand whether what we observe is expected behavior.

## What we do

Problems are hand-lowered QPs/LPs: `f(z, θ)` quadratic/linear, `g(z, θ) = b −
Az` affine, both MX. Extraction follows the same construction as
`qpsol_nlp` in `casadi/core/conic.cpp`:

```python
grad = ca.gradient(f, z)
H = ca.jacobian(grad, z)          # conic.cpp adds {"symmetric": True} here
c = ca.substitute(grad, z, ca.DM.zeros(z.shape))
A = -ca.jacobian(g, z)
b = ca.substitute(g, z, ca.DM.zeros(z.shape))
F = ca.Function("extract", psyms, [H, c, A, b])   # timed: this + one F.call
```

Outputs are verified entry-for-entry identical to a reference implementation
on all 25 benchmark problems.

## What we observe

`jacobian()` cost on `g = A·x` (A a constant `DM`) is driven by the coloring
of the Jacobian pattern, i.e. one seeded sweep per color — so a dense block
costs min(m, n) full-graph sweeps even though its entries sit in the graph as
a constant (casadi 3.7.2, single-threaded, i7-8850H):

| pattern of ∂g/∂x      | nnz  | `uni_coloring` (fwd/adj) | `jacobian` + eval |
|-----------------------|------|--------------------------|-------------------|
| dense 4000×2000       | 8 M  | 2000 / 4000              | 16.6 s            |
| same shape, 10/row    | 40 k | 55 / 60                  | 0.03 s            |
| tridiagonal n = 10⁶   | 3 M  | 3 / 3                    | 2.5 s             |

Minimal repro:

```python
import time
import numpy as np
import scipy.sparse as sps
import casadi as ca

def timed_jac(A_const):
    m, n = A_const.shape
    D = ca.DM(A_const.tocsc()) if sps.issparse(A_const) else ca.DM(A_const)
    x = ca.MX.sym("x", n)
    t0 = time.perf_counter()
    ca.Function("j", [], [ca.jacobian(ca.mtimes(D, x), x)]).call([])
    print(f"{m}x{n}: {time.perf_counter() - t0:.2f} s")

timed_jac(np.random.RandomState(0).randn(4000, 2000))          # ~18 s
timed_jac(sps.random(4000, 2000, density=10 / 2000, random_state=0))  # ~0.1 s
```

On our benchmark set this makes cold extraction of problems with dense data
blocks scale roughly O(n·nnz): a QP with dense 1600² H and dense 8000×1600
constraints takes ~54 s; a CVaR problem whose scenario matrix is 131072×768
dense (and whose sum-of-epigraph row makes all columns mutually conflicting
under coloring) did not finish in 10 h. Warm re-evaluation of the built
`Function` with new parameter values is fast and often beats the comparison
system — the cost is all in the one-time Jacobian construction.

## What we already ruled out

- **`qpsol` / `quadratic_coeff`:** both use the same construction internally;
  measured slower than the code above (n = 800 dense QP: 5.4 s for the path
  above vs 148 s for `hessian(f, x)`, 173 s for `qpsol('qrqp')` construction,
  140 s for `quadratic_coeff` + `linear_coeff`). The gap is `hessian()`'s
  star coloring on a dense Hessian pattern; `jacobian(gradient(f))` without
  the `symmetric` flag is much faster there. So we give CasADi the fastest
  route we could find.
- **SX:** same sweep counts, ~nnz scalar nodes per matrix op; 50–270× slower
  than MX on these graphs and cannot reach the benchmark sizes.

## Questions

1. Is `jacobian(gradient(f))` + `jacobian(g)` on MX indeed the fastest
   supported way to recover (H, c, A, b) from an expression graph, or is
   there an API/option we missed (`ad_weight`, `helper_options`,
   `GlobalOptions`, …) that materially changes the dense-block case?
2. Is one-sweep-per-color on dense *constant* blocks the expected behavior,
   i.e. there is deliberately no shortcut that reads the coefficient block of
   an affine subgraph directly off the constant node? (We understand the
   design center is repeated evaluation inside NLP solvers, where this
   construction cost is amortized — we want to state that fairly.)
3. Is the star-coloring blowup of `hessian()` on dense Hessian patterns
   (n = 800: 148 s vs 5.4 s via jacobian-of-gradient; n = 1600 did not finish
   in 6.5 h) known/expected? `qpsol` inherits it through
   `{"symmetric": true}` in `qpsol_nlp`.

Benchmark code, per-problem models, and probes:
https://github.com/SparseDifferentiation/Jacobian-accumulation
(`canonicalization/` — see `probes/probe_qpsol_path.py`,
`probes/probe_coloring_sweeps.py`, `probes/probe_sx_vs_mx.py`).

Environment: casadi 3.7.2 (PyPI), Python 3.14, macOS 15.7, NumPy 2.4, BLAS
pinned to 1 thread.
