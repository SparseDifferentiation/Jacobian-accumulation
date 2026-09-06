# Draft for a GitHub Issue on casadi/casadi (performance report)

Post the section below verbatim. Duplicate check done 2026-08-05: discussion
#3471 (general Jacobian slowness from C++), discussion #3486 (sparsity pattern
with MX.zeros), and the casadi-users threads on speeding up vertcat loops are
all different; no report of order dependence found. Latest PyPI release
(3.7.2) is affected. Local evidence:
`canonicalization/probes/probe_blocked_vs_stacked.py` (order-effect scaling
and diagnosis sections).

---

**Title:** MX `jacobian()` construction time depends on vertcat block order:
dense block first is ~10x slower than the row-permuted equivalent, and the
gap grows superlinearly

## Summary

Building an MX Jacobian of `g = vertcat(blocks)` takes ~10x longer when a
dense block is placed **before** elementwise rows than when the same blocks
are stacked in the opposite order, although the two orderings describe the
same Jacobian up to a row permutation. Sparsity detection time, the
`uni_coloring` results, the selected AD mode, and the evaluated matrix are
identical across the two orderings; only the symbolic construction inside
`jacobian()` differs. The gap grows superlinearly with size, reaching minutes
at moderate dimensions.

## Reproduction (casadi 3.7.2)

```python
import time
import numpy as np
import casadi as ca

m, n = 1200, 480
A = ca.DM(np.random.RandomState(0).randn(m, n))
t = ca.MX.sym("t", m)
x = ca.MX.sym("x", n)
z = ca.vertcat(t, x)
g = t - ca.mtimes(A, x)

for label, stack in (("dense-first", ca.vertcat(g, x, 1 - x)),
                     ("dense-last", ca.vertcat(x, 1 - x, g))):
    t0 = time.perf_counter()
    ca.jacobian(stack, z)
    print(label, round(time.perf_counter() - t0, 2), "s")
# dense-first 6.4 s
# dense-last  0.46 s      (same Jacobian up to a row permutation)
```

Scaling (median of 3, doubling both dimensions):

| (m, n)      | dense-first | dense-last | dense block alone |
|-------------|------------:|-----------:|------------------:|
| (600, 240)  |      0.82 s |     0.08 s |            0.07 s |
| (1200, 480) |       6.4 s |     0.46 s |            0.42 s |
| (2400, 960) |        52 s |      5.1 s |             3-4 s |

Extrapolating dense-first to (6000, 2400) predicts ~800 s; we measured 770 s
on a real problem of that size (a lowered parametric QP). This is how we hit
it: lowered conic/QP models are naturally written equalities first — the
dense equality block, then the bound rows — which is exactly the slow order.

## Localization

Everything that should determine the cost is identical across the orderings:

| quantity                                          | dense-first | dense-last |
|---------------------------------------------------|------------|------------|
| `jacobian_sparsity` time                          | 0.11 s     | 0.11 s     |
| `uni_coloring` sizes (fwd / adj)                  | 481 / 1202 | 481 / 1202 |
| mode chosen by the heuristic                      | forward    | forward    |
| `jacobian()` with forced `ad_weight: 0.0`         | 6.3 s      | 0.47 s     |
| `jacobian()` with forced `ad_weight: 1.0`         | 34.5 s     | 34.9 s     |
| evaluated matrix (up to the row permutation)      | equal      | equal      |

So the order effect lives in the forward-mode symbolic construction (forcing
forward reproduces it; forcing reverse is equally slow in both orders), not
in detection, coloring, or the mode heuristic. The trigger is also narrow —
the dense block must contain a second variable and be followed by many rows
in the other variable:

| first block `t - A@x` followed by      | time    |
|----------------------------------------|--------:|
| nothing                                | 0.43 s  |
| one row `x[0]`                         | 0.47 s  |
| `t, 1 - t` (rows in t)                 | 0.46 s  |
| `x, 1 - x` (rows in x)                 | 6.3 s   |

With a single variable (`z = x`, first block `A@x`), the order effect
disappears entirely, as it does when the big block is sparse (tridiagonal)
instead of dense.

## Environment

casadi 3.7.2 (PyPI, latest release), Python 3.14, macOS 15.7. Function
evaluation time is unaffected; only `jacobian()` construction. Workaround we
use: stack the dense block last and apply the row permutation to the
evaluated Jacobian afterwards (milliseconds in scipy, entry-identical
result).
