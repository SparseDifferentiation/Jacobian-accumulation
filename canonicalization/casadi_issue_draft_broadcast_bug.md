# Draft for a GitHub Issue on casadi/casadi (bug report)

Post the section below verbatim. Duplicate check done 2026-07-16: #3555
(broadcast semantics, raises errors), #3394 (repmat sparsity pattern), #683
(2013, closed) are all different; no report of this found. Latest PyPI
release (3.7.2) is affected.

---

**Title:** MX `jacobian()` silently returns wrong values in reverse mode for
scalar-broadcast operands (batched adjoint directions)

## Summary

When an MX Jacobian is computed in **reverse mode with batched directions**
(the default), Jacobian entries that correspond to a **scalar broadcast into a
vector expression** are silently zeroed for every direction after the first.
The sparsity pattern is correct; the stored values are wrong. No error or
warning is raised.

## Reproduction (casadi 3.7.2)

```python
import casadi as ca

x = ca.MX.sym("x", 7)
alpha = ca.MX.sym("alpha")
z = ca.vertcat(x, alpha)
g = alpha - 2 * x[:3]          # alpha broadcast across 3 rows

F = ca.Function("g", [z], [g], {"ad_weight": 1.0})   # force reverse mode
J = F.jacobian()(ca.DM.zeros(8), ca.DM.zeros(3))
print(J[:, 7])                 # expected [1, 1, 1] — casadi gives [1, 0, 0]
```

`ad_weight: 1.0` makes the repro deterministic, but the default heuristic
also selects reverse mode whenever the Jacobian is wide-ish and poorly
compressible — e.g. `g = alpha - mtimes(A, x)` with a dense 3×7 constant `A`
hits the same wrong values with **no options set**. That is how we found it:
lowered convex programs with dense data blocks flip the mode heuristic to
reverse, and the extracted constraint matrix silently loses the column of a
scalar epigraph variable (all rows except the first).

## Localization

Same expression, `d g/d alpha` (expected `[1, 1, 1]`):

| configuration                                  | result        |
|------------------------------------------------|---------------|
| forward mode (`ad_weight: 0.0`)                 | `[1, 1, 1]` ✓ |
| reverse mode, batched directions (default)      | `[1, 0, 0]` ✗ |
| reverse mode, `max_num_dir: 1`                  | `[1, 1, 1]` ✓ |
| explicit `F.reverse(3)` with 3 unit adjoint seeds | `[1, 1, 1]` ✓ |
| SX instead of MX                                | `[1, 1, 1]` ✓ |
| `repmat(alpha, 3, 1)` instead of broadcast      | `[1, 1, 1]` ✓ |

Since `F.reverse(3)` itself is correct, the multi-direction adjoint
propagation seems fine; the defect appears to be in `jacobian()`'s assembly of
batched reverse sweeps — entries that map to the *same* input nonzero (the
broadcast scalar) across several directions survive only from the first
direction.

## Environment

casadi 3.7.2 (PyPI, latest release), Python 3.14, macOS 15.7 (also checked:
not size-dependent; any shape whose mode heuristic lands on reverse is
affected). Workarounds we use: `repmat` the scalar explicitly, or force
`ad_weight: 0.0`.

Possibly related but distinct: #3394 (repmat sparsity pattern), #3555
(broadcast semantics).
