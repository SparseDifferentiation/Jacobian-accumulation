# Draft for a GitHub Issue on casadi/casadi (bug report)

Post the section below verbatim. Duplicate check done 2026-07-24: GitHub
search for `separate_linear` in casadi/casadi returns only #3639 (the feature
introduction, no crash discussion); search for `join_primitives` returns
nothing. Latest PyPI release (3.7.2) is affected. Keep this separate from the
extraction-performance discussion post (casadi_discussion_draft.md), which
cites `separate_linear` only on graphs where it works.

---

**Title:** `separate_linear` fails an internal assertion
(`MXNode::join_primitives`) when the expression contains a Function call node

## Summary

`MX::separate_linear` crashes with a developer assertion — the error message
itself says "Notify the CasADi developers", hence this report — as soon as the
expression graph contains a genuine (non-inlined) `Function` call node. Since
embedded function calls are how integrators, rootfinders, external functions,
and any `never_inline` helper appear in an MX graph, this limits
`separate_linear` to graphs built purely of inlined primitives.

## Reproduction (casadi 3.7.2)

```python
import casadi as ca

x = ca.MX.sym("x", 3)
th = ca.MX.sym("th", 2)
f = ca.Function("f", [x], [2 * x + 1])
[e] = f.call([x], False, True)   # never_inline=True -> keep a genuine call node
ca.separate_linear(e, x, th)
```

```
RuntimeError: .../casadi/core/mx_node.cpp:176: Assertion "ret.is_empty(true)" failed:
Notify the CasADi developers.
```

The same expression works fine if the call is inlined (`f(x)` lets the call
inline here, and `separate_linear` then returns the correct split
`const = ones(3x1)`, `lin = 2*x`, `nonlin = 0`), so the function *content* is
not the problem — the call node is.

## Localization

The assertion is `casadi_assert_dev(ret.is_empty(true))` in
`MXNode::join_primitives_gen` (`casadi/core/mx_node.cpp:176`). The default
`MXNode::eval_linear` fallback (which handles any node class without an
override, `Call` included) re-evaluates the node on the re-summed parts and
returns the result in the nonlinear slot; for a multi-output node the
per-output primitives that reach `join_primitives` no longer line up with the
expected sizes, and the assertion fires. Single-output uncovered nodes (e.g.
`solve`, `bilin`) take the same fallback without crashing — they are
classified as wholly nonlinear, which is conservative but safe.

## Environment

casadi 3.7.2 (PyPI, latest release), Python 3.14, macOS 15.7. Not
size-dependent; any expression whose graph retains a call node is affected.

Related: #3639 (introduction of `separate_linear`).
