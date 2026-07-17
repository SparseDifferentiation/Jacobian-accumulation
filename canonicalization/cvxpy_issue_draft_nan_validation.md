# Drafts for two upstream reports (NaN problem data)

Draft 1 goes to cvxpy/cvxpy, draft 2 to cvxpy/benchmarks; they cross-reference
each other, so post 2 first and paste its URL into 1. Behaviors below were
confirmed on our 1.10-dev tree (which includes #3044); re-run the snippet on a
clean upstream checkout before posting.

---

## Draft 1 — cvxpy/cvxpy

**Title:** Non-finite data validation is inconsistent: `Parameter` rejects
NaN, `Constant` accepts it, and `get_problem_data()` returns NaN data
unchecked

## Current behavior

```python
import numpy as np, cvxpy as cp

p = cp.Parameter(3)
p.value = np.array([1.0, np.nan, 2.0])   # ValueError: Parameter value must be real.

c = cp.Constant(np.array([[np.nan, 2.0, 3.0]]))   # silently accepted

x = cp.Variable(3)
prob = cp.Problem(cp.Minimize(cp.sum(x)),
                  [np.array([[np.nan, 2, 3]]) @ x <= 0, x >= -1])

data, _, _ = prob.get_problem_data(cp.CLARABEL)   # returns data, A contains NaN,
                                                  # no warning
prob.solve(solver=cp.CLARABEL)
# ValueError: Problem data contains NaN or Inf. Check your parameter values
# and constants.
```

So today (post-#3044, which added the `solve_via_data` check with the
`ignore_nan` opt-out):

- `Parameter` values are validated at assignment; `Constant` values are not
  validated anywhere.
- `solve()` catches non-finite data, but only at the very end, with a message
  that cannot say *which* constant is bad.
- `get_problem_data()` — the entry point for custom solver interfaces and
  benchmarking — returns NaN-laden matrices with no diagnostic at all.

## Why we hit this

While cross-validating canonicalization backends (in the context of #3348) we
bit-compared `(P, c, A, b)` across CPP, SCIPY, and an experimental backend on
the cvxpy/benchmarks suite. `OptimalAdvertisingBenchmark` turns out to
generate a NaN coefficient at its default size (0/0 in its `R0` data — filed
separately as [benchmarks issue link]), and nothing on the
`get_problem_data` path flagged it. It surfaced only as a one-entry
discrepancy between backends: with a NaN coefficient row, a numerically
evaluated constant term gives `0 * NaN = NaN` while structurally assembled
constants give 0 — i.e. on non-finite data the canonicalized output is not
even well-defined across implementations. Validation would make this a loud
error instead of a silent divergence.

## Proposal (happy to PR whichever direction you prefer)

1. Validate `Constant` values at construction, mirroring the existing
   `Parameter` check (one `np.isfinite` pass per constant, O(nnz), possibly
   behind a global setting if construction-time cost is a concern); this
   gives the earliest and most localizable error, or
2. run the #3044 check (or a warning) in `get_problem_data()` as well, so
   non-solve consumers see it, and/or
3. extend the solve-time message to name the offending leaf (the check could
   walk `problem.constants()` / parameters and report the first non-finite
   one).

Option 1 + 3 together would have turned our silent benchmark artifact into
`"Non-finite value (nan) in Constant of shape (1, 1000) used in expression
minimum(...)"` at model-build time.

---

## Draft 2 — cvxpy/benchmarks

**Title:** OptimalAdvertisingBenchmark generates NaN problem data (0/0 in
`R0`) at the default size

At the default size (m=250, n=1000), `benchmark/optimal_advertising.py`
produces a NaN coefficient:

```python
c0 = np.random.uniform(size=(m,))
c0 *= 0.6 * T0.sum() / c0.sum()
c0 = 1000 * np.round(c0 / 1000)      # <- one entry rounds to exactly 0
R0 = np.array([np.random.lognormal(c0.min() / c0[i]) for i in range(m)])
#                                    ^ 0/0 -> nan; lognormal(nan) -> nan
```

With seed 1 (`np.random.seed(1)` upstream), `c0[28]` rounds to 0, so
`R0[28] = nan`, and the NaN propagates into the canonicalized objective and
constraint data (numpy emits `RuntimeWarning: invalid value encountered in
scalar divide` during generation). Canonicalization-only benchmarks measure
the degenerate instance silently; with current cvxpy, actually *solving* the
instance raises `ValueError: Problem data contains NaN or Inf`.

Found while bit-comparing canonicalization backend outputs on the suite —
the NaN makes the instance's problem data implementation-defined (see
[cvxpy issue link]).

Suggested fix, preserving the intended data distribution: clamp the rounded
`c0` away from zero, e.g. `c0 = np.maximum(1000 * np.round(c0 / 1000),
1000.0)`, or guard the division. Happy to PR.
