"""
Problem families for the final-deposit experiments of the thesis, beyond the 25
problems of the cvxpy/benchmarks suite.

Every family is a builder ``f(variant, n, seed, **kw) -> cp.Problem``. A spec string
such as ``"lasso:variant=nondpp:n=1000:density=0.2:seed=3"`` names one instance;
``make_class(spec)`` turns it into an ASV-style class with ``setup()`` and
``self.problem``, so ``thesis_final_benchmarks.py`` runs it exactly like a suite
problem.

Variants (the non-DPP experiment, jury J1e / Legrain iii):
  dpp     parameters appear DPP-compliantly; CVXPY caches its parameter tensor.
  nondpp  the same model with one product of two parameters, which breaks DPP;
          CVXPY recompiles from scratch on every solve.
  free    the parameters are replaced by constants; a "re-solve" is a new problem,
          compiled from scratch by every backend.
In all three variants the problem data at the default parameter values are the
same, so the three re-solve times are directly comparable.

Families:
  lasso         min ||A x - b||^2 + lam ||x||_1, A of size (2n x n) with a given
                density; parameters b, lam (and alpha in nondpp: (alpha*lam)).
                Used for the (n, density) sensitivity sweep with variant=free.
  factor_cov    the suite's FactorCovarianceModel with n assets and m = n // 100
                factors; nondpp writes the risk aversion as gamma * delta.
  huber         the suite's HuberRegression at size n (parameter-free).
  advertising   the suite's OptimalAdvertising with n time slots, m = n // 4 ads
                (parameter-free).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _rng(seed):
    return np.random.default_rng(int(seed))


def lasso(variant="free", n=500, seed=0, density=1.0):
    import cvxpy as cp

    n = int(n)
    density = float(density)
    rng = _rng(seed)
    m = 2 * n
    A = rng.standard_normal((m, n))
    if density < 1.0:
        A *= rng.random((m, n)) < density
    x_true = rng.standard_normal(n) * (rng.random(n) < 0.1)
    b_val = A @ x_true + 0.1 * rng.standard_normal(m)
    lam_val = 0.1 * np.abs(A.T @ b_val).max() if A.any() else 1.0
    x = cp.Variable(n)
    if variant == "free":
        obj = cp.sum_squares(A @ x - b_val) + lam_val * cp.norm1(x)
    else:
        b = cp.Parameter(m, value=b_val)
        lam = cp.Parameter(nonneg=True, value=lam_val)
        if variant == "dpp":
            obj = cp.sum_squares(A @ x - b) + lam * cp.norm1(x)
        elif variant == "nondpp":
            alpha = cp.Parameter(nonneg=True, value=1.0)
            obj = cp.sum_squares(A @ x - b) + (alpha * lam) * cp.norm1(x)
        else:
            raise ValueError(variant)
    return cp.Problem(cp.Minimize(obj))


def factor_cov(variant="dpp", n=5000, seed=1):
    import cvxpy as cp

    n = int(n)
    m = max(n // 100, 2)
    rng = _rng(seed)
    mu = np.abs(rng.standard_normal((n, 1)))
    Sigma_tilde = rng.standard_normal((m, m))
    Sigma_tilde = Sigma_tilde.T @ Sigma_tilde
    D = sp.diags(rng.uniform(0, 0.9, size=n))
    F = rng.standard_normal((n, m))
    w = cp.Variable(n)
    f = cp.Variable(m)
    ret = mu.T @ w
    risk = cp.quad_form(f, Sigma_tilde, assume_PSD=True) + cp.sum_squares(np.sqrt(D) @ w)
    if variant == "free":
        gamma_risk = 0.1 * risk
        lmax = 2.0
    else:
        gamma = cp.Parameter(nonneg=True, value=0.1)
        lmax = cp.Parameter(value=2.0)
        if variant == "dpp":
            gamma_risk = gamma * risk
        elif variant == "nondpp":
            delta = cp.Parameter(nonneg=True, value=1.0)
            gamma_risk = (gamma * delta) * risk
        else:
            raise ValueError(variant)
    constraints = [cp.sum(w) == 1, f == F.T @ w, cp.norm(w, 1) <= lmax]
    return cp.Problem(cp.Maximize(ret - gamma_risk), constraints)


def huber(variant="free", n=3000, seed=1):
    import cvxpy as cp

    n = int(n)
    rng = _rng(seed)
    samples = int(1.5 * n)
    beta_true = 5 * rng.standard_normal((n, 1))
    X = rng.standard_normal((n, samples))
    v = rng.standard_normal((samples, 1))
    factor = 2 * rng.binomial(1, 1 - 0.12, size=(samples, 1)) - 1
    Y = factor * X.T.dot(beta_true) + v
    beta = cp.Variable((n, 1))
    return cp.Problem(cp.Minimize(cp.sum(cp.huber(X.T @ beta - Y, 1))))


def advertising(variant="free", n=1000, seed=1):
    import cvxpy as cp

    n = int(n)
    m = max(n // 4, 2)
    rng = _rng(seed)
    scale = 10000
    B = rng.lognormal(mean=8, size=(m, 1)) + 10000
    B = 1000 * np.round(B / 1000)
    P = rng.uniform(size=(m, 1)).dot(rng.uniform(size=(1, n)))
    T = np.sin(np.linspace(-np.pi, np.pi, n)) * scale
    T += -np.min(T) + scale
    c = rng.uniform(size=(m,))
    c *= 0.6 * T.sum() / c.sum()
    c = 1000 * np.round(c / 1000)
    R = np.array([rng.lognormal(c.min() / c[i]) for i in range(m)])
    D = cp.Variable((m, n))
    Si = [cp.minimum(R[i] * P[i, :] @ D[i, :].T, B[i]) for i in range(m)]
    constraints = [D >= 0, D.T @ np.ones(m) <= T, D @ np.ones(n) >= c]
    return cp.Problem(cp.Maximize(cp.sum(Si)), constraints)


FAMILIES = {"lasso": lasso, "factor_cov": factor_cov, "huber": huber,
            "advertising": advertising}


def parse_spec(spec):
    family, *parts = spec.split(":")
    kw = dict(p.split("=", 1) for p in parts)
    return family, kw


def make_class(spec):
    family, kw = parse_spec(spec)
    builder = FAMILIES[family]

    class Extra:
        def setup(self):
            self.problem = builder(**kw)

    Extra.__name__ = spec
    return Extra


# --------------------------------------------------------------------------- #
# The problem sets of the experiments
# --------------------------------------------------------------------------- #
SEEDS = range(5)

# Non-DPP table: every family with parameters, three variants each.
NONDPP = [f"{fam}:variant={v}:{size}:seed=0"
          for fam, size in (("lasso", "n=1000"), ("factor_cov", "n=5000"))
          for v in ("dpp", "nondpp", "free")]

# Sensitivity to size and density (parameter-free lasso), five seeds each.
SENSITIVITY = [f"lasso:variant=free:n={n}:density={d}:seed={s}"
               for n in (100, 300, 1000, 3000)
               for d in (0.01, 0.05, 0.2, 0.5, 1.0)
               for s in SEEDS]

# Scaling of three suite problems, five seeds each.
SCALING = ([f"huber:n={n}:seed={s}" for n in (300, 1000, 3000, 6000) for s in SEEDS]
           + [f"advertising:n={n}:seed={s}" for n in (200, 500, 1000, 2000) for s in SEEDS]
           + [f"factor_cov:variant=dpp:n={n}:seed={s}"
              for n in (1000, 5000, 25000, 50000) for s in SEEDS])

SETS = {"nondpp": NONDPP, "sensitivity": SENSITIVITY, "scaling": SCALING}
