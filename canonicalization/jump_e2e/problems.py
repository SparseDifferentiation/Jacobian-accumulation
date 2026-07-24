"""Raw problem data + CVXPY user models for the end-to-end JuMP comparison.

One entry per benchmark problem.  `data(full)` reproduces the exact arrays of
the casadi_compare.py Specs (same np.random.RandomState seeds, same draw
order), and `cvxpy_problem(d, ...)` builds the CVXPY user-level model FROM
those arrays -- so the CVXPY side and the JuMP side (models/<Name>.jl, which
receives the same arrays as npz) provably consume identical data.

Hand-verification: for each problem, compare `cvxpy_problem` here with
`build` in models/<Name>.jl; the docstrings state the mathematical model and
the epigraph rewriting used on the JuMP side (JuMP has no huber/norm atoms, so
the standard epigraph reformulations are written out explicitly -- the same
rewritings CVXPY's canonicalization performs internally).

Parametric problems take theta (drawn by `draw(rng)`, seeds matching the
suite: default_rng(42) for values, uniform/normal draws as in casadi_compare).
JuMP has no parameter objects, so theta is baked into the model at build time;
the CVXPY side keeps its cp.Parameter formulation (solved with
ignore_dpp=True, the suite's fairness setting).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class E2EProblem:
    name: str
    parametric: bool
    data: object            # data(full: bool) -> dict of numpy arrays
    cvxpy_problem: object   # cvxpy_problem(d) -> (cp.Problem, [cp.Parameter])
    draw: object = None     # draw(rng, d) -> [theta values]
    notes: str = ""


# --------------------------------------------------------------------------- #
# SimpleLP:  min c'x  s.t. 0 <= x <= 1
# --------------------------------------------------------------------------- #
def _simple_lp_data(full: bool) -> dict:
    n = int(1e7) if full else 7
    return {"c": np.arange(n).astype(float)}


def _simple_lp_cvxpy(d):
    import cvxpy as cp
    c = d["c"]
    x = cp.Variable(len(c))
    return cp.Problem(cp.Minimize(c @ x), [0 <= x, x <= 1]), []


# --------------------------------------------------------------------------- #
# LeastSquares:  min ||A0 x - b0||^2
# JuMP rewrite: r = A0 x - b0 (zero cone), min r'r  -- CVXPY's own
# sum_squares epigraph, written out.
# --------------------------------------------------------------------------- #
def _least_squares_data(full: bool) -> dict:
    m, n = (7000, 2000) if full else (3, 2)
    rng = np.random.RandomState(1)
    return {"A0": rng.randn(m, n), "b0": rng.randn(m)}


def _least_squares_cvxpy(d):
    import cvxpy as cp
    A0, b0 = d["A0"], d["b0"]
    x = cp.Variable(A0.shape[1])
    return cp.Problem(cp.Minimize(cp.sum_squares(A0 @ x - b0))), []


# --------------------------------------------------------------------------- #
# SimpleQP:  min 0.5 x'P0 x + q0'x  s.t. G0 x <= h0, Aeq x == beq
# --------------------------------------------------------------------------- #
def _simple_qp_data(full: bool) -> dict:
    m, n, p_ = (8000, 1600, 20) if full else (4, 3, 1)
    rng = np.random.RandomState(1)
    P0 = rng.randn(n, n)
    P0 = P0.T @ P0
    q0 = rng.randn(n)
    G0 = rng.randn(m, n)
    h0 = G0 @ rng.randn(n)
    Aeq = rng.randn(p_, n)
    beq = rng.randn(p_)
    return {"P0": P0, "q0": q0, "G0": G0, "h0": h0, "Aeq": Aeq, "beq": beq}


def _simple_qp_cvxpy(d):
    import cvxpy as cp
    x = cp.Variable(len(d["q0"]))
    prob = cp.Problem(
        cp.Minimize((1 / 2) * cp.quad_form(x, d["P0"], assume_PSD=True)
                    + d["q0"].T @ x),
        [d["G0"] @ x <= d["h0"], d["Aeq"] @ x == d["beq"]],
    )
    return prob, []


# --------------------------------------------------------------------------- #
# ParametrizedQP:  min ||A x - b||^2  s.t. 0 <= x <= 1,  (A, b) parameters
# JuMP rewrite: r = A x - b, min r'r; theta baked in at build time.
# --------------------------------------------------------------------------- #
def _parametrized_qp_data(full: bool) -> dict:
    m, n = (6000, 2400) if full else (3, 2)
    return {"m": np.int64(m), "n": np.int64(n)}


def _parametrized_qp_cvxpy(d):
    import cvxpy as cp
    m, n = int(d["m"]), int(d["n"])
    A = cp.Parameter((m, n))
    b = cp.Parameter((m,))
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [0 <= x, x <= 1])
    return prob, [A, b]


def _parametrized_qp_draw(rng, d):
    m, n = int(d["m"]), int(d["n"])
    return [rng.standard_normal((m, n)), rng.standard_normal(m)]


# --------------------------------------------------------------------------- #
# HuberRegression:  min sum(huber(Xt beta - y, M=1))
# JuMP rewrite: huber(r, 1) = min_{u,w} u^2 + 2|w|  s.t. r = u + w;
# |w| via t >= w, t >= -w  =>  min u'u + 2 sum(t).
# --------------------------------------------------------------------------- #
def _huber_data(full: bool) -> dict:
    n = 3000 if full else 2
    samples = int(1.5 * n) if full else 3
    rng = np.random.RandomState(1)
    X = rng.randn(n, samples)
    beta_true = 5 * rng.normal(size=(n, 1))
    v = rng.normal(size=(samples, 1))
    factor = 2 * rng.binomial(1, 0.88, size=(samples, 1)) - 1
    Y = factor * X.T.dot(beta_true) + v
    return {"Xt": X.T, "y": Y.ravel()}


def _huber_cvxpy(d):
    import cvxpy as cp
    Xt, y = d["Xt"], d["y"]
    beta = cp.Variable((Xt.shape[1], 1))
    Y = y.reshape(-1, 1)
    return cp.Problem(cp.Minimize(cp.sum(cp.huber(Xt @ beta - Y, 1)))), []


# --------------------------------------------------------------------------- #
# SVMWithL1Regularization:
#   min (1/m) sum(pos(1 - y .* (X beta - v))) + lambda ||beta||_1
# JuMP rewrite: hinge h >= 1 - y.*(X beta - v), h >= 0; |beta| via s >= beta,
# s >= -beta  =>  min sum(h)/m + lambda sum(s).
# --------------------------------------------------------------------------- #
def _svm_l1_data(full: bool) -> dict:
    n, m = (500, 25000) if full else (2, 3)
    rng = np.random.RandomState(1)
    beta_true = rng.randn(n, 1)
    idxs = rng.choice(range(n), int(0.8 * n), replace=False)
    beta_true[idxs] = 0
    X = rng.normal(0, 5, size=(m, n))
    Y = np.sign(X.dot(beta_true) + rng.normal(0, 45, size=(m, 1)))
    return {"X": X, "y": Y.ravel()}


def _svm_l1_cvxpy(d):
    import cvxpy as cp
    X, y = d["X"], d["y"]
    m, n = X.shape
    beta = cp.Variable((n, 1))
    v = cp.Variable()
    lambd = cp.Parameter(nonneg=True)
    Y = y.reshape(-1, 1)
    loss = cp.sum(cp.pos(1 - cp.multiply(Y, X @ beta - v)))
    return cp.Problem(cp.Minimize(loss / m + lambd * cp.norm(beta, 1))), [lambd]


def _svm_l1_draw(rng, d):
    return [rng.uniform(0.05, 1.0)]


PROBLEMS = {
    "SimpleLP": E2EProblem("SimpleLP", False, _simple_lp_data, _simple_lp_cvxpy),
    "LeastSquares": E2EProblem("LeastSquares", False, _least_squares_data,
                               _least_squares_cvxpy),
    "SimpleQP": E2EProblem("SimpleQP", False, _simple_qp_data, _simple_qp_cvxpy),
    "ParametrizedQP": E2EProblem("ParametrizedQP", True, _parametrized_qp_data,
                                 _parametrized_qp_cvxpy, _parametrized_qp_draw),
    "HuberRegression": E2EProblem("HuberRegression", False, _huber_data,
                                  _huber_cvxpy),
    "SVMWithL1Regularization": E2EProblem(
        "SVMWithL1Regularization", True, _svm_l1_data, _svm_l1_cvxpy,
        _svm_l1_draw),
}
