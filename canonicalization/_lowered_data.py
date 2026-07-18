"""Shared problem data for the Julia (SCT+SMC+DI) and AMPL/ASL comparisons.

Four representative problems from the CasADi comparison, with the SAME seeds
and lowered layouts as their Specs in casadi_compare.py / casadi_problems_ext_lp.py,
so results are directly comparable across all three tools.  Each entry provides

  data(size)        -> dict of numpy constants (npz-able) + "dims" metadata
  cvxpy_build(d)    -> (cp.Problem, [cp.Parameter, ...])   ground truth side
  draw(rng, d)      -> [param values ...] fresh parameter draws (same order)

Lowered layout (identical to the CasADi Specs; cones are metadata):
  SimpleQP        z = x;               rows: zero (Aeq), nonneg (G0)
  LeastSquares    z = [t, x];          rows: zero (b0 + t - A0 x)
  ParametrizedQP  z = [t, x];          rows: zero (b + t - A x), nonneg (x, 1-x)
  CVaRSlice       z = [x, alpha, u];   rows: all nonneg (see ext_lp make_cvar)

CVaRSlice is the benchmark CVaR generator at num_scenarios=8192 (1/16 of the
suite's 131072): the full size DNFs in CasADi (>10 h) and needs a ~100M-term
Pyomo model for ASL.  The CasADi reference at the slice size is the
probe_coloring_sweeps.py measurement, not the suite table.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sps

import cvxpy as cp


def _simple_qp(size: str) -> dict:
    m, n, p_ = {"small": (4, 3, 1), "full": (8000, 1600, 20)}[size]
    rng = np.random.RandomState(1)
    P0 = rng.randn(n, n)
    P0 = P0.T @ P0
    q0 = rng.randn(n)
    G0 = rng.randn(m, n)
    h0 = G0 @ rng.randn(n)
    Aeq = rng.randn(p_, n)
    beq = rng.randn(p_)
    return {"P0": P0, "q0": q0, "G0": G0, "h0": h0, "Aeq": Aeq, "beq": beq,
            "dims": {"m": m, "n": n, "p": p_}}


def _simple_qp_cvxpy(d):
    n = d["dims"]["n"]
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize((1 / 2) * cp.quad_form(x, d["P0"], assume_PSD=True) + d["q0"].T @ x),
        [d["G0"] @ x <= d["h0"], d["Aeq"] @ x == d["beq"]],
    )
    return prob, []


def _least_squares(size: str) -> dict:
    m, n = {"small": (3, 2), "full": (7000, 2000)}[size]
    rng = np.random.RandomState(1)
    return {"A0": rng.randn(m, n), "b0": rng.randn(m), "dims": {"m": m, "n": n}}


def _least_squares_cvxpy(d):
    n = d["dims"]["n"]
    x = cp.Variable(n)
    return cp.Problem(cp.Minimize(cp.sum_squares(d["A0"] @ x - d["b0"]))), []


def _parametrized_qp(size: str) -> dict:
    m, n = {"small": (3, 2), "full": (6000, 2400)}[size]
    return {"dims": {"m": m, "n": n}}


def _parametrized_qp_cvxpy(d):
    m, n = d["dims"]["m"], d["dims"]["n"]
    A = cp.Parameter((m, n))
    b = cp.Parameter((m,))
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [0 <= x, x <= 1])
    return prob, [A, b]


def _parametrized_qp_draw(rng, d):
    m, n = d["dims"]["m"], d["dims"]["n"]
    return [rng.standard_normal((m, n)), rng.standard_normal(m)]


def _cvar(size: str) -> dict:
    # generator copied from casadi_problems_ext_lp.make_cvar (RandomState(0)),
    # with num_scenarios reduced at "slice" size
    rs = np.random.RandomState(0)
    n_scen, n_ast = {"small": (6, 3), "slice": (8192, 192)}[size]
    price_scenarios = rs.randn(n_scen, n_ast)
    forward_price_scenarios = rs.randn(n_scen, n_ast)
    asset_energy_limits = rs.randn(n_ast, 2)
    bid_curve_prices = rs.randn(n_ast, 3)

    num_scenarios, num_assets = price_scenarios.shape
    num_energy_segments = bid_curve_prices.shape[1] + 1

    price_segments = np.sum(
        forward_price_scenarios[:, :, None] > bid_curve_prices[None], axis=2
    )
    price_segments_flat = (
        price_segments + np.arange(num_assets) * num_energy_segments
    ).reshape(-1)
    price_segments_sp = sps.coo_matrix(
        (
            np.ones(num_scenarios * num_assets),
            (np.arange(num_scenarios * num_assets), price_segments_flat),
        ),
        shape=(num_scenarios * num_assets, num_assets * num_energy_segments),
    )
    prices_flat = (price_scenarios - forward_price_scenarios).reshape(-1)
    scenario_sum = sps.coo_matrix(
        (
            np.ones(num_scenarios * num_assets),
            (
                np.repeat(np.arange(num_scenarios), num_assets),
                np.arange(num_scenarios * num_assets),
            ),
        )
    )
    A0 = np.asarray((scenario_sum @ sps.diags(prices_flat) @ price_segments_sp).todense())
    c0 = np.mean(A0, axis=0)
    gamma = 1.0 / (1.0 - 0.95) / num_scenarios
    x_min = np.tile(asset_energy_limits[:, 0:1], (1, num_energy_segments)).reshape(-1)
    x_max = np.tile(asset_energy_limits[:, 1:2], (1, num_energy_segments)).reshape(-1)
    nx = num_assets * num_energy_segments
    return {"A0": A0, "c0": c0, "x_min": x_min, "x_max": x_max,
            "gamma": np.float64(gamma), "kappa": np.float64(2.0),
            "dims": {"n_scen": num_scenarios, "nx": nx}}


def _cvar_cvxpy(d):
    nx = d["dims"]["nx"]
    alpha = cp.Variable()
    x = cp.Variable(nx)
    prob = cp.Problem(
        cp.Minimize(d["c0"].T @ x),
        [
            alpha + float(d["gamma"]) * cp.sum(cp.pos(d["A0"] @ x - alpha)) <= float(d["kappa"]),
            x >= d["x_min"],
            x <= d["x_max"],
        ],
    )
    return prob, []


class Entry:
    def __init__(self, name, sizes, data, cvxpy_build, draw=None):
        self.name = name
        self.sizes = sizes            # (small_key, benchmark_key)
        self.data = data
        self.cvxpy_build = cvxpy_build
        self.draw = draw
        self.parametric = draw is not None


PROBLEMS = {
    "SimpleQP": Entry("SimpleQP", ("small", "full"), _simple_qp, _simple_qp_cvxpy),
    "LeastSquares": Entry("LeastSquares", ("small", "full"),
                          _least_squares, _least_squares_cvxpy),
    "ParametrizedQP": Entry("ParametrizedQP", ("small", "full"),
                            _parametrized_qp, _parametrized_qp_cvxpy,
                            _parametrized_qp_draw),
    "CVaRSlice": Entry("CVaRSlice", ("small", "slice"), _cvar, _cvar_cvxpy),
}


def reference_data(name: str, size: str, param_vals=None):
    """CVXPY's (P, c, A, b) for this problem instance -- the ground truth."""
    e = PROBLEMS[name]
    d = e.data(size)
    prob, params = e.cvxpy_build(d)
    if e.parametric:
        for p, v in zip(params, param_vals):
            p.value = v
    data, _, _ = prob.get_problem_data(solver=cp.CLARABEL, ignore_dpp=True)
    return data
