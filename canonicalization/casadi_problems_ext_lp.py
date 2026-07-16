"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Extension problems for benchmarks/casadi_compare.py: the LP/QP benchmarks from
benchmarks/cvxpy_benchmarks/benchmark/ (matrix stuffing, slow pruning, Gini
portfolios, CVaR, optimal advertising).  Same Spec contract as the harness:
cvxpy_build() replicates the upstream formulation verbatim; casadi_build()
hand-writes the lowered model in cvxpy's exact column and row order (verified
entry-for-entry via --verify).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
import scipy.stats as st

import casadi as ca
import cvxpy as cp


# Mirrors casadi_compare.Spec (redefined here to avoid a circular import; the
# harness only reads these attributes).
@dataclass
class Spec:
    name: str
    parametric: bool
    cvxpy_build: object
    casadi_build: object
    draw: object = None
    notes: str = ""


# --------------------------------------------------------------------------- #
# Matrix stuffing family (benchmark/matrix_stuffing.py)
#   min sum(A x)  s.t.  C_i x_i <= b_i (i < m/2),  C_i x_{m/2+i} == b_{m/2+i}
# Lowered: columns = x; rows = zero cone (the equalities, in list order), then
# nonneg (the inequalities, in list order); g row for expr <= bnd is bnd - expr.
# Upstream draws data with the global np.random (no seed); RandomState(0) here
# so both sides share the same data.
# --------------------------------------------------------------------------- #
def _stuffing_spec(name: str, m: int) -> Spec:
    n = m
    rs = np.random.RandomState(0)
    A0 = rs.randn(m, n)
    C0 = rs.rand(m // 2)
    b0 = rs.randn(m)
    h = m // 2

    def cvxpy_build():
        x = cp.Variable(n)
        cost = cp.sum(A0 @ x)
        constraints = [C0[i] * x[i] <= b0[i] for i in range(h)]
        constraints.extend([C0[i] * x[h + i] == b0[h + i] for i in range(h)])
        return cp.Problem(cp.Minimize(cost), constraints), []

    def casadi_build():
        x = ca.MX.sym("x", n)
        f = ca.sum1(ca.mtimes(ca.DM(A0), x))
        g_zero = ca.DM(b0[h:]) - ca.DM(C0) * x[h:m]      # equalities first
        g_nonneg = ca.DM(b0[:h]) - ca.DM(C0) * x[:h]
        return x, [], f, ca.vertcat(g_zero, g_nonneg)

    return Spec(name, False, cvxpy_build, casadi_build)


def make_cone_matrix_stuffing(full: bool) -> Spec:
    return _stuffing_spec("ConeMatrixStuffing", 5000 if full else 4)


def make_small_matrix_stuffing(full: bool) -> Spec:
    return _stuffing_spec("SmallMatrixStuffing", 4000 if full else 4)


def _param_stuffing_spec(name: str, m: int) -> Spec:
    n = m
    h = m // 2

    def cvxpy_build():
        A = cp.Parameter((m, n))
        C = cp.Parameter(h)
        b = cp.Parameter(m)
        x = cp.Variable(n)
        cost = cp.sum(A @ x)
        constraints = [C[i] * x[i] <= b[i] for i in range(h)]
        constraints.extend([C[i] * x[h + i] == b[h + i] for i in range(h)])
        return cp.Problem(cp.Minimize(cost), constraints), [A, C, b]

    def casadi_build():
        x = ca.MX.sym("x", n)
        Ap = ca.MX.sym("A", m, n)
        Cp = ca.MX.sym("C", h)
        bp = ca.MX.sym("b", m)
        f = ca.sum1(ca.mtimes(Ap, x))
        g_zero = bp[h:m] - Cp * x[h:m]
        g_nonneg = bp[:h] - Cp * x[:h]
        return x, [Ap, Cp, bp], f, ca.vertcat(g_zero, g_nonneg)

    def draw(rng):
        return [rng.standard_normal((m, n)), rng.random(h), rng.standard_normal(m)]

    return Spec(name, True, cvxpy_build, casadi_build, draw)


def make_param_cone_matrix_stuffing(full: bool) -> Spec:
    return _param_stuffing_spec("ParamConeMatrixStuffing", 200 if full else 4)


def make_param_small_matrix_stuffing(full: bool) -> Spec:
    return _param_stuffing_spec("ParamSmallMatrixStuffing", 300 if full else 4)


# --------------------------------------------------------------------------- #
# SlowPruning (benchmark/slow_pruning_1668_benchmark.py, issue #1668)
#   min sum_squares(V M - Y),  V (rows, t), M (t, s), Y (rows, s)
# --------------------------------------------------------------------------- #
def make_slow_pruning(full: bool) -> Spec:
    rows, t, s = (100, 20, 4000) if full else (2, 2, 3)
    x = np.linspace(-100.0, 100.0, s)
    M0 = np.tile(np.array([x]), t).reshape((t, x.shape[0]))
    Y0 = np.tile(x, rows).reshape((rows, s))

    def cvxpy_build():
        var = cp.Variable(shape=(rows, t))
        cost = cp.sum_squares(var @ M0 - Y0)
        return cp.Problem(cp.Minimize(cost)), []

    def casadi_build():
        # lowered: min u'u  s.t.  u = vec_F(V M - Y); columns [u, vec_F(V)]
        u = ca.MX.sym("u", rows * s)
        v = ca.MX.sym("v", rows * t)
        z = ca.vertcat(u, v)
        V = ca.reshape(v, rows, t)                       # casadi is column-major
        f = ca.dot(u, u)
        g = ca.DM(Y0.flatten(order="F")) + u - ca.vec(ca.mtimes(V, ca.DM(M0)))
        return z, [], f, g

    return Spec("SlowPruning", False, cvxpy_build, casadi_build)


# --------------------------------------------------------------------------- #
# Gini portfolios (benchmark/gini_portfolio.py); shared data generator
# --------------------------------------------------------------------------- #
def _gini_returns(N: int, T: int) -> np.ndarray:
    rs = np.random.RandomState(123)
    cov = rs.rand(N, N) * 1.5 - 0.5
    cov = cov @ cov.T / 1000 + np.diag(rs.rand(N) * 0.7 + 0.3) / 1000
    mean = np.zeros(N) + 1 / 1000
    return st.multivariate_normal.rvs(mean=mean, cov=cov, size=T, random_state=rs)


def make_yitzhaki(full: bool) -> Spec:
    # Upstream Yitzhaki uses N=50, T=300.
    N, T = (50, 300) if full else (2, 3)
    returns = _gini_returns(N, T)
    D0 = np.array([]).reshape(0, N)
    for j in range(returns.shape[0] - 1):
        D0 = np.concatenate((D0, returns[j + 1:] - returns[j, :]), axis=0)
    pairs = T * (T - 1) // 2

    def cvxpy_build():
        d = cp.Variable((pairs, 1))
        w = cp.Variable((N, 1))
        apd = D0 @ w
        constraints = [d >= apd, d >= -apd, w >= 0, cp.sum(w) == 1]
        risk = cp.sum(d) / ((T - 1) * T)
        return cp.Problem(cp.Minimize(risk * 1000), constraints), []

    def casadi_build():
        # columns [d, w]; rows: zero (sum w = 1), then d - Dw, d + Dw, w
        d = ca.MX.sym("d", pairs)
        w = ca.MX.sym("w", N)
        z = ca.vertcat(d, w)
        f = 1000 * ca.sum1(d) / ((T - 1) * T)
        Dw = ca.mtimes(ca.DM(D0), w)
        g = ca.vertcat(1 - ca.sum1(w), d - Dw, d + Dw, w)
        return z, [], f, g

    return Spec("Yitzhaki", False, cvxpy_build, casadi_build)


def make_murray(full: bool) -> Spec:
    N, T = (50, 700) if full else (2, 3)
    returns = _gini_returns(N, T)
    pairs = T * (T - 1) // 2

    def cvxpy_build():
        d = cp.Variable((pairs, 1))
        w = cp.Variable((N, 1))
        ret_w = cp.Variable((T, 1))
        constraints = [ret_w == returns @ w]
        mat = np.zeros((pairs, T))                       # dense, as upstream
        ell = 0
        for j in range(T):
            for i in range(j + 1, T):
                mat[ell, i] = 1
                mat[ell, j] = -1
                ell += 1
        apd = mat @ ret_w
        constraints += [d >= apd, d >= -apd, w >= 0, cp.sum(w) == 1]
        risk = cp.sum(d) / ((T - 1) * T)
        return cp.Problem(cp.Minimize(risk * 1000), constraints), []

    def casadi_build():
        # columns [d, ret_w, w]; rows: zero (ret_w = R w rows, sum w = 1),
        # then d - mat ret_w, d + mat ret_w, w.  mat has 2 nnz per row, so it
        # is assembled sparse here (same values as the upstream dense loop).
        d = ca.MX.sym("d", pairs)
        rw = ca.MX.sym("ret_w", T)
        w = ca.MX.sym("w", N)
        z = ca.vertcat(d, rw, w)
        f = 1000 * ca.sum1(d) / ((T - 1) * T)
        r = np.arange(pairs)
        i_idx = np.concatenate([np.arange(j + 1, T) for j in range(T)])
        j_idx = np.repeat(np.arange(T), np.arange(T - 1, -1, -1))
        mat_sp = (
            sps.coo_matrix((np.ones(pairs), (r, i_idx)), shape=(pairs, T))
            + sps.coo_matrix((-np.ones(pairs), (r, j_idx)), shape=(pairs, T))
        ).tocsc()
        Mrw = ca.mtimes(ca.DM(mat_sp), rw)
        g = ca.vertcat(ca.mtimes(ca.DM(returns), w) - rw, 1 - ca.sum1(w),
                       d - Mrw, d + Mrw, w)
        return z, [], f, g

    return Spec("Murray", False, cvxpy_build, casadi_build)


def make_cajas(full: bool) -> Spec:
    N, T = (50, 800) if full else (2, 3)
    returns = _gini_returns(N, T)
    owa_w = []
    for i in range(1, T + 1):
        owa_w.append(2 * i - 1 - T)
    owa_w = np.array(owa_w) / (T * (T - 1))

    def cvxpy_build():
        w = cp.Variable((N, 1))
        a = cp.Variable((T, 1))
        b = cp.Variable((T, 1))
        y = cp.Variable((T, 1))
        constraints = [returns @ w == y, w >= 0, cp.sum(w) == 1]
        for i in range(T):
            constraints += [a[i] + b >= cp.multiply(owa_w[i], y)]
        risk = cp.sum(a + b)
        return cp.Problem(cp.Minimize(risk * 1000), constraints), []

    def casadi_build():
        # columns [a, b, w, y]; rows: zero (y = R w rows, sum w = 1), then
        # w >= 0, then T blocks a_i + b - owa_i y (each of length T)
        a = ca.MX.sym("a", T)
        b = ca.MX.sym("b", T)
        w = ca.MX.sym("w", N)
        y = ca.MX.sym("y", T)
        z = ca.vertcat(a, b, w, y)
        f = 1000 * (ca.sum1(a) + ca.sum1(b))
        blocks = [a[i] + b - owa_w[i] * y for i in range(T)]
        g = ca.vertcat(y - ca.mtimes(ca.DM(returns), w), 1 - ca.sum1(w), w, *blocks)
        return z, [], f, g

    return Spec("Cajas", False, cvxpy_build, casadi_build)


# --------------------------------------------------------------------------- #
# CVaR (benchmark/finance.py CVaRBenchmark)
#   min c'x  s.t.  alpha + gamma sum(pos(A x - alpha)) <= kappa, xmin<=x<=xmax
# Upstream seeds the global np.random with 0; RandomState(0) yields the same
# stream.  A (num_scenarios x num_assets*num_energy_segments) is genuinely
# dense (~805 MB at full size) and stays dense on both sides.
# --------------------------------------------------------------------------- #
def make_cvar(full: bool) -> Spec:
    rs = np.random.RandomState(0)
    n_scen, n_ast = (131072, 192) if full else (6, 3)
    price_scenarios = rs.randn(n_scen, n_ast)
    forward_price_scenarios = rs.randn(n_scen, n_ast)
    asset_energy_limits = rs.randn(n_ast, 2)
    bid_curve_prices = rs.randn(n_ast, 3)
    cvar_prob = 0.95
    cvar_kappa = 2.0

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
    gamma = 1.0 / (1.0 - cvar_prob) / num_scenarios
    kappa = cvar_kappa
    x_min = np.tile(asset_energy_limits[:, 0:1], (1, num_energy_segments)).reshape(-1)
    x_max = np.tile(asset_energy_limits[:, 1:2], (1, num_energy_segments)).reshape(-1)
    nx = num_assets * num_energy_segments

    def cvxpy_build():
        alpha = cp.Variable()
        x = cp.Variable(nx)
        prob = cp.Problem(
            cp.Minimize(c0.T @ x),
            [
                alpha + gamma * cp.sum(cp.pos(A0 @ x - alpha)) <= kappa,
                x >= x_min,
                x <= x_max,
            ],
        )
        return prob, []

    def casadi_build():
        # columns [x, alpha, u] (u = pos epigraph); rows (all nonneg):
        # u + alpha - A x, u, kappa - alpha - gamma sum(u), x - xmin, xmax - x
        x = ca.MX.sym("x", nx)
        alpha = ca.MX.sym("alpha")
        u = ca.MX.sym("u", num_scenarios)
        z = ca.vertcat(x, alpha, u)
        f = ca.dot(ca.DM(c0), x)
        # repmat instead of scalar broadcast: casadi 3.7.2 drops all but the
        # first alpha entry when jacobian() runs in reverse mode with batched
        # adjoint directions (the dense mtimes flips the ad-mode heuristic to
        # reverse); see casadi_issue_draft_broadcast_bug.md for the minimal
        # repro and localization
        g = ca.vertcat(
            u + ca.repmat(alpha, num_scenarios, 1) - ca.mtimes(ca.DM(A0), x),
            u,
            kappa - alpha - gamma * ca.sum1(u),
            x - ca.DM(x_min),
            ca.DM(x_max) - x,
        )
        return z, [], f, g

    return Spec("CVaR", False, cvxpy_build, casadi_build)


# --------------------------------------------------------------------------- #
# OptimalAdvertising (benchmark/optimal_advertising.py)
#   max sum_i min(R_i P_i D_i', B_i)  s.t.  D >= 0, D'1 <= T, D 1 >= c
# Upstream seeds the global np.random with 1; RandomState(1) is the same.
# --------------------------------------------------------------------------- #
def make_optimal_advertising(full: bool) -> Spec:
    m, n = (250, 1000) if full else (2, 3)
    rs = np.random.RandomState(1)
    SCALE = 10000
    B0 = rs.lognormal(mean=8, size=(m, 1)) + 10000
    B0 = 1000 * np.round(B0 / 1000)

    P_ad = rs.uniform(size=(m, 1))
    P_time = rs.uniform(size=(1, n))
    P0 = P_ad.dot(P_time)

    T0 = np.sin(np.linspace(-2 * np.pi / 2, 2 * np.pi - 2 * np.pi / 2, n)) * SCALE
    T0 += -np.min(T0) + SCALE
    c0 = rs.uniform(size=(m,))
    c0 *= 0.6 * T0.sum() / c0.sum()
    c0 = 1000 * np.round(c0 / 1000)
    R0 = np.array([rs.lognormal(c0.min() / c0[i]) for i in range(m)])

    def cvxpy_build():
        D = cp.Variable((m, n))
        Si = [cp.minimum(R0[i] * P0[i, :] @ D[i, :].T, B0[i]) for i in range(m)]
        objective = cp.Maximize(cp.sum(Si))
        constraints = [D >= 0, D.T @ np.ones(m) <= T0, D @ np.ones(n) >= c0]
        return cp.Problem(objective, constraints), []

    def casadi_build():
        # After FlipObjective each minimum becomes a hypograph var t_i with
        # min sum(t); columns [t, vec_F(D)]; rows (all nonneg): per i the pair
        # (t_i + R_i P_i D_i', B_i + t_i), then vec_F(D), T - D'1, D 1 - c
        t = ca.MX.sym("t", m)
        dvec = ca.MX.sym("D", m * n)
        z = ca.vertcat(t, dvec)
        Dmat = ca.reshape(dvec, m, n)                    # column-major = F order
        f = ca.sum1(t)
        RP = R0[:, None] * P0
        rows = []
        for i in range(m):
            rows.append(t[i] + ca.mtimes(ca.DM(RP[i:i + 1, :]), Dmat[i, :].T))
            rows.append(float(B0[i, 0]) + t[i])
        g = ca.vertcat(
            *rows,
            dvec,
            ca.DM(T0) - ca.mtimes(Dmat.T, ca.DM.ones(m)),
            ca.mtimes(Dmat, ca.DM.ones(n)) - ca.DM(c0),
        )
        return z, [], f, g

    return Spec("OptimalAdvertising", False, cvxpy_build, casadi_build)


MAKERS = {
    "ConeMatrixStuffing": make_cone_matrix_stuffing,
    "SmallMatrixStuffing": make_small_matrix_stuffing,
    "ParamConeMatrixStuffing": make_param_cone_matrix_stuffing,
    "ParamSmallMatrixStuffing": make_param_small_matrix_stuffing,
    "SlowPruning": make_slow_pruning,
    "Yitzhaki": make_yitzhaki,
    "Murray": make_murray,
    "Cajas": make_cajas,
    "CVaR": make_cvar,
    "OptimalAdvertising": make_optimal_advertising,
}
