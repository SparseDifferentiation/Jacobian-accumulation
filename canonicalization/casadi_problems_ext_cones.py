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

Cone-lowered problems for the CasADi-vs-CVXPY extraction comparison
(see casadi_compare.py for the Spec contract).  Cone membership is pure
metadata: the lowered slack expression g(z, p) = b - Az is affine in z
even for SOC/PSD problems, so it is expressible as CasADi MX.  Row and
column order below mirror cvxpy's CLARABEL lowering, decoded empirically
from get_problem_data at tiny sizes and checked by --verify.
"""
from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np
import scipy.sparse as sps

import cvxpy as cp


# Duck-typed copy of casadi_compare.Spec (importing it back would be circular:
# casadi_compare imports this module while it is executing as __main__).
@dataclass
class Spec:
    name: str
    parametric: bool
    cvxpy_build: object
    casadi_build: object
    draw: object = None
    notes: str = ""


def make_factor_covariance(full: bool) -> Spec:
    # benchmarks/cvxpy_benchmarks/benchmark/finance.py::FactorCovarianceModel
    n, m = (25000, 250) if full else (3, 2)
    rng = np.random.RandomState(1)  # same stream as upstream np.random.seed(1)
    mu = np.abs(rng.randn(n, 1))
    Sigma_tilde = rng.randn(m, m)
    Sigma_tilde = Sigma_tilde.T.dot(Sigma_tilde)
    d_diag = rng.uniform(0, 0.9, size=n)
    D = sps.diags(d_diag)
    F0 = rng.randn(n, m)

    def cvxpy_build():
        w = cp.Variable(n)
        f = cp.Variable(m)
        gamma = cp.Parameter(nonneg=True)
        Lmax = cp.Parameter()
        ret = mu.T @ w
        risk = cp.quad_form(f, Sigma_tilde, assume_PSD=True) + cp.sum_squares(np.sqrt(D) @ w)
        objective = cp.Maximize(ret - gamma * risk)
        constraints = [cp.sum(w) == 1, f == F0.T @ w, cp.norm(w, 1) <= Lmax]
        return cp.Problem(objective, constraints), [gamma, Lmax]

    def casadi_build():
        # lowered: min -mu'w + gamma*(f'St f + t't)
        #   s.t.  t = sqrt(D) w ; sum(w) = 1 ; f = F'w   (zero cone)
        #         |w| <= s ; sum(s) <= Lmax              (nonneg cone)
        # columns [w, f, t, s]; rows: t-rows, sum(w), f-rows, s-w, s+w, Lmax-sum(s)
        # gamma lands in P (parameter-dependent quadratic); Lmax lands in b.
        w = ca.MX.sym("w", n)
        fv = ca.MX.sym("f", m)
        t = ca.MX.sym("t", n)
        s = ca.MX.sym("s", n)
        gam = ca.MX.sym("gamma")
        Lm = ca.MX.sym("Lmax")
        z = ca.vertcat(w, fv, t, s)
        f = -ca.dot(ca.DM(mu.ravel()), w) \
            + gam * (ca.bilin(ca.DM(Sigma_tilde), fv, fv) + ca.dot(t, t))
        g_zero = ca.vertcat(
            t - ca.DM(np.sqrt(d_diag)) * w,
            1 - ca.sum1(w),
            ca.mtimes(ca.DM(F0.T), w) - fv,
        )
        g_nonneg = ca.vertcat(s - w, s + w, Lm - ca.sum1(s))
        return z, [gam, Lm], f, ca.vertcat(g_zero, g_nonneg)

    def draw(rng_):
        return [rng_.uniform(0.05, 1.0), rng_.uniform(1.0, 3.0)]

    return Spec("FactorCovarianceModel", True, cvxpy_build, casadi_build, draw)


def make_convex_plasticity(full: bool) -> Spec:
    # benchmarks/cvxpy_benchmarks/benchmark/high_dim_convex_plasticity.py
    N = 3000 if full else 2
    E_dim = 70e3
    E = 70e3 / E_dim
    nu = 0.3
    sig0 = 250 / E_dim
    Et = E / 100.0
    H = E * Et / (E - Et)
    lam, mus = E * nu / (1 + nu) / (1 - 2 * nu), E / 2 / (1 + nu)
    C = np.array(
        [
            [lam + 2 * mus, lam, lam, 0],
            [lam, lam + 2 * mus, lam, 0],
            [lam, lam, lam + 2 * mus, 0],
            [0, 0, 0, 2 * mus],
        ]
    )
    S = np.linalg.inv(C)
    dev = np.array(
        [
            [2 / 3.0, -1 / 3.0, -1 / 3.0, 0],
            [-1 / 3.0, 2 / 3.0, -1 / 3.0, 0],
            [-1 / 3.0, -1 / 3.0, 2 / 3.0, 0],
            [0, 0, 0, 1.0],
        ]
    )
    S_sparsed = sps.block_diag([S for _ in range(N)])

    def cvxpy_build():
        deps = cp.Parameter((4, N), name="deps")
        sig_old = cp.Parameter((4, N), name="sig_old")
        sig_elas = sig_old + C @ deps
        sig = cp.Variable((4, N), name="sig")
        p_old = cp.Parameter((N,), nonneg=True, name="p_old")
        p = cp.Variable((N,), nonneg=True, name="p")
        delta_sig = sig - sig_elas
        D = H * np.eye(N)
        delta_sig_vector = cp.reshape(delta_sig, (N * 4,), order="F")
        elastic_energy = cp.quad_form(delta_sig_vector, S_sparsed, assume_PSD=True)
        target = 0.5 * elastic_energy + 0.5 * cp.quad_form(p - p_old, D)
        sig0_vec = np.repeat(sig0, N)
        constraints = [np.sqrt(3 / 2) * cp.norm(dev @ sig, axis=0) <= sig0_vec + p * H]
        return cp.Problem(cp.Minimize(target), constraints), [deps, sig_old, p_old]

    def casadi_build():
        # lowered: min (1/2) d'S_blk d + (1/2) H e'e
        #   s.t.  d = vec(sig) - vec(sig_old + C deps) ; e = p - p_old  (zero cone)
        #         p >= 0 ; sig0 + H p - sqrt(3/2) t >= 0               (nonneg)
        #         N SOC blocks of size 5: [t_i; dev @ sig[:, i]]
        # columns [d (4N), e (N), p (N), vec(sig) (4N, F-order), t (N)]
        d = ca.MX.sym("d", 4 * N)
        e = ca.MX.sym("e", N)
        p = ca.MX.sym("p", N)
        sig = ca.MX.sym("sig", 4, N)
        t = ca.MX.sym("t", N)
        z = ca.vertcat(d, e, p, ca.reshape(sig, 4 * N, 1), t)
        deps = ca.MX.sym("deps", 4, N)
        sig_old = ca.MX.sym("sig_old", 4, N)
        p_old = ca.MX.sym("p_old", N)
        # cvxpy stores the p-block of P densely (the upstream model passes the
        # dense H*eye(N) array), but those are explicit zeros -- the comparator
        # ignores them, and a CasADi user would write the diagonal sparsely.
        f = 0.5 * ca.bilin(ca.DM(S_sparsed.tocsc()), d, d) \
            + 0.5 * ca.bilin(ca.DM(sps.eye(N, format="csc") * H), e, e)
        sig_elas = sig_old + ca.mtimes(ca.DM(C), deps)
        g_zero = ca.vertcat(
            ca.reshape(sig_elas, 4 * N, 1) + d - ca.reshape(sig, 4 * N, 1),
            p_old + e - p,
        )
        g_nonneg = ca.vertcat(p, sig0 + H * p - np.sqrt(3 / 2) * t)
        # dev sparsified: its exact zeros are pruned from cvxpy's A.
        dev_dm = ca.DM(sps.csc_matrix(dev))
        soc = ca.vertcat(ca.reshape(t, 1, N), ca.mtimes(dev_dm, sig))  # (5, N)
        g_soc = ca.reshape(soc, 5 * N, 1)  # column-major: one 5-block per i
        return z, [deps, sig_old, p_old], f, ca.vertcat(g_zero, g_nonneg, g_soc)

    def draw(rng_):
        return [
            1e-3 * rng_.standard_normal((4, N)),
            0.1 * rng_.standard_normal((4, N)),
            rng_.uniform(0.0, 0.1, size=N),  # p_old parameter is nonneg
        ]

    return Spec("ConvexPlasticity", True, cvxpy_build, casadi_build, draw)


def make_tv_inpainting(full: bool) -> Spec:
    # benchmarks/cvxpy_benchmarks/benchmark/tv_inpainting.py (upstream is
    # unseeded; we fix RandomState(1) with the upstream draw order so both
    # sides share identical data)
    rows, cols, colors = (512, 512, 3) if full else (3, 3, 3)
    rng = np.random.RandomState(1)
    Uorig = rng.randn(rows, cols, colors)
    mask = (rng.random_sample((rows, cols)) > 0.7).astype(float)
    known = np.repeat(mask[:, :, None], colors, axis=2)
    Ucorr = known * Uorig
    L = (rows - 1) * (cols - 1)

    def cvxpy_build():
        variables = []
        constraints = []
        for i in range(colors):
            U = cp.Variable(shape=(rows, cols))
            variables.append(U)
            constraints.append(
                cp.multiply(known[:, :, i], U) == cp.multiply(known[:, :, i], Ucorr[:, :, i])
            )
        return cp.Problem(cp.Minimize(cp.tv(*variables)), constraints), []

    def casadi_build():
        # lowered: min sum(t)
        #   s.t.  known .* U_k = known .* Ucorr_k per channel      (zero cone)
        #         L SOC blocks of size 7: [t_j; dx1; dy1; dx2; dy2; dx3; dy3]
        #         (pixel j runs column-major over the (rows-1)x(cols-1) grid)
        # columns [t (L), vec(U1), vec(U2), vec(U3)] (F-order vecs)
        t = ca.MX.sym("t", L)
        Us = [ca.MX.sym(f"U{k}", rows, cols) for k in range(colors)]
        z = ca.vertcat(t, *[ca.reshape(U, rows * cols, 1) for U in Us])
        f = ca.sum1(t)
        g_zero = []
        for k in range(colors):
            # sparse mask: zero-mask coefficients are pruned from cvxpy's A
            kv = ca.DM(sps.csc_matrix(known[:, :, k].flatten(order="F").reshape(-1, 1)))
            bk = ca.DM((known[:, :, k] * Ucorr[:, :, k]).flatten(order="F"))
            g_zero.append(bk - kv * ca.reshape(Us[k], rows * cols, 1))
        diffs = []
        for U in Us:  # same order as cvxpy's tv atom: (dx, dy) per channel
            dx = U[0:rows - 1, 1:cols] - U[0:rows - 1, 0:cols - 1]
            dy = U[1:rows, 0:cols - 1] - U[0:rows - 1, 0:cols - 1]
            diffs += [ca.reshape(dx, 1, L), ca.reshape(dy, 1, L)]
        stacked = ca.vertcat(ca.reshape(t, 1, L), *diffs)  # (1 + 2*colors, L)
        g_soc = ca.reshape(stacked, (1 + 2 * colors) * L, 1)
        return z, [], f, ca.vertcat(*g_zero, g_soc)

    return Spec("TvInpainting", False, cvxpy_build, casadi_build)


def make_semidefinite_programming(full: bool) -> Spec:
    # benchmarks/cvxpy_benchmarks/benchmark/semidefinite_programming.py
    n, p_ = (200, 120) if full else (3, 2)
    rng = np.random.RandomState(1)  # same stream/order as upstream np.random.seed(1)
    C0 = rng.randn(n, n)
    A_list = []
    b_list = []
    for _ in range(p_):
        A_list.append(rng.randn(n, n))
        b_list.append(rng.randn())
    b_arr = np.array(b_list)
    nsv = n * (n + 1) // 2

    # cvxpy's symmetric-variable columns are the lower-triangular entries of X
    # in COLUMN-major order: (0,0),(1,0),...,(n-1,0),(1,1),...  For trace(M@X)
    # the coefficient on column (i,j) is M[i,j]+M[j,i] off-diagonal, M[j,j] on
    # the diagonal.
    jj, ii = np.triu_indices(n)  # (j, i) pairs, j <= i, in column-major-lower order

    def tr_vec(M):
        v = (M + M.T)[ii, jj]
        v[ii == jj] /= 2.0
        return v

    # PSD slack rows use ROW-major lower-triangular (svec) order:
    # (0,0),(1,0),(1,1),(2,0),(2,1),(2,2),... with off-diagonals * sqrt(2).
    ri, rj = np.tril_indices(n)
    col_pos = (rj * n - rj * (rj - 1) // 2 + (ri - rj)).astype(int)
    scale = np.where(ri == rj, 1.0, np.sqrt(2.0))
    Aeq = np.stack([tr_vec(Ai) for Ai in A_list])  # (p_, nsv) dense
    c_obj = tr_vec(C0)

    def cvxpy_build():
        X = cp.Variable((n, n), symmetric=True)
        constraints = [X >> 0]
        constraints += [cp.trace(A_list[i] @ X) == b_list[i] for i in range(p_)]
        return cp.Problem(cp.Minimize(cp.trace(C0 @ X)), constraints), []

    def casadi_build():
        # columns: the nsv distinct entries of X (column-major lower order);
        # rows: p_ trace equalities (zero cone), then one PSD block of nsv
        # svec rows (row-major lower order, off-diagonals scaled by sqrt(2)).
        x = ca.MX.sym("x", nsv)
        f = ca.dot(ca.DM(c_obj), x)
        g_zero = ca.DM(b_arr) - ca.mtimes(ca.DM(Aeq), x)
        g_psd = ca.DM(scale) * x[col_pos.tolist()]
        return x, [], f, ca.vertcat(g_zero, g_psd)

    return Spec("SemidefiniteProgramming", False, cvxpy_build, casadi_build)


MAKERS = {
    "FactorCovarianceModel": make_factor_covariance,
    "ConvexPlasticity": make_convex_plasticity,
    "TvInpainting": make_tv_inpainting,
    "SemidefiniteProgramming": make_semidefinite_programming,
}
