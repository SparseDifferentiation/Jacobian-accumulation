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

Kron-structured problems for the CasADi-vs-CVXPY extraction comparison
(see casadi_compare.py for the Spec contract).  The CasADi models rebuild
the lowered expression chain symbolically: only data-independent index
permutations/selections (partial_transpose / partial_trace / diag / svec
layout) are precomputed -- never the numeric coefficient matrices of the
variables, which CasADi's AD must recover itself.  Row/column order mirrors
cvxpy's CLARABEL lowering, decoded empirically at tiny sizes.
"""
from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np
import scipy.sparse as sps
from scipy.linalg import dft

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


# --------------------------------------------------------------------------- #
# Structural (data-independent) layout helpers
# --------------------------------------------------------------------------- #
def svec_maps(n: int):
    """Layout maps for an n x n symmetric/PSD cvxpy variable.

    Columns are the n(n+1)/2 distinct entries in COLUMN-major lower-triangular
    order; the PSD slack rows are in ROW-major lower-triangular (svec) order
    with off-diagonals scaled by sqrt(2).

    Returns (dup_idx, diag_pos, psd_perm, psd_scale):
      dup_idx:  vecF(full matrix) index -> svec column (duplicates off-diag)
      diag_pos: svec columns holding the diagonal entries
      psd_perm: svec column supplying each PSD slack row
      psd_scale: per-PSD-row scaling (1 on diagonal, sqrt(2) off)
    """
    pos = np.zeros((n, n), dtype=int)
    k = 0
    for j in range(n):
        for i in range(j, n):
            pos[i, j] = pos[j, i] = k
            k += 1
    dup_idx = np.array([pos[i, j] for j in range(n) for i in range(n)])
    diag_pos = np.array([pos[j, j] for j in range(n)])
    ri, rj = np.tril_indices(n)
    psd_perm = pos[ri, rj]
    psd_scale = np.where(ri == rj, 1.0, np.sqrt(2.0))
    return dup_idx, diag_pos, psd_perm, psd_scale


def pt_perm(p: int, q: int) -> np.ndarray:
    """vecF permutation of cp.partial_transpose(X, dims=[p, q], axis=0)."""
    n = p * q
    perm = np.empty(n * n, dtype=int)
    for i1 in range(p):
        for i2 in range(q):
            for j1 in range(p):
                for j2 in range(q):
                    r_out = j1 * q + i2
                    c_out = i1 * q + j2
                    perm[c_out * n + r_out] = (j1 * q + j2) * n + (i1 * q + i2)
    return perm


def ptrace_sel(a: int, b: int, c: int) -> sps.csc_matrix:
    """vecF selection-sum matrix of cp.partial_trace(M, dims=[a,b,c], axis=1)."""
    i1, i3, j1, j3, kk = np.meshgrid(
        np.arange(a), np.arange(c), np.arange(a), np.arange(c), np.arange(b),
        indexing="ij",
    )
    k_out = (j1 * c + j3) * (a * c) + (i1 * c + i3)
    k_in = (j1 * b * c + kk * c + j3) * (a * b * c) + (i1 * b * c + kk * c + i3)
    return sps.coo_matrix(
        (np.ones(k_out.size), (k_out.ravel(), k_in.ravel())),
        shape=((a * c) ** 2, (a * b * c) ** 2),
    ).tocsc()


# --------------------------------------------------------------------------- #
# 1. UnconstrainedQP (simple_QP_benchmarks.py) -- complex variable + DFT
# --------------------------------------------------------------------------- #
def make_unconstrained_qp(full: bool) -> Spec:
    # tiny size uses dim=6: dft(4) is exactly unitary in floats, which produces
    # exact cross-term cancellations that cvxpy prunes from A but that remain
    # structural in casadi; dft(6) has no exact zeros beyond the data's own.
    N_r, N_t, N_s = (18, 2, 7) if full else (2, 1, 3)
    k = N_s * N_t
    dim = N_s * N_r * N_t
    rng = np.random.RandomState(1)  # upstream is unseeded; fixed for determinism
    x0 = rng.randint(2, size=dim)
    H = dft(dim) * 1j
    H_H = H.conj().T
    err = rng.random(N_r)
    Err = np.kron(np.diag(np.ones(k)), np.diag(err))
    y = H_H @ Err @ H @ x0

    def cvxpy_build():
        var = cp.Variable(shape=(N_r), complex=True)
        Err_est = cp.kron(np.diag(np.ones(k)), cp.diag(var))
        res = cp.norm2(H_H @ Err_est @ H @ x0 - y)
        return cp.Problem(cp.Minimize(res)), []

    def casadi_build():
        # Complex2Real lowering: r = H_H (tile(v) .* Hx) - y with v = vr + i*vi.
        # columns [t, m (dim), vr (N_r), vi (N_r)]
        # rows: dim SOC blocks of size 3, one per residual entry:
        #   [m_i; Re r_i = Mr.vr - Mi.vi - Re y; Im r_i = Mi.vr + Mr.vi - Im y]
        # then one SOC block of size 1 + dim: [t; m].
        t = ca.MX.sym("t")
        m = ca.MX.sym("m", dim)
        vr = ca.MX.sym("vr", N_r)
        vi = ca.MX.sym("vi", N_r)
        z = ca.vertcat(t, m, vr, vi)
        f = t
        # Constants sparsified (csc drops exact zeros only) so that data zeros
        # are structural, matching cvxpy's pruned coefficient support.
        hx = H @ x0  # constant vector of user data, NOT the coefficient matrix
        hr = ca.DM(sps.csc_matrix(hx.real.reshape(-1, 1)))
        hi = ca.DM(sps.csc_matrix(hx.imag.reshape(-1, 1)))
        vrt = ca.repmat(vr, k, 1)  # kron(I_k, diag(v)) @ w == tile(v, k) .* w
        vit = ca.repmat(vi, k, 1)
        ur = hr * vrt - hi * vit
        ui = hr * vit + hi * vrt
        Cr = ca.DM(sps.csc_matrix(H_H.real))
        Ci = ca.DM(sps.csc_matrix(H_H.imag))
        rr = ca.mtimes(Cr, ur) - ca.mtimes(Ci, ui) - ca.DM(y.real)
        ri = ca.mtimes(Cr, ui) + ca.mtimes(Ci, ur) - ca.DM(y.imag)
        blocks = ca.vertcat(
            ca.reshape(m, 1, dim), ca.reshape(rr, 1, dim), ca.reshape(ri, 1, dim)
        )  # column i is SOC block i
        g = ca.vertcat(ca.reshape(blocks, 3 * dim, 1), t, m)
        return z, [], f, g

    return Spec("UnconstrainedQP", False, cvxpy_build, casadi_build)


# --------------------------------------------------------------------------- #
# 2. QuantumHilbertMatrix (quantum_hilbert_matrix.py) -- PSD + kron + ptrace
# --------------------------------------------------------------------------- #
def make_quantum_hilbert(full: bool) -> Spec:
    d_, N_ = (2, 3) if full else (2, 1)
    dim = d_ ** N_
    n2 = dim ** 2
    I = sps.identity(dim)  # noqa: E741 -- mirrors upstream
    rng = np.random.default_rng(0)
    rhs = sps.random(m=n2, n=n2, density=0.05810546875, random_state=rng)
    A0 = sps.random(m=n2, n=n2, density=0.29058837890625, random_state=rng)
    AxI = sps.kron(A0, I)
    # scipy's kron stores explicit zeros (dense blocks); drop them so casadi's
    # structural pattern matches cvxpy's pruned coefficient support.
    AxI_csc = AxI.tocsc()
    AxI_csc.eliminate_zeros()

    dup_idx, _, psd_perm, psd_scale = svec_maps(n2)
    pt_idx = dup_idx[pt_perm(dim, dim)]  # x_sv -> vecF(partial_transpose(X))
    Tsel = ptrace_sel(dim, dim, dim)
    rhs_vec = rhs.toarray().flatten(order="F")

    def cvxpy_build():
        X = cp.Variable((n2, n2), PSD=True)
        lhs = cp.partial_trace(
            cp.kron(I, cp.partial_transpose(X, dims=[dim, dim], axis=0)) @ AxI,
            dims=[dim, dim, dim],
            axis=1,
        )
        return cp.Problem(cp.Minimize(cp.sum_squares(lhs - rhs))), []

    def casadi_build():
        # lowered: min t't  s.t.  t = vecF(lhs) - vecF(rhs)   (zero cone)
        #          one PSD block of n2(n2+1)/2 svec rows on X
        # columns [t (n2^2), x_sv (n2(n2+1)/2, column-major lower)]
        t = ca.MX.sym("t", n2 * n2)
        x = ca.MX.sym("x", n2 * (n2 + 1) // 2)
        z = ca.vertcat(t, x)
        f = ca.dot(t, t)
        ptx = ca.reshape(x[pt_idx.tolist()], n2, n2)  # partial_transpose(X)
        K = ca.mtimes(ca.kron(ca.DM(I.tocsc()), ptx), ca.DM(AxI_csc))
        lhs_vec = ca.mtimes(ca.DM(Tsel), ca.reshape(K, (dim * n2) ** 2, 1))
        g_zero = ca.DM(rhs_vec) + t - lhs_vec
        g_psd = ca.DM(psd_scale) * x[psd_perm.tolist()]
        return z, [], f, ca.vertcat(g_zero, g_psd)

    return Spec("QuantumHilbertMatrix", False, cvxpy_build, casadi_build)


# --------------------------------------------------------------------------- #
# 3. SDPSegfault1132 (sdp_segfault_1132_benchmark.py) -- kron of diag(VGV')
# --------------------------------------------------------------------------- #
def make_sdp_segfault_1132(full: bool) -> Spec:
    n = 100 if full else 4
    alpha = 1
    rng = np.random.RandomState(0)  # same stream as upstream np.random.seed(0)
    points = rng.rand(5, n)
    xtx = points.T @ points
    xtxd = np.diag(xtx)
    e1 = np.ones((n,))
    D = np.outer(e1, xtxd) - 2 * xtx + np.outer(xtxd, e1)
    W = np.ones((n, n))
    xv = -1 / (n + np.sqrt(n))
    yv = -1 / np.sqrt(n)
    V = np.ones((n, n - 1))
    V[0, :] *= yv
    V[1:, :] *= xv
    V[1:, :] += np.eye(n - 1)
    e = np.ones((n, 1))
    ng = n - 1
    dup_idx, diag_pos, psd_perm, psd_scale = svec_maps(ng)
    # Row-difference matrix: row (j*n + i) is V[i,:] - V[j,:] (vecF order of the
    # residual).  The identity M_ii + M_jj - 2 M_ij = (v_i - v_j) G (v_i - v_j)'
    # (M = VGV', G symmetric) holds symbolically; because V repeats the constant
    # x below row 0, these constant differences carry exact zeros, so csc
    # sparsification reproduces cvxpy's pruned coefficient support structurally.
    Ediff = sps.csc_matrix(
        np.vstack([V[i, :] - V[j, :] for j in range(n) for i in range(n)])
    )
    D_vec = D.flatten(order="F")
    W_vec = W.flatten(order="F")

    def cvxpy_build():
        G = cp.Variable((ng, ng), PSD=True)
        A = cp.kron(e, cp.reshape(cp.diag(V @ G @ V.T), (1, n), order="F"))
        B = cp.kron(e.T, cp.reshape(cp.diag(V @ G @ V.T), (n, 1), order="F"))
        C = alpha * cp.norm(cp.multiply(W, A + B - 2 * V @ G @ V.T - D), p="fro")
        return cp.Problem(cp.Maximize(cp.trace(G) - C)), []

    def casadi_build():
        # lowered (after FlipObjective): min -trace(G) + alpha*t
        #   rows: SOC block of size 1 + n^2: [t; vecF(W .* (A + B - 2VGV' - D))]
        #         then one PSD block of ng(ng+1)/2 svec rows on G
        # columns [g_sv (ng(ng+1)/2, column-major lower), t]
        g_sv = ca.MX.sym("g", ng * (ng + 1) // 2)
        t = ca.MX.sym("t")
        z = ca.vertcat(g_sv, t)
        f = alpha * t - ca.sum1(g_sv[diag_pos.tolist()])
        Gfull = ca.reshape(g_sv[dup_idx.tolist()], ng, ng)
        # vecF(A + B - 2VGV') via the row-difference identity: entry (i,j) is
        # (v_i - v_j) G (v_i - v_j)', i.e. rowwise bilinear forms in G.
        E = ca.DM(Ediff)
        r_vec = ca.sum2(ca.mtimes(E, Gfull) * E)
        resid = ca.DM(W_vec) * (r_vec - ca.DM(D_vec))
        g_soc = ca.vertcat(t, resid)
        g_psd = ca.DM(psd_scale) * g_sv[psd_perm.tolist()]
        return z, [], f, ca.vertcat(g_soc, g_psd)

    return Spec("SDPSegfault1132", False, cvxpy_build, casadi_build)


MAKERS = {
    "UnconstrainedQP": make_unconstrained_qp,
    "QuantumHilbertMatrix": make_quantum_hilbert,
    "SDPSegfault1132": make_sdp_segfault_1132,
}
