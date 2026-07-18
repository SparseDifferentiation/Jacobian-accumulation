"""Tool-neutral lowered forms of the CVXPY-suite problems, for the JuMP/MOI
and AMPL/ASL comparisons.

Each entry lowers a suite problem to

    minimize (1/2) z' P z + c' z      subject to    b - A z  in  K

with K given as an ordered list of row blocks (zero cone, nonneg cone, SOC
blocks, one PSD svec block) in CVXPY's CLARABEL row order, and z in CVXPY's
column order.  The definitions are direct translations of the verified CasADi
builders in casadi_compare.py / casadi_problems_ext_*.py (same seeds, same
layouts); the CVXPY reference side is reused from those modules' Specs, so
--verify compares against the identical ground truth.

Boundary note (deliberate, documented in the README): unlike the CasADi side
-- where coefficients sit in an expression graph and AD must *recover* them --
declared-structure tools take explicit coefficient lists as their input
contract.  Assembling these blocks from the user's data arrays is therefore
the *modeling* step for JuMP/ASL (reported as build time, excluded from the
extraction headline), exactly as `casadi_build` constructing DM constants is
excluded on the CasADi side.  Cone membership is pure metadata everywhere: it
never affects extraction cost.

Excluded here: the kron trio (index-machinery lowerings, follow-up work) and
CVaR at the full 131072-scenario size (the 100M-nnz dense block exceeds this
machine's memory through any modeling layer; the CVaRSlice instance in
_lowered_data.py is the compared size).

Block schema returned by build(name, full, theta):
    {
      "segments": [(name, length), ...],        # z layout
      "P": scipy sparse (full symmetric) or None,
      "c": 1-D ndarray,
      "rows": [
         {"kind": "zero"|"nonneg", "A": sparse, "b": 1-D ndarray}
         {"kind": "soc",  "A": ..., "b": ..., "dims": [k1, k2, ...]}
         {"kind": "psd",  "A": ..., "b": ..., "side": n}
      ],                                        # CVXPY row order
    }
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sps
import scipy.stats as st


def _eye(n, scale=1.0):
    return sps.identity(n, format="csr") * scale


def _z(*counts):
    """total width helper"""
    return int(sum(counts))


def _hstack(width_map, segments, **blocks):
    """Assemble a row block over full z from named segment blocks."""
    cols = []
    for name, length in segments:
        blk = blocks.get(name)
        if blk is None:
            m = None
            for b in blocks.values():
                if b is not None:
                    m = b.shape[0]
                    break
            cols.append(sps.csr_matrix((m, length)))
        else:
            cols.append(sps.csr_matrix(blk))
    return sps.hstack(cols, format="csr")


def _quad(segments, **Pblocks):
    """Full-symmetric P over z from named diagonal segment blocks."""
    total = sum(l for _, l in segments)
    P = sps.lil_matrix((total, total))
    off = 0
    offs = {}
    for name, length in segments:
        offs[name] = off
        off += length
    for name, blk in Pblocks.items():
        o = offs[name]
        blk = sps.coo_matrix(blk)
        P[o:o + blk.shape[0], o:o + blk.shape[1]] = blk
    return P.tocsr()


# --------------------------------------------------------------------------- #
# Base module problems (casadi_compare.py)
# --------------------------------------------------------------------------- #
def _simple_lp(full, theta):
    n = int(1e7) if full else 7
    c = np.arange(n).astype(float)
    segments = [("x", n)]
    rows = [
        {"kind": "nonneg", "A": _eye(n, -1.0), "b": np.zeros(n)},        # x >= 0
        {"kind": "nonneg", "A": _eye(n, 1.0), "b": np.ones(n)},          # 1 - x >= 0
    ]
    return {"segments": segments, "P": None, "c": c, "rows": rows}


def _scalar_param_lp(full, theta):
    n = int(2e6) if full else 7
    (p,) = theta
    c = float(p) * np.arange(n).astype(float)
    segments = [("x", n)]
    rows = [
        {"kind": "nonneg", "A": _eye(n, -1.0), "b": np.zeros(n)},
        {"kind": "nonneg", "A": _eye(n, 1.0), "b": np.ones(n)},
    ]
    return {"segments": segments, "P": None, "c": c, "rows": rows}


def _full_param_lp(full, theta):
    n = int(1e6) if full else 7
    (p,) = theta
    segments = [("x", n)]
    rows = [
        {"kind": "nonneg", "A": _eye(n, -1.0), "b": np.zeros(n)},
        {"kind": "nonneg", "A": _eye(n, 1.0), "b": np.ones(n)},
    ]
    return {"segments": segments, "P": None, "c": np.asarray(p, float).ravel(),
            "rows": rows}


def _least_squares(full, theta):
    m, n = (7000, 2000) if full else (3, 2)
    rng = np.random.RandomState(1)
    A0 = rng.randn(m, n)
    b0 = rng.randn(m)
    segments = [("t", m), ("x", n)]
    # g = b0 + t - A0 x in zero cone  =>  A = [-I, A0]
    rows = [{"kind": "zero",
             "A": _hstack(None, segments, t=_eye(m, -1.0), x=sps.csr_matrix(A0)),
             "b": b0}]
    P = _quad(segments, t=_eye(m, 2.0))
    c = np.zeros(m + n)
    return {"segments": segments, "P": P, "c": c, "rows": rows}


def _simple_qp(full, theta):
    m, n, p_ = (8000, 1600, 20) if full else (4, 3, 1)
    rng = np.random.RandomState(1)
    P0 = rng.randn(n, n)
    P0 = P0.T @ P0
    q0 = rng.randn(n)
    G0 = rng.randn(m, n)
    h0 = G0 @ rng.randn(n)
    Aeq = rng.randn(p_, n)
    beq = rng.randn(p_)
    segments = [("x", n)]
    rows = [
        {"kind": "zero", "A": sps.csr_matrix(Aeq), "b": beq},
        {"kind": "nonneg", "A": sps.csr_matrix(G0), "b": h0},
    ]
    return {"segments": segments, "P": sps.csr_matrix(P0), "c": q0, "rows": rows}


def _parametrized_qp(full, theta):
    m, n = (6000, 2400) if full else (3, 2)
    Ap, bp = theta
    segments = [("t", m), ("x", n)]
    rows = [
        {"kind": "zero",
         "A": _hstack(None, segments, t=_eye(m, -1.0), x=sps.csr_matrix(np.asarray(Ap))),
         "b": np.asarray(bp, float).ravel()},
        {"kind": "nonneg",
         "A": _hstack(None, segments, x=_eye(n, -1.0)), "b": np.zeros(n)},
        {"kind": "nonneg",
         "A": _hstack(None, segments, x=_eye(n, 1.0)), "b": np.ones(n)},
    ]
    P = _quad(segments, t=_eye(m, 2.0))
    return {"segments": segments, "P": P, "c": np.zeros(m + n), "rows": rows}


def _huber(full, theta):
    n = 3000 if full else 2
    samples = int(1.5 * n) if full else 3
    rng = np.random.RandomState(1)
    X = rng.randn(n, samples)
    beta_true = 5 * rng.normal(size=(n, 1))
    v = rng.normal(size=(samples, 1))
    factor = 2 * rng.binomial(1, 0.88, size=(samples, 1)) - 1
    Y = factor * X.T.dot(beta_true) + v
    Xt, y = X.T, Y.ravel()
    m = samples
    segments = [("u", m), ("t", m), ("w", m), ("beta", n)]
    # rows: zero (y + u + w - Xt beta), nonneg (t - w), nonneg (t + w)
    rows = [
        {"kind": "zero",
         "A": _hstack(None, segments, u=_eye(m, -1.0), w=_eye(m, -1.0),
                      beta=sps.csr_matrix(Xt)),
         "b": y},
        {"kind": "nonneg",
         "A": _hstack(None, segments, t=_eye(m, -1.0), w=_eye(m, 1.0)),
         "b": np.zeros(m)},
        {"kind": "nonneg",
         "A": _hstack(None, segments, t=_eye(m, -1.0), w=_eye(m, -1.0)),
         "b": np.zeros(m)},
    ]
    P = _quad(segments, u=_eye(m, 2.0))
    c = np.concatenate([np.zeros(m), 2 * np.ones(m), np.zeros(m), np.zeros(n)])
    return {"segments": segments, "P": P, "c": c, "rows": rows}


def _svm_l1(full, theta):
    n, m = (500, 25000) if full else (2, 3)
    rng = np.random.RandomState(1)
    beta_true = rng.randn(n, 1)
    idxs = rng.choice(range(n), int(0.8 * n), replace=False)
    beta_true[idxs] = 0
    X = rng.normal(0, 5, size=(m, n))
    Y = np.sign(X.dot(beta_true) + rng.normal(0, 45, size=(m, 1)))
    y = Y.ravel()
    (lam,) = theta
    segments = [("h", m), ("t", 1), ("beta", n), ("v", 1), ("s", n)]
    yX = sps.csr_matrix(y[:, None] * X)
    ones = np.ones((m, 1))
    # rows: hinge (-1 + h + y(Xb - v) >= 0), h >= 0, s - b, s + b, t - sum(s)
    rows = [
        {"kind": "nonneg",
         "A": _hstack(None, segments, h=_eye(m, -1.0), beta=-yX,
                      v=sps.csr_matrix(y.reshape(-1, 1))),
         "b": -np.ones(m)},
        {"kind": "nonneg", "A": _hstack(None, segments, h=_eye(m, -1.0)),
         "b": np.zeros(m)},
        {"kind": "nonneg",
         "A": _hstack(None, segments, s=_eye(n, -1.0), beta=_eye(n, 1.0)),
         "b": np.zeros(n)},
        {"kind": "nonneg",
         "A": _hstack(None, segments, s=_eye(n, -1.0), beta=_eye(n, -1.0)),
         "b": np.zeros(n)},
        {"kind": "nonneg",
         "A": _hstack(None, segments, t=sps.csr_matrix(np.array([[-1.0]])),
                      s=sps.csr_matrix(np.ones((1, n)))),
         "b": np.zeros(1)},
    ]
    c = np.concatenate([np.ones(m) / m, [float(lam)], np.zeros(n), [0.0],
                        np.zeros(n)])
    return {"segments": segments, "P": None, "c": c, "rows": rows}


# --------------------------------------------------------------------------- #
# ext_lp problems
# --------------------------------------------------------------------------- #
def _stuffing(m_full):
    def build(full, theta, m_full=m_full):
        m = m_full if full else 4
        n = m
        h = m // 2
        if theta:
            A0, C0, b0 = theta
            A0 = np.asarray(A0)
            C0 = np.asarray(C0).ravel()
            b0 = np.asarray(b0).ravel()
        else:
            rs = np.random.RandomState(0)
            A0 = rs.randn(m, n)
            C0 = rs.rand(h)
            b0 = rs.randn(m)
        segments = [("x", n)]
        # zero rows: C_i x_{h+i} = b_{h+i}; nonneg rows: b_i - C_i x_i >= 0
        Az = sps.csr_matrix(
            (C0, (np.arange(h), np.arange(h, m))), shape=(h, n))
        An = sps.csr_matrix(
            (C0, (np.arange(h), np.arange(h))), shape=(h, n))
        rows = [
            {"kind": "zero", "A": Az, "b": b0[h:]},
            {"kind": "nonneg", "A": An, "b": b0[:h]},
        ]
        return {"segments": segments, "P": None, "c": A0.sum(axis=0),
                "rows": rows}
    return build


def _slow_pruning(full, theta):
    rows_, t, s = (100, 20, 4000) if full else (2, 2, 3)
    x = np.linspace(-100.0, 100.0, s)
    M0 = np.tile(np.array([x]), t).reshape((t, x.shape[0]))
    Y0 = np.tile(x, rows_).reshape((rows_, s))
    segments = [("u", rows_ * s), ("v", rows_ * t)]
    # g = vecF(Y0) + u - vecF(V M0),  vecF(V M0) = (M0' (x) I_rows) v
    K = sps.kron(sps.csr_matrix(M0.T), sps.identity(rows_), format="csr")
    rows = [{"kind": "zero",
             "A": _hstack(None, segments, u=_eye(rows_ * s, -1.0), v=K),
             "b": Y0.flatten(order="F")}]
    P = _quad(segments, u=_eye(rows_ * s, 2.0))
    return {"segments": segments, "P": P, "c": np.zeros(rows_ * (s + t)),
            "rows": rows}


def _gini_returns(N, T):
    rs = np.random.RandomState(123)
    cov = rs.rand(N, N) * 1.5 - 0.5
    cov = cov @ cov.T / 1000 + np.diag(rs.rand(N) * 0.7 + 0.3) / 1000
    mean = np.zeros(N) + 1 / 1000
    return st.multivariate_normal.rvs(mean=mean, cov=cov, size=T,
                                      random_state=rs)


def _yitzhaki(full, theta):
    N, T = (50, 300) if full else (2, 3)
    returns = _gini_returns(N, T)
    D0 = np.array([]).reshape(0, N)
    for j in range(returns.shape[0] - 1):
        D0 = np.concatenate((D0, returns[j + 1:] - returns[j, :]), axis=0)
    pairs = T * (T - 1) // 2
    segments = [("d", pairs), ("w", N)]
    D0s = sps.csr_matrix(D0)
    rows = [
        {"kind": "zero",
         "A": _hstack(None, segments, w=sps.csr_matrix(np.ones((1, N)))),
         "b": np.ones(1)},
        {"kind": "nonneg",
         "A": _hstack(None, segments, d=_eye(pairs, -1.0), w=D0s),
         "b": np.zeros(pairs)},
        {"kind": "nonneg",
         "A": _hstack(None, segments, d=_eye(pairs, -1.0), w=-D0s),
         "b": np.zeros(pairs)},
        {"kind": "nonneg", "A": _hstack(None, segments, w=_eye(N, -1.0)),
         "b": np.zeros(N)},
    ]
    c = np.concatenate([np.full(pairs, 1000.0 / ((T - 1) * T)), np.zeros(N)])
    return {"segments": segments, "P": None, "c": c, "rows": rows}


def _pairs_mat(T):
    pairs = T * (T - 1) // 2
    r = np.arange(pairs)
    i_idx = np.concatenate([np.arange(j + 1, T) for j in range(T)])
    j_idx = np.repeat(np.arange(T), np.arange(T - 1, -1, -1))
    return (sps.coo_matrix((np.ones(pairs), (r, i_idx)), shape=(pairs, T))
            + sps.coo_matrix((-np.ones(pairs), (r, j_idx)), shape=(pairs, T))
            ).tocsr()


def _murray(full, theta):
    N, T = (50, 700) if full else (2, 3)
    returns = _gini_returns(N, T)
    pairs = T * (T - 1) // 2
    mat_sp = _pairs_mat(T)
    segments = [("d", pairs), ("rw", T), ("w", N)]
    rows = [
        {"kind": "zero",
         "A": _hstack(None, segments, rw=_eye(T, 1.0),
                      w=sps.csr_matrix(-returns)),
         "b": np.zeros(T)},
        {"kind": "zero",
         "A": _hstack(None, segments, w=sps.csr_matrix(np.ones((1, N)))),
         "b": np.ones(1)},
        {"kind": "nonneg",
         "A": _hstack(None, segments, d=_eye(pairs, -1.0), rw=mat_sp),
         "b": np.zeros(pairs)},
        {"kind": "nonneg",
         "A": _hstack(None, segments, d=_eye(pairs, -1.0), rw=-mat_sp),
         "b": np.zeros(pairs)},
        {"kind": "nonneg", "A": _hstack(None, segments, w=_eye(N, -1.0)),
         "b": np.zeros(N)},
    ]
    c = np.concatenate([np.full(pairs, 1000.0 / ((T - 1) * T)),
                        np.zeros(T + N)])
    return {"segments": segments, "P": None, "c": c, "rows": rows}


def _cajas(full, theta):
    N, T = (50, 800) if full else (2, 3)
    returns = _gini_returns(N, T)
    owa_w = np.array([2 * i - 1 - T for i in range(1, T + 1)]) / (T * (T - 1))
    segments = [("a", T), ("b", T), ("w", N), ("y", T)]
    # zero rows: y - R w = 0 (T rows), sum w = 1
    # nonneg: w >= 0, then T blocks (a_i + b - owa_i y >= 0), each length T
    blocks = []
    for i in range(T):
        Ablk = sps.hstack([
            sps.csr_matrix((np.full(T, -1.0), (np.arange(T), np.full(T, i))),
                           shape=(T, T)),                    # -a_i column
            _eye(T, -1.0),                                   # -b
            sps.csr_matrix((T, N)),
            sps.identity(T, format="csr") * owa_w[i],        # +owa_i y
        ], format="csr")
        blocks.append({"kind": "nonneg", "A": Ablk, "b": np.zeros(T)})
    rows = [
        {"kind": "zero",
         "A": _hstack(None, segments, y=_eye(T, -1.0),
                      w=sps.csr_matrix(returns)),
         "b": np.zeros(T)},
        {"kind": "zero",
         "A": _hstack(None, segments, w=sps.csr_matrix(np.ones((1, N)))),
         "b": np.ones(1)},
        {"kind": "nonneg", "A": _hstack(None, segments, w=_eye(N, -1.0)),
         "b": np.zeros(N)},
    ] + blocks
    c = np.concatenate([np.full(T, 1000.0), np.full(T, 1000.0),
                        np.zeros(N + T)])
    return {"segments": segments, "P": None, "c": c, "rows": rows}


def _optimal_advertising(full, theta):
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
    RP = R0[:, None] * P0                                     # (m, n)

    segments = [("t", m), ("D", m * n)]                       # D is vec_F
    # interleaved pairs per i: (t_i + RP_i . D_i:) then (B_i + t_i)
    # D_i: in F-order sits at indices i + m*j for j in range(n)
    ri = np.repeat(np.arange(m), n)
    ci = (ri + m * np.tile(np.arange(n), m))
    A_pairs_D = sps.csr_matrix((-RP.ravel(), (2 * ri, ci)),
                               shape=(2 * m, m * n))
    tcols = np.arange(m)
    A_pairs_t = sps.csr_matrix(
        (np.full(2 * m, -1.0),
         (np.arange(2 * m), np.repeat(tcols, 2))),
        shape=(2 * m, m))
    b_pairs = np.zeros(2 * m)
    b_pairs[1::2] = B0.ravel()
    # T - D'1 rows: row j has -1 on D columns i + m*j ... coefficient +1 => A has +1
    rj = np.repeat(np.arange(n), m)
    cj = np.tile(np.arange(m), n) + m * rj
    A_T = sps.csr_matrix((np.ones(n * m), (rj, cj)), shape=(n, m * n))
    # D 1 - c rows: row i has -1 coefs on D columns i + m*j
    A_c = sps.csr_matrix((-np.ones(m * n), (ri, ci)), shape=(m, m * n))
    rows = [
        {"kind": "nonneg",
         "A": sps.hstack([A_pairs_t, A_pairs_D], format="csr"), "b": b_pairs},
        {"kind": "nonneg",
         "A": _hstack(None, segments, D=_eye(m * n, -1.0)),
         "b": np.zeros(m * n)},
        {"kind": "nonneg",
         "A": sps.hstack([sps.csr_matrix((n, m)), A_T], format="csr"),
         "b": T0},
        {"kind": "nonneg",
         "A": sps.hstack([sps.csr_matrix((m, m)), A_c], format="csr"),
         "b": -c0},
    ]
    c = np.concatenate([np.full(m, 1.0), np.zeros(m * n)])
    return {"segments": segments, "P": None, "c": c, "rows": rows}


# --------------------------------------------------------------------------- #
# ext_cones problems
# --------------------------------------------------------------------------- #
def _factor_covariance(full, theta):
    n, m = (25000, 250) if full else (3, 2)
    rng = np.random.RandomState(1)
    mu = np.abs(rng.randn(n, 1))
    Sigma_tilde = rng.randn(m, m)
    Sigma_tilde = Sigma_tilde.T.dot(Sigma_tilde)
    d_diag = rng.uniform(0, 0.9, size=n)
    F0 = rng.randn(n, m)
    gam, Lm = float(theta[0]), float(theta[1])
    segments = [("w", n), ("f", m), ("t", n), ("s", n)]
    rows = [
        {"kind": "zero",
         "A": _hstack(None, segments, t=_eye(n, -1.0),
                      w=sps.diags(np.sqrt(d_diag), format="csr")),
         "b": np.zeros(n)},
        {"kind": "zero",
         "A": _hstack(None, segments, w=sps.csr_matrix(np.ones((1, n)))),
         "b": np.ones(1)},
        {"kind": "zero",
         "A": _hstack(None, segments, w=sps.csr_matrix(-F0.T), f=_eye(m, 1.0)),
         "b": np.zeros(m)},
        {"kind": "nonneg",
         "A": _hstack(None, segments, s=_eye(n, -1.0), w=_eye(n, 1.0)),
         "b": np.zeros(n)},
        {"kind": "nonneg",
         "A": _hstack(None, segments, s=_eye(n, -1.0), w=_eye(n, -1.0)),
         "b": np.zeros(n)},
        {"kind": "nonneg",
         "A": _hstack(None, segments, s=sps.csr_matrix(np.ones((1, n)))),
         "b": np.array([Lm])},
    ]
    P = _quad(segments, f=sps.csr_matrix(2 * gam * Sigma_tilde),
              t=_eye(n, 2 * gam))
    c = np.concatenate([-mu.ravel(), np.zeros(m + 2 * n)])
    return {"segments": segments, "P": P, "c": c, "rows": rows}


def _convex_plasticity(full, theta):
    N = 3000 if full else 2
    E_dim = 70e3
    E = 70e3 / E_dim
    nu = 0.3
    sig0 = 250 / E_dim
    Et = E / 100.0
    H = E * Et / (E - Et)
    lam, mus = E * nu / (1 + nu) / (1 - 2 * nu), E / 2 / (1 + nu)
    C = np.array([
        [lam + 2 * mus, lam, lam, 0],
        [lam, lam + 2 * mus, lam, 0],
        [lam, lam, lam + 2 * mus, 0],
        [0, 0, 0, 2 * mus],
    ])
    S = np.linalg.inv(C)
    dev = np.array([
        [2 / 3.0, -1 / 3.0, -1 / 3.0, 0],
        [-1 / 3.0, 2 / 3.0, -1 / 3.0, 0],
        [-1 / 3.0, -1 / 3.0, 2 / 3.0, 0],
        [0, 0, 0, 1.0],
    ])
    S_sparsed = sps.block_diag([S for _ in range(N)], format="csr")
    deps, sig_old, p_old = (np.asarray(t) for t in theta)
    sig_elas = sig_old + C @ deps                             # (4, N)
    segments = [("d", 4 * N), ("e", N), ("p", N), ("sig", 4 * N), ("t", N)]
    # zero rows: vecF(sig_elas) + d - vecF(sig); p_old + e - p
    rows = [
        {"kind": "zero",
         "A": _hstack(None, segments, d=_eye(4 * N, -1.0), sig=_eye(4 * N, 1.0)),
         "b": sig_elas.flatten(order="F")},
        {"kind": "zero",
         "A": _hstack(None, segments, e=_eye(N, -1.0), p=_eye(N, 1.0)),
         "b": p_old.ravel()},
        {"kind": "nonneg", "A": _hstack(None, segments, p=_eye(N, -1.0)),
         "b": np.zeros(N)},
        {"kind": "nonneg",
         "A": _hstack(None, segments, p=_eye(N, -H),
                      t=_eye(N, np.sqrt(3 / 2))),
         "b": np.full(N, sig0)},
    ]
    # SOC: N blocks of size 5, block i = [t_i; dev @ sig[:, i]], interleaved
    # g_soc = reshape([t'; dev sig], 5N) column-major
    dev_sp = sps.csr_matrix(dev)
    dev_blk = sps.block_diag([dev_sp] * N, format="csr")      # (4N, 4N) on vecF(sig)
    # rows of the soc block in order: for i: t_i, then (dev sig)_[:,i]
    sel_t = sps.csr_matrix(
        (np.ones(N), (5 * np.arange(N), np.arange(N))), shape=(5 * N, N))
    sel_ds = sps.csr_matrix(
        (np.ones(4 * N),
         (np.repeat(5 * np.arange(N), 4) + np.tile(np.arange(1, 5), N),
          np.arange(4 * N))),
        shape=(5 * N, 4 * N))
    Asoc_full = _hstack(None, segments, t=-sel_t,
                        sig=-(sel_ds @ dev_blk))
    rows.append({"kind": "soc", "A": Asoc_full, "b": np.zeros(5 * N),
                 "dims": [5] * N})
    P = _quad(segments, d=S_sparsed, e=_eye(N, H))
    c = np.zeros(sum(l for _, l in segments))
    return {"segments": segments, "P": P, "c": c, "rows": rows}


def _tv_inpainting(full, theta):
    rows_, cols, colors = (512, 512, 3) if full else (3, 3, 3)
    rng = np.random.RandomState(1)
    Uorig = rng.randn(rows_, cols, colors)
    mask = (rng.random_sample((rows_, cols)) > 0.7).astype(float)
    known = np.repeat(mask[:, :, None], colors, axis=2)
    Ucorr = known * Uorig
    L = (rows_ - 1) * (cols - 1)
    npix = rows_ * cols
    segments = [("t", L)] + [(f"U{k}", npix) for k in range(colors)]

    row_blocks = []
    for k in range(colors):
        kf = known[:, :, k].flatten(order="F")
        diag = sps.diags(kf, format="csr")
        A_k = {f"U{k}": diag}
        row_blocks.append({
            "kind": "zero",
            "A": _hstack(None, segments, **A_k),
            "b": (known[:, :, k] * Ucorr[:, :, k]).flatten(order="F")})

    # SOC: L blocks of size 1+2*colors; block j (column-major over the grid):
    # [t_j; dx1_j; dy1_j; dx2_j; dy2_j; dx3_j; dy3_j]
    # dx over U: dx[r,cc] = U[r, cc+1] - U[r, cc]  for r<rows-1, cc<cols-1
    # dy over U: dy[r,cc] = U[r+1, cc] - U[r, cc]
    rr = np.tile(np.arange(rows_ - 1), cols - 1)
    cc = np.repeat(np.arange(cols - 1), rows_ - 1)
    base = rr + rows_ * cc                                    # U index of (r, cc)
    right = rr + rows_ * (cc + 1)
    down = (rr + 1) + rows_ * cc
    j = np.arange(L)
    blocklen = 1 + 2 * colors
    sel_rows_t = blocklen * j                                 # position of t_j
    Asoc_parts = {"t": sps.csr_matrix(
        (-np.ones(L), (sel_rows_t, j)), shape=(blocklen * L, L))}
    for k in range(colors):
        dx_rows = blocklen * j + 1 + 2 * k
        dy_rows = blocklen * j + 2 + 2 * k
        data = np.concatenate([
            -np.ones(L), np.ones(L),      # dx: +U_right - U_base -> A = -coef
            -np.ones(L), np.ones(L),      # dy: +U_down  - U_base
        ])
        rws = np.concatenate([dx_rows, dx_rows, dy_rows, dy_rows])
        cls = np.concatenate([right, base, down, base])
        Asoc_parts[f"U{k}"] = sps.csr_matrix(
            (data, (rws, cls)), shape=(blocklen * L, npix))
    row_blocks.append({"kind": "soc",
                       "A": _hstack(None, segments, **Asoc_parts),
                       "b": np.zeros(blocklen * L),
                       "dims": [blocklen] * L})
    c = np.concatenate([np.ones(L)] + [np.zeros(npix)] * colors)
    return {"segments": segments, "P": None, "c": c, "rows": row_blocks}


def _semidefinite_programming(full, theta):
    n, p_ = (200, 120) if full else (3, 2)
    rng = np.random.RandomState(1)
    C0 = rng.randn(n, n)
    A_list = []
    b_list = []
    for _ in range(p_):
        A_list.append(rng.randn(n, n))
        b_list.append(rng.randn())
    b_arr = np.array(b_list)
    nsv = n * (n + 1) // 2
    jj, ii = np.triu_indices(n)

    def tr_vec(M):
        v = (M + M.T)[ii, jj]
        v[ii == jj] /= 2.0
        return v

    ri, rj = np.tril_indices(n)
    col_pos = (rj * n - rj * (rj - 1) // 2 + (ri - rj)).astype(int)
    scale = np.where(ri == rj, 1.0, np.sqrt(2.0))
    Aeq = np.stack([tr_vec(Ai) for Ai in A_list])
    segments = [("x", nsv)]
    A_psd = sps.csr_matrix(
        (-scale, (np.arange(nsv), col_pos)), shape=(nsv, nsv))
    rows = [
        {"kind": "zero", "A": sps.csr_matrix(Aeq), "b": b_arr},
        {"kind": "psd", "A": A_psd, "b": np.zeros(nsv), "side": n},
    ]
    return {"segments": segments, "P": None, "c": tr_vec(C0), "rows": rows}


# --------------------------------------------------------------------------- #
# Registry.  draw comes from the casadi Specs (same seeds/order); theta is the
# list of drawn values for parametric problems, or [] for the rest.
# --------------------------------------------------------------------------- #
SUITE = {
    "SimpleLP": _simple_lp,
    "SimpleScalarParametrizedLP": _scalar_param_lp,
    "SimpleFullyParametrizedLP": _full_param_lp,
    "LeastSquares": _least_squares,
    "SimpleQP": _simple_qp,
    "ParametrizedQP": _parametrized_qp,
    "HuberRegression": _huber,
    "SVMWithL1Regularization": _svm_l1,
    "ConeMatrixStuffing": _stuffing(5000),
    "SmallMatrixStuffing": _stuffing(4000),
    "ParamConeMatrixStuffing": _stuffing(200),
    "ParamSmallMatrixStuffing": _stuffing(300),
    "SlowPruning": _slow_pruning,
    "Yitzhaki": _yitzhaki,
    "Murray": _murray,
    "Cajas": _cajas,
    "OptimalAdvertising": _optimal_advertising,
    "FactorCovarianceModel": _factor_covariance,
    "ConvexPlasticity": _convex_plasticity,
    "TvInpainting": _tv_inpainting,
    "SemidefiniteProgramming": _semidefinite_programming,
}

# CVaR: full size excluded (100M-nnz dense block; see module docstring).
# The compared instance is CVaRSlice in _lowered_data.py / julia_compare.py.

# Small-size stuffing specs use the SAME m for small (m=4): handled by the
# maker's `full` flag mirroring the casadi Specs (m is fixed per problem name;
# small instances use m=4 via casadi_compare's spec sizes).


def build_blocks(name: str, full: bool, theta):
    return SUITE[name](full, theta or [])
