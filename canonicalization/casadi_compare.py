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

Compare problem-data extraction (P, c, A, b) between the CVXPY diff engine and
CasADi, on the LOWERED problem formulations.

Methodology
-----------
CVXPY side: ``prob.get_problem_data(solver=cp.CLARABEL, ignore_dpp=True)`` on the
high-level model -- canonicalization + stuffing produce Clarabel data

    minimize   (1/2) z'Pz + c'z    subject to   Az + s = b,  s in K.

CasADi side: we hand-write the *already-lowered* model (same epigraph variables,
same row order as CVXPY's canonicalization -- cones are pure metadata) as MX
expressions f(z, p) and g(z, p) := b - Az (the slack expression), and extract

    P, grad = casadi.hessian(f, z)          c = grad|_{z=0}
    A = -casadi.jacobian(g, z)              b = g|_{z=0}

wrapped in a casadi.Function(p -> (P, c, A, b)).  This is exactly how CasADi's
own qpsol/conic path recovers problem data from an NLP-form model, without
paying for a solver.  "single" times AD + Function construction + first numeric
evaluation on a freshly built graph (the analogue of a cold get_problem_data on
an existing expression tree); "re-solve" times one more numeric evaluation with
fresh parameter values (the analogue of the cached-engine re-solve).

Verification: casadi and cvxpy matrices are compared entry-for-entry (same row
and column order; sparse, no densification) at a small instance size and, with
--verify-full, at the benchmark size.

Usage:
    python canonicalization/casadi_compare.py --verify        # small-size checks
    python canonicalization/casadi_compare.py --verify-full   # benchmark-size checks
    python canonicalization/casadi_compare.py --time          # the timing tables
    ... --only SimpleLP,ParametrizedQP     restrict problems
Env: CASADI_ITERS (default 3 cold iters), CASADI_RESOLVES (default 20).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sps

import casadi as ca
import cvxpy as cp

ITERS = int(os.environ.get("CASADI_ITERS", "3"))
RESOLVES = int(os.environ.get("CASADI_RESOLVES", "20"))
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Problem specs.  Each build(full) returns:
#   cvxpy_build() -> (Problem, [cp.Parameter, ...])
#   casadi_build() -> (z, [MX param syms, ...], f, g)   with g = b - Az
#   draw(rng) -> [numpy values ...]  fresh parameter values (same order)
# Column/row order mirrors cvxpy's CLARABEL lowering (verified by --verify).
# --------------------------------------------------------------------------- #
@dataclass
class Spec:
    name: str
    parametric: bool
    cvxpy_build: object
    casadi_build: object
    draw: object = None
    notes: str = ""


def make_simple_lp(full: bool) -> Spec:
    n = int(1e7) if full else 7
    c = np.arange(n).astype(float)

    def cvxpy_build():
        x = cp.Variable(n)
        return cp.Problem(cp.Minimize(c @ x), [0 <= x, x <= 1]), []

    def casadi_build():
        x = ca.MX.sym("x", n)
        f = ca.dot(ca.DM(c), x)
        g = ca.vertcat(x, 1 - x)  # rows: x >= 0 block, then x <= 1 block
        return x, [], f, g

    return Spec("SimpleLP", False, cvxpy_build, casadi_build)


def make_scalar_param_lp(full: bool) -> Spec:
    n = int(2e6) if full else 7
    c = np.arange(n).astype(float)

    def cvxpy_build():
        p = cp.Parameter()
        x = cp.Variable(n)
        return cp.Problem(cp.Minimize((p * c) @ x), [0 <= x, x <= 1]), [p]

    def casadi_build():
        x = ca.MX.sym("x", n)
        p = ca.MX.sym("p")
        f = p * ca.dot(ca.DM(c), x)
        g = ca.vertcat(x, 1 - x)
        return x, [p], f, g

    def draw(rng):
        return [rng.uniform(0.5, 2.0)]

    return Spec("SimpleScalarParametrizedLP", True, cvxpy_build, casadi_build, draw)


def make_full_param_lp(full: bool) -> Spec:
    n = int(1e6) if full else 7

    def cvxpy_build():
        p = cp.Parameter(n)
        x = cp.Variable(n)
        return cp.Problem(cp.Minimize(p @ x), [0 <= x, x <= 1]), [p]

    def casadi_build():
        x = ca.MX.sym("x", n)
        p = ca.MX.sym("p", n)
        f = ca.dot(p, x)
        g = ca.vertcat(x, 1 - x)
        return x, [p], f, g

    def draw(rng):
        return [rng.standard_normal(n)]

    return Spec("SimpleFullyParametrizedLP", True, cvxpy_build, casadi_build, draw)


def make_least_squares(full: bool) -> Spec:
    m, n = (7000, 2000) if full else (3, 2)
    rng = np.random.RandomState(1)
    A0 = rng.randn(m, n)
    b0 = rng.randn(m)

    def cvxpy_build():
        x = cp.Variable(n)
        return cp.Problem(cp.Minimize(cp.sum_squares(A0 @ x - b0))), []

    def casadi_build():
        # lowered: min t't  s.t.  t = A0 x - b0 ; columns [t, x]
        t = ca.MX.sym("t", m)
        x = ca.MX.sym("x", n)
        z = ca.vertcat(t, x)
        f = ca.dot(t, t)
        g = ca.DM(b0) + t - ca.mtimes(ca.DM(A0), x)  # zero cone
        return z, [], f, g

    return Spec("LeastSquares", False, cvxpy_build, casadi_build)


def make_simple_qp(full: bool) -> Spec:
    m, n, p_ = (8000, 1600, 20) if full else (4, 3, 1)
    rng = np.random.RandomState(1)
    P0 = rng.randn(n, n)
    P0 = P0.T @ P0
    q0 = rng.randn(n)
    G0 = rng.randn(m, n)
    h0 = G0 @ rng.randn(n)
    Aeq = rng.randn(p_, n)
    beq = rng.randn(p_)

    def cvxpy_build():
        x = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize((1 / 2) * cp.quad_form(x, P0, assume_PSD=True) + q0.T @ x),
            [G0 @ x <= h0, Aeq @ x == beq],
        )
        return prob, []

    def casadi_build():
        x = ca.MX.sym("x", n)
        f = 0.5 * ca.bilin(ca.DM(P0), x, x) + ca.dot(ca.DM(q0), x)
        g_zero = ca.DM(beq) - ca.mtimes(ca.DM(Aeq), x)      # equalities first
        g_ineq = ca.DM(h0) - ca.mtimes(ca.DM(G0), x)
        return x, [], f, ca.vertcat(g_zero, g_ineq)

    return Spec("SimpleQP", False, cvxpy_build, casadi_build)


def make_parametrized_qp(full: bool) -> Spec:
    m, n = (6000, 2400) if full else (3, 2)

    def cvxpy_build():
        A = cp.Parameter((m, n))
        b = cp.Parameter((m,))
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [0 <= x, x <= 1])
        return prob, [A, b]

    def casadi_build():
        # lowered: min t't s.t. t = A(p) x - b(p); 0 <= x <= 1; columns [t, x]
        t = ca.MX.sym("t", m)
        x = ca.MX.sym("x", n)
        z = ca.vertcat(t, x)
        Ap = ca.MX.sym("A", m, n)
        bp = ca.MX.sym("b", m)
        f = ca.dot(t, t)
        g_zero = bp + t - ca.mtimes(Ap, x)
        g_nonneg = ca.vertcat(x, 1 - x)
        return z, [Ap, bp], f, ca.vertcat(g_zero, g_nonneg)

    def draw(rng):
        return [rng.standard_normal((m, n)), rng.standard_normal(m)]

    return Spec("ParametrizedQP", True, cvxpy_build, casadi_build, draw)


def make_huber(full: bool) -> Spec:
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

    def cvxpy_build():
        beta = cp.Variable((n, 1))
        return cp.Problem(cp.Minimize(cp.sum(cp.huber(Xt @ beta - Y, 1)))), []

    def casadi_build():
        # lowered huber(r, M=1): min u'u + 2*sum(t)  s.t. r = u + w, |w| <= t
        # columns [u, t, w, beta]; rows: zero cone (r), then w<=t, then -w<=t
        u = ca.MX.sym("u", m)
        t = ca.MX.sym("t", m)
        w = ca.MX.sym("w", m)
        beta = ca.MX.sym("beta", n)
        z = ca.vertcat(u, t, w, beta)
        f = ca.dot(u, u) + 2 * ca.sum1(t)
        g_zero = ca.DM(y) + u + w - ca.mtimes(ca.DM(Xt), beta)
        g = ca.vertcat(g_zero, t - w, t + w)
        return z, [], f, g

    return Spec("HuberRegression", False, cvxpy_build, casadi_build)


def make_svm_l1(full: bool) -> Spec:
    n, m = (500, 25000) if full else (2, 3)
    rng = np.random.RandomState(1)
    beta_true = rng.randn(n, 1)
    idxs = rng.choice(range(n), int(0.8 * n), replace=False)
    beta_true[idxs] = 0
    X = rng.normal(0, 5, size=(m, n))
    Y = np.sign(X.dot(beta_true) + rng.normal(0, 45, size=(m, 1)))
    y = Y.ravel()

    def cvxpy_build():
        beta = cp.Variable((n, 1))
        v = cp.Variable()
        lambd = cp.Parameter(nonneg=True)
        loss = cp.sum(cp.pos(1 - cp.multiply(Y, X @ beta - v)))
        prob = cp.Problem(cp.Minimize(loss / m + lambd * cp.norm(beta, 1)))
        return prob, [lambd]

    def casadi_build():
        # lowered: min sum(h)/m + lam*t
        #   h >= 1 - y.(Xb - v), h >= 0, |b| <= s, sum(s) <= t
        # columns [h, t, beta, v, s]; rows: hinge, h>=0, s-b, s+b, t-sum(s)
        h = ca.MX.sym("h", m)
        t = ca.MX.sym("t")
        beta = ca.MX.sym("beta", n)
        v = ca.MX.sym("v")
        s = ca.MX.sym("s", n)
        lam = ca.MX.sym("lam")
        z = ca.vertcat(h, t, beta, v, s)
        f = ca.sum1(h) / m + lam * t
        g_hinge = -1 + h + ca.DM(y) * (ca.mtimes(ca.DM(X), beta) - v)
        g = ca.vertcat(g_hinge, h, s - beta, s + beta, t - ca.sum1(s))
        return z, [lam], f, g

    def draw(rng_):
        return [rng_.uniform(0.05, 1.0)]

    return Spec("SVMWithL1Regularization", True, cvxpy_build, casadi_build, draw)


MAKERS = {
    "SimpleLP": make_simple_lp,
    "SimpleScalarParametrizedLP": make_scalar_param_lp,
    "SimpleFullyParametrizedLP": make_full_param_lp,
    "LeastSquares": make_least_squares,
    "SimpleQP": make_simple_qp,
    "ParametrizedQP": make_parametrized_qp,
    "HuberRegression": make_huber,
    "SVMWithL1Regularization": make_svm_l1,
}

# Extension modules contribute additional problems (same Spec contract) without
# editing this file; they live next to this script.
for _ext in ("casadi_problems_ext_lp", "casadi_problems_ext_cones", "casadi_problems_ext_kron"):
    try:
        import importlib as _importlib

        MAKERS.update(_importlib.import_module(_ext).MAKERS)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# CasADi extraction
# --------------------------------------------------------------------------- #
def casadi_extract_function(z, psyms, f, g):
    """AD + Function construction: the symbolic part of extraction.

    P is computed as jacobian(gradient(f)) rather than casadi.hessian(f, z):
    hessian()'s symmetry-exploiting star coloring is pathological on dense
    Hessian patterns (n=800 dense P: 126 s vs 2.6 s for the same matrix via
    jacobian-of-gradient; n=1600 did not finish in 6.5 h), so this gives
    CasADi its faster documented path.  The resulting P is identical.
    """
    grad = ca.gradient(f, z)
    P = ca.jacobian(grad, z)
    zero = ca.DM.zeros(z.shape)
    c = ca.substitute(grad, z, zero)
    A = -ca.jacobian(g, z)
    b = ca.substitute(g, z, zero)
    return ca.Function("extract", psyms, [P, c, A, b])


def dm_to_csc(D) -> sps.csc_matrix:
    rows, cols = D.sparsity().get_triplet()
    return sps.coo_matrix(
        (np.asarray(D.nonzeros(), dtype=float), (rows, cols)),
        shape=(D.size1(), D.size2()),
    ).tocsc()


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _sparse_close(M1, M2, tol=1e-9):
    if M1.shape != M2.shape:
        return False, f"shape {M1.shape} vs {M2.shape}"
    # Explicit stored zeros are a storage artifact (e.g. cvxpy keeps a dense
    # user-supplied D = H*eye block dense in P; DM constants may carry
    # structural zeros): compare modulo them, but report the count.
    A1, A2 = M1.tocsr(), M2.tocsr()
    nnz1, nnz2 = A1.nnz, A2.nnz
    A1.eliminate_zeros(); A2.eliminate_zeros()
    A1.sort_indices(); A2.sort_indices()
    n_explicit = (nnz1 - A1.nnz) + (nnz2 - A2.nnz)
    zeros_note = f" (modulo {n_explicit} explicit zeros)" if n_explicit else ""
    if not ((A1.indptr == A2.indptr).all() and (A1.indices == A2.indices).all()):
        d = abs(M1 - M2)
        err = d.max() if d.nnz else 0.0
        return False, f"pattern differs, max|diff|={err:.3g}"
    # identical NaNs (e.g. NaN already present in the problem data) count as equal
    close = np.isclose(A1.data, A2.data, rtol=tol, atol=tol, equal_nan=True)
    if close.all():
        diff = np.abs(A1.data - A2.data)
        err = np.nanmax(diff) if diff.size else 0.0
        return True, f"max|diff|={err:.3g}{zeros_note}"
    return False, f"{(~close).sum()}/{close.size} entries differ"

def verify(spec: Spec, full: bool) -> bool:
    rng = np.random.default_rng(42)
    prob, params = spec.cvxpy_build()
    vals = spec.draw(rng) if spec.parametric else []
    for p, v in zip(params, vals):
        p.value = v
    data, _, _ = prob.get_problem_data(solver=cp.CLARABEL, ignore_dpp=True)

    F = casadi_extract_function(*spec.casadi_build())
    out = F.call([ca.DM(v) for v in vals])
    P_ca, c_ca, A_ca, b_ca = out

    ok = True
    msgs = []
    # A
    good, msg = _sparse_close(dm_to_csc(A_ca), sps.csc_matrix(data["A"]))
    ok &= good
    msgs.append(f"A[{'OK' if good else 'FAIL'} {msg}]")
    # b, c
    for name, ca_v, cv_v in (("b", b_ca, data["b"]), ("c", c_ca, data["c"])):
        cvv = np.asarray(cv_v).ravel()
        cav = np.asarray(ca.DM(ca_v)).ravel()
        if cav.shape != cvv.shape:
            good, msg = False, "shape"
        else:
            close = np.isclose(cav, cvv, atol=1e-9, rtol=1e-9, equal_nan=True)
            good = bool(close.all())
            msg = "OK" if good else f"{(~close).sum()}/{close.size} differ"
        ok &= good
        msgs.append(f"{name}[{msg if not good else 'OK'}]")
    # P
    P_cv = data.get("P")
    if P_cv is not None:
        good, msg = _sparse_close(dm_to_csc(P_ca), sps.csc_matrix(P_cv))
        ok &= good
        msgs.append(f"P[{'OK' if good else 'FAIL'} {msg}]")
    else:
        good = P_ca.nnz() == 0
        ok &= good
        msgs.append(f"P[{'OK (both absent)' if good else 'FAIL (casadi P nonzero)'}]")
    size = "full" if full else "small"
    print(f"  {spec.name:<28s} ({size}): {'MATCH' if ok else 'MISMATCH'}  " + " ".join(msgs))
    return bool(ok)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def _median(xs):
    return float(np.median(xs)) if xs else float("nan")

def time_cvxpy(spec: Spec):
    rng = np.random.default_rng(7)
    singles = []
    for _ in range(ITERS):
        prob, params = spec.cvxpy_build()
        for p, v in zip(params, spec.draw(rng) if spec.parametric else []):
            p.value = v
        gc.collect()
        t0 = time.perf_counter()
        prob.get_problem_data(solver=cp.CLARABEL, ignore_dpp=True)
        singles.append(time.perf_counter() - t0)
    resolves = []
    if spec.parametric:
        prob, params = spec.cvxpy_build()
        for p, v in zip(params, spec.draw(rng)):
            p.value = v
        prob.get_problem_data(solver=cp.CLARABEL, ignore_dpp=True)  # warm the cache
        for _ in range(RESOLVES):
            for p, v in zip(params, spec.draw(rng)):
                p.value = v
            gc.collect()
            t0 = time.perf_counter()
            prob.get_problem_data(solver=cp.CLARABEL, ignore_dpp=True)
            resolves.append(time.perf_counter() - t0)
    return _median(singles), _median(resolves)


def time_casadi(spec: Spec):
    rng = np.random.default_rng(7)
    singles, builds = [], []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        z, psyms, f, g = spec.casadi_build()
        builds.append(time.perf_counter() - t0)
        vals = [ca.DM(v) for v in (spec.draw(rng) if spec.parametric else [])]
        gc.collect()
        t0 = time.perf_counter()
        F = casadi_extract_function(z, psyms, f, g)
        F.call(vals)
        singles.append(time.perf_counter() - t0)
    resolves = []
    if spec.parametric:
        z, psyms, f, g = spec.casadi_build()
        F = casadi_extract_function(z, psyms, f, g)
        F.call([ca.DM(v) for v in spec.draw(rng)])
        for _ in range(RESOLVES):
            vals = [ca.DM(v) for v in spec.draw(rng)]
            gc.collect()
            t0 = time.perf_counter()
            F.call(vals)
            resolves.append(time.perf_counter() - t0)
    return _median(singles), _median(resolves), _median(builds)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="small-size matrix checks")
    ap.add_argument("--verify-full", action="store_true", help="benchmark-size checks")
    ap.add_argument("--time", action="store_true", help="run the timing comparison")
    ap.add_argument("--only", default="", help="comma-separated problem names")
    ap.add_argument("--out", default="results_casadi_compare.json")
    args = ap.parse_args()

    names = [n for n in args.only.split(",") if n] or list(MAKERS)
    for n in names:
        if n not in MAKERS:
            raise SystemExit(f"unknown problem {n!r}; choose from {list(MAKERS)}")

    if args.verify or args.verify_full:
        print(f"casadi {ca.__version__}, cvxpy {cp.__version__}")
        print("Verifying casadi-extracted (P, c, A, b) == cvxpy ignore_dpp data:")
        all_ok = True
        for n in names:
            all_ok &= verify(MAKERS[n](full=args.verify_full), full=args.verify_full)
        print("ALL MATCH" if all_ok else "MISMATCHES FOUND")
        if not all_ok:
            raise SystemExit(1)

    if args.time:
        rows = []
        for n in names:
            spec = MAKERS[n](full=True)
            cs, cr, cb = time_casadi(spec)
            vs, vr = time_cvxpy(spec)
            rows.append({
                "name": n, "parametric": spec.parametric,
                "casadi_single_s": cs, "casadi_resolve_s": cr, "casadi_build_s": cb,
                "engine_single_s": vs, "engine_resolve_s": vr,
            })
            print(f"{n:<28s} single: casadi {cs*1e3:9.1f}ms  engine {vs*1e3:9.1f}ms"
                  + (f"   resolve: casadi {cr*1e3:9.2f}ms  engine {vr*1e3:9.2f}ms"
                     if spec.parametric else ""))
            (HERE / args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {HERE / args.out}")


if __name__ == "__main__":
    main()
