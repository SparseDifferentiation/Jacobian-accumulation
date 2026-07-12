"""How does CasADi extraction time scale on SimpleQP (dense P) with n?

SimpleQP full size is n=1600 (m=8000, p=20); extraction DNF'd at >6.5 h.
Probe n = 100/200/400 (m = 5n, p = 20) to characterize the growth, engine
alongside for the same instances.
"""
import sys
import time
import warnings

import numpy as np

import casadi as ca
import cvxpy as cp

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from casadi_compare import casadi_extract_function  # noqa: E402

warnings.filterwarnings("ignore")


def build(n):
    m, p_ = 5 * n, 20
    rng = np.random.RandomState(1)
    P0 = rng.randn(n, n)
    P0 = P0.T @ P0
    q0 = rng.randn(n)
    G0 = rng.randn(m, n)
    h0 = G0 @ rng.randn(n)
    Aeq = rng.randn(p_, n)
    beq = rng.randn(p_)

    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize((1 / 2) * cp.quad_form(x, P0, assume_PSD=True) + q0.T @ x),
        [G0 @ x <= h0, Aeq @ x == beq],
    )

    xs = ca.MX.sym("x", n)
    f = 0.5 * ca.bilin(ca.DM(P0), xs, xs) + ca.dot(ca.DM(q0), xs)
    g = ca.vertcat(ca.DM(beq) - ca.mtimes(ca.DM(Aeq), xs),
                   ca.DM(h0) - ca.mtimes(ca.DM(G0), xs))
    return prob, (xs, [], f, g)


for n in (100, 200, 400, 800):
    prob, model = build(n)
    t0 = time.perf_counter()
    F = casadi_extract_function(*model)
    F.call([])
    t_ca = time.perf_counter() - t0
    t0 = time.perf_counter()
    prob.get_problem_data(solver=cp.CLARABEL, ignore_dpp=True)
    t_en = time.perf_counter() - t0
    print(f"n={n:5d}  casadi {t_ca:9.2f}s   engine {t_en:7.2f}s   ratio {t_ca/t_en:8.1f}x",
          flush=True)
