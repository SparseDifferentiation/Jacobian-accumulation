"""Where does CasADi extraction time go on the dense-P QP? Stage-by-stage at n=800.

Also tries alternative formulations/APIs to make sure we give CasADi its best shot:
  A. bilin(P, x, x)                    (harness formulation)
  B. 0.5 * mtimes(x.T, mtimes(P, x))  (most common user spelling)
  C. hessian via jacobian(gradient(f))
  D. A-matrix only (no Hessian) to isolate the constraint side
"""
import time

import numpy as np

import casadi as ca

n = 800
m, p_ = 5 * n, 20
rng = np.random.RandomState(1)
P0 = rng.randn(n, n)
P0 = P0.T @ P0
q0 = rng.randn(n)
G0 = rng.randn(m, n)
h0 = G0 @ rng.randn(n)
Aeq = rng.randn(p_, n)
beq = rng.randn(p_)


def stage(label, fn):
    t0 = time.perf_counter()
    out = fn()
    print(f"  {label:<42s} {time.perf_counter()-t0:9.2f}s", flush=True)
    return out


for spelling in ("A: bilin", "B: xT_P_x"):
    print(spelling)
    x = ca.MX.sym("x", n)
    if spelling.startswith("A"):
        f = 0.5 * ca.bilin(ca.DM(P0), x, x) + ca.dot(ca.DM(q0), x)
    else:
        f = 0.5 * ca.mtimes(x.T, ca.mtimes(ca.DM(P0), x)) + ca.dot(ca.DM(q0), x)
    g = ca.vertcat(ca.DM(beq) - ca.mtimes(ca.DM(Aeq), x),
                   ca.DM(h0) - ca.mtimes(ca.DM(G0), x))
    H, grad = stage("hessian(f, x)", lambda: ca.hessian(f, x))
    c = stage("substitute(grad, x, 0)", lambda: ca.substitute(grad, x, ca.DM.zeros(n)))
    A = stage("jacobian(g, x)", lambda: -ca.jacobian(g, x))
    b = stage("substitute(g, x, 0)", lambda: ca.substitute(g, x, ca.DM.zeros(n)))
    F = stage("Function ctor", lambda: ca.Function("extract", [], [H, c, A, b]))
    stage("first eval", lambda: F.call([]))

print("C: jacobian(jacobian(f).T) instead of hessian")
x = ca.MX.sym("x", n)
f = 0.5 * ca.bilin(ca.DM(P0), x, x) + ca.dot(ca.DM(q0), x)
J = stage("jacobian(f, x)", lambda: ca.jacobian(f, x))
H2 = stage("jacobian(J.T, x)", lambda: ca.jacobian(ca.transpose(J), x))
F2 = stage("Function + eval", lambda: ca.Function("h2", [], [H2]).call([]))
