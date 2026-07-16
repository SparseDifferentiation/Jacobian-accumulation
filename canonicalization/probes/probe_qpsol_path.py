"""Would CasADi's own qpsol interface extract the QP data faster than the
harness's jacobian-of-gradient path?  (Verification of the fairness claim.)

qpsol({'x','f','g'}) must itself recover (H, c, A) from the expression graph
before it can call any QP plugin; casadi's conic.cpp does this with
MX::hessian(f, x) and MX::jacobian(g, x).  The harness deliberately replaces
hessian() with jacobian(gradient(f)) because hessian()'s star coloring is
pathological on dense Hessians -- so if anything the harness UNDERSTATES
casadi's cost.  This probe measures all candidate paths on the SimpleQP family
(dense n x n quadratic, m = 5n inequality rows, 20 equality rows):

  A. harness path      jacobian(gradient(f)) + jacobian(g) + Function + eval
  B. hessian path      casadi.hessian(f, x)  (what conic.cpp/qpsol uses for H)
  C. qpsol ctor        ca.qpsol('S', 'qrqp', {...}) construction only
  D. coeff helpers     casadi.quadratic_coeff(f, x) + linear_coeff(g, x)
                       (documented "extract QP data" API; also hessian-based)

Path C constructs the solver instance without solving (construction is where
conic.cpp performs the AD).  Run with --big to include n=800; the hessian-based
paths grow super-cubically, so n=1600 (the benchmark size) is out of reach.

Usage: python canonicalization/probes/probe_qpsol_path.py [--big]
"""
import argparse
import time

import numpy as np

import casadi as ca


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
    x = ca.MX.sym("x", n)
    f = 0.5 * ca.bilin(ca.DM(P0), x, x) + ca.dot(ca.DM(q0), x)
    g = ca.vertcat(ca.DM(beq) - ca.mtimes(ca.DM(Aeq), x),
                   ca.DM(h0) - ca.mtimes(ca.DM(G0), x))
    return x, f, g


def timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"    {label:<44s} {dt:9.2f}s", flush=True)
    return out, dt


def path_harness(x, f, g):
    grad = ca.gradient(f, x)
    P = ca.jacobian(grad, x)
    zero = ca.DM.zeros(x.shape)
    c = ca.substitute(grad, x, zero)
    A = -ca.jacobian(g, x)
    b = ca.substitute(g, x, zero)
    ca.Function("extract", [], [P, c, A, b]).call([])


def path_hessian(x, f, g):
    H, _ = ca.hessian(f, x)
    A = -ca.jacobian(g, x)
    ca.Function("extract", [], [H, A]).call([])


def path_qpsol(x, f, g):
    ca.qpsol("S", "qrqp", {"x": x, "f": f, "g": g},
             {"print_header": False, "print_iter": False, "print_info": False,
              "error_on_fail": False})


def path_coeff(x, f, g):
    P, c, _ = ca.quadratic_coeff(f, x, True)
    A, b = ca.linear_coeff(g, x, True)
    ca.Function("extract", [], [P, c, A, b]).call([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true", help="include n=800")
    args = ap.parse_args()

    sizes = (100, 200, 400) + ((800,) if args.big else ())
    for n in sizes:
        print(f"n={n} (m={5*n}, dense P and G):", flush=True)
        x, f, g = build(n)
        timed("A harness: jac(grad(f)) + jac(g) + eval", lambda: path_harness(x, f, g))
        timed("B hessian(f,x) + jac(g) + eval", lambda: path_hessian(x, f, g))
        timed("C qpsol('qrqp') construction", lambda: path_qpsol(x, f, g))
        timed("D quadratic_coeff + linear_coeff + eval", lambda: path_coeff(x, f, g))


if __name__ == "__main__":
    main()
