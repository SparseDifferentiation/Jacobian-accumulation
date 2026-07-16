"""Would SX expressions beat MX for problem-data extraction?

Both SX and MX go through the same AD pipeline (sparsity sweeps -> coloring ->
one sweep per color; function_internal.cpp), so the dense-block sweep count is
identical.  What differs is the graph: MX keeps a dense constant as ONE mtimes
node; SX scalar-expands it into ~nnz multiply-add nodes.  casadi itself frames
SX as a small-problem optimization (the 'expand' option of qpsol).

Measured here on the SimpleQP extraction (harness path: jacobian(gradient(f))
+ jacobian(g) + Function + eval) and on a bare dense-block jacobian, MX vs SX,
with the SX graph-construction cost reported separately (MX build is ~free).
Node counts show why SX cannot reach benchmark sizes: nodes ~ nnz, so CVaR's
1e8-nnz block or SimpleLP's 1e7 variables would exhaust memory before AD runs.

Usage: python canonicalization/probes/probe_sx_vs_mx.py
"""
import time

import numpy as np

import casadi as ca


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def extract(sym, x, f, g):
    grad = ca.gradient(f, x)
    P = ca.jacobian(grad, x)
    zero = ca.DM.zeros(x.shape)
    c = ca.substitute(grad, x, zero)
    A = -ca.jacobian(g, x)
    b = ca.substitute(g, x, zero)
    ca.Function("extract", [], [P, c, A, b]).call([])


def qp_model(sym, n):
    m, p_ = 5 * n, 20
    rng = np.random.RandomState(1)
    P0 = rng.randn(n, n)
    P0 = P0.T @ P0
    q0 = rng.randn(n)
    G0 = rng.randn(m, n)
    h0 = G0 @ rng.randn(n)
    Aeq = rng.randn(p_, n)
    beq = rng.randn(p_)
    x = sym.sym("x", n)
    f = 0.5 * ca.bilin(ca.DM(P0), x, x) + ca.dot(ca.DM(q0), x)
    g = ca.vertcat(ca.DM(beq) - ca.mtimes(ca.DM(Aeq), x),
                   ca.DM(h0) - ca.mtimes(ca.DM(G0), x))
    return x, f, g


print("SimpleQP family, harness extraction path, MX vs SX")
print(f"  {'n':>5s} {'kind':>4s} {'build(s)':>9s} {'extract(s)':>10s} {'g nodes':>10s}")
for n in (50, 100, 200, 400):
    for sym in (ca.MX, ca.SX):
        (xfg), t_build = timed(lambda: qp_model(sym, n))
        x, f, g = xfg
        _, t_ex = timed(lambda: extract(sym, x, f, g))
        nodes = ca.Function("g", [x], [g]).n_nodes()
        print(f"  {n:>5d} {sym.__name__:>4s} {t_build:>9.2f} {t_ex:>10.2f} {nodes:>10d}",
              flush=True)

print("bare dense-block jacobian, g = A x (m = 2n)")
print(f"  {'n':>5s} {'kind':>4s} {'build(s)':>9s} {'jac+eval(s)':>11s} {'g nodes':>10s}")
rng = np.random.RandomState(0)
for n in (250, 500, 1000):
    A0 = rng.randn(2 * n, n)
    for sym in (ca.MX, ca.SX):
        (xg), t_build = timed(lambda: (sym.sym("x", n),))
        x = xg[0]
        g, t_g = timed(lambda: ca.mtimes(ca.DM(A0), x))
        _, t_jac = timed(lambda: ca.Function("j", [], [ca.jacobian(g, x)]).call([]))
        nodes = ca.Function("g", [x], [g]).n_nodes()
        print(f"  {n:>5d} {sym.__name__:>4s} {t_g:>9.2f} {t_jac:>11.2f} {nodes:>10d}",
              flush=True)
