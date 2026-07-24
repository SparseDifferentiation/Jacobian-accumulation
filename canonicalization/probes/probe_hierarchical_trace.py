"""Trace CasADi's Jacobian pipeline stage by stage on the suite's slow shapes:
is the paper's hierarchical sparsity algorithm running, and where do the
minutes actually go?

The CasADi paper (Andersson et al. 2019, Sect. 5) claims, for jacobian():
  stage 1  sparsity DETECTION: bitvector propagation, 64 columns per sweep,
           with hierarchical refinement that exploits large all-zero regions
           (Example 1: a 1e5 tridiagonal pattern in 7 sweeps instead of 1563);
  stage 2  graph COLORING of the detected pattern (unidirectional);
  stage 3  ACCUMULATION: one AD sweep of the whole graph per color.
It also states the disclaimer that the approach "is not efficient if the
nonzero entries are spread out randomly and CasADi assumes that the user takes
this into account when e.g. formulating a large NLP" -- i.e. patterns without
large empty regions are the known-bad case.

This probe checks both halves against the benchmark problems:
  (a) CONTROL: the paper's Example-1 tridiagonal reproduces (detection fast,
      3 colors) -- the hierarchical algorithm works as advertised;
  (b) the actual ParametrizedQP rows  g = [b + t - A x; x; 1 - x]  with A an
      m x n dense MX *parameter* (and the constant-DM twin, the LeastSquares /
      SimpleQP case), at growing scale, timing each stage separately:

        t_pattern   ca.jacobian_sparsity(g, z)      -- stage 1 alone
        colors      sp.uni_coloring fwd / adj       -- stage 2 (sweep counts)
        t_jacsym    ca.jacobian(g, z)               -- stages 1+2+3, symbolic
        t_ctor      casadi.Function construction
        t_eval      one numeric evaluation

      Dense blocks admit no compression, so colors = full column dimension and
      stage 3 degenerates to n whole-graph sweeps: the time should track
      colors, not nnz.  Detection alone should stay comparatively small.

  (c) --full: the real ParametrizedQP (m, n) = (6000, 2400) harness path
      (gradient, jacobian for P, jacobian for A, substitutes, Function ctor,
      evaluation at a drawn theta), splitting the suite's ~770 s headline
      number into its stages.

Usage:
    python canonicalization/probes/probe_hierarchical_trace.py [--full]
Pin BLAS to 1 thread as usual for timing runs.
"""
import argparse
import time

import numpy as np
import scipy.sparse as sps

import casadi as ca


def _fmt(t):
    return f"{t:8.2f}s"


def example1_control():
    print("(a) paper Example-1 control: 100000x100000 tridiagonal, constant")
    n = 100_000
    T = sps.diags([np.ones(n - 1), np.ones(n), np.ones(n - 1)], [-1, 0, 1],
                  format="csc")
    x = ca.MX.sym("x", n)
    g = ca.mtimes(ca.DM(T), x)
    t0 = time.perf_counter()
    sp = ca.jacobian_sparsity(g, x)
    t_pattern = time.perf_counter() - t0
    fwd = sp.uni_coloring().size2()
    adj = sp.T.uni_coloring().size2()
    t0 = time.perf_counter()
    ca.jacobian(g, x)
    t_jac = time.perf_counter() - t0
    print(f"    pattern {_fmt(t_pattern)}  colors fwd/adj {fwd}/{adj}  "
          f"jac(sym) {_fmt(t_jac)}   [paper: 7 sweeps, 3 colors]\n")


def parametrized_qp_rows(m, n, symbolic_A):
    """The exact ParametrizedQP constraint rows of casadi_compare.py."""
    t = ca.MX.sym("t", m)
    x = ca.MX.sym("x", n)
    z = ca.vertcat(t, x)
    if symbolic_A:
        Ap = ca.MX.sym("A", m, n)
        bp = ca.MX.sym("b", m)
    else:
        rng = np.random.RandomState(1)
        Ap, bp = ca.DM(rng.randn(m, n)), ca.DM(rng.randn(m))
    g = ca.vertcat(bp + t - ca.mtimes(Ap, x), x, 1 - x)
    return z, g, ([Ap, bp] if symbolic_A else [])


def stage_trace(m, n, symbolic_A):
    z, g, psyms = parametrized_qp_rows(m, n, symbolic_A)
    nrow, ncol = 2 * n + m, m + n

    t0 = time.perf_counter()
    sp = ca.jacobian_sparsity(g, z)
    t_pattern = time.perf_counter() - t0

    t0 = time.perf_counter()
    fwd = sp.uni_coloring().size2()
    adj = sp.T.uni_coloring().size2()
    t_color = time.perf_counter() - t0

    t0 = time.perf_counter()
    J = ca.jacobian(g, z)
    t_jacsym = time.perf_counter() - t0

    t0 = time.perf_counter()
    F = ca.Function("J", psyms, [J])
    t_ctor = time.perf_counter() - t0

    rng = np.random.RandomState(7)
    argv = [rng.randn(m, n), rng.randn(m)] if symbolic_A else []
    t0 = time.perf_counter()
    F.call([ca.DM(v) for v in argv])
    t_eval = time.perf_counter() - t0

    kind = "MX param A" if symbolic_A else "constant A"
    naive_sweeps = -(-ncol // 64)  # ceil: bitvector batch, no hierarchy
    print(f"  {kind}  {m}x{n}: J is {nrow}x{ncol}, nnz {sp.nnz()}\n"
          f"    detection {_fmt(t_pattern)}  (<= {naive_sweeps} bitvector "
          f"sweeps even without hierarchy)\n"
          f"    coloring  {_fmt(t_color)}   colors fwd/adj {fwd}/{adj}"
          f"  -> accumulation = {min(fwd, adj)} whole-graph AD sweeps\n"
          f"    jac(sym)  {_fmt(t_jacsym)}  fn ctor {_fmt(t_ctor)}  "
          f"eval {_fmt(t_eval)}   total {_fmt(t_pattern + t_jacsym + t_ctor + t_eval)}",
          flush=True)


def full_harness_split():
    """Decompose the suite's ParametrizedQP single_s into harness stages."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from casadi_compare import MAKERS

    print("\n(c) real ParametrizedQP (6000x2400), harness extraction stages")
    spec = MAKERS["ParametrizedQP"](full=True)
    z, psyms, f, g = spec.casadi_build()

    t0 = time.perf_counter()
    grad = ca.gradient(f, z)
    t_grad = time.perf_counter() - t0

    t0 = time.perf_counter()
    P = ca.jacobian(grad, z)
    t_jacP = time.perf_counter() - t0

    t0 = time.perf_counter()
    A = -ca.jacobian(g, z)
    t_jacA = time.perf_counter() - t0

    zero = ca.DM.zeros(z.shape)
    t0 = time.perf_counter()
    c = ca.substitute(grad, z, zero)
    b = ca.substitute(g, z, zero)
    t_subst = time.perf_counter() - t0

    t0 = time.perf_counter()
    F = ca.Function("extract", psyms, [P, c, A, b])
    t_ctor = time.perf_counter() - t0

    theta = spec.draw(np.random.default_rng(42))
    t0 = time.perf_counter()
    F.call([ca.DM(v) for v in theta])
    t_eval = time.perf_counter() - t0

    total = t_grad + t_jacP + t_jacA + t_subst + t_ctor + t_eval
    print(f"    gradient(f)      {_fmt(t_grad)}\n"
          f"    jacobian->P      {_fmt(t_jacP)}   (identity block: cheap)\n"
          f"    jacobian->A      {_fmt(t_jacA)}   (dense 6000x2400 param block)\n"
          f"    substitutes      {_fmt(t_subst)}\n"
          f"    Function ctor    {_fmt(t_ctor)}\n"
          f"    eval (1 theta)   {_fmt(t_eval)}\n"
          f"    TOTAL            {_fmt(total)}   "
          f"(suite casadi_single_s was ~771.5 s = ctor-onward, x1 draw, med 3)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="add the 6000x2400 suite-size harness split (slow)")
    args = ap.parse_args()

    example1_control()

    print("(b) ParametrizedQP-shaped rows, stage-by-stage")
    for m, n in ((750, 300), (1500, 600), (3000, 1200)):
        stage_trace(m, n, symbolic_A=True)
    print()
    for m, n in ((750, 300), (1500, 600), (3000, 1200)):
        stage_trace(m, n, symbolic_A=False)
    print("\n  detection+coloring only at suite size (6000x2400):")
    z, g, _ = parametrized_qp_rows(6000, 2400, True)
    t0 = time.perf_counter()
    sp = ca.jacobian_sparsity(g, z)
    t_pattern = time.perf_counter() - t0
    fwd = sp.uni_coloring().size2()
    adj = sp.T.uni_coloring().size2()
    print(f"    detection {_fmt(t_pattern)}  colors fwd/adj {fwd}/{adj}  "
          f"nnz {sp.nnz()}")

    if args.full:
        full_harness_split()


if __name__ == "__main__":
    main()
