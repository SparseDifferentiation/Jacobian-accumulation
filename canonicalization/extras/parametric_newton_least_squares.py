"""EARLY DRAFT — kept for reference, not part of the thesis tables.

A different experiment from the extraction harness one directory up: it uses the
diff engine as a derivative oracle inside a Newton loop to *solve* parametric
least squares end-to-end, against CasADi's qpsol. The canonical extraction-only
comparison (no solving) is canonicalization/casadi_compare.py.
"""

"""Parametric least squares: CVXPY diff engine vs CasADi's QP solver interface.

Problem (parametric in b):

    minimize_x  || A x - b ||_2^2          A in R^{m x n} fixed,  b in R^m the parameter

This is the canonical warm-start / "solve the same structure for many right-hand
sides" scenario. We sweep over K random parameter vectors b and, for each one,
recover the optimal x with two different toolchains, timing the per-parameter cost.

--------------------------------------------------------------------------------
Pipeline A -- CVXPY diff engine (SparseDiffEngine via the DNLP fork)
--------------------------------------------------------------------------------
The diff engine is a *derivative oracle*, not a solver: it canonicalizes the
problem once and then, for any x, hands back the objective gradient g = nabla f(x)
and the Lagrangian Hessian H. For this quadratic objective:

    f(x) = ||A x - b||^2,   nabla f(x) = 2 A^T (A x - b),   H = 2 A^T A  (constant)

Because f is quadratic, a single Newton step from any x0 is exact:

    x* = x0 - H^{-1} nabla f(x0)

So per parameter we: update_params(b) -> objective_forward -> gradient -> one
prefactored linear solve. H is constant in b, so we extract + factor it once.

--------------------------------------------------------------------------------
Pipeline B -- CasADi QP solver interface (ca.conic, OSQP backend)
--------------------------------------------------------------------------------
We build a conic QP solver once for the fixed Hessian sparsity, then per parameter
form the linear term g = -2 A^T b and call the solver:

    minimize_x  0.5 x^T (2 A^T A) x + (-2 A^T b)^T x

Both pipelines are checked against numpy.linalg.lstsq.

Run:  uv run python benchmarks/parametric_least_squares.py
      uv run python benchmarks/parametric_least_squares.py --sizes 50x20 200x80 --params 200
"""

from __future__ import annotations

import argparse
import time

import numpy as np


# --------------------------------------------------------------------------- #
# Timing helper
# --------------------------------------------------------------------------- #
def best_of(fn, repeats: int) -> float:
    """Run ``fn`` ``repeats`` times, return the best (minimum) wall-clock time."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# --------------------------------------------------------------------------- #
# Problem generation
# --------------------------------------------------------------------------- #
def make_problem(m: int, n: int, n_params: int, seed: int):
    """A fixed (m x n) design matrix plus ``n_params`` random right-hand sides."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    B = rng.standard_normal((n_params, m))  # one b per row
    x0 = np.zeros(n)                         # evaluation point for the oracle
    return A, B, x0


# --------------------------------------------------------------------------- #
# Pipeline A: CVXPY diff engine
# --------------------------------------------------------------------------- #
def build_diffengine(A, b_init):
    """Canonicalize once and extract the constant Hessian. Returns (oracle, factor)."""
    import cvxpy as cp
    from scipy.linalg import cho_factor
    from cvxpy.reductions.dnlp2smooth.dnlp2smooth import Dnlp2Smooth
    from cvxpy.reductions.solvers.nlp_solvers.diff_engine.c_problem import C_problem

    m, n = A.shape
    x = cp.Variable(n)
    b = cp.Parameter(m)
    b.value = b_init
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

    # Dnlp2Smooth lowers the DCP problem to the smooth-NLP form the C engine
    # expects; skipping it leaves the DAG malformed and segfaults the extension.
    canon = Dnlp2Smooth().apply(prob)[0]
    c = C_problem(canon, verbose=False)
    c.init_jacobian_coo()
    c.init_hessian_coo_lower_tri()

    # Extract the (constant) Hessian H = 2 A^T A once and prefactor it.
    # Call order matters: objective_forward -> gradient -> constraint_forward
    # before eval_hessian (gradient() primes internal buffers).
    n_con = len(c.constraint_forward(np.zeros(n)))
    c.objective_forward(np.zeros(n))
    c.gradient()
    c.constraint_forward(np.zeros(n))
    hr, hc = c.get_problem_hessian_sparsity_coo()
    hv = c.eval_hessian_vals_coo_lower_tri(1.0, np.zeros(n_con))
    H = np.zeros((n, n))
    H[hr, hc] = hv
    H = H + H.T - np.diag(np.diag(H))
    factor = cho_factor(H)
    return c, factor


def solve_diffengine(c, factor, x0, b):
    """One parameter update + gradient extraction + exact Newton step."""
    from scipy.linalg import cho_solve

    c.update_params(b)
    c.objective_forward(x0)
    g = c.gradient()
    return x0 - cho_solve(factor, g)


# --------------------------------------------------------------------------- #
# Pipeline B: CasADi conic QP interface
# --------------------------------------------------------------------------- #
def build_casadi(A):
    """Build a conic QP solver once for the fixed Hessian sparsity."""
    import casadi as ca
    from scipy import sparse

    n = A.shape[1]
    H = 2.0 * A.T @ A
    Hsp = sparse.csc_matrix(H)
    spar = ca.Sparsity(n, n, Hsp.indptr.tolist(), Hsp.indices.tolist())
    opts = {"print_time": False, "osqp": {"verbose": False}}
    solver = ca.conic("ls_qp", "osqp", {"h": spar}, opts)
    return solver, ca.DM(H)


def solve_casadi(solver, H_dm, A, b):
    """Form the parameter-dependent linear term and solve the QP."""
    import casadi as ca

    g = -2.0 * A.T @ b
    res = solver(h=H_dm, g=ca.DM(g))
    return np.asarray(res["x"]).flatten()


# --------------------------------------------------------------------------- #
# Benchmark driver
# --------------------------------------------------------------------------- #
def run_size(m: int, n: int, n_params: int, repeats: int, seed: int):
    A, B, x0 = make_problem(m, n, n_params, seed)
    reference = np.linalg.lstsq(A, B[0], rcond=None)[0]

    # ---- setup (one-time) ----
    t_setup_de = best_of(lambda: build_diffengine(A, B[0]), repeats)
    c, factor = build_diffengine(A, B[0])

    t_setup_ca = best_of(lambda: build_casadi(A), repeats)
    solver, H_dm = build_casadi(A)

    # ---- correctness ----
    x_de = solve_diffengine(c, factor, x0, B[0])
    x_ca = solve_casadi(solver, H_dm, A, B[0])
    err_de = np.linalg.norm(x_de - reference)
    err_ca = np.linalg.norm(x_ca - reference)

    # ---- per-parameter sweep cost (best-of over the full K-sweep) ----
    def sweep_de():
        for b in B:
            solve_diffengine(c, factor, x0, b)

    def sweep_ca():
        for b in B:
            solve_casadi(solver, H_dm, A, b)

    t_sweep_de = best_of(sweep_de, repeats)
    t_sweep_ca = best_of(sweep_ca, repeats)

    return {
        "m": m, "n": n, "K": n_params,
        "setup_de": t_setup_de, "setup_ca": t_setup_ca,
        "per_de": t_sweep_de / n_params, "per_ca": t_sweep_ca / n_params,
        "sweep_de": t_sweep_de, "sweep_ca": t_sweep_ca,
        "err_de": err_de, "err_ca": err_ca,
    }


def fmt_us(seconds: float) -> str:
    return f"{seconds * 1e6:8.1f}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", nargs="+", default=["20x10", "100x40", "400x150"],
                   help="problem sizes as MxN (rows x cols), space separated")
    p.add_argument("--params", type=int, default=200,
                   help="number of parameter vectors b in the sweep (K)")
    p.add_argument("--repeats", type=int, default=5,
                   help="best-of repeats for each timed block")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sizes = []
    for s in args.sizes:
        m, n = (int(v) for v in s.lower().split("x"))
        sizes.append((m, n))

    print(f"\nParametric least squares  ||A x - b||^2,  K = {args.params} parameter "
          f"sweeps,  best-of-{args.repeats}\n")
    print("Setup is one-time; per-param is the amortized sweep cost (microseconds).\n")

    head = (f"{'size (mxn)':>12} | {'setup DE':>9} {'setup CA':>9} (us) | "
            f"{'per-b DE':>9} {'per-b CA':>9} (us) | {'speedup':>7} | "
            f"{'err DE':>8} {'err CA':>8}")
    print(head)
    print("-" * len(head))

    for m, n in sizes:
        r = run_size(m, n, args.params, args.repeats, args.seed)
        speedup = r["per_ca"] / r["per_de"]
        print(f"{m:5d}x{n:<6d} | {fmt_us(r['setup_de'])} {fmt_us(r['setup_ca'])}      | "
              f"{fmt_us(r['per_de'])} {fmt_us(r['per_ca'])}      | "
              f"{speedup:6.1f}x | {r['err_de']:.1e} {r['err_ca']:.1e}")

    print("\nDE = CVXPY diff engine (update_params + gradient + prefactored Newton step)")
    print("CA = CasADi conic QP interface (form linear term + OSQP solve)")
    print("err = ||x - numpy.lstsq||_2\n")


if __name__ == "__main__":
    main()
