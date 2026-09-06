"""Would per-constraint jacobians beat the harness's single stacked g?
(Sensitivity check on the casadi_compare.py methodology.)

The harness hands CasADi one stacked g = vertcat(blocks) and extracts
A = -jacobian(g, z) in one call, exactly like CasADi's own qpsol path
(probe_qpsol_path.py).  The alternative: call jacobian(g_k, z) once per
constraint block and apply the row offsets ourselves (scipy vstack), the way
CVXPY's canonicalization places per-constraint blocks.  Does that change the
cost?

Sweep-count analysis says mostly no.  Columns collide only if they share a
row, so blocks over DISJOINT variables reuse each other's colors -- the
stacked jacobian needs ~max(colors_k) sweeps, not the sum, and stacking wins.
The one coloring regime where blocking wins: uni_coloring picks ONE direction
(forward = columns, reverse = rows) for the whole stacked matrix, so a
tall-dense block (cheap forward) stacked with a wide-dense block (cheap
reverse) forces one into its expensive direction.

Four instances, one per regime:

  qp-shape      dense zero-cone block + identity bounds (ParametrizedQP
                shape): same coloring either way (241 vs 243 sweeps).
  mode-conflict tall dense on x (cheap forward) + wide dense on y (cheap
                reverse): stacked pays max-dimension sweeps in either single
                mode; blocked pays ~n_x + rows_wide.
  same-cols     two dense blocks over the same x (SimpleQP shape): the columns
                are a clique either way; sweep totals roughly match.
  disjoint      two tall dense blocks on disjoint x, y, same preferred mode:
                stacked shares colors across the blocks (max, not sum) --
                stacking WINS; per-block pays the sum.

FINDING (casadi 3.7.2): the timings do NOT follow the sweep counts.
qp-shape is ~6x slower stacked despite equal sweeps, and the second section
shows why: ca.jacobian on g = vertcat(blocks) is order-dependent -- ~10x at
small sizes, growing superlinearly (0.83/6.4/52 s vs 0.09/0.47/5.0 s as
(m, n) doubles) -- for identical mathematical content.  Diagnosis: sparsity
detection time, coloring (481/1202 at m=1200), chosen mode (forward, both
orders, ad_weight-forced and auto), and sweep count are all identical across
orders; the gap sits entirely in casadi's symbolic construction of the
forward-mode Jacobian, i.e. an implementation artifact, not AD cost.  The
trigger is specific: a block of the form [identity on t + dense on x] placed
BEFORE the x-only bound rows (dense alone, dense + one trailing row, and
dense + bounds-on-t are all fast).  CVXPY's CLARABEL row order (zero cone
first) produces exactly the triggering order in every dense-block suite
problem: extrapolating to ParametrizedQP's benchmark size (m=6000, n=2400)
predicts ~800 s -- the recorded casadi_single is 771 s.  The same matrix in
the same row order is obtained dense-last + a scipy row permutation (~4 ms,
verified entry-identical), so the stacked-g contract (qpsol-style) stands
while the recorded dense-block singles overstate CasADi by ~10x.

Accounting: the stacked path delivers A as one DM straight from F.call, so its
total is jacobian construction + Function ctor + eval, matching the harness's
"single".  The blocked path delivers per-block DMs; assembling the single
solver-ready A (row offsets, scipy vstack to CSC) is work the harness's g
vector does inside CasADi, so it counts toward the blocked total (also shown
separately).  Blocked A is asserted equal to stacked A entry-for-entry before
timing.

Usage: python canonicalization/probes/probe_blocked_vs_stacked.py
"""
import time

import numpy as np
import scipy.sparse as sps

import casadi as ca

rng = np.random.RandomState(0)


def dm_to_csc(D) -> sps.csc_matrix:
    rows, cols = D.sparsity().get_triplet()
    return sps.coo_matrix(
        (np.asarray(D.nonzeros(), dtype=float), (rows, cols)),
        shape=(D.size1(), D.size2()),
    ).tocsc()


def make_qp_shape():
    m, n = 600, 240
    A0, b0 = rng.randn(m, n), rng.randn(m)
    t = ca.MX.sym("t", m)
    x = ca.MX.sym("x", n)
    z = ca.vertcat(t, x)
    blocks = [ca.DM(b0) + t - ca.mtimes(ca.DM(A0), x), x, 1 - x]
    return "qp-shape", z, blocks


def make_mode_conflict():
    n_x, m_tall, n_y, m_wide = 100, 2000, 2000, 10
    A0, C0 = rng.randn(m_tall, n_x), rng.randn(m_wide, n_y)
    x = ca.MX.sym("x", n_x)
    y = ca.MX.sym("y", n_y)
    z = ca.vertcat(x, y)
    blocks = [ca.mtimes(ca.DM(A0), x) - ca.DM(rng.randn(m_tall)),
              ca.mtimes(ca.DM(C0), y) - ca.DM(rng.randn(m_wide))]
    return "mode-conflict", z, blocks


def make_same_cols():
    m, n, p = 1200, 240, 20
    G0, Aeq = rng.randn(m, n), rng.randn(p, n)
    x = ca.MX.sym("x", n)
    blocks = [ca.DM(rng.randn(p)) - ca.mtimes(ca.DM(Aeq), x),
              ca.DM(rng.randn(m)) - ca.mtimes(ca.DM(G0), x)]
    return "same-cols", x, blocks


def make_disjoint():
    m, n = 600, 150
    A0, B0 = rng.randn(m, n), rng.randn(m, n)
    x = ca.MX.sym("x", n)
    y = ca.MX.sym("y", n)
    z = ca.vertcat(x, y)
    blocks = [ca.mtimes(ca.DM(A0), x), ca.mtimes(ca.DM(B0), y)]
    return "disjoint", z, blocks


def colors(sp):
    """(forward, reverse) sweep counts casadi's uni_coloring would use."""
    return sp.uni_coloring().size2(), sp.T.uni_coloring().size2()


ITERS = 3


def run(label, z, blocks):
    # --- stacked: one jacobian on vertcat(blocks), A delivered as one DM
    g = ca.vertcat(*blocks)
    stacked_times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        J = ca.jacobian(g, z)
        F = ca.Function("stacked", [], [J])
        (A_stacked,) = F.call([])
        stacked_times.append(time.perf_counter() - t0)
    t_stacked = float(np.median(stacked_times))
    fwd, adj = colors(ca.jacobian_sparsity(g, z))

    # --- blocked: one jacobian per block, offsets applied by us in scipy
    ad_times, asm_times = [], []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        Js = [ca.jacobian(gk, z) for gk in blocks]
        Fb = ca.Function("blocked", [], Js)
        outs = Fb.call([])
        ad_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        A_blocked = sps.vstack([dm_to_csc(D) for D in outs], format="csc")
        asm_times.append(time.perf_counter() - t0)
    t_blocked_ad = float(np.median(ad_times))
    t_asm = float(np.median(asm_times))
    per_block = [colors(ca.jacobian_sparsity(gk, z)) for gk in blocks]

    S, B = dm_to_csc(A_stacked), A_blocked.copy()
    for M in (S, B):
        M.eliminate_zeros()
        M.sort_indices()
    assert S.shape == B.shape and (S.indptr == B.indptr).all() \
        and (S.indices == B.indices).all() and np.allclose(S.data, B.data), \
        f"{label}: blocked assembly != stacked A"

    blocked_sweeps = " + ".join(str(min(f, a)) for f, a in per_block)
    print(f"  {label:<14s} stacked {t_stacked:7.2f}s  "
          f"blocked {t_blocked_ad + t_asm:7.2f}s "
          f"(ad {t_blocked_ad:6.2f}s + asm {t_asm:5.2f}s)   "
          f"sweeps: stacked min{fwd, adj} = {min(fwd, adj)}, "
          f"blocked {blocked_sweeps} = {sum(min(f, a) for f, a in per_block)}",
          flush=True)
    return t_stacked, t_blocked_ad + t_asm


print(f"casadi {ca.__version__}: stacked vertcat(g) vs per-block jacobian + manual offsets")
_wz = ca.MX.sym("w", 8)
ca.Function("warmup", [], [ca.jacobian(ca.mtimes(ca.DM(rng.randn(8, 8)), _wz), _wz)]).call([])
results = {}
for make in (make_qp_shape, make_mode_conflict, make_same_cols, make_disjoint):
    label, z, blocks = make()
    results[label] = run(label, z, blocks)

print("\nvertcat order effect (same sparsity, same coloring; only the block order differs):")
for m, n in ((600, 240), (1200, 480), (2400, 960)):
    A0, b0 = rng.randn(m, n), rng.randn(m)
    t = ca.MX.sym("t", m)
    x = ca.MX.sym("x", n)
    z = ca.vertcat(t, x)
    g_dense = ca.DM(b0) + t - ca.mtimes(ca.DM(A0), x)
    row = [f"m={m:5d} n={n:4d} "]
    for order_label, g in (("dense-first", ca.vertcat(g_dense, x, 1 - x)),
                           ("dense-last", ca.vertcat(x, 1 - x, g_dense)),
                           ("dense-only", g_dense)):
        ts = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            ca.jacobian(g, z)
            ts.append(time.perf_counter() - t0)
        row.append(f"{order_label} {float(np.median(ts)):7.3f}s")
    print("  " + "  ".join(row), flush=True)

print("\ndiagnosis at m=1200, n=480 (order effect is not detection, coloring, or mode):")
m, n = 1200, 480
A0, b0 = rng.randn(m, n), rng.randn(m)
t = ca.MX.sym("t", m)
x = ca.MX.sym("x", n)
z = ca.vertcat(t, x)
g_dense = ca.DM(b0) + t - ca.mtimes(ca.DM(A0), x)


def _tsym(g):
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        ca.jacobian(g, z)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


for label, wval in (("forced-forward", 0), ("forced-reverse", 1)):
    ts_first, ts_last = [], []
    for order, sink in ((ca.vertcat(g_dense, x, 1 - x), ts_first),
                        (ca.vertcat(x, 1 - x, g_dense), ts_last)):
        F = ca.Function("F", [z], [order], {"ad_weight": wval})
        for _ in range(ITERS):
            t0 = time.perf_counter()
            F.jacobian()
            sink.append(time.perf_counter() - t0)
    print(f"  {label}: dense-first {float(np.median(ts_first)):7.3f}s  "
          f"dense-last {float(np.median(ts_last)):7.3f}s", flush=True)
print(f"  trigger: dense alone {_tsym(g_dense):.3f}s, + one x row "
      f"{_tsym(ca.vertcat(g_dense, x[0])):.3f}s, + bounds on t "
      f"{_tsym(ca.vertcat(g_dense, t, 1 - t)):.3f}s, + bounds on x "
      f"{_tsym(ca.vertcat(g_dense, x, 1 - x)):.3f}s", flush=True)

(A_last,) = ca.Function("fl", [], [ca.jacobian(ca.vertcat(x, 1 - x, g_dense), z)]).call([])
(A_first,) = ca.Function("ff", [], [ca.jacobian(ca.vertcat(g_dense, x, 1 - x), z)]).call([])
Al = dm_to_csc(A_last)
t0 = time.perf_counter()
A_fixed = Al[np.r_[2 * n + np.arange(m), np.arange(2 * n)], :].tocsc()
t_perm = time.perf_counter() - t0
Ref = dm_to_csc(A_first)
for M in (A_fixed, Ref):
    M.eliminate_zeros()
    M.sort_indices()
assert (Ref.indptr == A_fixed.indptr).all() and (Ref.indices == A_fixed.indices).all() \
    and np.allclose(Ref.data, A_fixed.data)
print(f"  fix: dense-last + scipy row permutation back to CVXPY order = "
      f"{t_perm * 1e3:.1f} ms, entry-identical to dense-first", flush=True)

s, b = results["mode-conflict"]
print(f"""
Conclusion: the coloring regimes behave as sweep counts predict only once the
order effect is factored out.  Coloring already shares sweeps across disjoint
blocks ('disjoint': stacked takes the max, blocked pays the sum) and
same-column dense blocks tie, so the stacked-g contract is not what inflates
the suite; a genuine forward/reverse mode conflict helps blocking
('mode-conflict': {s:.2f}s vs {b:.2f}s), but the suite's stacks are dominated
by a single dense block, not by conflicting-mode pairs (not audited across
all 25 problems).  What inflates the suite is the dense-first vertcat
pathology above:
CVXPY's row order puts the dense zero-cone block first, and that ordering
alone is worth ~10x on the cold jacobian at benchmark sizes.  Either
reordering the stack dense-last and permuting rows afterwards, or per-block
extraction with manual offsets, removes it.""")
