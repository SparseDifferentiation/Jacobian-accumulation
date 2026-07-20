# SimpleQP -- cvxpy benchmark suite, lowered CLARABEL form.
#
#   original:  minimize (1/2) x'P0x + q0'x
#              s.t.     G0 x <= h0,  Aeq x == beq
#
# The lowering is the identity here (no epigraph variables): z = x, and the
# CLARABEL rows are the equalities (zero cone) first, then the inequalities
# (nonneg cone) -- mirroring casadi_compare.make_simple_qp exactly.
#
# Data (P0 dense n x n PSD, q0, G0 dense m x n, h0, Aeq p x n, beq) comes from
# SimpleQP.dat, generated with the suite's RandomState(1) stream; the small
# instance (m,n,p) = (4,3,1) is committed for hand verification, the benchmark
# instance is (8000, 1600, 20).
#
# Extraction identities (what the harness reads back through ASL):
#   P = Hessian of obj = P0        c = grad obj at 0 = q0
#   A = -d(body)/dx rows           b = body(0) - lb rows
# with constraint bodies written as b - A x {=,>=} 0 below.

param n integer > 0;
param m integer > 0;
param p integer > 0;

param P0 {0..n-1, 0..n-1};
param q0 {0..n-1};
param G0 {0..m-1, 0..n-1};
param h0 {0..m-1};
param Aeq {0..p-1, 0..n-1};
param beq {0..p-1};

var x {0..n-1};

minimize obj:
    0.5 * sum {i in 0..n-1, j in 0..n-1} P0[i,j] * x[i] * x[j]
  + sum {j in 0..n-1} q0[j] * x[j];

# zero cone: beq - Aeq x = 0
s.t. zero_eq {i in 0..p-1}:
    beq[i] - sum {j in 0..n-1} Aeq[i,j] * x[j] = 0;

# nonneg cone: h0 - G0 x >= 0
s.t. nonneg_ineq {i in 0..m-1}:
    h0[i] - sum {j in 0..n-1} G0[i,j] * x[j] >= 0;
