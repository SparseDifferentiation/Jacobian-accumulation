# =============================================================================
# Sparse Jacobian computation with the AMPL Solver Library (ASL)
# =============================================================================
# Same tridiagonal system as the casadi/jax/diffengine examples, driven through
# the *real* ASL: Pyomo writes an AMPL .nl file, PyNumero's AslNLP loads it and
# evaluates derivatives through the compiled AMPL Solver Library.
#
# The point of this example: ASL has NO sparsity-detection and NO graph-coloring
# stage.  The .nl format stores each constraint's linear coefficients as
# explicit sparse J-segments, and the nonlinear parts as per-constraint
# expression graphs whose variable sets are known at read time (Gay, "Hooking
# Your Solver to AMPL"; Gay 1996 for Hessians via partial separability).  The
# Jacobian pattern therefore exists the moment the file is read -- structure is
# a priori, like SparseDiffEngine's per-atom structure, not discovered by
# tracing sweeps like CasADi/SCT.  Gradients per constraint use reverse AD on
# tapes allocated during the read.
#
# Requires: uv sync --extra asl && pyomo build-extensions (see repo README).

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP

n = 10

model = pyo.ConcreteModel()
model.x = pyo.Var(range(n), initialize=1.0)


def f_rule(m, i):
    expr = pyo.sin(m.x[i])
    if i > 0:
        expr = expr + m.x[i - 1] * m.x[i]
    if i < n - 1:
        expr = expr + pyo.exp(-m.x[i] * m.x[i + 1])
    return expr == 0


model.f = pyo.Constraint(range(n), rule=f_rule)
# ASL requires an objective; a constant one adds nothing to the Jacobian.
model.obj = pyo.Objective(expr=0.0)

# PyomoNLP writes the .nl file to a temp dir and loads it through AslNLP.
nlp = PyomoNLP(model)

# The nl writer orders variables (nonlinear-before-linear, else declaration
# order); print the mapping so the columns below can be read unambiguously.
print("nl column order:", nlp.primals_names())
print("nl row order:   ", nlp.constraint_names())

# =============================================================================
# Sparsity: available immediately after the .nl read -- no detection pass, no
# coloring.  ASL evaluates the Jacobian row-by-row (one reverse sweep per
# constraint tape); cost is O(sum of per-row tape sizes), independent of any
# column-orthogonality structure.
# =============================================================================
nlp.set_primals(np.ones(n))
J = nlp.evaluate_jacobian().tocoo()

print(f"\nJacobian sparsity: {J.nnz} non-zeros out of {n * n}")
print("Coloring: none -- ASL reads the pattern off the .nl file "
      "(no compression step exists or is needed)")

print("\nSparse Jacobian values:")
for r, c, v in sorted(zip(J.row, J.col, J.data)):
    print(f"  J[{r},{c}] = {v:.6f}")

print(f"\nFull Jacobian:\n{J.toarray()}")
