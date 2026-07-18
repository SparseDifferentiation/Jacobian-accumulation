"""Does AMPL/ASL share CasADi's dense-block pathology?  (Spoiler: it cannot.)

Mirror of probe_why_slow.py, driven through the real AMPL Solver Library:
Pyomo builds the constraints g = A x (as an expression tree), writes an AMPL
.nl file, and PyNumero's AslNLP loads it.  For linear constraints the .nl
format stores the coefficients as explicit sparse J-segments, so the Jacobian
pattern AND values exist the moment the file is read -- there is no sparsity
detection, no coloring, and no AD sweep at all.  The expected result is cost
linear in nnz for dense and sparse alike (pure I/O + ASL setup), i.e. the
CasADi mechanism (one seeded sweep per color, dense block => min(m,n) colors)
has no analog here.

Stages timed:
  t_model_build  Pyomo constraint construction (graph-construction analog --
                 reported, excluded from the extraction total, exactly like
                 casadi_build_s in the main harness)
  t_nl_write     model.write(*.nl)  -- AMPL's "canonicalization"
  t_asl_read     AslNLP(*.nl)       -- ASL read + AD structure setup
  t_first_jac    first evaluate_jacobian()
  t_warm_jac     median of 5 more evaluate_jacobian() calls
plus the .nl file size.  Correctness: J is compared exactly against A (up to
the nl writer's row/col permutation, recovered from the .row/.col files).

E1/E3 dense cases stop at n=2000 (4000x2000 = 8M coefficients); E4 stops at
1e6 (Pyomo model *construction* at 4e6 variables is the binding constraint,
not ASL).

Run: .venv/bin/python canonicalization/probes/probe_dense_block_asl.py
"""
import gc
import json
import os
import re
import time
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import numpy as np
import scipy.sparse as sps

import pyomo.environ as pyo
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP

HERE = Path(__file__).resolve().parent
SCRATCH = os.environ.get("ASL_SCRATCH", "")  # default: TemporaryDirectory


def _name_index(path: Path) -> np.ndarray:
    """Map nl-file row/col order -> model index, from a .row/.col file."""
    order = []
    for line in path.read_text().splitlines():
        m = re.search(r"\[(\d+)\]", line)
        if m:
            order.append(int(m.group(1)))
    return np.asarray(order)


def probe(A, label, results, warm_evals=5):
    A_csr = sps.csr_matrix(A)
    m, n = A_csr.shape

    gc.collect()
    t0 = time.perf_counter()
    model = pyo.ConcreteModel()
    model.x = pyo.Var(range(n))
    rows = [
        LinearExpression(
            constant=0.0,
            linear_coefs=A_csr.data[A_csr.indptr[i]:A_csr.indptr[i + 1]].tolist(),
            linear_vars=[model.x[j] for j in
                         A_csr.indices[A_csr.indptr[i]:A_csr.indptr[i + 1]]],
        )
        for i in range(m)
    ]
    model.g = pyo.Constraint(range(m), rule=lambda mo, i: rows[i] == 0.0)
    model.obj = pyo.Objective(expr=0.0)
    t_build = time.perf_counter() - t0

    with TemporaryDirectory(dir=SCRATCH or None) as td:
        nl = Path(td) / "g.nl"
        gc.collect()
        t0 = time.perf_counter()
        model.write(str(nl))          # timed write: no symbolic labels --
        t_write = time.perf_counter() - t0  # .row/.col are harness bookkeeping
        nl_bytes = nl.stat().st_size

        gc.collect()
        t0 = time.perf_counter()
        nlp = AslNLP(str(nl))
        t_read = time.perf_counter() - t0

        nlp.set_primals(np.ones(nlp.n_primals()))
        gc.collect()
        t0 = time.perf_counter()
        J = nlp.evaluate_jacobian()
        t_first = time.perf_counter() - t0

        warm = []
        for _ in range(warm_evals):
            gc.collect()
            t0 = time.perf_counter()
            nlp.evaluate_jacobian()
            warm.append(time.perf_counter() - t0)
        t_warm = median(warm)

        # correctness: undo the nl writer's permutation, then J must equal A
        # (a second, untimed write emits the .row/.col mapping files)
        model.write(str(Path(td) / "chk.nl"),
                    io_options={"symbolic_solver_labels": True})
        row_order = _name_index(Path(td) / "chk.row")
        col_order = _name_index(Path(td) / "chk.col")
        J = sps.csr_matrix(J)
        pr = np.argsort(row_order)  # nl row for model row i
        pc = np.argsort(col_order)
        J_model = J[pr][:, pc]
        diff = abs(J_model - A_csr)
        err = diff.max() if diff.nnz else 0.0
        assert err <= 1e-12, f"J != A for {label}: max|diff|={err}"

    print(f"  {label:<34s} build {t_build:7.2f}s | write {t_write:7.2f}s  "
          f"read {t_read:7.2f}s  jac1 {t_first:8.4f}s  warm {t_warm:8.4f}s  "
          f"nnz {J.nnz}  nl {nl_bytes/1e6:.1f}MB", flush=True)
    results.append({
        "label": label, "m": m, "n": n, "nnz": int(A_csr.nnz),
        "t_model_build_s": t_build, "t_nl_write_s": t_write,
        "t_asl_read_s": t_read, "t_first_jac_s": t_first,
        "t_warm_jac_s": t_warm, "nl_bytes": nl_bytes,
    })


def main():
    import pyomo
    rng = np.random.RandomState(0)
    results = []

    print("E1: dense tall block (m=2n), n doubling")
    for n in (250, 500, 1000, 2000):
        probe(rng.randn(2 * n, n), f"dense {2*n}x{n}", results)

    print("E2: same shapes, sparse ~10 nnz/row")
    for n in (250, 500, 1000, 2000):
        probe(sps.random(2 * n, n, density=10.0 / n, random_state=0, format="csr"),
              f"sparse {2*n}x{n} (10/row)", results)

    print("E3: dense wide block (n=2m), m doubling")
    for m in (250, 500, 1000, 2000):
        probe(rng.randn(m, 2 * m), f"dense {m}x{2*m}", results)

    print("E4: diagonal, LP scale (linear regime; 4e6 skipped -- Pyomo model "
          "construction, not ASL, is the binding constraint)")
    for n in (10**5, 10**6):
        probe(sps.eye(n, format="csr"), f"diag {n}", results, warm_evals=3)

    out = HERE.parent / "results_probe_dense_block_asl.json"
    out.write_text(json.dumps({"pyomo": pyomo.__version__, "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
