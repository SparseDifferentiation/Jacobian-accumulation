"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Run the upstream cvxpy benchmark suite (https://github.com/cvxpy/benchmarks)
against the canonicalization backends, on both the DPP path and the
``ignore_dpp`` path.

Clone the suite first (next to this script; gitignored):
    git clone --depth 1 https://github.com/cvxpy/benchmarks canonicalization/cvxpy_benchmarks

Then:
    python canonicalization/run_backend_benchmarks.py

What it does
------------
Every benchmark problem is run *as is* -- we import the upstream ASV classes
unmodified and call ``setup()``. We do NOT call the benchmark's own
``time_compile_problem``: each class declares its own solver (SCS, OSQP, ...),
which would make the columns incomparable. Instead every measurement is a
uniform ``get_problem_data(solver=CLARABEL, **target_kwargs)``, and that single
call is the entire timed region.

Which target each column selects is defined once, in TARGET_KWARGS; see
TARGET_LEGEND for what they mean. The comparison that motivates the ``*_ND``
targets: on the DPP path a backend builds a parameter->data *tensor*, whereas
``ignore_dpp=True`` bakes the parameters into constants (EvalParams) and builds
a plain matrix -- the same artefact the diff engine produces. Comparing a
tensor build against a tree evaluation flatters whichever side is doing less
work, so the ``*_ND`` columns exist to compare like with like. They require
upstream cvxpy: the diff-engine fork raises ValueError for an explicit
canon_backend on the ignore_dpp path.

For problems that have ``cp.Parameter`` objects we additionally run the
re-compilation comparison, K re-compiles with fresh parameter values on a FRESH
benchmark instance per strategy: the chain cache key is
(solver, gp, ignore_dpp, use_quad_obj) -- it excludes canon_backend and is_dpp -- so
sharing one Problem across strategies silently reuses the previous strategy's cached
chain / param_prog and times the wrong thing.

Robustness: several upstream problems are gigantic (1e6-1e7 var LPs, 5000^2 cone
stuffing, 6000x2400 QP) and the diff engine can segfault on some shapes, so each
benchmark runs in its own subprocess with a wall-clock timeout and streams partial
results to a JSON file. A hang / OOM / segfault in one problem is isolated and
reported, never aborting the run. The timeout is enforced by
_run_worker_subprocess, which kills the worker's whole process group and bounds
the reap -- plain subprocess.run(timeout=) blocks forever on a child wedged in an
uninterruptible syscall, which is how one problem ran 77 min against a 40 min
budget. Worker stderr goes to _worker_stderr.log and its tail is folded into the
status column, since a worker that dies without writing JSON (segfault, OOM kill)
leaves no other evidence.

Memory mode (BENCH_MEMORY=1)
----------------------------
Reports peak RSS instead of time. One isolated subprocess per measurement -- per
(problem, backend) cold extraction and per (problem, strategy) warm re-solve loop
-- so backends never share a heap. The metric is the ru_maxrss high-water DELTA
around the measured region (gc.collect + snapshot after setup / after the warmup
compile, snapshot again after the region); tracemalloc would miss cvxcore's C++
heap entirely. ru_maxrss is monotone, so a region that never exceeds the prior
high-water mark reports 0 -- flagged as a floor, not a true peak. Engine targets
(DIFFENGINE/DE_DPP/diffengine/de_cached) additionally record the diff engine's
own sp_malloc allocation counters (g_peak_bytes/g_allocated_bytes via ctypes),
separating engine-internal memory from CVXPY front-end / scipy glue memory.
Output defaults to results_backends_memory.txt (+ raw .json sibling); timing
reference files are never touched. Full suite is ~165 subprocesses (hours);
quick test:
    BENCH_MEMORY=1 BENCH_ONLY=HuberRegression BENCH_BACKENDS=CPP,DIFFENGINE \
        BENCH_ITERS=1 python canonicalization/run_backend_benchmarks.py

The ignore_dpp sweep on upstream cvxpy (see canonicalization/.venv-upstream):
    BENCH_BACKENDS=CPP,SCIPY,COO,CPP_ND,SCIPY_ND,COO_ND \
    BENCH_STRATEGIES=dpp,dpp_coo,dpp_scipy,nodpp_cpp,nodpp_scipy,nodpp_coo \
    BENCH_RATIO=CPP_ND/CPP BENCH_RATIO_WARM=nodpp_cpp/dpp \
        canonicalization/.venv-upstream/bin/python canonicalization/run_backend_benchmarks.py

Env knobs: BENCH_TIMEOUT (s, default 240; per-measurement in memory mode),
           BENCH_ITERS (default 3), BENCH_ONLY (comma-separated class names),
           BENCH_SKIP, BENCH_BACKENDS, BENCH_STRATEGIES, BENCH_OUT,
           BENCH_RATIO / BENCH_RATIO_WARM ("NUM/DEN", the last table column),
           BENCH_COLD=0 (skip cold),
           BENCH_MEMORY=1 (memory mode), BENCH_WARM=0 (memory mode: skip warm).
"""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE / "cvxpy_benchmarks" / "benchmark"

TIMEOUT = float(os.environ.get("BENCH_TIMEOUT", "240"))
ITERS = int(os.environ.get("BENCH_ITERS", "3"))
COMPARE_SOLVER = "CLARABEL"  # uniform solver so all backends are comparable

# Every measurable target -- cold canon backend (Table 1) or warm re-compile
# strategy (Table 2) -- maps to the get_problem_data kwargs that select it.
# One table, so the timing and memory workers cannot drift apart.
#
# The `_ND` / `nodpp_` targets are the ignore_dpp=True variants of the stock
# tensor backends: parameters are baked into constants (EvalParams) and the
# backend builds a plain non-parametric matrix, i.e. the same artefact the
# diff engine produces. That is the apples-to-apples comparison; the plain
# CPP/SCIPY/COO targets build a parameter->data tensor, which is strictly more
# work. NOTE these only run on upstream cvxpy: the diff-engine fork raises
# ValueError for an explicit canon_backend on the ignore_dpp path (parametrized
# <=2-D problems are force-routed to DIFFENGINE).
TARGET_KWARGS = {
    # -- cold canon backends --------------------------------------------- #
    "CPP": {"canon_backend": "CPP"},
    "SCIPY": {"canon_backend": "SCIPY"},
    "COO": {"canon_backend": "COO"},
    "CPP_ND": {"canon_backend": "CPP", "ignore_dpp": True},
    "SCIPY_ND": {"canon_backend": "SCIPY", "ignore_dpp": True},
    "COO_ND": {"canon_backend": "COO", "ignore_dpp": True},
    # No canon_backend at all: whatever cvxpy picks. Kept only for continuity
    # with results published before the CPP column was pinned -- see below.
    "CPP_DEFAULT": {},
    "DIFFENGINE": {"ignore_dpp": True},
    # Explicit-arg selection for forks where DIFFENGINE is opt-in only (the
    # re-scoped pr-a has no ignore_dpp default yet).
    "DIFFENGINE_EXPL": {"canon_backend": "DIFFENGINE", "ignore_dpp": True},
    "DE_DPP": {"canon_backend": "DIFFENGINE"},
    # -- warm re-compile strategies -------------------------------------- #
    "dpp": {},
    "diffengine": {"ignore_dpp": True},
    "de_cached": {"canon_backend": "DIFFENGINE"},
    "dpp_scipy": {"canon_backend": "SCIPY"},
    "dpp_coo": {"canon_backend": "COO"},
    "nodpp_cpp": {"canon_backend": "CPP", "ignore_dpp": True},
    "nodpp_scipy": {"canon_backend": "SCIPY", "ignore_dpp": True},
    "nodpp_coo": {"canon_backend": "COO", "ignore_dpp": True},
}

# Warm strategies that only mean anything on the DPP path (they exist to time
# the cached-tensor re-apply). On a non-DPP problem cvxpy would route them
# somewhere else entirely, so they are skipped rather than silently measuring
# a different code path under the old label.
REQUIRES_DPP = {"de_cached", "dpp_scipy", "dpp_coo"}

# Why CPP is pinned: with no canon_backend, cvxpy substitutes COO on the DPP
# path whenever the total parameter size reaches DPP_PARAM_THRESHOLD (1000).
# That silently made the CPP and COO columns the same code on 4 of the 8
# parametric problems in results_backends_all_20260724_repl.txt. Passing
# canon_backend="CPP" explicitly keeps the baseline honest; CPP_DEFAULT
# reproduces the old behaviour if a published table needs to be replicated.
BACKENDS = [b.strip() for b in os.environ.get(
    "BENCH_BACKENDS", "CPP,DIFFENGINE,DE_DPP,SCIPY,COO").split(",") if b.strip()]

# Re-compilation strategies (parametric problems only). Order matters: the
# established comparison runs first so it survives a crash or worker timeout
# in a later backend (SCIPY has done both on cold compiles).
STRATEGIES = [s.strip() for s in os.environ.get(
    "BENCH_STRATEGIES",
    "diffengine,de_cached,dpp,dpp_coo,dpp_scipy").split(",") if s.strip()]


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover():
    """Return [(module_name, class_name), ...] for every benchmark class."""
    import importlib

    sys.path.insert(0, str(BENCH_DIR))
    found = []
    for path in sorted(BENCH_DIR.glob("*.py")):
        if path.stem == "__init__":
            continue
        try:
            mod = importlib.import_module(path.stem)
        except Exception as exc:  # noqa: BLE001
            found.append((path.stem, None, f"import error: {type(exc).__name__}: {exc}"))
            continue
        for name in dir(mod):
            obj = getattr(mod, name)
            if (
                isinstance(obj, type)
                and obj.__module__ == mod.__name__
                and hasattr(obj, "setup")
                and any(m.startswith("time_") for m in dir(obj))
            ):
                found.append((path.stem, name, None))
    return found


# --------------------------------------------------------------------------- #
# Worker helpers
# --------------------------------------------------------------------------- #
def _assign_param(param, rng):
    """Assign a fresh, attribute-valid random value to a cp.Parameter."""
    shape = param.shape
    attrs = getattr(param, "attributes", {}) or {}
    if attrs.get("PSD") or attrs.get("NSD"):
        n = shape[0]
        M = rng.standard_normal((n, n))
        val = M @ M.T + np.eye(n)
        if attrs.get("NSD"):
            val = -val
    elif attrs.get("symmetric") or attrs.get("hermitian"):
        n = shape[0]
        M = rng.standard_normal((n, n))
        val = (M + M.T) / 2.0
    elif attrs.get("nonneg"):
        val = np.abs(rng.standard_normal(shape))
    elif attrs.get("nonpos"):
        val = -np.abs(rng.standard_normal(shape))
    else:
        val = rng.standard_normal(shape)
    param.value = float(val) if shape == () else val


def _find_problem(inst):
    import cvxpy as cp

    prob = getattr(inst, "problem", None)
    if isinstance(prob, cp.Problem):
        return prob
    for v in vars(inst).values():
        if isinstance(v, cp.Problem):
            return v
    return None


def _chain_is_dpp(prob):
    """is_dpp() as construct_solving_chain evaluates it, not the bare default.

    The chain uses quad_form_dpp='qp' whenever the target solver accepts a
    quadratic objective (CLARABEL does), which changes the verdict for
    quad_form atoms: it additionally requires the x in quad_form(x, P) to be
    parameter-free. Plain prob.is_dpp() omits that and disagrees -- on
    ConvexPlasticity it reports DPP for a problem the chain actually sends
    down the non-DPP branch, so the reported "CPP baseline" for that row was
    really the diff engine.
    """
    try:
        from cvxpy.reductions.solvers.defines import SOLVER_MAP_CONIC, SOLVER_MAP_QP

        si = SOLVER_MAP_CONIC.get(COMPARE_SOLVER) or SOLVER_MAP_QP.get(COMPARE_SOLVER)
        quad_form_dpp = "qp" if si is not None and si.supports_quad_obj() else None
        return bool(prob.is_dpp("dcp", quad_form_dpp=quad_form_dpp))
    except Exception:  # noqa: BLE001
        return bool(prob.is_dpp())


def _kwargs_for(target, prob=None):
    """get_problem_data kwargs selecting one cold backend / warm strategy.

    Single source of truth for both the timing and the memory worker; they
    used to carry four copies of this branch and could drift apart silently.
    """
    try:
        extra = TARGET_KWARGS[target]
    except KeyError:
        raise ValueError(f"unknown target {target!r}") from None
    if prob is not None and target in REQUIRES_DPP and not _chain_is_dpp(prob):
        raise ValueError(f"not DPP: {target} needs the DPP path")
    return {"solver": COMPARE_SOLVER, **extra}


def _time_strategy(Cls, strategy, rng):
    """Return (mean_s, std_s) for K re-compilations under one strategy.

    Builds a fresh benchmark instance: the chain cache key excludes
    canon_backend and is_dpp, so a shared Problem would silently reuse the
    previous strategy's cached chain / param_prog.
    """
    import cvxpy as cp

    inst = Cls()
    inst.setup()
    prob = _find_problem(inst)
    params = prob.parameters()

    kwargs = _kwargs_for(strategy, prob)
    for p in params:  # one warmup compile to populate caches
        _assign_param(p, rng)
    prob.get_problem_data(**kwargs)
    times = []
    for _ in range(ITERS):
        for p in params:
            _assign_param(p, rng)
        t0 = time.perf_counter()
        prob.get_problem_data(**kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def _cold_compile(Cls, backend):
    """Cold first-compile under one canon backend; returns (seconds, problem).

    A fresh instance/setup is required per backend because the chain cache key is
    (solver, gp, ignore_dpp, use_quad_obj) -- it does NOT include canon_backend,
    so reusing one problem would silently return the first backend's cached chain.
    """
    inst = Cls()
    inst.setup()
    prob = _find_problem(inst)
    # Initialize any unspecified parameters so the cold compile doesn't fail on
    # problems whose setup() leaves Parameter values unset (deterministic seed).
    if prob is not None:
        rng = np.random.default_rng(0)
        for p in prob.parameters():
            if p.value is None:
                _assign_param(p, rng)
    kwargs = _kwargs_for(backend, prob)
    t0 = time.perf_counter()
    prob.get_problem_data(**kwargs)
    return time.perf_counter() - t0, prob


def run_worker(module_name, class_name, out_path):
    """Run one benchmark; stream partial results to out_path as JSON."""
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(BENCH_DIR))
    import importlib

    import cvxpy as cp  # noqa: F401

    result = {"module": module_name, "class": class_name}

    def flush():
        Path(out_path).write_text(json.dumps(result))

    flush()
    mod = importlib.import_module(module_name)
    Cls = getattr(mod, class_name)

    # Cold first-compile under each canonicalization backend (fresh setup each).
    # BENCH_COLD=0 skips Table 1 (e.g. when only the re-compile comparison matters).
    result["cold"] = {}
    meta_prob = None
    backends = BACKENDS if os.environ.get("BENCH_COLD", "1") != "0" else BACKENDS[:1]
    for backend in backends:
        try:
            secs, prob = _cold_compile(Cls, backend)
            result["cold"][backend] = secs
            if meta_prob is None:
                meta_prob = prob
        except Exception as exc:  # noqa: BLE001
            result["cold"][backend] = f"ERR:{type(exc).__name__}"
        flush()

    if meta_prob is not None:
        result["n_vars"] = int(sum(v.size for v in meta_prob.variables()))
        result["n_params"] = len(meta_prob.parameters())
        result["sense"] = type(meta_prob.objective).__name__
        try:
            # The chain's predicate, so the dpp? column matches the branch the
            # problem actually takes; is_dpp_plain keeps the value older
            # results reported, for continuity when they disagree.
            result["is_dpp"] = _chain_is_dpp(meta_prob)
            result["is_dpp_plain"] = bool(meta_prob.is_dpp())
        except Exception:  # noqa: BLE001
            result["is_dpp"] = None
    flush()

    # Re-compilation comparison only for parametric problems (secondary).
    params = meta_prob.parameters() if meta_prob is not None else []
    if params:
        result["comparison"] = {}
        for strat in STRATEGIES:
            rng = np.random.default_rng(0)  # same value stream across strategies
            try:
                mean, std = _time_strategy(Cls, strat, rng)
                result["comparison"][strat] = {"mean_s": mean, "std_s": std}
            except Exception as exc:  # noqa: BLE001
                result["comparison"][strat] = {"error": f"{type(exc).__name__}: {exc}"[:120]}
            flush()
    result["done"] = True
    flush()


# --------------------------------------------------------------------------- #
# Parent orchestration
# --------------------------------------------------------------------------- #
def _run_worker_subprocess(cmd, timeout, stderr_path=None):
    """Run one worker, and actually come back within roughly `timeout`.

    `subprocess.run(timeout=...)` is not sufficient here. On timeout it sends
    SIGKILL to the direct child and then blocks in wait() until that child is
    reaped -- which never happens if the child is wedged in an uninterruptible
    syscall. A 1.4e7-parameter problem thrashing swap did exactly that and ran
    77 minutes against a 40 minute budget, taking the rest of the sweep with it.

    Three changes: the child gets its own process group so the whole group can
    be signalled; termination escalates SIGTERM -> SIGKILL with bounded waits,
    so an unreapable child is reported rather than waited on forever; and
    stderr goes to a file instead of a pipe, since a pipe that fills up is
    itself a way to deadlock, and the previous code captured the output only
    to discard it.

    Returns a status string: "ok", "TIMEOUT(>Ns)", or one of those plus
    "+UNREAPED" when the process could not be killed.
    """
    err = open(stderr_path, "wb") if stderr_path else subprocess.DEVNULL
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err,
                                start_new_session=True)
    finally:
        if stderr_path:
            err.close()

    try:
        proc.wait(timeout=timeout)
        return "ok"
    except subprocess.TimeoutExpired:
        pass

    status = f"TIMEOUT(>{timeout:.0f}s)"
    for sig, grace in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 10.0)):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            return status  # already gone
        try:
            proc.wait(timeout=grace)
            return status
        except subprocess.TimeoutExpired:
            continue
    # Wedged past SIGKILL. Say so and keep going rather than block the sweep.
    print(f"  WARNING: worker pid {proc.pid} survived SIGKILL; continuing", flush=True)
    return status + "+UNREAPED"


def _stderr_tail(path, limit=300):
    """Last few characters of a worker's stderr, for the status column.

    Worth keeping: a worker that dies without writing JSON leaves no Python
    traceback anywhere else, which is exactly the case that was hardest to
    diagnose (a segfault and an OOM kill look identical from the parent).
    """
    try:
        text = Path(path).read_text(errors="replace").strip()
    except OSError:
        return ""
    if not text:
        return ""
    return text[-limit:].replace("\n", " ")


def _fmt(v):
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return str(v)


def run_parent():
    only = {s.strip() for s in os.environ.get("BENCH_ONLY", "").split(",") if s.strip()}
    skip = {s.strip() for s in os.environ.get("BENCH_SKIP", "").split(",") if s.strip()}
    classes = discover()
    rows = []
    tmp = HERE / "_worker_result.json"
    errlog = HERE / "_worker_stderr.log"
    out = HERE / os.environ.get("BENCH_OUT", "results_upstream_benchmarks.txt")
    raw = out.with_suffix(".json")

    print(f"Upstream cvxpy/benchmarks suite. timeout={TIMEOUT:.0f}s/problem, "
          f"comparison: {ITERS} re-compiles via {COMPARE_SOLVER}.\n")

    for module_name, class_name, err in classes:
        if class_name is None:
            rows.append({"module": module_name, "class": "(module)", "status": err})
            continue
        if only and class_name not in only:
            continue
        if class_name in skip:
            rows.append({"module": module_name, "class": class_name, "status": "skipped (too large)"})
            out.write_text(header() + "\n\n" + render(rows))
            continue
        label = f"{module_name}.{class_name}"
        print(f"running {label} ...", flush=True)
        if tmp.exists():
            tmp.unlink()
        cmd = [sys.executable, __file__, "--worker", module_name, class_name, str(tmp)]
        status = _run_worker_subprocess(cmd, TIMEOUT, stderr_path=errlog)
        res = {}
        if tmp.exists():
            try:
                res = json.loads(tmp.read_text())
            except Exception:  # noqa: BLE001
                res = {}
        if not res:
            res = {"module": module_name, "class": class_name}
        if not res.get("done"):
            if status == "ok":
                # Exited without finishing and without an exception: a segfault
                # or an OOM kill. The worker's stderr is the only evidence.
                tail = _stderr_tail(errlog)
                status = f"crashed/killed{': ' + tail if tail else ''}"
            res.setdefault("status", status)
        else:
            res["status"] = "ok"
        rows.append(res)
        # Persist after every problem so an interrupted run keeps partial results.
        out.write_text(header() + "\n\n" + render(rows))
        # Raw sibling, as memory mode already does: the .txt is a rendering, and
        # re-deriving a summary from it later means parsing fixed-width columns.
        raw.write_text(json.dumps(rows, indent=1))
    if tmp.exists():
        tmp.unlink()

    report = render(rows)
    print("\n" + report)
    print(f"\nSaved to {out} (raw JSON: {raw})")


# The 8 parametric problems of the upstream suite. Needed by --summarize, which
# reads a rendered table where n_params is not a column; a live run reads
# n_params off the Problem instead and never consults this list.
PARAMETRIC_PROBLEMS = {
    "FactorCovarianceModel", "ConvexPlasticity", "ParamConeMatrixStuffing",
    "ParamSmallMatrixStuffing", "SimpleFullyParametrizedLPBenchmark",
    "SimpleScalarParametrizedLPBenchmark", "ParametrizedQPBenchmark",
    "SVMWithL1Regularization",
}


def _parse_table(path):
    """{problem_short_name: {column: seconds}} from an ALREADY RENDERED Table 1."""
    txt = Path(path).read_text()
    blk = txt[txt.index("Table 1 -- "):]
    blk = blk[:blk.index("Table 2 -- ")]
    header_line = next(ln for ln in blk.splitlines() if ln.startswith("benchmark"))
    cols = header_line.split()[3:-2]  # between dpp? and the ratio/status pair
    # Accept either the target name or the abbreviation the table prints.
    canon = {}
    for c in cols:
        canon[c] = c
    for target, label in COL_LABEL.items():
        if label in cols:
            canon[target] = label
    out = {}
    for ln in blk.splitlines():
        if not ln or ln.startswith(("Table", "---", "benchmark", "  geometric")):
            continue
        toks = ln.split()
        cells = toks[3:3 + len(cols)]
        if len(cells) < len(cols):
            continue
        row = {}
        for c, v in zip(cols, cells):
            try:
                row[c] = float(v)
            except ValueError:
                pass
        out[toks[0].split(".")[-1]] = row
    return canon, out


def summarize(path_num, path_den, num, den):
    """Geometric means of `num`/`den`, optionally across two result files.

    Runs after the fact on committed .txt files, so a published figure can be
    re-derived without re-running the benchmark. Two paths are allowed because
    the like-for-like parametric comparison spans environments: the engine only
    exists in the fork, the *_ND columns only in stock upstream cvxpy.
    """
    try:
        canon_a, A = _parse_table(path_num)
        canon_b, B = _parse_table(path_den)
    except (ValueError, StopIteration) as exc:
        print(f"cannot parse a Table 1 out of the inputs: {exc}")
        return 1
    if num not in canon_a:
        print(f"{path_num}: column {num!r} not found")
        return 1
    if den not in canon_b:
        print(f"{path_den}: column {den!r} not found")
        return 1
    ka, kb = canon_a[num], canon_b[den]

    buckets = {"all": [], "parametric": [], "parameter-free": []}
    missing = []
    for name, row in A.items():
        a, b = row.get(ka), B.get(name, {}).get(kb)
        if not (isinstance(a, float) and isinstance(b, float) and a > 0 and b > 0):
            missing.append(name)
            continue
        buckets["all"].append(a / b)
        buckets["parametric" if name in PARAMETRIC_PROBLEMS
                else "parameter-free"].append(a / b)

    same = path_num == path_den
    print(f"{path_num}" + ("" if same else f"\n  vs {path_den}"))
    print(f"  ratio = {num}/{den}")
    for key in ("all", "parametric", "parameter-free"):
        g, n = _geomean_rows(buckets[key])
        if g is not None:
            print(f"  [{key:14}] geometric mean = {g:.3f}x  (n={n})")
    if missing:
        # Never let a shrinking denominator pass unnoticed.
        print(f"  excluded ({len(missing)}): {', '.join(sorted(missing))}")
    return 0


TARGET_LEGEND = (
    "Targets (the get_problem_data kwargs each column selects):\n"
    "  CPP/SCIPY/COO      explicit canon_backend, DPP path -- builds the\n"
    "                     parameter->data TENSOR (only worth it across re-solves).\n"
    "  CPP_ND/SCIPY_ND/   same backend with ignore_dpp=True: EvalParams bakes the\n"
    "  COO_ND             parameters into constants, so the backend builds a plain\n"
    "                     non-parametric matrix -- the SAME artefact the diff engine\n"
    "                     produces, hence the like-for-like comparison. Upstream\n"
    "                     cvxpy only: the fork rejects an explicit backend here.\n"
    "  CPP_DEFAULT        no canon_backend at all. NOT the same as CPP: cvxpy\n"
    "                     silently switches to COO once total parameter size >=\n"
    "                     DPP_PARAM_THRESHOLD (1000). Kept only to replicate\n"
    "                     results published before the CPP column was pinned.\n"
    "  DIFFENGINE         ignore_dpp=True with no explicit backend (diff-engine fork).\n"
    "  DE_DPP             explicit canon_backend=DIFFENGINE on the DPP path.\n"
    "  dpp/diffengine/    warm strategies; nodpp_* are the ignore_dpp=True variants,\n"
    "  de_cached/dpp_*/   where cvxpy sets uncached_param_prog and every re-solve is\n"
    "  nodpp_*            a FULL recompile -- i.e. the cost of not having DPP.\n"
    "'dpp?' is is_dpp() as the solving chain evaluates it (quad_form_dpp='qp' for\n"
    "solvers that take a quadratic objective), which is what decides the branch --\n"
    "not the bare problem.is_dpp() older tables reported.\n"
)


def header():
    import platform

    import cvxpy as cp
    return (
        "Upstream cvxpy/benchmarks suite: canonicalization backends + ignore_dpp\n"
        "=======================================================================\n"
        f"Source:  https://github.com/cvxpy/benchmarks (cloned, run as is)\n"
        f"cvxpy:   {cp.__version__}   python: {platform.python_version()}   "
        f"platform: {sys.platform}\n"
        f"timeout: {TIMEOUT:.0f}s/problem   solver={COMPARE_SOLVER} (uniform, for comparability)\n"
        f"backends: {','.join(BACKENDS)}\n"
        f"strategies: {','.join(STRATEGIES)}\n"
        "\n"
        "Table 1 = COLD first-compile time (s) of each problem under each canon backend\n"
        "  (fresh setup + get_problem_data per backend, since the chain cache key\n"
        "  excludes canon_backend).\n"
        "Table 2 = re-compile time (s) with changing parameters (parametric problems\n"
        f"  only), mean of {ITERS}, fresh instance per strategy.\n"
        "The last column of each table is a configurable ratio (BENCH_RATIO,\n"
        "BENCH_RATIO_WARM); <1 means the numerator is faster.\n"
        "\n"
        + TARGET_LEGEND
    )


def _cell(v):
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    if isinstance(v, str) and v.startswith("ERR"):
        return v
    return "-"


# Display abbreviations, so the historic column headings survive the switch to
# generated columns and the long names still fit.
COL_LABEL = {"DIFFENGINE": "DIFFENG", "diffengine": "diffeng"}


def _columns(rows, section, preferred):
    """Column order for a table: the configured targets first, then any other
    target present in the data. Driven by the rows rather than a hardcoded
    list, so an old results JSON re-renders with exactly its own columns."""
    seen = []
    for r in rows:
        for key in (r.get(section, {}) or {}):
            if key not in seen:
                seen.append(key)
    cols = [t for t in preferred if t in seen]
    cols += [t for t in seen if t not in cols]
    return cols


def _ratio_spec(env_key, default):
    """'NUM/DEN' -> (num, den); the ratio column is configurable because the
    interesting pair changes with the target set (DIFFENGINE/CPP for the engine
    comparison, CPP_ND/CPP for the DPP-tensor overhead)."""
    num, _, den = os.environ.get(env_key, default).partition("/")
    return num.strip(), den.strip()


def _ratio_value(get, num, den):
    a, b = get(num), get(den)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b > 0 and a > 0:
        return a / b
    return None


def _ratio(get, num, den):
    v = _ratio_value(get, num, den)
    return f"{v:.2f}x" if v is not None else "-"


def _geomean_rows(vals):
    """Geometric mean of per-problem ratios, the summary the thesis quotes.

    Computed here rather than by hand so the published figure is reproducible
    from the results file. Rows where either operand is missing or errored are
    excluded, and the count is reported alongside so a shrinking denominator
    cannot pass unnoticed.
    """
    vals = [v for v in vals if v is not None and v > 0]
    if not vals:
        return None, 0
    return math.exp(sum(math.log(v) for v in vals) / len(vals)), len(vals)


def _geomean_lines(rows, section, rnum, rden, getter, label):
    """Geometric-mean summary split by problem class.

    Parametric and parameter-free problems answer different questions -- for a
    parameter-free problem ignore_dpp is a no-op and no parameter tensor is
    built -- so a pooled mean mixes two populations. Split by n_params.
    """
    buckets = {"all": [], "parametric": [], "parameter-free": []}
    for r in rows:
        v = _ratio_value(getter(r), rnum, rden)
        if v is None:
            continue
        buckets["all"].append(v)
        np_ = r.get("n_params")
        if isinstance(np_, int):
            buckets["parametric" if np_ else "parameter-free"].append(v)
    out = []
    for key in ("all", "parametric", "parameter-free"):
        g, n = _geomean_rows(buckets[key])
        if g is not None:
            out.append(f"  geometric mean {label} [{key:14}] = {g:.3f}x  (n={n})")
    return out


def render(rows):
    lines = []

    # ---- Table 1: cold first-compile per canon backend ----
    cols = _columns(rows, "cold", BACKENDS)
    labels = [COL_LABEL.get(c, c) for c in cols]
    w = max(9, max((len(x) for x in labels), default=9))
    rnum, rden = _ratio_spec("BENCH_RATIO", "DIFFENGINE/CPP")
    rlabel = f"{COL_LABEL.get(rnum, rnum)}/{COL_LABEL.get(rden, rden)}"
    rw = max(7, len(rlabel))

    lines.append("Table 1 -- cold first-compile (s) by canon backend")
    h1 = (f"{'benchmark':44} {'n_vars':>10} {'dpp?':>5} "
          + " ".join(f"{x:>{w}}" for x in labels)
          + f" {rlabel:>{rw}} {'status':>14}")
    lines.append(h1)
    lines.append("-" * len(h1))
    for r in rows:
        label = f"{r.get('module','?')}.{r.get('class','?')}"
        nv = r.get("n_vars", "-")
        is_dpp = r.get("is_dpp")
        dpp_s = "-" if is_dpp is None else ("yes" if is_dpp else "no")
        cold = r.get("cold", {}) or {}
        nv_s = f"{nv:,}" if isinstance(nv, int) else str(nv)
        lines.append(
            f"{label:44} {nv_s:>10} {dpp_s:>5} "
            + " ".join(f"{_cell(cold.get(c, '-')):>{w}}" for c in cols)
            + f" {_ratio(cold.get, rnum, rden):>{rw}} {r.get('status','-'):>14}"
        )
    lines += _geomean_lines(rows, "cold", rnum, rden,
                            lambda r: (r.get("cold", {}) or {}).get, rlabel)

    # ---- Table 2: parametric re-compile comparison ----
    param_rows = [r for r in rows if r.get("comparison")]
    scols = _columns(param_rows, "comparison", STRATEGIES)
    slabels = [COL_LABEL.get(c, c) for c in scols]
    sw = max(10, max((len(x) for x in slabels), default=10))
    snum, sden = _ratio_spec("BENCH_RATIO_WARM", "de_cached/dpp")
    srlabel = f"{COL_LABEL.get(snum, snum)} vs {COL_LABEL.get(sden, sden)}"
    srw = max(11, len(srlabel))

    lines.append("")
    lines.append("Table 2 -- re-compile (s) with changing parameters (parametric problems only)")
    h2 = (f"{'benchmark':44} {'n_vars':>10} "
          + " ".join(f"{x:>{sw}}" for x in slabels)
          + f" {srlabel:>{srw}}")
    lines.append(h2)
    lines.append("-" * len(h2))
    for r in param_rows:
        label = f"{r.get('module','?')}.{r.get('class','?')}"
        nv = r.get("n_vars", "-")
        comp = r.get("comparison", {})

        def mean(key, _comp=comp):
            return (_comp.get(key) or {}).get("mean_s")

        def cv(key, _comp=comp):
            c = _comp.get(key, {})
            if "mean_s" in c:
                return f"{c['mean_s']:.4f}"
            return "ERR" if "error" in c else "-"

        nv_s = f"{nv:,}" if isinstance(nv, int) else str(nv)
        lines.append(
            f"{label:44} {nv_s:>10} "
            + " ".join(f"{cv(c):>{sw}}" for c in scols)
            + f" {_ratio(mean, snum, sden):>{srw}}"
        )
    lines += _geomean_lines(
        param_rows, "comparison", snum, sden,
        lambda r: (lambda k: ((r.get("comparison", {}) or {}).get(k) or {}).get("mean_s")),
        srlabel)

    # ---- Error detail ----
    errnotes = []
    for r in rows:
        for key, v in (r.get("cold", {}) or {}).items():
            if isinstance(v, str) and v.startswith("ERR"):
                errnotes.append(f"  cold {r.get('module')}.{r.get('class')} [{key}]: {v}")
        for key, c in (r.get("comparison", {}) or {}).items():
            if isinstance(c, dict) and "error" in c:
                errnotes.append(f"  recompile {r.get('module')}.{r.get('class')} [{key}]: {c['error']}")
    if errnotes:
        lines.append("\nErrors:")
        lines.extend(errnotes)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Memory mode (BENCH_MEMORY=1): peak-RSS per (problem, backend/strategy)
# --------------------------------------------------------------------------- #
# Targets whose measured region runs the diff engine, so the engine's own
# allocation counters are worth reading alongside the process RSS.
ENGINE_TARGETS = {"DIFFENGINE", "DIFFENGINE_EXPL", "DE_DPP", "diffengine", "de_cached"}


def _maxrss_mb():
    """Peak RSS high-water mark of this process, MB (ru_maxrss is bytes on
    macOS, kilobytes on Linux)."""
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def _rss_now_mb():
    """Current RSS in MB: /proc on Linux, `ps -o rss=` (KB) on macOS."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except OSError:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True)
        return int(out.stdout.strip()) / 1024


def _engine_counters_mb():
    """(live_mb, peak_mb) from the diff engine's sp_malloc tracker.

    g_allocated_bytes / g_peak_bytes are process-wide globals in
    _sparsediffengine.so (usable-size accounting, engine-internal allocations
    only; CVXPY front-end / scipy / numpy memory is invisible to them).
    g_peak_bytes is reset to the current live bytes at each capsule
    construction (new_problem). Returns (None, None) if unresolvable.
    """
    try:
        import ctypes

        import sparsediffpy._sparsediffengine as se

        lib = ctypes.CDLL(se.__file__)
        live = ctypes.c_size_t.in_dll(lib, "g_allocated_bytes").value
        peak = ctypes.c_size_t.in_dll(lib, "g_peak_bytes").value
        return live / 2**20, peak / 2**20
    except Exception:  # noqa: BLE001
        return None, None


def run_memory_worker(module_name, class_name, out_path, kind, target):
    """One isolated memory measurement, streamed to out_path as JSON.

    kind="cold":  fresh setup, snapshot, ONE get_problem_data under canon
                  backend `target`, snapshot.
    kind="warm":  fresh setup, warmup compile under strategy `target` (cache
                  population), snapshot, ITERS re-solves with fresh parameter
                  values, snapshot -- the delta is the ADDITIONAL peak of warm
                  re-solves.
    Target kwargs come from _kwargs_for, the same helper the timing worker
    uses, so the two modes cannot measure different things under one label.
    """
    import gc
    import importlib

    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(BENCH_DIR))
    import cvxpy as cp  # noqa: F401

    result = {"module": module_name, "class": class_name, "mode": "memory",
              "kind": kind, "target": target}

    def flush():
        Path(out_path).write_text(json.dumps(result))

    def fail(exc):
        result["error"] = f"{type(exc).__name__}: {exc}"[:120]
        result["done"] = True
        flush()

    flush()
    mod = importlib.import_module(module_name)
    Cls = getattr(mod, class_name)

    inst = Cls()
    inst.setup()
    prob = _find_problem(inst)
    rng = np.random.default_rng(0)  # same value stream as the timing mode
    if kind == "cold" and prob is not None:
        for p in prob.parameters():
            if p.value is None:
                _assign_param(p, rng)
    result["n_vars"] = int(sum(v.size for v in prob.variables()))
    result["n_params"] = len(prob.parameters())
    try:
        result["is_dpp"] = _chain_is_dpp(prob)
        result["is_dpp_plain"] = bool(prob.is_dpp())
    except Exception:  # noqa: BLE001
        result["is_dpp"] = None
    flush()

    try:
        if kind == "warm" and not prob.parameters():
            result["skip"] = "no parameters"
            result["done"] = True
            flush()
            return
        kwargs = _kwargs_for(target, prob)
        if kind == "warm":
            for p in prob.parameters():  # warmup compile to populate caches
                _assign_param(p, rng)
            prob.get_problem_data(**kwargs)
            result["iters"] = ITERS
    except Exception as exc:  # noqa: BLE001
        fail(exc)
        return

    engine = target in ENGINE_TARGETS
    gc.collect()
    result["pre_maxrss_mb"] = _maxrss_mb()
    result["pre_rss_mb"] = _rss_now_mb()
    if engine:
        result["engine_pre_live_mb"], result["engine_peak_pre_mb"] = _engine_counters_mb()
    flush()

    try:
        if kind == "cold":
            prob.get_problem_data(**kwargs)
        else:
            for _ in range(ITERS):
                for p in prob.parameters():
                    _assign_param(p, rng)
                prob.get_problem_data(**kwargs)
    except Exception as exc:  # noqa: BLE001
        fail(exc)
        return

    result["post_maxrss_mb"] = _maxrss_mb()
    result["post_rss_mb"] = _rss_now_mb()
    if engine:
        result["engine_post_live_mb"], result["engine_peak_mb"] = _engine_counters_mb()
    delta = result["post_maxrss_mb"] - result["pre_maxrss_mb"]
    result["delta_mb"] = delta
    result["floor"] = delta <= 0.0
    result["retained_mb"] = result["post_rss_mb"] - result["pre_rss_mb"]
    result["done"] = True
    flush()


def run_parent_memory():
    only = {s.strip() for s in os.environ.get("BENCH_ONLY", "").split(",") if s.strip()}
    skip = {s.strip() for s in os.environ.get("BENCH_SKIP", "").split(",") if s.strip()}
    cold_on = os.environ.get("BENCH_COLD", "1") != "0"
    warm_on = os.environ.get("BENCH_WARM", "1") != "0"
    classes = discover()
    rows = []
    tmp = HERE / "_worker_result.json"
    errlog = HERE / "_worker_stderr.log"
    out = HERE / os.environ.get("BENCH_OUT", "results_backends_memory.txt")
    raw = out.with_suffix(".json")

    n_classes = sum(1 for _, c, _ in classes
                    if c is not None and (not only or c in only) and c not in skip)
    est = n_classes * (len(BACKENDS) if cold_on else 0) \
        + (n_classes * len(STRATEGIES) if warm_on else 0)
    print(f"Memory mode: peak-RSS delta, ONE subprocess per measurement "
          f"(<= {est} children for {n_classes} problems; warm children only run "
          f"for parametric problems). timeout={TIMEOUT:.0f}s/measurement, "
          f"warm iters={ITERS}.\n"
          f"Quick tests: BENCH_ONLY=<Name> BENCH_BACKENDS=CPP,DIFFENGINE BENCH_ITERS=1\n",
          flush=True)

    def persist():
        out.write_text(header_memory() + "\n\n" + render_memory(rows))
        raw.write_text(json.dumps(rows, indent=2))

    def spawn(kind, target):
        if tmp.exists():
            tmp.unlink()
        cmd = [sys.executable, __file__, "--worker", module_name, class_name,
               str(tmp), kind, target]
        status = _run_worker_subprocess(cmd, TIMEOUT, stderr_path=errlog)
        res = {}
        if tmp.exists():
            try:
                res = json.loads(tmp.read_text())
            except Exception:  # noqa: BLE001
                res = {}
        if status != "ok":
            res["error"] = status
            res.pop("done", None)
        elif not res.get("done"):
            tail = _stderr_tail(errlog)
            res.setdefault("error", f"crashed/killed{': ' + tail if tail else ''}")
        return res

    for module_name, class_name, err in classes:
        if class_name is None:
            rows.append({"module": module_name, "class": "(module)", "status": err})
            continue
        if only and class_name not in only:
            continue
        if class_name in skip:
            rows.append({"module": module_name, "class": class_name,
                         "status": "skipped (too large)"})
            persist()
            continue
        label = f"{module_name}.{class_name}"
        row = {"module": module_name, "class": class_name, "status": "ok",
               "mem_cold": {}, "mem_warm": {}}
        rows.append(row)

        def hoist(res):
            for k in ("n_vars", "n_params", "is_dpp"):
                if k not in row and k in res:
                    row[k] = res[k]

        if cold_on:
            for backend in BACKENDS:
                print(f"running {label} cold {backend} ...", flush=True)
                res = spawn("cold", backend)
                row["mem_cold"][backend] = res
                hoist(res)
                persist()
        if warm_on and row.get("n_params") != 0:
            for strat in STRATEGIES:
                print(f"running {label} warm {strat} ...", flush=True)
                res = spawn("warm", strat)
                row["mem_warm"][strat] = res
                hoist(res)
                persist()
                if res.get("skip") == "no parameters":
                    # parametricity was unknown (cold skipped/crashed): stop here
                    break
    if tmp.exists():
        tmp.unlink()

    report = render_memory(rows)
    print("\n" + report)
    print(f"\nSaved to {out} (raw JSON: {raw})")


def header_memory():
    import platform

    import cvxpy as cp
    return (
        "Upstream cvxpy/benchmarks suite: MEMORY of canonicalization backends\n"
        "=====================================================================\n"
        f"Source:  https://github.com/cvxpy/benchmarks (cloned, run as is)\n"
        f"cvxpy:   {cp.__version__}   python: {platform.python_version()}   "
        f"platform: {sys.platform}\n"
        f"timeout: {TIMEOUT:.0f}s/measurement   solver={COMPARE_SOLVER}   "
        f"warm iters={ITERS}\n"
        f"backends: {','.join(BACKENDS)}\n"
        f"strategies: {','.join(STRATEGIES)}\n"
        "\n"
        "Metric: peak-RSS high-water delta (MB) of ONE isolated subprocess per\n"
        "(problem, backend/strategy). gc.collect + ru_maxrss snapshot after setup\n"
        "(cold) / after the cache-populating warmup compile (warm), run the measured\n"
        "region, snapshot again; the delta is the memory attributable to the region.\n"
        "ru_maxrss is monotone: '*' marks regions that never exceeded the prior\n"
        "high-water mark -- the value is a floor, not a true peak (see retained_mb\n"
        "in the raw JSON).\n"
        "\n"
        "Table 1M = COLD first-extraction peak (MB) per canon backend. The last\n"
        "  column is a configurable ratio (BENCH_RATIO); <1 = numerator leaner.\n"
        "  It is suppressed when either side is a floor rather than a true peak.\n"
        f"Table 2M = ADDITIONAL peak (MB) during {ITERS} warm re-solves with fresh\n"
        "  parameter values, after the warmup compile (parametric problems only).\n"
        "Table 3M = the diff engine's self-reported allocation peak (MB), from its\n"
        "  sp_malloc counters (g_peak_bytes; engine-internal allocations only --\n"
        "  CVXPY front-end / scipy / numpy memory is invisible to it). 'rss' is the\n"
        "  matching process-level delta from Tables 1M/2M; the gap is glue memory\n"
        "  outside the engine. g_peak_bytes resets at each capsule construction, so\n"
        "  warm diffengine values are the per-re-solve capsule peak, warm de_cached\n"
        "  values the peak since the cached capsule was built.\n"
        "  Empty on upstream cvxpy, which has no diff-engine backend.\n"
        "\n"
        + TARGET_LEGEND
    )


def _mcell(d, width_fmt="{:.1f}"):
    if not isinstance(d, dict):
        return "-"
    if "error" in d:
        return "ERR"
    if d.get("skip"):
        return "-"
    v = d.get("delta_mb")
    if v is None:
        return "-"
    return width_fmt.format(v) + ("*" if d.get("floor") else "")


def _mratio(section, num, den):
    """delta_mb ratio, suppressed when either side is a floor rather than a
    true peak (ru_maxrss is monotone, so a 0 delta means 'never exceeded the
    prior high-water mark', which no ratio can honestly use)."""
    a, b = section.get(num) or {}, section.get(den) or {}
    da, db = a.get("delta_mb"), b.get("delta_mb")
    if (isinstance(da, (int, float)) and isinstance(db, (int, float)) and db > 0
            and not (a.get("floor") or b.get("floor"))):
        return f"{da / db:.2f}x"
    return "-"


def render_memory(rows):
    lines = []

    # ---- Table 1M: cold extraction peak per canon backend ----
    cols = _columns(rows, "mem_cold", BACKENDS)
    labels = [COL_LABEL.get(c, c) for c in cols]
    w = max(9, max((len(x) for x in labels), default=9))
    rnum, rden = _ratio_spec("BENCH_RATIO", "DIFFENGINE/CPP")
    rlabel = f"{COL_LABEL.get(rnum, rnum)}/{COL_LABEL.get(rden, rden)}"
    rw = max(7, len(rlabel))

    lines.append("Table 1M -- cold first-extraction peak RSS delta (MB) by canon backend")
    h1 = (f"{'benchmark':44} {'n_vars':>10} {'dpp?':>5} "
          + " ".join(f"{x:>{w}}" for x in labels)
          + f" {rlabel:>{rw}} {'status':>14}")
    lines.append(h1)
    lines.append("-" * len(h1))
    for r in rows:
        label = f"{r.get('module','?')}.{r.get('class','?')}"
        nv = r.get("n_vars", "-")
        is_dpp = r.get("is_dpp")
        dpp_s = "-" if is_dpp is None else ("yes" if is_dpp else "no")
        cold = r.get("mem_cold", {}) or {}
        nv_s = f"{nv:,}" if isinstance(nv, int) else str(nv)
        lines.append(
            f"{label:44} {nv_s:>10} {dpp_s:>5} "
            + " ".join(f"{_mcell(cold.get(c)):>{w}}" for c in cols)
            + f" {_mratio(cold, rnum, rden):>{rw}} {r.get('status','-'):>14}"
        )
    lines += _geomean_lines(rows, "mem_cold", rnum, rden,
                            lambda r: (lambda k: (((r.get("mem_cold", {}) or {}).get(k) or {})
                                                  .get("delta_mb"))), rlabel)

    # ---- Table 2M: additional peak during warm re-solves ----
    warm_rows = [r for r in rows
                 if any(isinstance(d, dict) and not d.get("skip")
                        for d in (r.get("mem_warm", {}) or {}).values())]
    scols = _columns(warm_rows, "mem_warm", STRATEGIES)
    slabels = [COL_LABEL.get(c, c) for c in scols]
    sw = max(10, max((len(x) for x in slabels), default=10))
    snum, sden = _ratio_spec("BENCH_RATIO_WARM", "de_cached/dpp")
    srlabel = f"{COL_LABEL.get(snum, snum)} vs {COL_LABEL.get(sden, sden)}"
    srw = max(11, len(srlabel))

    lines.append("")
    lines.append(f"Table 2M -- ADDITIONAL peak RSS delta (MB) during {ITERS} warm "
                 "re-solves (parametric problems only)")
    h2 = (f"{'benchmark':44} {'n_vars':>10} "
          + " ".join(f"{x:>{sw}}" for x in slabels)
          + f" {srlabel:>{srw}}")
    lines.append(h2)
    lines.append("-" * len(h2))
    for r in warm_rows:
        label = f"{r.get('module','?')}.{r.get('class','?')}"
        nv = r.get("n_vars", "-")
        warm = r.get("mem_warm", {}) or {}
        nv_s = f"{nv:,}" if isinstance(nv, int) else str(nv)
        lines.append(
            f"{label:44} {nv_s:>10} "
            + " ".join(f"{_mcell(warm.get(c)):>{sw}}" for c in scols)
            + f" {_mratio(warm, snum, sden):>{srw}}"
        )

    # ---- Table 3M: engine self-reported allocation peak vs process RSS ----
    def eng(d):
        v = (d or {}).get("engine_peak_mb")
        return f"{v:.1f}" if isinstance(v, (int, float)) else "-"

    eng_rows = []
    for r in rows:
        cells = {
            "cDE": (r.get("mem_cold", {}) or {}).get("DIFFENGINE"),
            "cDEDPP": (r.get("mem_cold", {}) or {}).get("DE_DPP"),
            "wDE": (r.get("mem_warm", {}) or {}).get("diffengine"),
            "wDEC": (r.get("mem_warm", {}) or {}).get("de_cached"),
        }
        if any(isinstance(d, dict) and d.get("engine_peak_mb") is not None
               for d in cells.values()):
            eng_rows.append((r, cells))
    if eng_rows:
        lines.append("")
        lines.append("Table 3M -- diff-engine self-reported allocation peak (MB) "
                     "vs process RSS delta")
        h3 = (f"{'benchmark':44} "
              f"{'cold-DE':>9} {'rss':>9} {'cold-DEDPP':>11} {'rss':>9} "
              f"{'warm-de':>9} {'rss':>9} {'warm-dec':>9} {'rss':>9}")
        lines.append(h3)
        lines.append("-" * len(h3))
        for r, cells in eng_rows:
            label = f"{r.get('module','?')}.{r.get('class','?')}"
            lines.append(
                f"{label:44} "
                f"{eng(cells['cDE']):>9} {_mcell(cells['cDE']):>9} "
                f"{eng(cells['cDEDPP']):>11} {_mcell(cells['cDEDPP']):>9} "
                f"{eng(cells['wDE']):>9} {_mcell(cells['wDE']):>9} "
                f"{eng(cells['wDEC']):>9} {_mcell(cells['wDEC']):>9}"
            )

    lines.append("")
    lines.append("* = region never exceeded the prior high-water mark; the value is a "
                 "floor, not a true peak (see retained_mb in the raw JSON)")

    # ---- Error detail ----
    errnotes = []
    for r in rows:
        for section, name in (("mem_cold", "cold"), ("mem_warm", "warm")):
            for key, d in (r.get(section, {}) or {}).items():
                if isinstance(d, dict) and "error" in d:
                    errnotes.append(f"  {name} {r.get('module')}.{r.get('class')} "
                                    f"[{key}]: {d['error']}")
    if errnotes:
        lines.append("\nErrors:")
        lines.extend(errnotes)
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--summarize":
        # --summarize <numerator.txt> [denominator.txt] [NUM/DEN]
        _args = sys.argv[2:]
        _spec = _args.pop() if _args and "/" in _args[-1] else "DIFFENGINE/CPP"
        _pa = _args[0]
        _pb = _args[1] if len(_args) > 1 else _pa
        _n, _, _d = _spec.partition("/")
        sys.exit(summarize(_pa, _pb, _n.strip(), _d.strip()))
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        if len(sys.argv) >= 7:
            run_memory_worker(sys.argv[2], sys.argv[3], sys.argv[4],
                              sys.argv[5], sys.argv[6])
        else:
            run_worker(sys.argv[2], sys.argv[3], sys.argv[4])
    elif os.environ.get("BENCH_MEMORY", "0") == "1":
        run_parent_memory()
    else:
        run_parent()
