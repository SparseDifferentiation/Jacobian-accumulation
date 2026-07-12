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
against the current cvxpy, focusing on the new ``ignore_dpp`` DIFFENGINE backend.

Clone the suite first (next to this script; gitignored):
    git clone --depth 1 https://github.com/cvxpy/benchmarks canonicalization/cvxpy_benchmarks

Then:
    python canonicalization/run_backend_benchmarks.py

What it does
------------
Every benchmark problem is run *as is* -- we import the upstream ASV classes
unmodified, call ``setup()``, and time the benchmark's own ``time_compile_problem``
(its declared solver / ConeMatrixStuffing.apply). That is the "baseline" column.

For problems that actually have ``cp.Parameter`` objects, we additionally run the
re-compilation comparison:
    dpp         - default get_problem_data (DPP-cached tensor when the problem is DPP).
    diffengine  - ignore_dpp=True => the C diff engine; parametric problems rebuild
                  the capsule each solve (uncached_param_prog keeps the chain uncached).
    de_cached   - canon_backend="DIFFENGINE" on the normal DPP path: the chain and the
                  DiffengineConeProgram (C problem) are cached, so re-compiles only
                  re-evaluate the expression tree at the new parameter values.
                  DPP problems only (non-DPP problems route to the diffengine strategy).
Each strategy re-compiles K times with fresh parameter values (CLARABEL, uniform, so
they are comparable) on a FRESH benchmark instance: the chain cache key is
(solver, gp, ignore_dpp, use_quad_obj) -- it excludes canon_backend and is_dpp -- so
sharing one Problem across strategies silently reuses the previous strategy's cached
chain / param_prog and times the wrong thing.

Robustness: several upstream problems are gigantic (1e6-1e7 var LPs, 5000^2 cone
stuffing, 6000x2400 QP) and the diff engine can segfault on some shapes, so each
benchmark runs in its own subprocess with a wall-clock timeout and streams partial
results to a JSON file. A hang / OOM / segfault in one problem is isolated and
reported, never aborting the run.

Env knobs: BENCH_TIMEOUT (s, default 150), BENCH_ITERS (default 3),
           BENCH_ONLY (comma-separated class names to restrict to).
"""
from __future__ import annotations

import json
import os
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

# Cold first-compile is timed under each canonicalization backend. CPP is the
# default; DIFFENGINE is the ignore_dpp path (selected via ignore_dpp=True);
# DE_DPP is explicit canon_backend="DIFFENGINE" on the normal DPP path (no
# fold reduction, capsule cached -- non-DPP problems are force-routed through
# the ignore_dpp path regardless, see the is_dpp column). CPP + the diffengine
# variants are timed first so the key comparison survives even if a giant
# problem times the worker out mid-run. Override with BENCH_BACKENDS.
BACKENDS = [b.strip() for b in os.environ.get(
    "BENCH_BACKENDS", "CPP,DIFFENGINE,DE_DPP,SCIPY,COO").split(",") if b.strip()]

# Re-compilation strategies (parametric problems only).
STRATEGIES = ["diffengine", "de_cached", "dpp"]


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

    kwargs = {"solver": COMPARE_SOLVER}
    if strategy == "diffengine":
        kwargs["ignore_dpp"] = True
    elif strategy == "de_cached":
        if not prob.is_dpp():
            raise ValueError("not DPP: de_cached == diffengine here")
        kwargs["canon_backend"] = "DIFFENGINE"
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
    kwargs = {"solver": COMPARE_SOLVER}
    if backend == "DIFFENGINE":
        kwargs["ignore_dpp"] = True  # diffengine via the ignore_dpp path
    elif backend == "DE_DPP":
        kwargs["canon_backend"] = "DIFFENGINE"  # explicit, normal DPP path
    elif backend != "CPP":
        kwargs["canon_backend"] = backend
    # CPP is the default backend: omit canon_backend so non-DPP parametric
    # problems take the default EvalParams route instead of raising the
    # explicit-backend ValueError (the baseline the comparison wants anyway).
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
            result["is_dpp"] = bool(meta_prob.is_dpp())
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
    out = HERE / os.environ.get("BENCH_OUT", "results_upstream_benchmarks.txt")

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
        status = "ok"
        try:
            subprocess.run(cmd, timeout=TIMEOUT, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            status = f"TIMEOUT(>{TIMEOUT:.0f}s)"
        res = {}
        if tmp.exists():
            try:
                res = json.loads(tmp.read_text())
            except Exception:  # noqa: BLE001
                res = {}
        if not res:
            res = {"module": module_name, "class": class_name}
        if not res.get("done"):
            res.setdefault("status", status if status != "ok" else "crashed/killed")
        else:
            res["status"] = "ok"
        rows.append(res)
        # Persist after every problem so an interrupted run keeps partial results.
        out.write_text(header() + "\n\n" + render(rows))
    if tmp.exists():
        tmp.unlink()

    report = render(rows)
    print("\n" + report)
    print(f"\nSaved to {out}")


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
        "\n"
        "Table 1 = COLD first-compile time (s) of each problem under each canon backend\n"
        "  (fresh setup + get_problem_data per backend). CPP is the default backend;\n"
        "  DIFFENG is the ignore_dpp=True path; DE_DPP is explicit canon_backend=DIFFENGINE\n"
        "  on the normal DPP path (no fold, capsule cached; non-DPP rows fall back to the\n"
        "  ignore_dpp path). 'd/CPP' = DIFFENG / CPP (<1 = diffengine faster).\n"
        "Table 2 = re-compile time (s) with changing parameters (parametric problems only),\n"
        f"  mean of {ITERS}, fresh instance per strategy: dpp=default (cached DPP tensor),\n"
        "  diffengine=ignore_dpp (capsule rebuilt\n"
        "  per solve), de_cached=canon_backend=DIFFENGINE on the DPP path (cached C problem,\n"
        "  re-evaluates the expression tree only)."
    )


def _cell(v):
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    if isinstance(v, str) and v.startswith("ERR"):
        return v.replace("ERR:", "ERR:")
    return "-"


def render(rows):
    lines = []

    # ---- Table 1: cold first-compile per canon backend ----
    lines.append("Table 1 -- cold first-compile (s) by canon backend")
    h1 = (f"{'benchmark':44} {'n_vars':>10} {'dpp?':>5} "
          f"{'CPP':>9} {'SCIPY':>9} {'COO':>9} {'DIFFENG':>9} {'DE_DPP':>9} "
          f"{'d/CPP':>7} {'status':>14}")
    lines.append(h1)
    lines.append("-" * len(h1))
    for r in rows:
        label = f"{r.get('module','?')}.{r.get('class','?')}"
        nv = r.get("n_vars", "-")
        is_dpp = r.get("is_dpp")
        dpp_s = "-" if is_dpp is None else ("yes" if is_dpp else "no")
        cold = r.get("cold", {}) or {}

        def cb(key):
            return _cell(cold.get(key, "-"))

        cpp, scipy, coo = cb("CPP"), cb("SCIPY"), cb("COO")
        de, de_dpp = cb("DIFFENGINE"), cb("DE_DPP")
        ratio = "-"
        cv_cpp, cv_de = cold.get("CPP"), cold.get("DIFFENGINE")
        if isinstance(cv_cpp, (int, float)) and isinstance(cv_de, (int, float)) and cv_cpp > 0:
            ratio = f"{cv_de / cv_cpp:.2f}x"
        nv_s = f"{nv:,}" if isinstance(nv, int) else str(nv)
        lines.append(
            f"{label:44} {nv_s:>10} {dpp_s:>5} "
            f"{cpp:>9} {scipy:>9} {coo:>9} {de:>9} {de_dpp:>9} "
            f"{ratio:>7} {r.get('status','-'):>14}"
        )

    # ---- Table 2: parametric re-compile comparison ----
    param_rows = [r for r in rows if r.get("comparison")]
    lines.append("")
    lines.append("Table 2 -- re-compile (s) with changing parameters (parametric problems only)")
    h2 = (f"{'benchmark':44} {'n_vars':>10} {'dpp':>10} {'eval_par':>10} "
          f"{'diffeng':>10} {'de_cached':>10} {'dec vs dpp':>11}")
    lines.append(h2)
    lines.append("-" * len(h2))
    for r in param_rows:
        label = f"{r.get('module','?')}.{r.get('class','?')}"
        nv = r.get("n_vars", "-")
        comp = r.get("comparison", {})

        def cv(key):
            c = comp.get(key, {})
            if "mean_s" in c:
                return f"{c['mean_s']:.4f}"
            if "error" in c:
                return "ERR"
            return "-"

        def ratio(num, den):
            cn, cd = comp.get(num, {}), comp.get(den, {})
            if "mean_s" in cn and "mean_s" in cd and cd["mean_s"] > 0:
                return f"{cn['mean_s'] / cd['mean_s']:.2f}x"
            return "-"

        nv_s = f"{nv:,}" if isinstance(nv, int) else str(nv)
        lines.append(
            f"{label:44} {nv_s:>10} {cv('dpp'):>10} "
            f"{cv('diffengine'):>10} {cv('de_cached'):>10} "
            f"{ratio('de_cached','dpp'):>11}"
        )

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


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        run_worker(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        run_parent()
