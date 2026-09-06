"""Four thesis figures from the committed benchmark results. No new runs.

1. quadrant_time_memory      One point per problem: x = engine/CPP_ND cold time,
                             y = engine/CPP cold memory (delta max-RSS), log-log
                             quadrants. The one-figure summary of the benchmark.
2. cold_vs_warm_parametric   The 6 fully-replicated parametric problems: cold
                             compile vs warm re-solve for the engine, the DPP
                             tensor path, and the upstream no-DPP rebuild, with
                             min/max error bars over the 3 replicate runs.
3. dense_block_scaling       Dense-tall (2n x n) vs sparse (10 nnz/row) cost of
                             the first Jacobian as n grows, for CasADi, Julia
                             SCT, and ASL: generic sparse-AD accumulation cost
                             tracks the color count (colors = n on dense
                             columns); a-priori-structure ASL stays flat.
4. performance_profile       Dolan-More profile of cold canonicalization over
                             the 25 problems: DIFFENGINE vs CPP_ND / SCIPY_ND /
                             COO_ND. Crashed/missing cells count as failures.

Sources (all committed under results/): sparsity_correlations.json,
repl/{fork,upstream}_param_rep{1,2,3}.json, results_probes_20260716.txt,
results_probe_dense_block_{julia,asl}_20260718.json,
results_backends_all_20260724_repl.txt,
results_backends_upstream192_nodpp_20260808_merged.txt,
results_backends_fork_bigparam_20260808.json.

Usage: .venv/bin/python canonicalization/plot_thesis_figures.py
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from plot_sparsity_correlations import (  # noqa: E402
    AXIS, BLUE, GRID, INK, INK2, MUTED,
    ENGINE_TABLE, ND_TABLE, BIGPARAM_JSON,
    _place_labels, _short, _style,
)
from run_backend_benchmarks import _parse_table  # noqa: E402

RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"
CORR_JSON = RESULTS / "sparsity_correlations.json"
REPL_DIR = RESULTS / "repl"
PROBES_TXT = RESULTS / "results_probes_20260716.txt"
JULIA_PROBE = RESULTS / "results_probe_dense_block_julia_20260718.json"
ASL_PROBE = RESULTS / "results_probe_dense_block_asl_20260718.json"
BIGPARAM_TXT = RESULTS / "results_backends_upstream192_nodpp_20260808_bigparam.txt"
ND_BIGPARAM_TXT = RESULTS / "results_backends_upstream192_nd_bigparam_20260811.txt"
MEM_ENGINE_JSON = RESULTS / "results_backends_memory_20260802.json"
MEM_ND_JSON = RESULTS / "results_backends_memory_nd_20260811.json"
# ConvexPlasticity retime (2026-08-19) on the build that converts the mostly-zero
# dense quad-form constant to sparse form (fork ce554d88 + sparsediffpy 0.7.0):
# six cold runs, median 0.91 s. Thesis Table 4.1 carries this value with a
# footnote; the time profile uses the same override so table and figure agree.
CP_RETIME_DIR = RESULTS / "retime_cp_quadform_20260819"

ORANGE, AQUA, VIOLET = "#eb6834", "#1baf7a", "#4a3aa7"

# Profile series: color + dash follow the backend everywhere. The four hues are
# reference-palette categorical slots validated together under --pairs all
# (step lines cross, so every pair is adjacent); slot-4 yellow fails the
# normal-vision floor against orange there, hence violet for COO. The dash
# pattern is the identity channel for grayscale print and full-severity CVD.
PROFILE_STYLE = {
    "diffengine": (BLUE, "solid"),
    "CPP": (ORANGE, (0, (5, 2))),
    "SCIPY": (AQUA, (0, (3, 1.5, 1, 1.5))),
    "COO": (VIOLET, (0, (1.2, 1.4))),
}

# --thesis mode: bare figures (captions carry titles/caveats), CM-style serif,
# generated at the document's text width so LaTeX includes them 1:1 and the
# point sizes stay true. Put \the\textwidth in the document to confirm the
# width (pt / 72.27 = inches) and adjust TEXTWIDTH_IN if it differs.
THESIS = False
TEXTWIDTH_IN = 6.53  # wzz-thesis: \textwidth = 16.59 cm (MemoireThese.sty)


def _figsize(normal, thesis):
    return thesis if THESIS else normal


def _save(fig, stem, written):
    import matplotlib.pyplot as plt
    if THESIS:
        # No bbox_inches="tight": it would change the physical width and LaTeX
        # would rescale; tight_layout fits everything inside the fixed canvas.
        fig.tight_layout()
        fig.savefig(PLOTS / f"{stem}_thesis.pdf")
        fig.savefig(PLOTS / f"{stem}_thesis.png", dpi=200)  # preview only
        written.append(f"{stem}_thesis")
    else:
        for ext in ("png", "pdf"):
            fig.savefig(PLOTS / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
        written.append(stem)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 1. Time x memory quadrant
# --------------------------------------------------------------------------- #
def fig_quadrant(plt, written):
    records = json.loads(CORR_JSON.read_text())["records"]
    pts = [(r["ratio_cpp"], r["mem_ratio_cpp"], _short(r["name"])) for r in records
           if r["ratio_cpp"] and r["mem_ratio_cpp"]]
    dropped = [r["name"] for r in records if not (r["ratio_cpp"] and r["mem_ratio_cpp"])]

    fig, ax = plt.subplots(figsize=_figsize((7.6, 6.4), (TEXTWIDTH_IN, 4.9)))
    _style(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(1.0, color=AXIS, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.axhline(1.0, color=AXIS, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    xs, ys, _ = zip(*pts)
    ax.scatter(xs, ys, s=38, color=BLUE, edgecolors="white", linewidths=0.6, zorder=3)
    _place_labels(ax, pts, fontsize=6.5)
    # Quadrant labels in the corners, in axes coordinates.
    for xa, ya, ha, va, txt in [
            (0.02, 0.97, "left", "top", "engine faster, heavier"),
            (0.98, 0.90, "right", "top", "engine slower, heavier"),
            (0.02, 0.03, "left", "bottom", "engine faster, lighter"),
            (0.98, 0.03, "right", "bottom", "engine slower, lighter")]:
        ax.text(xa, ya, txt, transform=ax.transAxes, ha=ha, va=va,
                fontsize=7.5, color=MUTED, style="italic")
    ax.set_xlabel("cold canonicalization time, engine / CPP_ND", fontsize=9, color=INK2)
    ax.set_ylabel("cold memory (delta max-RSS), engine / CPP", fontsize=9, color=INK2)
    if not THESIS:
        ax.set_title("Engine vs cvxcore, per problem: time against memory\n"
                     f"n = {len(pts)}; time from 2026-07-24/08-08 runs (~±20% floor); "
                     "memory baseline is the DPP path", fontsize=10, color=INK, loc="left")
    _save(fig, "quadrant_time_memory", written)
    print(f"quadrant_time_memory: {len(pts)} points; dropped (no time ratio): {dropped}")


# --------------------------------------------------------------------------- #
# 2. Cold vs warm, parametric problems
# --------------------------------------------------------------------------- #
def _repl_series(side, cold_key, warm_key):
    """{class: (cold_means[3], warm_means[3])} across the 3 replicate files."""
    out = {}
    for rep in (1, 2, 3):
        data = json.loads((REPL_DIR / f"{side}_param_rep{rep}.json").read_text())
        for row in data:
            cold = row.get("cold", {}).get(cold_key)
            warm = (row.get("comparison", {}).get(warm_key) or {}).get("mean_s")
            if isinstance(cold, (int, float)) and isinstance(warm, (int, float)):
                out.setdefault(row["class"], ([], []))
                out[row["class"]][0].append(cold)
                out[row["class"]][1].append(warm)
    return out


def fig_cold_warm(plt, written):
    series = [
        ("engine (fork, ignore_dpp)", BLUE, _repl_series("fork", "DIFFENGINE", "diffengine")),
        ("DPP tensor (upstream CPP)", ORANGE, _repl_series("upstream", "CPP", "dpp")),
        ("no-DPP rebuild (upstream CPP_ND)", AQUA, _repl_series("upstream", "CPP_ND", "nodpp_cpp")),
    ]
    problems = sorted(set().union(*(s[2] for s in series)))

    fig, ax = plt.subplots(figsize=_figsize((7.6, 6.4), (TEXTWIDTH_IN, 4.9)))
    _style(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    label_pts = []
    for cls in problems:
        # Connector through this problem's points across the three strategies.
        chain = [(np.mean(s[2][cls][0]), np.mean(s[2][cls][1]))
                 for s in series if cls in s[2]]
        ax.plot(*zip(*chain), color=GRID, linewidth=0.8, zorder=2)
        eng = series[0][2].get(cls)
        if eng:
            label_pts.append((np.mean(eng[0]), np.mean(eng[1]), _short(cls)))
    for name, color, data in series:
        cx = [np.mean(v[0]) for v in data.values()]
        cy = [np.mean(v[1]) for v in data.values()]
        xerr = np.array([[np.mean(v[0]) - min(v[0]) for v in data.values()],
                         [max(v[0]) - np.mean(v[0]) for v in data.values()]])
        yerr = np.array([[np.mean(v[1]) - min(v[1]) for v in data.values()],
                         [max(v[1]) - np.mean(v[1]) for v in data.values()]])
        ax.errorbar(cx, cy, xerr=xerr, yerr=yerr, fmt="o", ms=6, color=color,
                    ecolor=color, elinewidth=0.9, capsize=2, mec="white",
                    mew=0.6, label=name, zorder=3)
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=0.9,
            linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.97, 0.90, "warm = cold\n(no re-use benefit)", transform=ax.transAxes,
            ha="right", fontsize=7.5, color=MUTED, style="italic")
    _place_labels(ax, label_pts, fontsize=6.5)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK2)
    ax.set_xlabel("cold first compile (s)", fontsize=9, color=INK2)
    ax.set_ylabel("warm re-solve, new parameter values (s)", fontsize=9, color=INK2)
    if not THESIS:
        ax.set_title("Cold compile vs warm re-solve on the parametric problems\n"
                     "mean over 3 replicate runs, error bars min/max; labels at the "
                     "engine's point", fontsize=10, color=INK, loc="left")
    _save(fig, "cold_vs_warm_parametric", written)
    print(f"cold_vs_warm_parametric: {len(problems)} problems x {len(series)} strategies "
          "(SimpleFullyParametrizedLP/ParametrizedQP excluded: no upstream replicates)")


# --------------------------------------------------------------------------- #
# 3. Dense-block scaling across three stacks
# --------------------------------------------------------------------------- #
def _casadi_sweep():
    """{(regime, n): (total_s, fwd_colors)} from the coloring-sweeps transcript."""
    rx = re.compile(
        r"(dense|sparse) (\d+)x(\d+)\s+nnz\s+\d+\s+pattern\s+([\d.]+)s\s+"
        r"colors fwd/adj\s+(\d+)/\s*\d+\s+jac\(sym\)\s+([\d.]+)s\s+"
        r"ctor\+eval\s+([\d.]+)s")
    out = {}
    for m in rx.finditer(PROBES_TXT.read_text()):
        regime, mm, nn = m.group(1), int(m.group(2)), int(m.group(3))
        if mm != 2 * nn:  # only the tall-block sweep (skip wide/tridiag rows)
            continue
        total = float(m.group(4)) + float(m.group(6)) + float(m.group(7))
        out[(regime, nn)] = (total, int(m.group(5)))
    return out


def _probe_rows(path):
    d = json.loads(path.read_text())
    res = d["results"]
    if isinstance(res, dict) and "results" in res:
        res = res["results"]
    return res


def _julia_sweep():
    out = {}
    for r in _probe_rows(JULIA_PROBE):
        m = re.match(r"(dense|sparse) (\d+)x(\d+)", r["label"])
        if not m or int(m.group(2)) != 2 * int(m.group(3)):
            continue
        # t_first_eval_s carries Julia JIT compilation charged to whichever row
        # first hits a code path (sparse 500x250 shows 0.93 s, 4000x2000 shows
        # 0.008 s) -- use the post-JIT warm eval instead, so the curve reflects
        # the algorithm, not compilation order.
        total = (r["t_pattern_s"] + r["t_coloring_s"] + r["t_prep_s"]
                 + r["t_warm_eval_s"])
        out[(m.group(1), int(m.group(3)))] = (total, r.get("ncolors_fwd"))
    return out


def _asl_sweep():
    out = {}
    for r in _probe_rows(ASL_PROBE):
        m = re.match(r"(dense|sparse) (\d+)x(\d+)", r["label"])
        if not m or int(m.group(2)) != 2 * int(m.group(3)):
            continue
        out[(m.group(1), int(m.group(3)))] = (r["t_asl_read_s"] + r["t_first_jac_s"], None)
    return out


def fig_scaling(plt, written):
    stacks = [
        ("CasADi", " (MX, uni_coloring)\n(pattern + jacobian + ctor/eval)",
         _casadi_sweep()),
        ("Julia SCT", " (SparseConnectivityTracer\n+ SparseMatrixColorings; post-JIT eval)",
         _julia_sweep()),
        ("ASL", " (a-priori structure)\n(.nl read + first Jacobian)",
         _asl_sweep()),
    ]
    fig, axes = plt.subplots(1, 3, figsize=_figsize((12.6, 4.4), (TEXTWIDTH_IN, 2.6)),
                             sharey=True)
    for ax, (name, detail, data) in zip(axes, stacks):
        _style(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        for regime, color in (("dense", BLUE), ("sparse", AQUA)):
            ns = sorted(n for r, n in data if r == regime)
            ys = [data[(regime, n)][0] for n in ns]
            ax.plot(ns, ys, "o-", color=color, ms=5 if not THESIS else 3.5,
                    linewidth=1.6, mec="white", mew=0.5,
                    label=r"dense $2n{\times}n$" if regime == "dense"
                    else "sparse, 10 nnz/row")
            # The mechanism label: forward color count. Thesis layout is tight,
            # so label the mechanism mid-line instead of the largest point.
            colors = data[(regime, ns[-1])][1]
            if colors is None:
                pass
            elif THESIS:
                ax.annotate("colors $= n$" if regime == "dense"
                            else f"≈{colors} colors",
                            (ns[-2], ys[-2]), textcoords="offset points",
                            xytext=(0, 7 if regime == "dense" else -12),
                            ha="center", fontsize=6.5, color=INK2)
            else:
                ax.annotate(f"{colors} colors" if regime == "dense"
                            else f"≈{colors} colors",
                            (ns[-1], ys[-1]), textcoords="offset points",
                            xytext=(-2, 8 if regime == "dense" else -13),
                            ha="right", fontsize=7, color=INK2)
        from matplotlib.ticker import NullFormatter, NullLocator
        ax.set_xticks([250, 500, 1000, 2000])
        ax.set_xticklabels(["250", "500", "1k", "2k"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("block width n", fontsize=9, color=INK2)
        ax.set_title(name if THESIS else name + detail,
                     fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("time to first Jacobian (s)", fontsize=9, color=INK2)
    axes[0].legend(loc="upper left", fontsize=7.5 if not THESIS else 6.5,
                   frameon=False, labelcolor=INK2)
    if not THESIS:
        fig.suptitle("Dense tall blocks defeat coloring (colors = n, so sweeps scale "
                     "with n); a-priori structure pays only O(nnz)", fontsize=11.5,
                     color=INK, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "dense_block_scaling", written)
    n_pts = sum(len(s[1]) for s in stacks)
    print(f"dense_block_scaling: {n_pts} points across 3 stacks x 2 regimes")


# --------------------------------------------------------------------------- #
# 4. Performance profile (Dolan-More)
# --------------------------------------------------------------------------- #
BRIDGE_TABLE = RESULTS / "results_backends_fork_bridge_20260808.txt"
# The five parametric problems whose Table 4.1 / 4.2 cells are the fastest of the
# three replicate runs in results/repl (thesis protocol, 2026-08-22).
_REPL_PROBLEMS = ("ParamSmallMatrixStuffing", "ParamConeMatrixStuffing",
                  "SimpleScalarParametrizedLP", "SVMWithL1Regularization",
                  "FactorCovarianceModel")
_REPL_CACHE = {}


def _short_cls(cls):
    return cls.replace("Benchmark", "").replace("Bench", "")


def _repl_min(cls, side, key):
    """Fastest of the three replicate runs of `cls` for cold target `key`."""
    if not _REPL_CACHE:
        for s_ in ("fork", "upstream"):
            for rep in (1, 2, 3):
                for row in json.loads((REPL_DIR / f"{s_}_param_rep{rep}.json").read_text()):
                    for k, v in row.get("cold", {}).items():
                        if isinstance(v, (int, float)):
                            _REPL_CACHE.setdefault((_short_cls(row["class"]), s_, k), []).append(v)
    vals = _REPL_CACHE.get((_short_cls(cls), side, key))
    return float(min(vals)) if vals else None


def _thesis_table_cells(cls, cell):
    """Overwrite the one-shot cells of `cls` with the ones printed in thesis
    Table 4.1, so fig:profiles(a) and the table are built from the same numbers:
    17 parameter-free rows = the single 2026-08-08 bridge run (fork build, all
    four backends); the five replicated parametric rows and ConvexPlasticity's
    CVXPY columns = fastest of three replicate runs; ConvexPlasticity diffengine
    = fastest of the six later-build retimes (set by the caller); the two
    big-parameter rows keep the single-measurement patches."""
    short = _short_cls(cls)
    if short in _REPL_PROBLEMS:
        cell["CPP"] = _repl_min(cls, "upstream", "CPP_ND")
        cell["SCIPY"] = _repl_min(cls, "upstream", "SCIPY_ND")
        cell["COO"] = _repl_min(cls, "upstream", "COO_ND")
        cell["diffengine"] = _repl_min(cls, "fork", "DIFFENGINE")
        return
    if short == "ConvexPlasticity":
        cell["CPP"] = _repl_min(cls, "upstream", "CPP_ND")
        cell["SCIPY"] = _repl_min(cls, "upstream", "SCIPY_ND")
        cell["COO"] = _repl_min(cls, "upstream", "COO_ND")
        return
    if short in ("SimpleFullyParametrizedLP", "ParametrizedQP"):
        # Single measurements (dagger rows): CPP from the 2026-08-11 ND run,
        # SCIPY/COO from the 2026-08-08 big-parameter run, diffengine from the
        # fork big-parameter run -- not the 07-24 engine table, whose
        # SimpleFullyParametrizedLP cell (1.35 s) was a transient.
        _, nd11 = _parse_table(ND_BIGPARAM_TXT)
        _, nd08 = _parse_table(BIGPARAM_TXT)
        big = {r["class"]: r.get("cold") or {}
               for r in json.loads(BIGPARAM_JSON.read_text())}
        cell["CPP"] = nd11.get(cls, {}).get("CPP_ND")
        cell["SCIPY"] = nd08.get(cls, {}).get("SCIPY_ND")
        cell["COO"] = nd08.get(cls, {}).get("COO_ND")
        cell["diffengine"] = big.get(cls, {}).get("DIFFENGINE")
        return
    canon_b, br = _parse_table(BRIDGE_TABLE)
    row = br.get(cls, {})
    for lab, key in (("CPP", "CPP"), ("SCIPY", "SCIPY"), ("COO", "COO"),
                     ("diffengine", canon_b.get("DIFFENGINE", "DIFFENG"))):
        v = row.get(key)
        if isinstance(v, (int, float)):
            cell[lab] = v


def _profile_times():
    """Two panels of cold times, {class: {series_label: seconds}} each.

    One-shot: engine (ignore_dpp) vs upstream CPP_ND/SCIPY_ND/COO_ND -- all
    emit a concrete (P, c, A, b). Parametric: engine DE_DPP vs upstream
    CPP/SCIPY/COO on the DPP path -- the first compile paid before warm
    re-solves (the artifacts differ: parameter tensor vs engine tree; both
    are that compile). On the 17 parameter-free problems the two jobs
    coincide.
    """
    canon_e, eng = _parse_table(ENGINE_TABLE)
    de_label = canon_e.get("DIFFENGINE", "DIFFENG")
    _, nd = _parse_table(ND_TABLE)
    # The two >=1e6-parameter problems are all-dash in the merged table (the
    # main sweep's worker died at pinned CPP -- the DPP tensor build -- before
    # trying the others). Two patch runs fill them in: the 08-08 bigparam run
    # (COO_ND/SCIPY_ND/COO) and the 08-11 ND run (adds CPP_ND, which turns out
    # to COMPLETE: the old "CPP_ND crashes" was an inference from cells that
    # were never attempted; only DPP-path CPP/SCIPY truly fail here).
    for patch_txt in (BIGPARAM_TXT, ND_BIGPARAM_TXT):
        _, big = _parse_table(patch_txt)
        for cls, row in big.items():
            for key, v in row.items():
                nd.setdefault(cls, {}).setdefault(key, v)
    patch = {r["class"]: r.get("cold") or {}
             for r in json.loads(BIGPARAM_JSON.read_text())}
    problems = sorted(set(eng) | set(nd) | set(patch))

    def col(table, key, cls, patch_key=None):
        v = table.get(cls, {}).get(key)
        if not isinstance(v, (int, float)) and patch_key:
            v = patch.get(cls, {}).get(patch_key)
        return v if isinstance(v, (int, float)) else None

    oneshot, dpp = {}, {}
    for cls in problems:
        oneshot[cls] = {
            # Labels match the thesis tables: diffengine vs the stock backends,
            # all on the ignore_dpp (ND) path, as the caption states.
            "diffengine": col(eng, de_label, cls, "DIFFENGINE"),
            "CPP": col(nd, "CPP_ND", cls),
            "SCIPY": col(nd, "SCIPY_ND", cls),
            "COO": col(nd, "COO_ND", cls),
        }
        if cls == "ConvexPlasticity" and CP_RETIME_DIR.is_dir():
            # Later-build retime replaces the padded-P pinned-build time (2.07 s),
            # matching the footnoted row of the thesis table.
            vals = [r["cold"]["DIFFENGINE"]
                    for f in sorted(CP_RETIME_DIR.glob("*.json"))
                    for r in json.loads(f.read_text())
                    if r["class"] == "ConvexPlasticity"
                    and isinstance((r.get("cold") or {}).get("DIFFENGINE"), (int, float))]
            if vals:
                oneshot[cls]["diffengine"] = float(np.min(vals))
        _thesis_table_cells(cls, oneshot[cls])
        dpp[cls] = {
            "engine": col(eng, "DE_DPP", cls, "DE_DPP"),
            "CPP": col(nd, "CPP", cls),
            "SCIPY": col(nd, "SCIPY", cls),
            "COO": col(nd, "COO", cls),
        }
    return problems, oneshot, dpp


def _oneshot_ratios():
    """Per-backend ratio-to-best on the one-shot job, {label: [ratio]}.
    Missing cells count as failures (ratio inf)."""
    problems, oneshot, _dpp = _profile_times()
    series = list(next(iter(oneshot.values())))
    ratios = {s: [] for s in series}
    for cls in problems:
        avail = [v for v in oneshot[cls].values() if v]
        best = min(avail) if avail else None
        for s in series:
            v = oneshot[cls].get(s)
            ratios[s].append(v / best if v and best else math.inf)
    return len(problems), series, ratios


def fig_profile(plt, written):
    """One-shot extraction only (ignore_dpp): all backends emit a concrete
    (P, c, A, b), so all 25 problems are the same job. The DPP-path variant
    (engine DE_DPP vs upstream CPP/SCIPY/COO, meaningful on the 8 parametric
    problems) was tried and dropped -- _profile_times still returns it."""
    nprob, series, ratios = _oneshot_ratios()
    tick_taus = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    taus = np.geomspace(1, 512, 600)

    fig, ax = plt.subplots(figsize=_figsize((7.4, 5.2), (TEXTWIDTH_IN, 4.0)))
    _style(ax)
    ax.set_xscale("log", base=2)
    for s in series:
        r = np.array(ratios[s])
        rho = [(r <= tau).mean() for tau in taus]
        # All series share one weight (no self-highlight, Clarabel-paper style).
        # All four curves reach 1.0, so end-labels would stack -- the legend
        # (color + dash, ink text) is the identity channel instead.
        color, ls = PROFILE_STYLE[s]
        ax.step(taus, rho, where="post", color=color, linewidth=1.5,
                linestyle=ls, zorder=3, label=s)
        print(f"performance_profile {s:10} rho(1)={rho[0]:.2f} "
              f"plateau={rho[-1]:.2f}")
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK2,
              handlelength=2.8)
    ax.set_xlim(1, 512)
    ax.set_ylim(0, 1.03)
    ax.set_xticks(tick_taus)
    ax.set_xticklabels([rf"${t}{{\times}}$" for t in tick_taus], fontsize=8)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(r"within factor $\tau$ of the best backend", fontsize=9, color=INK2)
    ax.set_ylabel(f"share of the {nprob} problems", fontsize=9, color=INK2)
    if not THESIS:
        ax.set_title("Performance profile, cold one-shot canonicalization "
                     f"({nprob} problems)\nignore_dpp: every backend emits a "
                     "concrete (P, c, A, b); complete data, all backends",
                     fontsize=10, color=INK, loc="left")
        fig.text(0.02, -0.05,
                 "Engine column from the 2026-07-24 fork sweep; stock columns from the "
                 "2026-08-08 upstream-1.9.2 sweeps plus the 2026-08-11 bigparam ND "
                 "patch.\nCross-date ~±20% noise floor affects the small-τ region only. "
                 "All four backends complete all 25 problems on this path.",
                 fontsize=7.5, color=MUTED, linespacing=1.5)
        fig.tight_layout()
    _save(fig, "performance_profile", written)


CAPTIONS = {
    "performance_profile":
        r"Dolan--Mor\'e performance profile of canonicalization time "
        r"(\texttt{ignore\_dpp}: every backend emits a concrete $(P, c, A, b)$) over "
        r"the 25 problems of the CVXPY benchmark suite. All four backends complete "
        r"all 25 problems on this path. Engine times from the 2026-07-24 sweep, stock "
        r"times from the 2026-08-08 upstream-1.9.2 sweeps and the 2026-08-11 "
        r"big-parameter patch run; the ${\sim}{\pm}20\%$ cross-date noise floor "
        r"affects only the small-$\tau$ region.",
    "memory_profile":
        r"Dolan--Mor\'e profile of peak memory during canonicalization "
        r"($\Delta$max-RSS, every backend on the \texttt{ignore\_dpp} job) over all "
        r"25 problems. On ConvexPlasticity the CVXPY backends' growth sits below the "
        r"measurement floor, as is the engine's on the later build used for its time "
        r"cell, so the four are scored as tied there.",
    "profiles_combined":
        r"Dolan--Mor\'e profiles over the 25 problems, every backend on the "
        r"\texttt{ignore\_dpp} path: canonicalization time (top) and peak memory "
        r"($\Delta$max-RSS, bottom). On ConvexPlasticity the CVXPY backends' memory "
        r"growth sits below the measurement floor; the three are scored as tied at "
        r"the floor, as is the engine's on the later build used for its time cell, "
        r"so the four are scored as tied there.",
    "quadrant_time_memory":
        r"Cold canonicalization, engine relative to cvxcore, per problem ($n = 23$): "
        r"time ratio (engine\,/\,CPP\_ND, like-for-like \texttt{ignore\_dpp} path) "
        r"against memory ratio ($\Delta$\,max-RSS, engine\,/\,CPP; the memory "
        r"baseline is the DPP path). Time ratios carry a ${\sim}{\pm}20\%$ cross-date "
        r"noise floor.",
    "cold_vs_warm_parametric":
        r"Cold first compile against warm re-solve (fresh parameter values) on the "
        r"six fully replicated parametric problems: the diff engine "
        r"(\texttt{ignore\_dpp}), the DPP parameter-tensor path (upstream CPP), and "
        r"the upstream no-DPP rebuild (CPP\_ND). Points are means over three "
        r"replicate runs; error bars span min/max and are often smaller than the "
        r"markers. Labels sit at the engine's point.",
    "dense_block_scaling":
        r"Cost of the first Jacobian on a dense tall $2n \times n$ block versus a "
        r"same-shape sparse pattern (10 nonzeros per row), as $n$ grows. On the dense "
        r"block no coloring can compress the Jacobian (forward colors $= n$), so the "
        r"accumulation cost of the generic sparse-AD stacks (CasADi; Julia "
        r"SparseConnectivityTracer + SparseMatrixColorings, post-JIT evaluation) "
        r"grows with an extra factor of $n$, while ASL, which knows the structure a "
        r"priori, pays only $O(\mathrm{nnz})$. The sparse pattern colors to "
        r"${\approx}54$ and stays flat in every stack.",
}


def _write_tex(stems):
    lines = [
        "% Auto-generated by plot_thesis_figures.py --thesis. Ready to \\input or paste.",
        f"% Figures are {TEXTWIDTH_IN} in wide. Confirm your document's \\the\\textwidth",
        "% (pt / 72.27 = in); if it differs, change TEXTWIDTH_IN and regenerate, and",
        "% keep \\includegraphics WITHOUT width= so LaTeX never rescales the fonts.",
        "",
    ]
    for stem, caption in ((s, CAPTIONS[s]) for s in stems):
        lines += [
            r"\begin{figure}[tbp]",
            r"  \centering",
            rf"  \includegraphics{{{stem}_thesis.pdf}}",
            rf"  \caption{{{caption}}}",
            rf"  \label{{fig:{stem}}}",
            r"\end{figure}",
            "",
        ]
    out = PLOTS / "thesis_figures.tex"
    out.write_text("\n".join(lines))
    return out


# --------------------------------------------------------------------------- #
# 5. Memory profile (like-for-like ignore_dpp)
# --------------------------------------------------------------------------- #
def _memory_ratios():
    """Per-backend ratio-to-best on cold delta max-RSS, {label: [ratio]}.
    ConvexPlasticity: on the later engine build used for its Table 4.1 time
    cell the engine grows 1.4 MB (results/retime_cp_quadform_20260819/
    mem_cp_devbuild.json) against 3.4-4.1 MB for the CVXPY backends, all below
    the resolution of the measurement, so the four are scored as tied (ratio 1).
    Before 2026-08-22 the pinned build's 1.2 GB was censored as inf."""
    eng = {r["class"]: r["mem_cold"]["DIFFENGINE"]["delta_mb"]
           for r in json.loads(MEM_ENGINE_JSON.read_text())}
    nd = {r["class"]: r["mem_cold"] for r in json.loads(MEM_ND_JSON.read_text())}
    floor = {"ConvexPlasticity"}
    vals = {c: {"diffengine": eng[c],
                "CPP": nd[c]["CPP_ND"]["delta_mb"],
                "SCIPY": nd[c]["SCIPY_ND"]["delta_mb"],
                "COO": nd[c]["COO_ND"]["delta_mb"]}
            for c in eng}
    series = ["diffengine", "CPP", "SCIPY", "COO"]
    ratios = {lbl: [] for lbl in series}
    for cls, row in vals.items():
        if cls in floor:
            for lbl in ratios:
                ratios[lbl].append(1.0)
            continue
        best = min(row.values())
        for lbl, v in row.items():
            ratios[lbl].append(v / best)
    return len(vals), series, ratios


def fig_memory_profile(plt, written):
    """Dolan-More profile over cold delta max-RSS, all backends on the
    ignore_dpp job: engine from the 2026-08-02 fork memory run, stock backends
    from the 2026-08-11 upstream ND memory run (cross-run drift on same-job
    problems is sub-MB, so mixing the two runs is sound). On ConvexPlasticity
    the stock cells are measurement floors -- extraction never exceeded
    setup's max-RSS high-water mark (retained_mb is negative; the 3-4 MB
    deltas are transient noise) -- so no fair denominator exists there. The
    problem stays in with the three stock backends scored as a tie at the
    floor (ratio 1); since 2026-08-22 the engine is tied there too (1.4 MB on the
    later build); before that it was censored (ratio inf: any finite ratio
    would have been an artifact of the floor chosen) and its curve plateaued
    at 24/25."""
    n, series_lbls, ratios = _memory_ratios()
    series = [(lbl, PROFILE_STYLE[lbl]) for lbl in series_lbls]
    taus = np.geomspace(1, 64, 600)
    tick_taus = [1, 2, 4, 8, 16, 32, 64]
    fig, ax = plt.subplots(figsize=_figsize((7.4, 5.2), (TEXTWIDTH_IN, 4.0)))
    _style(ax)
    ax.set_xscale("log", base=2)
    for lbl, (color, ls) in series:
        r = np.array(ratios[lbl])
        rho = [(r <= t).mean() for t in taus]
        # All series share one weight (no self-highlight, Clarabel-paper style).
        # Three curves converge at 1.0, so end-labels would stack -- the legend
        # (color + dash, ink text) is the identity channel instead.
        ax.step(taus, rho, where="post", color=color, linewidth=1.5,
                linestyle=ls, zorder=3, label=lbl)
        print(f"memory_profile {lbl:8} rho(1)={rho[0]:.2f} "
              f"rho(2)={float(np.interp(2, taus, rho)):.2f} max={r.max():.1f}x")
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK2,
              handlelength=2.8)
    ax.set_xlim(1, 64)
    ax.set_ylim(0, 1.03)
    ax.set_xticks(tick_taus)
    ax.set_xticklabels([rf"${t}{{\times}}$" for t in tick_taus], fontsize=8)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(r"within factor $\tau$ of the lowest-memory backend", fontsize=9,
                  color=INK2)
    ax.set_ylabel(f"share of the {n} problems", fontsize=9, color=INK2)
    if not THESIS:
        ax.set_title(f"Memory profile, cold one-shot canonicalization — {n} problems, "
                     "like-for-like\ncold delta max-RSS; every backend on the "
                     "ignore_dpp job", fontsize=10, color=INK, loc="left")
        fig.text(0.02, -0.05,
                 "Engine from the 2026-08-02 fork run; stock backends from the "
                 "2026-08-11 upstream ND run (cross-run drift on the 17 same-job "
                 "problems:\nsub-MB — memory deltas are essentially deterministic). "
                 "ConvexPlasticity: stock cells are measurement floors (extraction "
                 "never exceeded\nsetup's high-water mark), scored as a 3-way tie at "
                 "ratio 1; the engine's real +1.2 GB — the 274×-padded-P issue — is "
                 "tied with them on the later build (1.4 MB).",
                 fontsize=7.5, color=MUTED, linespacing=1.5)
    _save(fig, "memory_profile", written)


def fig_profiles_combined(plt, written):
    """Time and memory profiles stacked in one canvas so the thesis places
    them as a single float (both panels guaranteed on one page). Same data
    and styling as fig_profile / fig_memory_profile; each panel carries its
    own boxed legend."""
    n_t, series, ratios_t = _oneshot_ratios()
    n_m, _lbls, ratios_m = _memory_ratios()
    # Thesis height fills a float page: \textheight is 9.03 in and the caption
    # takes ~1.2 in, so 7.7 in puts each panel back at the singles' ~4 in.
    fig, axes = plt.subplots(2, 1, figsize=_figsize((7.4, 9.6),
                                                    (TEXTWIDTH_IN, 7.7)))
    panels = [
        (axes[0], ratios_t, n_t, 512, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
         "(a) canonicalization time", r"within factor $\tau$ of the best backend", True),
        (axes[1], ratios_m, n_m, 64, [1, 2, 4, 8, 16, 32, 64],
         "(b) peak memory", r"within factor $\tau$ of the lowest-memory backend", False),
    ]
    for ax, ratios, n, hi, tick_taus, tag, xlabel, legend in panels:
        _style(ax)
        # Boxed axes (advisor, Document 9 p. 65): re-enable the top/right
        # spines that _style hides.
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(AXIS)
        ax.set_xscale("log", base=2)
        taus = np.geomspace(1, hi, 600)
        for s in series:
            r = np.array(ratios[s])
            rho = [(r <= tau).mean() for tau in taus]
            color, ls = PROFILE_STYLE[s]
            # Typewriter legend labels, matching \texttt{} in the thesis text.
            ax.step(taus, rho, where="post", color=color, linewidth=1.5,
                    linestyle=ls, zorder=3, label=rf"$\mathtt{{{s}}}$")
        # One boxed legend per panel (advisor, Document 8 p. 48).
        ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=1.0,
                  edgecolor=GRID, fancybox=False, labelcolor=INK2, handlelength=2.8)
        ax.set_xlim(1, hi)
        ax.set_ylim(0, 1.03)
        ax.set_xticks(tick_taus)
        ax.set_xticklabels([rf"${t}{{\times}}$" for t in tick_taus], fontsize=8)
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        # Panel tag "(a) ..." below the panel, as a second line of the x label
        # (tight_layout accounts for it there; a free-floating text would not).
        ax.set_xlabel(f"{xlabel}\n{tag}", fontsize=9, color=INK2, linespacing=1.8)
        ax.set_ylabel(f"share of the {n} problems", fontsize=9, color=INK2)
    _save(fig, "profiles_combined", written)


def main():
    import argparse

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thesis", action="store_true",
                    help="bare figures for LaTeX: *_thesis.pdf at TEXTWIDTH_IN wide, "
                         "CM-style serif, titles/footnotes left to \\caption "
                         "(snippets in thesis_figures.tex)")
    args = ap.parse_args()
    global THESIS
    THESIS = args.thesis

    rc = {"text.color": INK, "axes.labelcolor": INK2,
          "figure.facecolor": "white", "savefig.facecolor": "white"}
    if THESIS:
        rc.update({"font.family": "serif", "mathtext.fontset": "cm"})
    else:
        rc.update({"font.family": "sans-serif"})
    plt.rcParams.update(rc)

    PLOTS.mkdir(parents=True, exist_ok=True)
    written = []
    if THESIS:
        # Only the two profiles go in the thesis; the other figures keep
        # their screen versions but get no thesis export.
        fig_profile(plt, written)
        fig_memory_profile(plt, written)
        fig_profiles_combined(plt, written)
        print(f"Caption snippets: "
              f"{_write_tex(['performance_profile', 'memory_profile', 'profiles_combined'])}")
    else:
        fig_quadrant(plt, written)
        fig_cold_warm(plt, written)
        fig_scaling(plt, written)
        fig_profile(plt, written)
        fig_memory_profile(plt, written)
    print(f"Figures: {', '.join(written)} -> {PLOTS}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
