"""Correlate lowered-matrix structure metrics with engine-relative cold time.

Joins results/sparsity_metrics.json (per-problem structural metrics of the
lowered A and P, numeric pattern) with the committed cold-compile tables:

  t_DE      DIFFENG column of results_backends_all_20260724_repl.txt (fork),
            with SimpleFullyParametrizedLPBenchmark / ParametrizedQPBenchmark
            patched from results_backends_fork_bigparam_20260808.json (the
            repl table's 1.35 s SimpleFullyParametrizedLP cell is a documented
            transient; ParametrizedQP is absent from the full sweep).
  t_*_ND    CPP_ND / SCIPY_ND / COO_ND columns of
            results_backends_upstream192_nodpp_20260808_merged.txt (upstream
            1.9.2, ignore_dpp) -- the like-for-like baselines: both sides emit
            a concrete (P, c, A, b) for one parameter value.

Y axes (ratio framing): ratio_cpp = t_DE / t_CPP_ND (headline) and
ratio_best = t_DE / min(t_CPP_ND, t_SCIPY_ND, t_COO_ND). CAVEAT: engine and
ND columns come from different run dates on the same machine; the noise floor
is roughly +/-20%, so only ratios beyond ~1.25x are meaningful.

X axes: numeric-pattern A metrics (block_gain, gini_row, entropy_norm,
density, diag_dist_mean). Hypothesis: dense-block-ness (high block_gain /
gini, low entropy) correlates with engine-relative slowdown.

Outputs: results/sparsity_correlations.{json,txt} (Spearman rho per pair,
dropped problems with reasons) and results/plots/*.{png,pdf}.

Usage: .venv/bin/python canonicalization/plot_sparsity_correlations.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_backend_benchmarks import _parse_table  # noqa: E402

RESULTS = HERE / "results"
METRICS_JSON = RESULTS / "sparsity_metrics.json"
ENGINE_TABLE = RESULTS / "results_backends_all_20260724_repl.txt"
ND_TABLE = RESULTS / "results_backends_upstream192_nodpp_20260808_merged.txt"
BIGPARAM_JSON = RESULTS / "results_backends_fork_bigparam_20260808.json"
MEM_JSON = RESULTS / "results_backends_memory_20260802.json"
PLOTS = RESULTS / "plots"

# (metric key in the numeric block, x-scale, axis label)
X_METRICS = [
    ("block_gain", "log", "block gain (fill in touched 16x16 blocks / density)"),
    ("gini_row", "linear", "Gini of per-row nnz"),
    ("entropy_norm", "linear", "normalized spatial entropy (64x64 grid)"),
    ("density", "log", "density nnz/(m*n)"),
    ("diag_dist_mean", "linear", "mean scaled diagonal distance"),
]
Y_AXES = [
    ("ratio_cpp", "engine / CPP_ND cold time"),
    ("ratio_best", "engine / best stock ND cold time"),
    ("mem_ratio_cpp", "engine / CPP cold memory (delta max-RSS)"),
]

# Reference-palette light-mode roles (dataviz skill): single series, ink text.
BLUE, INK, INK2, MUTED, GRID, AXIS = (
    "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")


def _short(name):
    return name[:-len("Benchmark")] if name.endswith("Benchmark") else name


def build_records():
    """One row per problem: metrics + timings + ratios; plus dropped reasons."""
    canon_e, eng = _parse_table(ENGINE_TABLE)
    de_label = canon_e.get("DIFFENGINE", "DIFFENG")
    _, nd = _parse_table(ND_TABLE)
    t_de = {cls: row[de_label] for cls, row in eng.items() if de_label in row}
    for r in json.loads(BIGPARAM_JSON.read_text()):
        cold = (r.get("cold") or {}).get("DIFFENGINE")
        if isinstance(cold, (int, float)):
            t_de[r["class"]] = cold

    # Cold memory (2026-08-02 run): delta max-RSS per target. CAVEAT: the
    # stock targets there are the DPP path (no ND memory sweep exists), which
    # builds a parameter->data tensor -- a different artifact on the 8
    # parametric problems; on the 17 parameter-free ones the paths coincide.
    mem = {}
    for r in json.loads(MEM_JSON.read_text()):
        mc = r.get("mem_cold") or {}
        de, cpp = mc.get("DIFFENGINE"), mc.get("CPP")
        if isinstance(de, dict) and isinstance(cpp, dict):
            mem[r["class"]] = {
                "mem_de_mb": de.get("delta_mb"),
                "mem_cpp_mb": cpp.get("delta_mb"),
                "engine_peak_mb": de.get("engine_peak_mb"),
                "mem_floor": bool(de.get("floor") or cpp.get("floor")),
            }

    metrics = json.loads(METRICS_JSON.read_text())
    records, dropped = [], {}
    for cls, rec in sorted(metrics["problems"].items()):
        if rec.get("status") != "ok":
            dropped[cls] = f"metrics: {rec.get('status')}"
            continue
        row = {"name": cls,
               "A": rec["A"]["numeric"], "A_shape": rec["A"]["shape"],
               "A_zero_pad": rec["A"]["zero_pad_ratio"],
               "P": rec["P"]["numeric"] if rec.get("P") else None,
               "P_zero_pad": rec["P"]["zero_pad_ratio"] if rec.get("P") else None,
               "t_de": t_de.get(cls)}
        nd_row = nd.get(cls, {})
        for col in ("CPP_ND", "SCIPY_ND", "COO_ND"):
            row["t_" + col.lower()] = nd_row.get(col)
        if row["t_de"] is None:
            dropped[cls] = "no engine cold time in the source tables"
        stocks = [row[k] for k in ("t_cpp_nd", "t_scipy_nd", "t_coo_nd") if row[k]]
        row["ratio_cpp"] = (row["t_de"] / row["t_cpp_nd"]
                            if row["t_de"] and row["t_cpp_nd"] else None)
        row["ratio_best"] = (row["t_de"] / min(stocks)
                             if row["t_de"] and stocks else None)
        if row["ratio_cpp"] is None and cls not in dropped:
            dropped[cls] = "no CPP_ND baseline (crashed/excluded in the ND sweep)"
        mrow = mem.get(cls, {})
        row.update({k: mrow.get(k) for k in
                    ("mem_de_mb", "mem_cpp_mb", "engine_peak_mb")})
        usable = (mrow and not mrow["mem_floor"]
                  and mrow["mem_de_mb"] and mrow["mem_cpp_mb"]
                  and mrow["mem_cpp_mb"] > 0)
        row["mem_ratio_cpp"] = (mrow["mem_de_mb"] / mrow["mem_cpp_mb"]
                                if usable else None)
        records.append(row)
    return records, dropped


def spearman_table(records):
    from scipy.stats import spearmanr

    out = {}
    for y_key, _ in Y_AXES:
        out[y_key] = {}
        for mkey, _, _ in X_METRICS:
            xs, ys = [], []
            for r in records:
                x, y = r["A"].get(mkey), r[y_key]
                if x is not None and y is not None and math.isfinite(x):
                    xs.append(x)
                    ys.append(y)
            if len(xs) >= 3:
                rho, p = spearmanr(xs, ys)
                out[y_key][mkey] = {"rho": float(rho), "p": float(p), "n": len(xs)}
            else:
                out[y_key][mkey] = {"rho": None, "p": None, "n": len(xs)}
    return out


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _style(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _scatter(ax, records, mkey, xscale, xlabel, y_key, annotate=True, source="A"):
    pts = [(r[source][mkey], r[y_key], _short(r["name"])) for r in records
           if r.get(source) and r[source].get(mkey) is not None
           and math.isfinite(r[source][mkey]) and r[y_key] is not None]
    if not pts:
        return 0
    xs, ys, names = zip(*pts)
    _style(ax)
    ax.set_xscale(xscale)
    ax.set_yscale("log")
    ax.axhline(1.0, color=AXIS, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.scatter(xs, ys, s=34, color=BLUE, edgecolors="white", linewidths=0.6,
               zorder=3)
    if annotate:
        _place_labels(ax, pts)
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    return len(pts)


def _place_labels(ax, pts, fontsize=6.0):
    """Greedy collision-avoiding point labels, working in pixel space.

    Candidate offsets around each marker are tried in order; the first whose
    approximate text box does not overlap an already-placed box wins. Falls
    back to the default offset (accepting the overlap) when every slot is
    taken, so no label is silently dropped.
    """
    ax.figure.canvas.draw()  # realize the autoscaled limits for transData
    xy = ax.transData.transform([(x, y) for x, y, _ in pts])
    char_w, box_h = fontsize * 0.62, fontsize + 3.0
    candidates = [(5, 2, "left"), (5, -9, "left"), (-5, 2, "right"),
                  (-5, -9, "right"), (5, 11, "left"), (-5, 11, "right"),
                  (5, -19, "left"), (-5, -19, "right")]
    placed = []
    order = sorted(range(len(pts)), key=lambda i: (-xy[i][1], xy[i][0]))
    for i in order:
        px, py = xy[i]
        name = pts[i][2]
        w = char_w * len(name)
        for dx, dy, ha in candidates:
            x0 = px + dx - (w if ha == "right" else 0)
            box = (x0, py + dy, x0 + w, py + dy + box_h)
            if all(box[0] > b[2] or box[2] < b[0] or box[1] > b[3] or box[3] < b[1]
                   for b in placed):
                break
        else:
            dx, dy, ha = candidates[0]
            x0 = px + dx
            box = (x0, py + dy, x0 + w, py + dy + box_h)
        placed.append(box)
        ax.annotate(name, pts[i][:2], textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=fontsize, color=INK2,
                    zorder=4)


def make_plots(records, corr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "text.color": INK,
                         "axes.labelcolor": INK2, "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    PLOTS.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig, stem):
        for ext in ("png", "pdf"):
            fig.savefig(PLOTS / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(stem)

    def grid(y_key, y_label, stem, title, notes_text, source="A"):
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))
        for ax, (mkey, xscale, xlabel) in zip(axes.flat, X_METRICS):
            _scatter(ax, records, mkey, xscale, xlabel, y_key, source=source)
            c = corr[y_key][mkey] if source == "A" else None
            ax.set_title(f"{mkey}   rho={c['rho']:+.2f}  n={c['n']}"
                         if c and c["rho"] is not None else f"{source}: {mkey}",
                         fontsize=9.5, color=INK, loc="left")
            ax.set_ylabel(y_label if ax in (axes.flat[0], axes.flat[3]) else "",
                          fontsize=8.5, color=INK2)
        notes = axes.flat[5]
        notes.axis("off")
        notes.text(0.0, 0.95, "Notes", fontsize=10, color=INK, va="top")
        notes.text(0.0, 0.82, notes_text, fontsize=8, color=INK2, va="top",
                   linespacing=1.5)
        fig.suptitle(title, fontsize=12, color=INK, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save(fig, stem)

    def panels(y_key, y_label, prefix, title_fmt):
        for mkey, xscale, xlabel in X_METRICS:
            fig, ax = plt.subplots(figsize=(6.4, 4.6))
            _scatter(ax, records, mkey, xscale, xlabel, y_key)
            c = corr[y_key][mkey]
            rho = (f"Spearman rho = {c['rho']:+.2f} (n = {c['n']})"
                   if c["rho"] is not None else "")
            ax.set_ylabel(y_label, fontsize=9, color=INK2)
            ax.set_title(title_fmt.format(mkey=mkey) + f"\n{rho}",
                         fontsize=10, color=INK, loc="left")
            save(fig, f"{prefix}_vs_{mkey}")

    time_notes = (
        "y: engine cold (fork, 2026-07-24 repl + bigparam patch)\n"
        "   / upstream 1.9.2 CPP_ND cold (2026-08-08).\n"
        "Different run dates, same machine: ~+/-20% noise floor;\n"
        "only ratios beyond ~1.25x are meaningful.\n"
        "x: numeric pattern of the lowered A (stored zeros dropped).\n"
        "Dashed line: engine = CPP_ND.\n"
        "Missing: problems without an ND baseline (crashed/excluded).")
    mem_notes = (
        "y: cold delta max-RSS, engine / CPP, single-run\n"
        "   (results_backends_memory_20260802.json).\n"
        "CAVEAT: the CPP baseline is the DPP path (no ND memory\n"
        "sweep exists) -- a different artifact on the 8 parametric\n"
        "problems; identical job on the 17 parameter-free ones.\n"
        "x: numeric pattern of the lowered A (stored zeros dropped).\n"
        "Dashed line: engine = CPP.")

    panels("ratio_cpp", Y_AXES[0][1], "ratio",
           "Engine-relative cold canonicalization vs {mkey}")
    panels("mem_ratio_cpp", Y_AXES[2][1], "mem_ratio",
           "Engine-relative cold memory vs {mkey}")
    grid("ratio_cpp", Y_AXES[0][1], "ratio_summary_grid",
         "Structure of lowered A vs engine-relative cold canonicalization time",
         time_notes)
    grid("mem_ratio_cpp", Y_AXES[2][1], "mem_ratio_summary_grid",
         "Structure of lowered A vs engine-relative cold memory (delta max-RSS)",
         mem_notes)

    # P subset (QP/SDP problems only), same metrics on P's numeric pattern.
    p_records = [r for r in records if r.get("P")]
    if p_records:
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))
        for ax, (mkey, xscale, xlabel) in zip(axes.flat, X_METRICS):
            _scatter(ax, p_records, mkey, xscale, xlabel, y_key, source="P")
            ax.set_title(f"P: {mkey}", fontsize=9.5, color=INK, loc="left")
            ax.set_ylabel(y_label if ax is axes.flat[0] or ax is axes.flat[3] else "",
                          fontsize=8.5, color=INK2)
        axes.flat[5].axis("off")
        axes.flat[5].text(0.0, 0.9, f"P-bearing problems only (n = {len(p_records)}).\n"
                          "Same y and caveats as the A grid.",
                          fontsize=8, color=INK2, va="top", linespacing=1.5)
        fig.suptitle("Structure of lowered P vs engine-relative cold canonicalization time",
                     fontsize=12, color=INK, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save(fig, "P_ratio_summary_grid")
    return written


# --------------------------------------------------------------------------- #
def render_txt(records, corr, dropped):
    lines = [
        "Sparsity-structure metrics vs engine-relative cold canonicalization time",
        "=" * 74,
        "",
        f"Sources: t_DE = {ENGINE_TABLE.name} (DIFFENG col; SimpleFullyParametrizedLP/",
        f"         ParametrizedQP patched from {BIGPARAM_JSON.name});",
        f"         t_*_ND = {ND_TABLE.name};",
        f"         metrics = {METRICS_JSON.name} (numeric pattern of A).",
        "Caveat:  engine and ND columns are different run dates on the same machine",
        "         (~+/-20% noise floor): ratios beyond ~1.25x are the meaningful regime.",
        f"Memory:  mem_ratio = cold delta max-RSS engine/CPP from {MEM_JSON.name};",
        "         the CPP baseline there is the DPP path (no ND memory sweep exists),",
        "         a different artifact on the 8 parametric problems.",
        "Spearman rank correlations (n ~ 22): p-values are indicative only.",
        "",
    ]
    for y_key, y_label in Y_AXES:
        lines.append(f"y = {y_key}  ({y_label})")
        lines.append(f"  {'metric':22} {'rho':>7} {'p':>8} {'n':>4}")
        for mkey, _, _ in X_METRICS:
            c = corr[y_key][mkey]
            if c["rho"] is None:
                lines.append(f"  {mkey:22} {'-':>7} {'-':>8} {c['n']:>4}")
            else:
                lines.append(f"  {mkey:22} {c['rho']:>+7.3f} {c['p']:>8.3f} {c['n']:>4}")
        lines.append("")
    lines.append(f"{'problem':38} {'ratio_cpp':>9} {'mem_ratio':>9} {'block_gain':>10} "
                 f"{'gini_row':>8} {'entropy':>7} {'density':>9} {'diag_d':>7}")
    for r in sorted(records, key=lambda r: -(r["ratio_cpp"] or 0)):
        a = r["A"]

        def f(v, fmt):
            if v is not None and math.isfinite(v):
                return format(v, fmt)
            return "-".rjust(int(fmt.split(".")[0]))
        lines.append(
            f"{r['name']:38} {f(r['ratio_cpp'], '9.2f')} {f(r['mem_ratio_cpp'], '9.2f')} "
            f"{f(a.get('block_gain'), '10.3g')} "
            f"{f(a.get('gini_row'), '8.3f')} {f(a.get('entropy_norm'), '7.3f')} "
            f"{f(a.get('density'), '9.2g')} {f(a.get('diag_dist_mean'), '7.3f')}")
    if dropped:
        lines += ["", "Dropped from ratio plots/correlations:"]
        lines += [f"  {k:38} {v}" for k, v in sorted(dropped.items())]
    return "\n".join(lines) + "\n"


def main():
    records, dropped = build_records()
    corr = spearman_table(records)
    out = {
        "meta": {
            "engine_table": ENGINE_TABLE.name, "nd_table": ND_TABLE.name,
            "bigparam_patch": BIGPARAM_JSON.name, "metrics": METRICS_JSON.name,
            "memory": MEM_JSON.name,
            "note": "different-date runs, same machine, ~+/-20% noise floor",
        },
        "spearman": corr,
        "dropped": dropped,
        "records": [{k: v for k, v in r.items()} for r in records],
    }
    (RESULTS / "sparsity_correlations.json").write_text(json.dumps(out, indent=1))
    txt = render_txt(records, corr, dropped)
    (RESULTS / "sparsity_correlations.txt").write_text(txt)
    print(txt)
    written = make_plots(records, corr)
    n_ratio = sum(1 for r in records if r["ratio_cpp"] is not None)
    print(f"{len(records)} problems with metrics, {n_ratio} with a CPP_ND ratio.")
    print(f"Plots: {', '.join(written)} -> {PLOTS}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
