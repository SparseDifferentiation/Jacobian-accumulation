"""Deck copies of the two thesis profiles (defence slides, advisor 2026-09-29).

Same data as plot_thesis_figures.fig_profile / fig_memory_profile, restyled for a
projector: thicker lines, boxed axes, sans-serif labels (in the deck's typewriter
font COO reads as C-zero-zero). Both figures share one canvas size so the two
slides render at the same scale. Writes straight into the thesis repo:

    uv run --with matplotlib --with numpy python plot_deck_profiles.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from plot_sparsity_correlations import AXIS, GRID, INK, INK2, _style  # noqa: E402
from plot_thesis_figures import PROFILE_STYLE, _memory_ratios  # noqa: E402

OUT = Path.home() / "Documents" / "wzz-thesis" / "images"
FIGSIZE = (5.6, 2.75)  # legend outside the axes on the right; about 0.85\textwidth, 1:1
LINEWIDTH = 1.9
THESIS_TABLE = Path.home() / "Documents" / "wzz-thesis" / "7-Theme2.tex"


def _table_ratios():
    """Ratio-to-best per backend from the rows of thesis Table 4.1
    (tab:oneshot-compile). plot_thesis_figures builds fig:profiles(a) from
    exactly these cells (_thesis_table_cells), but some of its sources (the
    results/repl replicates) are not in every checkout; the printed table is.
    Two-decimal rounding can move a step by a hair at small tau, nothing more."""
    import re
    tex = THESIS_TABLE.read_text()
    body = tex[tex.index(r"\label{tab:oneshot-compile}"):]
    body = body[body.index(r"\midrule"):body.index(r"\textbf{Geometric mean}")]
    series = ["diffengine", "CPP", "SCIPY", "COO"]
    ratios = {s: [] for s in series}
    n = 0
    for line in body.splitlines():
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        if len(cells) != 6:
            continue
        t = [float(re.sub(r"[^0-9.]", "", c)) for c in cells[1:5]]
        row = {"CPP": t[0], "SCIPY": t[1], "COO": t[2], "diffengine": t[3]}
        best = min(row.values())
        for s in series:
            ratios[s].append(row[s] / best)
        n += 1
    return n, series, ratios


def _profile(plt, ratios, series, n, hi, tick_taus, xlabel, stem):
    taus = np.geomspace(1, hi, 600)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    _style(ax)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    for side in ax.spines:
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.set_xscale("log", base=2)
    for s in series:
        r = np.array(ratios[s])
        rho = [(r <= t).mean() for t in taus]
        color, ls = PROFILE_STYLE[s]
        ax.step(taus, rho, where="post", color=color, linewidth=LINEWIDTH,
                linestyle=ls, zorder=3, label=s)
    # outside the axes: inside, it covered the CVXPY curves' climb and plateau
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9,
              frameon=True, framealpha=1.0, edgecolor=GRID, fancybox=False,
              labelcolor=INK, handlelength=3.2, borderaxespad=0)
    ax.set_xlim(1, hi)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xticks(tick_taus)
    ax.set_xticklabels([rf"{t}×" for t in tick_taus], fontsize=9)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10, color=INK2)
    ax.set_ylabel(f"share of the {n} problems", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}_deck.pdf")
    plt.close(fig)
    print(f"wrote {OUT / f'{stem}_deck.pdf'}")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "text.color": INK,
                         "axes.labelcolor": INK2, "figure.facecolor": "white",
                         "savefig.facecolor": "white"})

    n_t, series, ratios_t = _table_ratios()
    assert n_t == 25, n_t
    print("diffengine fastest on", sum(r == 1.0 for r in ratios_t["diffengine"]))
    # every other power of two: at slide size ten labels run into each other
    _profile(plt, ratios_t, series, n_t, 512, [1, 4, 16, 64, 256],
             r"within factor $\tau$ of the fastest backend", "performance_profile")
    n_m, series_m, ratios_m = _memory_ratios()
    _profile(plt, ratios_m, series_m, n_m, 64, [1, 2, 4, 8, 16, 32, 64],
             r"within factor $\tau$ of the lowest-memory backend", "memory_profile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
