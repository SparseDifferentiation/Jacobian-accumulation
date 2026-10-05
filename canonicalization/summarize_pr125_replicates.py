"""Combine the SparseDiffEngine #125 A/B replicate sweeps into one table.

`run_backend_benchmarks.py` writes a rendered fixed-width `.txt` plus a raw
`.json` per invocation. Re-deriving a cross-replicate summary by parsing the
`.txt` would mean parsing fixed-width columns, so this reads the `.json`
siblings instead:

    results/results_backends_pr125_<build>_rep<k>[big]_*.json

for both engine builds (`engbase` = SparseDiffEngine main 4dbb53b,
`pr125` = #125 head b9b5771) and prints, per (problem, strategy), the median
of the per-replicate means, the de_cached/dpp and de_cached/best-tensor
ratios under each build, the #125 speedup on the engine columns, any
replicate spread above 10 %, and the geometric mean.

The `dpp`/`dpp_coo`/`dpp_scipy` rows must agree across the two builds -- they
share one cvxpy and only the engine wheel differs -- so a drift there means
the A/B is contaminated.

Produces the tables in `results/results_backends_pr125_summary_20260919.md`.

Usage:
    python canonicalization/summarize_pr125_replicates.py [results dir]
"""
import json
import statistics as st
import sys
from pathlib import Path

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("canonicalization/results")
STRATS = ["dpp", "dpp_coo", "dpp_scipy", "diffengine", "de_cached"]
BUILDS = ["engbase", "pr125"]


def load(build):
    """{problem: {strategy: [mean_s per replicate]}} plus {problem: meta}."""
    per, meta = {}, {}
    files = sorted(RES.glob(f"results_backends_pr125_{build}_rep*.json"))
    for f in files:
        for row in json.loads(f.read_text()):
            name = row["class"]
            meta.setdefault(name, row)
            d = per.setdefault(name, {})
            for s, v in (row.get("comparison") or {}).items():
                if v and v.get("mean_s") is not None:
                    d.setdefault(s, []).append(v["mean_s"])
    return per, meta, [f.name for f in files]


def med(xs):
    return st.median(xs) if xs else None


def fmt(v, w=10):
    return f"{v:{w}.4f}" if v is not None else " " * (w - 3) + "  -"


def main():
    data = {b: load(b) for b in BUILDS}
    for b in BUILDS:
        print(f"{b}: {len(data[b][2])} replicate files -> {data[b][2]}")
    names = sorted(set(data["engbase"][0]) | set(data["pr125"][0]))
    print()

    hdr = f"{'benchmark':<40s}{'build':<9s}" + "".join(f"{s:>11s}" for s in STRATS)
    hdr += f"{'dec/dpp':>10s}{'dec/best':>10s}{'reps':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for name in names:
        for b in BUILDS:
            per = data[b][0].get(name, {})
            vals = {s: med(per.get(s, [])) for s in STRATS}
            tensors = [vals[s] for s in ("dpp", "dpp_coo", "dpp_scipy") if vals[s]]
            best = min(tensors) if tensors else None
            de = vals["de_cached"] or vals["diffengine"]
            r1 = f"{de / vals['dpp']:9.2f}x" if de and vals["dpp"] else "        -"
            r2 = f"{de / best:9.2f}x" if de and best else "        -"
            n = max((len(v) for v in per.values()), default=0)
            print(f"{name:<40s}{b:<9s}" + "".join(fmt(vals[s], 11) for s in STRATS)
                  + r1 + r2 + f"{n:>6d}")
        # PR-125 effect on the engine columns
        for s in ("de_cached", "diffengine"):
            a = med(data["engbase"][0].get(name, {}).get(s, []))
            c = med(data["pr125"][0].get(name, {}).get(s, []))
            if a and c:
                print(f"{'':<40s}{'#125 ' + s:<9s}"
                      f"  base={a:.4f}  pr125={c:.4f}  speedup={a / c:.2f}x")
        print()

    # spread report
    print("\nreplicate spread (max-min)/median per build/strategy:")
    for b in BUILDS:
        for name in names:
            for s, xs in sorted(data[b][0].get(name, {}).items()):
                if len(xs) > 1 and st.median(xs):
                    sp = (max(xs) - min(xs)) / st.median(xs)
                    if sp > 0.10:
                        print(f"  {b:<8s} {name:<40s} {s:<11s} {sp:6.1%}  {xs}")

    # geometric means
    import math
    print("\ngeometric mean de_cached/dpp (DPP problems only):")
    for b in BUILDS:
        rs = []
        for name in names:
            per = data[b][0].get(name, {})
            de, dpp = med(per.get("de_cached", [])), med(per.get("dpp", []))
            if de and dpp:
                rs.append(de / dpp)
        if rs:
            g = math.exp(sum(math.log(r) for r in rs) / len(rs))
            print(f"  {b:<8s} {g:.3f}x  (n={len(rs)})")


if __name__ == "__main__":
    main()
