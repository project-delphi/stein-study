"""Reproduce the temporal-tier panel of Figure 7 (left) from freshly trained runs, and compare.

    STEIN_RUNS=runs_repro .venv/bin/python -m stein.sweep heldout_h5 --workers 6 --out results/repro/h5.parquet
    .venv/bin/python tools/repro_h5.py results/repro/h5.parquet

Writes results/repro/fig7_left_repro.pdf (reported vs reproduced, side by side) and
results/repro/h5_compare.md (per-arm means with 95% bootstrap CIs, reported vs reproduced).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stein.analyze import boot_mean

ARMS = ["ema", "ema+sksd", "ema+sigreg", "sksd", "vicreg"]
NAMES = {"ema": "EMA", "ema+sksd": "EMA+\nsliced KSD", "ema+sigreg": "EMA+\nSIGReg", "sksd": "sliced KSD",
         "vicreg": "VICReg"}


def summarise(df):
    out = {}
    for a in ARMS:
        q = df[(df.tier == "temporal") & (df.arm == a)]
        out[a] = {m: boot_mean(q[m].values) for m in ("leak_r2", "leak_knn_r2", "content_r2")}
    return out


def main(path):
    rep = summarise(pd.read_parquet("results/heldout.parquet"))
    new_df = pd.read_parquet(path)
    new = summarise(new_df)
    os.makedirs("results/repro", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
    for ax, S, t in zip(axes, [rep, new], ["reported (results/heldout.parquet)", f"reproduced ({path})"]):
        for i, a in enumerate(ARMS):
            for m, off, mk, c in [("leak_r2", -0.15, "o", "k"), ("leak_knn_r2", 0.15, "s", "0.5")]:
                y, lo, hi, _ = S[a][m]
                ax.errorbar([i + off], [y], yerr=[[y - lo], [hi - y]], color=c, marker=mk, ms=4, capsize=2,
                            label=(("linear" if m == "leak_r2" else "kNN") if i == 0 else None))
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([NAMES[a] for a in ARMS], fontsize=7)
        ax.set_title(t, fontsize=8); ax.set_ylim(0, 1)
    axes[0].set_ylabel("temporal-tier leakage R²"); axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig("results/repro/fig7_left_repro.pdf", bbox_inches="tight")
    lines = ["| arm | metric | reported mean [95% CI] | reproduced mean [95% CI] | n seeds | reproduced mean inside reported CI |",
             "|:--|:--|--:|--:|--:|:-:|"]
    inside = 0
    for a in ARMS:
        for m in ("leak_r2", "leak_knn_r2", "content_r2"):
            r, n = rep[a][m], new[a][m]
            ok = r[1] <= n[0] <= r[2]
            inside += ok
            lines.append(f"| {a} | {m} | {r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}] | {n[0]:.3f} [{n[1]:.3f}, {n[2]:.3f}] "
                         f"| {n[3]} | {'yes' if ok else 'no'} |")
    lines.append(f"\n{inside}/{3 * len(ARMS)} reproduced means fall inside the reported 95% CI.")
    open("results/repro/h5_compare.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/repro/h5.parquet")
