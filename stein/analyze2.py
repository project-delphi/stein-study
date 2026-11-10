"""Study-2 analysis: per-run table, per-config means with seed-bootstrap CIs, threshold estimates,
content-leakage frontiers -> results2/.

    .venv/bin/python -W ignore -m stein.analyze2 [--grid s2_all] [--out results2]

Threshold estimator (fixed in theory/predictions.md before any study-2 network was trained):
lambda_hat* = smallest lambda at which the 5-seed mean linear leakage R^2 exceeds tau = 0.10,
interpolated linearly in log lambda; 95 % percentile bootstrap over seeds (2000 draws, seeds resampled
with replacement, the same draw used at every lambda). Censored values are reported as "<lam_min" /
">lam_max". Secondary: nuisance variance share with tau = 0.05. Supplementary (GATE_1, defined
before the held-out runs): the same crossing searched upward from the lambda of minimum mean leakage,
for paths that start high in a low-lambda collapse regime.
"""
import argparse, json, os

import numpy as np
import pandas as pd

from .train import resolve, cfg_hash
from . import grids

METRICS = ["leak_r2", "leak_knn_r2", "nuis_share", "nuis_share_oh", "content_r2", "content_knn_r2",
           "content_share", "content_share_oh", "unexpl_share", "unexpl_share_oh", "nuis_dir_ratio",
           "mcc", "ident_orth_r2", "eff_rank", "emb_std_mean", "total_var", "shape_acc"]
SETTING = ["tier", "encoder", "pred_arch", "arm", "d", "alpha", "sigma", "noise_mode", "law", "prior", "steps"]
N_BOOT = 2000


def load_runs(cfgs, runs_dir):
    rows = []
    memb = grids.s2_membership()
    for c in cfgs:
        rc = resolve(c)
        p = os.path.join(runs_dir, cfg_hash(rc), "result.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        row = dict(r["config"])
        row["cell"] = cfg_hash(resolve({**c, "seed": -1}))
        row["blocks"] = ",".join(memb.get(row["cell"], []))
        for k, v in r.items():
            if k != "config":
                row[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
        rows.append(row)
    return pd.DataFrame(rows)


def boot_ci(x, rng, n=N_BOOT):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def cells(df, rng):
    out = []
    for cell, g in df.groupby("cell"):
        row = {k: g[k].iloc[0] for k in SETTING + ["lam", "blocks"]}
        row.update(cell=cell, n_seeds=len(g))
        for m in METRICS:
            if m in g:
                row[m], row[m + "_lo"], row[m + "_hi"] = boot_ci(g[m].astype(float), rng)
        out.append(row)
    return pd.DataFrame(out)


def crossing(lams, y, tau, start=0):
    """log-lambda interpolated first crossing of y > tau at or after index `start`.
    Returns (value, censor) with censor in {'', '<', '>'}."""
    for i in range(start, len(lams)):
        if y[i] > tau:
            if i == start:
                return lams[i], "<"
            a, b = np.log(lams[i - 1]), np.log(lams[i])
            t = (tau - y[i - 1]) / (y[i] - y[i - 1])
            return float(np.exp(a + t * (b - a))), ""
    return lams[-1], ">"


def thresholds(df, rng, metric="leak_r2", tau=0.10, upward=False):
    """One row per setting (all keys but lam): lambda_hat* and its seed-bootstrap CI."""
    out = []
    keys = [k for k in SETTING if k in df]
    for key, g in df.groupby(keys):
        lams = np.sort(g["lam"].unique())
        if len(lams) < 3:
            continue
        P = g.pivot_table(index="seed", columns="lam", values=metric)[lams]
        P = P.dropna()
        if len(P) < 2:
            continue
        Y = P.to_numpy(float)

        def est(Yb):
            m = Yb.mean(0)
            s = int(np.argmin(m)) if upward else 0
            return crossing(lams, m, tau, s)
        v, cen = est(Y)
        draws = []
        for _ in range(N_BOOT):
            lv, c = est(Y[rng.integers(0, len(Y), len(Y))])
            draws.append(-np.inf if c == "<" else (np.inf if c == ">" else np.log(lv)))
        draws = np.array(draws)
        lo, hi = np.exp(np.percentile(draws, [2.5, 97.5]))
        row = dict(zip(keys, key))
        row.update(metric=metric, tau=tau, upward=upward, lam_hat=v, censor=cen, lo=lo, hi=hi,
                   frac_censored=float(np.mean(~np.isfinite(draws))), n_seeds=len(Y),
                   lam_min=lams[0], lam_max=lams[-1], n_levels=len(lams))
        out.append(row)
    return pd.DataFrame(out)


def frontier(df, rng, ceilings=(0.05, 0.10, 0.20)):
    """Best mean content R^2 reachable over the lambda path with mean leakage <= ceiling, per setting,
    with a seed-bootstrap CI (same draw across lambda)."""
    out = []
    keys = [k for k in SETTING if k in df]
    for key, g in df.groupby(keys):
        lams = np.sort(g["lam"].unique())
        C = g.pivot_table(index="seed", columns="lam", values="content_r2")[lams].dropna()
        L = g.pivot_table(index="seed", columns="lam", values="leak_r2")[lams].loc[C.index]
        if len(C) < 2:
            continue
        Cn, Ln = C.to_numpy(float), L.to_numpy(float)

        def best(ix, ceil):
            c, l = Cn[ix].mean(0), Ln[ix].mean(0)
            ok = l <= ceil
            return float(c[ok].max()) if ok.any() else np.nan
        for ceil in ceilings:
            allix = np.arange(len(Cn))
            b = np.array([best(rng.integers(0, len(Cn), len(Cn)), ceil) for _ in range(N_BOOT)])
            row = dict(zip(keys, key))
            row.update(ceiling=ceil, best_content=best(allix, ceil),
                       lo=float(np.nanpercentile(b, 2.5)) if np.isfinite(b).any() else np.nan,
                       hi=float(np.nanpercentile(b, 97.5)) if np.isfinite(b).any() else np.nan,
                       frac_none=float(np.mean(~np.isfinite(b))))
            out.append(row)
    return pd.DataFrame(out)


def md(df):
    """Markdown table without the optional `tabulate` dependency."""
    if df.empty:
        return "(none)"
    fmt = lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


SCHEMA = {
    "runs2.parquet": "one row per run: resolved config, `cell` (hash without seed), `blocks` (study-2 blocks "
                     "containing the config), every metric of stein.train.evaluate (lists JSON-encoded)",
    "cells2.parquet": "one row per config over seeds: mean and 95% seed-bootstrap CI (`_lo`, `_hi`) of METRICS",
    "thresholds.parquet": "one row per setting x estimator: lam_hat (censor '<'/'>' when outside the grid), "
                          "bootstrap CI lo/hi (+-inf draws when censored), metric, tau, upward",
    "frontier.parquet": "best mean content R^2 over the lambda path subject to mean leak_r2 <= ceiling, with CI",
    "metrics": {
        "leak_r2 / content_r2": "linear ridge probe R^2 (mean over code dims), held out",
        "nuis_share / content_share": "fraction of total embedding variance explained by OLS on the linear "
                                      "nuisance / content code, held out (raw units)",
        "*_share_oh": "same with one-hot factor levels (additive per factor)",
        "unexpl_share(_oh)": "1 - joint share of content and nuisance codes",
        "nuis_eigs / content_eigs": "eigenvalues of the covariance of the code-explained part of z",
        "nuis_dir_ratio": "mean of the top min(5, d-6) nuis_eigs / mean of the top min(6, d) content_eigs "
                          "(theory: v = 1 - lam*/lam)",
        "pc_var / pc_nuis_var / pc_content_var": "variance along the embedding's principal axes, total and "
                                                 "explained by each code",
        "mcc / mcc_per": "mean |corr| between embedding coordinates and the 6 content code dims after "
                         "Hungarian matching; per code dim (cos hue, sin hue, scale, shape simplex x3)",
    },
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="s2_all")
    ap.add_argument("--out", default="results2")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    runs_dir = os.environ.get("STEIN_RUNS", "runs")
    df = load_runs(getattr(grids, a.grid)(), runs_dir)
    os.makedirs(a.out, exist_ok=True)
    if df.empty:
        print("no finished runs"); return
    df.to_parquet(f"{a.out}/runs2.parquet", index=False)
    C = cells(df, rng)
    C.to_parquet(f"{a.out}/cells2.parquet", index=False)
    # thresholds: lambda paths only (sigma = 0, base law/prior, 2000 steps; arms with a path)
    path = df[(df.sigma == 0) & (df.law == "uniform") & (df.prior == "gaussian") & (df.steps == 2000)]
    T = pd.concat([thresholds(path, rng, "leak_r2", 0.10), thresholds(path, rng, "nuis_share", 0.05),
                   thresholds(path, rng, "leak_r2", 0.10, upward=True)], ignore_index=True)
    T.to_parquet(f"{a.out}/thresholds.parquet", index=False)
    F = frontier(path, rng)
    F.to_parquet(f"{a.out}/frontier.parquet", index=False)
    json.dump(SCHEMA, open(f"{a.out}/schema.json", "w"), indent=1)
    with open(f"{a.out}/analysis.md", "w") as f:
        f.write(f"# Study 2 analysis (generated by stein/analyze2.py)\n\n{len(df)} runs, {len(C)} configs.\n\n")
        f.write("## Thresholds\n\n" + md(T) + "\n\n")
        f.write("## Frontier\n\n" + md(F) + "\n\n")
        cols = ["blocks", "arm", "encoder", "d", "alpha", "sigma", "noise_mode", "law", "prior", "steps", "lam",
                "n_seeds", "leak_r2", "nuis_share", "content_r2", "unexpl_share_oh", "nuis_dir_ratio", "mcc"]
        f.write("## Cells\n\n" + md(C[[c for c in cols if c in C]].sort_values([c for c in cols[:11] if c in C])) + "\n")
    print(f"{len(df)} runs, {len(C)} configs, {len(T)} threshold rows -> {a.out}/")


if __name__ == "__main__":
    main()
