"""Study-2 P2: the statistics behind each pre-registered prediction (theory/predictions.md P1-P18),
read from results2/runs2.parquet -> results2/verdicts.json.

    .venv/bin/python -W ignore -m stein.verdicts2

Every number in the P2 gate file and the verdict table comes from here. The registered criteria are
coded as written in theory/predictions.md; the verdict strings are this script's mechanical reading of
them, and GATE_2.md adds the judgement where a criterion is ambiguous (censored thresholds).
Bootstrap: seeds resampled with replacement (2000 draws); two different settings are resampled
independently (different runs), the same draw is used across lambda within a setting.
"""
import json

import numpy as np
import pandas as pd

from .analyze2 import crossing

N_BOOT = 2000
THEORY = dict(cov=0.0427, sigreg=0.0389, ksd=0.939, sksd=7.33, mmd=17.1, vicreg=7.0)
OUT = {}


def base(R, **kw):
    q = (R.sigma == 0) & (R.law == "uniform") & (R.prior == "gaussian") & (R.steps == 2000) & \
        R.blocks.str.contains(r"E1_|E2_|E6_", regex=True)
    sel = {"encoder": "conv", "d": 12, "alpha": 1.0, **kw}
    for k, v in sel.items():
        q &= R[k] == v
    return R[q]


def mat(g, metric):
    lams = np.sort(g.lam.unique())
    P = g.pivot_table(index="seed", columns="lam", values=metric)[lams].dropna()
    return lams, P.to_numpy(float)


def lam_hat_draws(g, rng, metric="leak_r2", tau=0.10):
    """point estimate (value, censor) and bootstrap draws of log lambda_hat (+-inf when censored)."""
    lams, Y = mat(g, metric)
    v, c = crossing(lams, Y.mean(0), tau)
    d = []
    for _ in range(N_BOOT):
        lv, cc = crossing(lams, Y[rng.integers(0, len(Y), len(Y))].mean(0), tau)
        d.append(-np.inf if cc == "<" else (np.inf if cc == ">" else np.log(lv)))
    return (v, c), np.array(d)


def ci(x):
    x = np.asarray(x, float)
    with np.errstate(invalid="ignore"):
        return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def ratio(a, b):
    """ratio of two lambda_hat estimates with CI from independent draws; None if a point is censored."""
    (va, ca), da = a
    (vb, cb), db = b
    pt = None if (ca or cb) else va / vb
    r = np.exp(da - db)
    r = r[np.isfinite(r)]
    return dict(ratio=pt, ci=ci(r) if len(r) else None, frac_finite=float(len(r) / N_BOOT),
                num=[va, ca], den=[vb, cb])


def interp_logl(lams, y, lam):
    return float(np.interp(np.log(lam), np.log(lams), y))


def rise_width(lams, y):
    """log10-lambda width between the first crossings of 10 % and 90 % of the path's range
    (measured from the path's minimum upward, after the minimum)."""
    s = int(np.argmin(y))
    lo, hi = y[s:].min(), y[s:].max()
    if hi - lo <= 0:
        return np.nan
    out = []
    for f in (0.1, 0.9):
        v, c = crossing(lams, y, lo + f * (hi - lo), s)
        out.append(np.log10(v))
    return float(out[1] - out[0])


def boot_mean(x, rng):
    x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return float(x.mean()), ci(b)


def boot_diff(x, y, rng):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1) - y[rng.integers(0, len(y), (N_BOOT, len(y)))].mean(1)
    return dict(diff=float(x.mean() - y.mean()), ci=ci(b))


def main():
    rng = np.random.default_rng(0)
    R = pd.read_parquet("results2/runs2.parquet")
    arms = ["cov", "vicreg", "sigreg", "sksd", "ksd", "mmd"]
    LH = {}

    # ---------------- P1 step-like onset (alpha 1, d 12)
    p1 = {}
    for a in arms:
        g = base(R, arm=a)
        lams, Y = mat(g, "leak_r2")
        m = Y.mean(0)
        half = 0.5 * m.max()
        # last level below 0.10 before the first level reaching >= 50 % of max
        j = int(np.argmax(m >= half))
        below = [i for i in range(j) if m[i] < 0.10]
        width = float(np.log10(lams[j] / lams[below[-1]])) if below else None
        p1[a] = dict(leak_lowest=float(m[0]), max=float(m.max()), lam_half=float(lams[j]),
                     lam_last_below_0p10=float(lams[below[-1]]) if below else None, decades=width,
                     lowest_ok=bool(m[0] < 0.05), within_decade=bool(width is not None and width <= 1.0 + 1e-9))
    OUT["P1"] = p1

    # ---------------- thresholds (primary) and P2/P3
    for a in arms:
        for key, kw in {"d12a1": {}, "d12a0.5": dict(alpha=0.5), "d12a0.25": dict(alpha=0.25),
                        "d6": dict(d=6), "d24": dict(d=24)}.items():
            LH[(a, key)] = lam_hat_draws(base(R, arm=a, **kw), rng)
        LH[(a, "share")] = lam_hat_draws(base(R, arm=a), rng, "nuis_share", 0.05)
    for a in ["cov", "ksd"]:
        for al in (1.0, 0.5):
            LH[(a, f"lin{al}")] = lam_hat_draws(base(R, arm=a, encoder="linear", alpha=al), rng)
    OUT["lam_hat"] = {f"{a}|{k}": dict(value=v[0][0], censor=v[0][1], ci=ci(np.exp(v[1][np.isfinite(v[1])])),
                                       frac_censored=float(np.mean(~np.isfinite(v[1]))))
                      for (a, k), v in LH.items()}
    OUT["P2"] = {a: dict(lam_hat=LH[(a, "d12a1")][0], theory=THEORY[a],
                         ratio=(None if LH[(a, "d12a1")][0][1] else LH[(a, "d12a1")][0][0] / THEORY[a]))
                 for a in arms}
    OUT["P3"] = {f"{x}/{y}": ratio(LH[(x, "d12a1")], LH[(y, "d12a1")])
                 for x, y in [("sigreg", "cov"), ("ksd", "sigreg"), ("sksd", "ksd"), ("mmd", "sksd")]}
    # ---------------- P4 alpha scaling, P5 d = 6, P6 d = 24
    OUT["P4"] = {a: {str(al): ratio(LH[(a, f"d12a{al}")], LH[(a, "d12a1")]) for al in (0.5, 0.25)} for a in arms}
    p5 = {}
    for a in arms:
        g6, g12 = base(R, arm=a, d=6), base(R, arm=a)
        l6, Y6 = mat(g6, "leak_r2")
        m12 = g12.groupby("lam").leak_r2.mean()
        p5[a] = dict(lams=l6.tolist(), leak_d6=Y6.mean(0).tolist(),
                     leak_d12=[float(m12.get(l, np.nan)) for l in l6],
                     max_d6=float(Y6.mean(0).max()), any_ge_0p10=bool((Y6.mean(0) >= 0.10).any()),
                     content_d6=mat(g6, "content_r2")[1].mean(0).tolist(),
                     content_d12=[float(base(R, arm=a).groupby("lam").content_r2.mean().get(l, np.nan)) for l in l6])
    OUT["P5"] = p5
    OUT["P6"] = {a: ratio(LH[(a, "d24")], LH[(a, "d12a1")]) for a in arms}

    # ---------------- P7 steepness: leakage R^2 vs nuisance share
    p7 = {}
    for a in arms:
        g = base(R, arm=a)
        lams, L = mat(g, "leak_r2")
        _, S = mat(g, "nuis_share")
        p7[a] = dict(width_leak=rise_width(lams, L.mean(0)), width_share=rise_width(lams, S.mean(0)))
        p7[a]["leak_steeper"] = bool(p7[a]["width_leak"] < p7[a]["width_share"])
    OUT["P7"] = p7

    # ---------------- P8 linear rung, P10 KSD/cov ratio
    OUT["P8"] = {a: dict(lam_hat_a1=LH[(a, "lin1.0")][0], theory=THEORY[a],
                         ratio_to_theory=None if LH[(a, "lin1.0")][0][1] else LH[(a, "lin1.0")][0][0] / THEORY[a],
                         alpha_ratio=ratio(LH[(a, "lin0.5")], LH[(a, "lin1.0")])) for a in ["cov", "ksd"]}
    OUT["P10"] = dict(linear=ratio(LH[("ksd", "lin1.0")], LH[("cov", "lin1.0")]),
                      conv=ratio(LH[("ksd", "d12a1")], LH[("cov", "d12a1")]),
                      conv_share=ratio(LH[("ksd", "share")], LH[("cov", "share")]))

    # ---------------- P9 per-direction nuisance variance ratio at 2x and 5x lambda_hat
    p9 = {}
    for a, enc in [("cov", "conv"), ("cov", "linear"), ("ksd", "conv"), ("sigreg", "conv"), ("mmd", "conv")]:
        key = "d12a1" if enc == "conv" else "lin1.0"
        (v, c), _ = LH[(a, key)]
        g = base(R, arm=a, encoder=enc)
        lams, V = mat(g, "nuis_dir_ratio")
        row = dict(lam_hat=v, censor=c)
        if not c:
            for q in (2, 5):
                lq = q * v
                row[f"v_at_{q}x"] = interp_logl(lams, V.mean(0), lq) if lams[0] <= lq <= lams[-1] else None
                row[f"theory_{q}x"] = 1 - 1 / q
        (vs, cs), _ = LH[(a, "share")] if enc == "conv" else ((None, "n/a"), None)
        if enc == "conv" and not cs:
            row["share_lam_hat"] = vs
            for q in (2, 5):
                lq = q * vs
                row[f"v_at_{q}x_share_est"] = interp_logl(lams, V.mean(0), lq) if lams[0] <= lq <= lams[-1] else None
        row["path"] = dict(zip(map(float, lams), V.mean(0).tolist()))
        p9[f"{a}|{enc}"] = row
    OUT["P9"] = p9

    # ---------------- E3 / P11, P12: training length
    e3 = {}
    for a, ls in {"ksd": [0.8, 3.2], "sigreg": [0.08, 0.32]}.items():
        for l in ls:
            q = (R.arm == a) & (R.lam == l) & (R.encoder == "conv") & (R.d == 12) & (R.alpha == 1) & \
                (R.sigma == 0) & (R.law == "uniform") & (R.prior == "gaussian")
            g2, g8 = R[q & (R.steps == 2000)], R[q & (R.steps == 8000)]
            row = {"leak_8000_minus_2000": boot_diff(g8.leak_r2, g2.leak_r2, rng)}
            for m in ["leak_r2", "content_r2", "nuis_share", "content_share", "unexpl_share",
                      "nuis_share_oh", "content_share_oh", "unexpl_share_oh", "leak_knn_r2", "content_knn_r2"]:
                row[m] = {"2000": boot_mean(g2[m], rng), "8000": boot_mean(g8[m], rng)}
            e3[f"{a}|{l}"] = row
    OUT["E3"] = e3

    # ---------------- E4 / P13-P15: noise
    e4 = {}
    for a, ls in {"ksd": [0.1, 0.2], "vicreg": [1.25, 5.0]}.items():
        for l in ls:
            q = (R.arm == a) & (R.lam == l) & (R.encoder == "conv") & (R.d == 12) & (R.alpha == 1) & \
                (R.steps == 2000) & (R.law == "uniform") & (R.prior == "gaussian")
            g0 = R[q & (R.sigma == 0)]
            row = {}
            for mode in ("batch", "fixed"):
                for s in (0.5, 1.0):
                    gs = R[q & (R.sigma == s) & (R.noise_mode == mode)]
                    row[f"{mode}{s}"] = {m: boot_mean(gs[m], rng) for m in
                                         ["leak_r2", "content_r2", "content_share", "nuis_share", "emb_std_mean"]}
                    row[f"{mode}{s}"]["leak_minus_sigma0"] = boot_diff(gs.leak_r2, g0.leak_r2, rng)
            row["sigma0"] = {m: boot_mean(g0[m], rng) for m in
                             ["leak_r2", "content_r2", "content_share", "nuis_share", "emb_std_mean"]}
            for s in (0.5, 1.0):
                gb = R[q & (R.sigma == s) & (R.noise_mode == "batch")]
                gf = R[q & (R.sigma == s) & (R.noise_mode == "fixed")]
                row[f"fixed_minus_batch{s}"] = {m: boot_diff(gf[m], gb[m], rng) for m in ["leak_r2", "emb_std_mean"]}
            e4[f"{a}|{l}"] = row
    OUT["E4"] = e4

    # ---------------- E5 / P16, P17
    e5 = {}
    hue = lambda s: float(np.mean(json.loads(s)[:2]))
    scale = lambda s: float(json.loads(s)[2])
    R5 = R[R.blocks.str.contains("E5")].copy()
    R5["mcc_hue"] = R5.mcc_per.map(hue)
    R5["mcc_scale"] = R5.mcc_per.map(scale)
    for law, lm in [("uniform", {0.2: 0.2, 0.04: 0.04}), ("laplace", {0.2: 1.0, 0.04: 0.2})]:
        for lg, lmatch in lm.items():
            gG = R5[(R5.law == law) & (R5.arm == "ksd") & (R5.prior == "gaussian") & (R5.lam == lg)]
            gM = R5[(R5.law == law) & (R5.arm == "ksd") & (R5.prior == law) & (R5.lam == lmatch)]
            e5[f"{law}|gauss{lg}|matched{lmatch}"] = {
                m: dict(gauss=boot_mean(gG[m], rng), matched=boot_mean(gM[m], rng),
                        matched_minus_gauss=boot_diff(gM[m], gG[m], rng))
                for m in ["ident_orth_r2", "mcc", "mcc_hue", "mcc_scale", "leak_r2", "content_r2"]}
    e5["all_cells"] = {f"{r.law}|{r.arm}|{r.prior}|{r.lam}": dict(mcc=float(r.mcc), orth=float(r.ident_orth_r2))
                       for r in R5.groupby(["law", "arm", "prior", "lam"])[["mcc", "ident_orth_r2"]].mean()
                       .reset_index().itertuples()}
    OUT["E5"] = e5

    # ---------------- frontier: leakage at matched content R^2 (along each arm's path after its
    # content maximum), alpha 1 d 12
    fr = {}
    for a in arms:
        g = base(R, arm=a)
        lams, C = mat(g, "content_r2")
        _, L = mat(g, "leak_r2")

        def at(Cb, Lb, c):
            cm, lm = Cb.mean(0), Lb.mean(0)
            s = int(np.argmax(cm))
            if cm[s] < c:
                return np.nan
            for i in range(s + 1, len(lams)):
                if cm[i] <= c:
                    t = (cm[i - 1] - c) / (cm[i - 1] - cm[i])
                    return float(lm[i - 1] + t * (lm[i] - lm[i - 1]))
            return np.nan
        row = {"max_content": float(C.mean(0).max()), "lam_at_max_content": float(lams[int(np.argmax(C.mean(0)))]),
               "leak_at_max_content": float(L.mean(0)[int(np.argmax(C.mean(0)))])}
        for c in (0.5, 0.4):
            b = []
            for _ in range(N_BOOT):
                ix = rng.integers(0, len(C), len(C))
                b.append(at(C[ix], L[ix], c))
            row[f"leak_at_content_{c}"] = dict(value=at(C, L, c), ci=ci(b), frac_undefined=float(np.mean(np.isnan(b))))
        fr[a] = row
    OUT["frontier_matched"] = fr

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else round(float(o), 5)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    json.dump(clean(OUT), open("results2/verdicts.json", "w"), indent=1)
    print("wrote results2/verdicts.json")


if __name__ == "__main__":
    main()
