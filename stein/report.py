"""P3 report builder: figures, tables and in-text numbers for the paper and the blog.

    .venv/bin/python -m stein.report            # everything (needs data/processed for fig 1)
    .venv/bin/python -m stein.report --no-views # skip the image-panel figure

Reads only results/*.parquet, results/tuning_selected.json and (for figure 1) the processed
3DShapes array. Writes
    paper/figures/fig*.pdf|svg       vector figures for the paper
    paper/_generated/numbers.yml     every number quoted in the paper text ({{< meta n.KEY >}})
    paper/_generated/tab_*.md        tables included by the paper
    blog/data/*.csv, blog/data/numbers.yml, blog/figures/*.svg   precomputed blog inputs
No number in the paper or the blog is typed by hand: they all pass through this file.
"""
import argparse, json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
PFIG, PGEN = "paper/figures", "paper/_generated"
BDATA, BFIG = "blog/data", "blog/figures"
K = 6
NAMES = dict(none="none", vicreg="VICReg", sigreg="SIGReg", sksd="sliced KSD", ksd="KSD", mmd="MMD",
             ema="EMA", oracle="oracle", betavae="β-VAE")
TNAMES = {"ema": "EMA", "ema+sksd": "EMA + sliced KSD", "ema+sigreg": "EMA + SIGReg", "sksd": "sliced KSD",
          "vicreg": "VICReg"}
VNAMES = {"betavae": "β-VAE", "ksd": "KSD-AE", "mmd": "MMD-AE"}
COL = dict(vicreg="#1b9e77", sigreg="#d95f02", sksd="#7570b3", ksd="#e7298a", mmd="#66a61e", ema="#000000",
           oracle="#a6761d", none="#999999")
MARK = dict(vicreg="s", sigreg="^", sksd="v", ksd="o", mmd="D", ema="X", oracle="P", none=".")
ARMS8 = ["vicreg", "sigreg", "sksd", "ksd", "mmd", "ema", "oracle"]
ARMS6 = ["ema", "vicreg", "sigreg", "sksd", "ksd", "mmd"]
MINUS = "−"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


# ----------------------------------------------------------------------------- formatting

def f(x, nd=3):
    if x is None or not np.isfinite(x):
        return "n/a"
    s = f"{x:.{nd}f}"
    if s.startswith("-"):
        s = MINUS + s[1:]
    if float(s.replace(MINUS, "-")) == 0:
        s = s.replace(MINUS, "")
    return s


def fs(x, nd=3):
    """Signed (explicit + for positive) for contrasts."""
    s = f(x, nd)
    return s if s.startswith(MINUS) or s == "n/a" or float(s) == 0 else "+" + s


def ci(m, lo, hi, nd=3, signed=False):
    g = fs if signed else f
    return f"{g(m, nd)} [{f(lo, nd)}, {f(hi, nd)}]"


def slug(s):
    s = s.lower().replace("α", "a").replace("σ", "s").replace("λ", "l").replace("+", "p")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def num_slug(x):
    return f"{x:g}".replace(".", "")


# ----------------------------------------------------------------------------- loading

def load():
    S = pd.read_parquet(f"{R}/summary.parquet")
    C = pd.read_parquet(f"{R}/contrasts.parquet")
    L = pd.read_parquet(f"{R}/lam_sens_summary.parquet")
    H = pd.read_parquet(f"{R}/heldout.parquet")
    LR = pd.read_parquet(f"{R}/lam_sens.parquet")
    TU = pd.read_parquet(f"{R}/tuning.parquet")
    SEL = json.load(open(f"{R}/tuning_selected.json"))
    return S, C, L, H, LR, TU, SEL


def row_key(r):
    """Short key of a held-out config: tier_arm plus every field that differs from the base."""
    k = f"{r.tier}_{r.arm}".replace("+", "_")
    if r.d != 12: k += f"_d{r.d}"
    if r.sigma != 0: k += f"_s{num_slug(r.sigma)}"
    if r.alpha != 1: k += f"_a{num_slug(r.alpha)}"
    if r.law != "uniform": k += f"_law{r.law}"
    if r.prior != "gaussian": k += f"_prior{r.prior}"
    return k


def sel(S, **kw):
    q = S
    for k, v in kw.items():
        q = q[q[k] == v]
    return q


def one(S, **kw):
    q = sel(S, **kw)
    assert len(q) == 1, (kw, len(q))
    return q.iloc[0]


# ----------------------------------------------------------------------------- numbers

MET = dict(content_r2=3, content_knn_r2=3, leak_r2=3, leak_knn_r2=3, ident_orth_r2=3, shape_acc=3, class_acc=3,
           eff_rank=1, emb_std_mean=3, npksd_prior=2, gauss_ep_raw=1, loss_cv_last20=2)


def numbers(S, C, L, H, LR, TU, SEL):
    N = {}
    # every held-out config x metric: mean and "mean [lo, hi]"
    keys = set()
    for r in S.itertuples():
        k = row_key(r)
        assert k not in keys, k
        keys.add(k)
        for m, nd in MET.items():
            v = getattr(r, m)
            if np.isfinite(v):
                N[f"{k}_{m}"] = f(v, nd)
                N[f"{k}_{m}_ci"] = ci(v, getattr(r, m + "_lo"), getattr(r, m + "_hi"), nd)
        N[f"{k}_collapsed"] = f"{int(round(r.collapsed_frac * r.n_seeds))}/{r.n_seeds}"
        N[f"{k}_lam"] = f"{r.lam:g}"
    # lambda sensitivity
    for r in L.itertuples():
        k = f"lam_{r.arm}_l{num_slug(r.lam)}"
        for m in ("leak_r2", "content_r2", "leak_knn_r2", "ident_orth_r2"):
            N[f"{k}_{m}"] = f(getattr(r, m))
            N[f"{k}_{m}_ci"] = ci(getattr(r, m), getattr(r, m + "_lo"), getattr(r, m + "_hi"))
    # contrasts
    for r in C.itertuples():
        k = "c_" + slug(f"{r.hyp} {r.label} {r.metric}")
        assert k not in N, k
        N[k] = ci(r.est, r.lo, r.hi, signed=True)
        N[k + "_est"] = fs(r.est)
    # counts
    N["n_heldout_runs"] = str(len(H)); N["n_heldout_configs"] = str(len(S))
    N["n_lam_runs"] = str(len(LR)); N["n_lam_configs"] = str(len(L))
    N["n_lam_new_runs"] = str(int((~LR.hash.isin(H.hash)).sum()))
    N["n_tuning_runs"] = str(len(TU)); N["n_contrasts"] = str(len(C))
    N["n_failed"] = str(int((H.loss_nonfinite > 0).sum()))
    N["n_seeds"] = str(H.seed.nunique())
    allruns = pd.concat([H, LR[~LR.hash.isin(H.hash)], TU])
    N["n_total_runs"] = str(len(allruns))
    N["run_hours"] = f"{(allruns.train_time_s + allruns.eval_time_s).sum() / 3600:.1f}"
    for t in ["shapes", "temporal", "cifar", "vae"]:
        q = H[H.tier == t]
        N[f"time_{t}_median_s"] = f"{q.train_time_s.median():.0f}"
    N["time_sksd_median_s"] = f"{H[(H.tier == 'shapes') & (H.arm == 'sksd')].train_time_s.median():.0f}"
    # P3 reproduction check (tools/repro_h5.py): H5 block retrained from scratch in a scratch run cache
    rp = f"{R}/repro/h5.parquet"
    if os.path.exists(rp):
        RP = pd.read_parquet(rp).set_index("hash")
        ms = ["content_r2", "leak_r2", "leak_knn_r2", "loss_final"]
        dd = (RP[ms] - H.set_index("hash").loc[RP.index, ms]).abs()
        N["repro_n"] = str(len(RP)); N["repro_bitwise"] = str(int((dd.max(1) == 0).sum()))
        N["repro_maxdiff"] = f"{dd.values.max():.1e}"
    # contrast sign counts used in the text
    h1 = C[(C.hyp == "H1") & (C.metric == "leak_r2")]
    vic = h1[h1.label.str.endswith("VICReg") & h1.label.str.match(r"d=(12|24):")]
    N["h1_vicreg_neg"] = str(int((vic.sign == MINUS).sum())); N["h1_vicreg_tot"] = str(len(vic))
    h6 = C[(C.hyp == "H6") & C.label.str.contains("JEPA")]
    for m in ("leak_r2", "leak_knn_r2"):
        q = h6[h6.metric == m]
        N[f"h6_pos_{m}"] = str(int((q.sign == "+").sum())); N[f"h6_tot_{m}"] = str(len(q))
    lamc = C[(C.hyp == "H1-lam") & C.label.str.contains(r" − λ=") & (C.metric == "leak_r2")]
    N["lam_pos"] = str(int((lamc.sign == "+").sum())); N["lam_tot"] = str(len(lamc))
    # tuning levels
    for k, v in SEL.items():
        N["sel_" + slug(k)] = f"{v['level'].get('lam', 0):g}" if "ema" not in k.split("/")[1] or "+" in k \
            else f"{v['level'].get('ema_tau'):g}"
    # collapse
    base_none = one(S, tier="shapes", arm="none", d=12, sigma=0.0, alpha=1.0, law="uniform", prior="gaussian")
    N["none_collapsed_frac"] = f"{(S[S.arm == 'none'].collapsed_frac == 1).mean() * 100:.0f}%"
    N["base_none_emb_std"] = f(base_none.emb_std_mean, 4)
    # validation numbers (P1, results/validation.json)
    V = json.load(open(f"{R}/validation.json"))
    sp, pr = V["sampler"], V["probes"]
    N["val_a1_maxcorr"] = f(sp["alpha=1.0"]["max_abs_cross_corr"], 4)
    N["val_a05_maxcorr"] = f(sp["alpha=0.5"]["max_abs_cross_corr"], 2)
    N["val_cifar_maxcorr"] = f(sp["cifar alpha=1 aug-param cross-view max |corr|"], 3)
    N["val_sigreg_reldiff"] = f(max(v["rel_diff"] for v in V["sigreg_vs_lejepa"].values()), 1)
    N["val_probe_content"] = f(pr["given true factors (rotated content+nuisance): content R2"])
    N["val_probe_leak_from_content"] = f(pr["given content only: leakage R2"], 4)
    N["val_orth_mix"] = f(pr["orth identifiability of random linear mix"], 2)
    reg = V["regularisers_d12_n256"]
    N["val_ksd_unif_exact"] = f(reg["ksd[uniform]"]["uniform sample (exact)"], 4)
    N["val_ksd_unif_gauss"] = f(reg["ksd[uniform]"]["N(0,I)"], 2)
    N["val_ksd_lap_exact"] = f(reg["ksd[laplace]"]["laplace sample (exact)"], 4)
    N["val_ksd_lap_gauss"] = f(reg["ksd[laplace]"]["N(0,I)"], 4)
    N["val_oracle_own"] = f(reg["oracle"]["oracle target N(0, diag(1_6, 0.09))"][0], 3)
    N["val_sigreg_null"] = f(reg["sigreg"]["N(0,I)"][0], 2)
    return N


# ----------------------------------------------------------------------------- tables

def md_table(header, rows, align=None):
    align = align or ["l"] + ["r"] * (len(header) - 1)
    # pandoc sizes wide pipe tables by the dash count of each column, so make it proportional to content
    w = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    sep = [(":" if x == "l" else "-") + "-" * max(w_ - 2, 1) + (":" if x == "r" else "-") for x, w_ in zip(align, w)]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def base(S, arm, tier="shapes", **kw):
    p = dict(tier=tier, arm=arm, d=12, sigma=0.0, alpha=1.0, law="uniform", prior="gaussian")
    p.update(kw)
    return one(S, **p)


def tables(S, C, L, SEL):
    T = {}
    rows = []
    for a in ["none"] + ARMS8:
        r = base(S, a)
        lam = "—" if a in ("none",) else (f"τ = {r.ema_tau:g}" if a == "ema" else f"{r.lam:g}")
        rows.append([NAMES[a], lam, ci(r.content_r2, r.content_r2_lo, r.content_r2_hi),
                     f(r.content_knn_r2), ci(r.leak_r2, r.leak_r2_lo, r.leak_r2_hi), f(r.leak_knn_r2),
                     ci(r.ident_orth_r2, r.ident_orth_r2_lo, r.ident_orth_r2_hi), f(r.eff_rank, 1),
                     f(r.emb_std_mean, 3)])
    T["tab_base"] = md_table(["arm", "λ", "content R²", "kNN cont.", "leakage R²", "kNN leak.", "ident.",
                              "eff. rank", "emb. sd"], rows)
    # temporal
    rows = []
    for a in ["ema", "ema+sksd", "ema+sigreg", "sksd", "vicreg"]:
        r = base(S, a, tier="temporal")
        lam = f"τ = {r.ema_tau:g}" if a == "ema" else (f"{r.lam:g} (τ = {r.ema_tau:g})" if "+" in a
                                                         else f"{r.lam:g}")
        rows.append([TNAMES[a], lam, ci(r.content_r2, r.content_r2_lo, r.content_r2_hi),
                     ci(r.leak_r2, r.leak_r2_lo, r.leak_r2_hi), ci(r.leak_knn_r2, r.leak_knn_r2_lo, r.leak_knn_r2_hi),
                     f(r.eff_rank, 1)])
    T["tab_temporal"] = md_table(["arm", "λ", "content R²", "leakage R²", "kNN leakage R²", "eff. rank"], rows)
    # CIFAR
    rows = []
    for a in ["none", "vicreg", "sigreg", "sksd", "ksd", "mmd", "ema"]:
        r = base(S, a, tier="cifar")
        lam = "—" if a == "none" else (f"τ = {r.ema_tau:g}" if a == "ema" else f"{r.lam:g}")
        rows.append([NAMES[a], lam, ci(r.class_acc, r.class_acc_lo, r.class_acc_hi),
                     ci(r.leak_r2, r.leak_r2_lo, r.leak_r2_hi), f(r.eff_rank, 1), f(r.emb_std_mean, 3)])
    T["tab_cifar"] = md_table(["arm", "λ", "class accuracy", "augmentation leakage R²", "eff. rank", "emb. sd"],
                              rows)
    # H3: identifiability by law
    rows = []
    for lab, a, pr in [("SIGReg", "sigreg", "gaussian"), ("sliced KSD → N(0, I)", "sksd", "gaussian"),
                       ("KSD → N(0, I)", "ksd", "gaussian"), ("MMD", "mmd", "gaussian"), ("oracle", "oracle", "gaussian"),
                       ("KSD → matched prior", "ksd", "matched"), ("sliced KSD → matched prior", "sksd", "matched")]:
        cells = [lab]
        for law in ["uniform", "gaussian", "laplace"]:
            p = law if pr == "matched" else pr
            if pr == "matched" and law == "gaussian":
                cells.append("(= N(0, I) row)"); continue
            q = sel(S, tier="shapes", arm=a, d=12, sigma=0.0, alpha=1.0, law=law, prior=p)
            cells.append(ci(q.iloc[0].ident_orth_r2, q.iloc[0].ident_orth_r2_lo, q.iloc[0].ident_orth_r2_hi)
                         if len(q) else "not run")
        rows.append(cells)
    T["tab_h3"] = md_table(["arm (target)", "uniform law", "Gaussian law", "Laplace law"], rows)
    # VAE tier
    rows = []
    for a, law, pr, lab in [("betavae", "uniform", "gaussian", "β-VAE (KL to N(0, I))"),
                            ("ksd", "uniform", "gaussian", "KSD-AE → N(0, I)"),
                            ("mmd", "uniform", "gaussian", "MMD-AE → N(0, I)"),
                            ("ksd", "laplace", "gaussian", "KSD-AE → N(0, I), Laplace law"),
                            ("ksd", "laplace", "laplace", "KSD-AE → Laplace prior, Laplace law")]:
        r = base(S, a, tier="vae", law=law, prior=pr)
        rows.append([lab, f"{r.lam:g}", ci(r.content_r2, r.content_r2_lo, r.content_r2_hi),
                     ci(r.leak_r2, r.leak_r2_lo, r.leak_r2_hi), f(r.leak_knn_r2),
                     ci(r.ident_orth_r2, r.ident_orth_r2_lo, r.ident_orth_r2_hi)])
    T["tab_vae"] = md_table(["model", "λ / β", "content R²", "leakage R²", "kNN leak.", "ident."], rows)
    # tuning
    order = ["shapes/vicreg", "shapes/sigreg", "shapes/sksd", "shapes/ksd", "shapes/mmd", "shapes/oracle", "shapes/ema"]
    rows = []
    for k in order:
        v = SEL[k]
        lvl = f"τ = {v['level']['ema_tau']:g}" if "ema" in k else f"{v['level']['lam']:g}"
        rows.append([NAMES[k.split("/")[1]], lvl, f(v["score"])])
    T["tab_tuning"] = md_table(["arm (3DShapes base)", "selected level", "validation content R²"], rows)
    # validation gate (P1): regulariser value on N(0, I) and on degenerate samples (mean of 5 draws)
    V = json.load(open(f"{R}/validation.json"))["regularisers_d12_n256"]
    cols = ["N(0,I)", "collapsed (1 + 0.01 N)", "rank-1", "anisotropic (sd 3/1/0.1)", "shifted N(0.5,I)"]
    rows = [[NAMES[a]] + [f(V[a][c][0], 3 if abs(V[a][c][0]) < 10 else 1) for c in cols]
            for a in ["vicreg", "sigreg", "sksd", "ksd", "mmd"]]
    T["tab_validation"] = md_table(["regulariser", "N(0, I)", "collapsed", "rank 1", "anisotropic", "shifted mean"],
                                   rows)
    # verdicts are words; the evidence numbers come from the contrasts
    return T


# ----------------------------------------------------------------------------- figures

def save(fig, name):
    os.makedirs(PFIG, exist_ok=True); os.makedirs(BFIG, exist_ok=True)
    fig.savefig(f"{PFIG}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{PFIG}/{name}.svg", bbox_inches="tight")
    fig.savefig(f"{BFIG}/{name}.svg", bbox_inches="tight")
    plt.close(fig)


def bottom_legend(fig, ax, ncol=8):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=6.8)


def err(ax, x, r, m, a, label=True, **kw):
    y = np.array([getattr(q, m) for q in r]); lo = np.array([getattr(q, m + "_lo") for q in r])
    hi = np.array([getattr(q, m + "_hi") for q in r])
    ax.errorbar(x, y, yerr=[y - lo, hi - y], color=COL.get(a, "k"), marker=MARK.get(a, "o"), ms=4, lw=1.1,
                capsize=2, label=NAMES.get(a, a) if label else None, **kw)


def fig_views(seed=7):
    from stein.data import Shapes
    import torch; torch.manual_seed(seed)  # images_for draws pixel noise from the global torch RNG
    sh = Shapes(law="uniform", seed=seed)
    alphas = [0.0, 0.5, 1.0]
    npair = 4
    fig, axes = plt.subplots(len(alphas), 2 * npair, figsize=(6.6, 2.75))
    for i, a in enumerate(alphas):
        f1, f2 = sh.sample_pair_factors(npair, a)
        x1 = sh.images_for(f1, noise=True) + 0.5; x2 = sh.images_for(f2, noise=True) + 0.5
        for j in range(npair):
            for v, x in enumerate((x1, x2)):
                ax = axes[i, 2 * j + v]
                ax.imshow(x[j].permute(1, 2, 0).clamp(0, 1).numpy(), interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if i == 0:
                    ax.set_title(f"pair {j + 1}: view {v + 1}" if v == 0 else "view 2", fontsize=6)
        axes[i, 0].set_ylabel(f"α = {a:g}", fontsize=8)
        axes[i, 0].yaxis.set_visible(True)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    save(fig, "fig1_views")


def fig_h1(S):
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.3))
    ds = [3, 6, 12, 24]
    for a in ARMS8:
        r = [base(S, a, d=d) for d in ds]
        jit = (ARMS8.index(a) - 3) * 0.025
        x = np.array(ds) * 2 ** jit
        for ax, m in zip(axes, ["leak_r2", "leak_knn_r2", "content_r2"]):
            err(ax, x, r, m, a)
    for ax, t in zip(axes, ["linear leakage R² (↓)", "kNN leakage R² (↓)", "linear content R² (↑)"]):
        ax.set_xscale("log", base=2); ax.set_xticks(ds); ax.set_xticklabels([str(d) for d in ds])
        ax.axvline(K, color="0.6", ls=":", lw=0.8); ax.set_xlabel("embedding dim d (k = 6 dotted)"); ax.set_title(t)
    fig.tight_layout(); bottom_legend(fig, axes[0])
    save(fig, "fig2_h1_dim")


def fig_lam(L, S):
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.3))
    ema = base(S, "ema")
    for a in ["vicreg", "sigreg", "sksd", "ksd", "mmd", "oracle"]:
        q = L[L.arm == a].sort_values("lam")
        lam_sel = base(S, a).lam
        x = q.lam.values / lam_sel * 2 ** ((["vicreg", "sigreg", "sksd", "ksd", "mmd", "oracle"].index(a) - 2.5) * 0.04)
        for ax, m in zip(axes, ["leak_r2", "content_r2"]):
            err(ax, x, list(q.itertuples()), m, a)
    for ax, m, t in zip(axes, ["leak_r2", "content_r2"], ["linear leakage R² (↓)", "linear content R² (↑)"]):
        ax.axhspan(getattr(ema, m + "_lo"), getattr(ema, m + "_hi"), color="0.85", lw=0, label="EMA (95% CI)")
        ax.set_xscale("log"); ax.set_xticks([0.2, 1, 5, 25]); ax.set_xticklabels(["1/5", "1", "5", "25"])
        ax.axvline(1, color="0.6", ls=":", lw=0.8)
        ax.set_xlabel("λ / tuned λ"); ax.set_title(t)
    fig.tight_layout(); bottom_legend(fig, axes[0], ncol=7)
    save(fig, "fig3_lambda")


def fig_h2(S):
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.1), sharey=False)
    sig = [0.0, 0.5, 1.0]
    for j, d in enumerate([12, 24]):
        for a in ARMS6:
            r = [base(S, a, d=d, sigma=s) for s in sig]
            x = np.array(sig) + (ARMS6.index(a) - 2.5) * 0.02
            err(axes[j], x, r, "leak_r2", a); err(axes[2 + j], x, r, "content_r2", a)
        axes[j].set_title(f"leakage R² (↓), d = {d}"); axes[2 + j].set_title(f"content R² (↑), d = {d}")
    for ax in axes:
        ax.set_xticks(sig); ax.set_xlabel("noise σ")
    for ax in axes[:2]:
        ax.set_ylim(-0.05, 0.95)
    for ax in axes[2:]:
        ax.set_ylim(-0.05, 0.85)
    fig.tight_layout(); bottom_legend(fig, axes[0])
    save(fig, "fig4_h2_sigma")


def fig_h3(S):
    fig, ax = plt.subplots(figsize=(6.0, 2.3))
    laws = ["uniform", "gaussian", "laplace"]
    spec = [("sigreg", "gaussian", "SIGReg"), ("sksd", "gaussian", "sliced KSD → N(0,I)"),
            ("ksd", "gaussian", "KSD → N(0,I)"), ("mmd", "gaussian", "MMD"), ("oracle", "gaussian", "oracle"),
            ("ksd", "matched", "KSD → matched prior"), ("sksd", "matched", "sliced KSD → matched prior")]
    for i, (a, pr, lab) in enumerate(spec):
        xs, rs = [], []
        for j, law in enumerate(laws):
            p = law if pr == "matched" else pr
            if pr == "matched" and law == "gaussian":
                continue
            q = sel(S, tier="shapes", arm=a, d=12, sigma=0.0, alpha=1.0, law=law, prior=p)
            if len(q):
                xs.append(j + (i - 3) * 0.1); rs.append(q.iloc[0])
        y = np.array([r.ident_orth_r2 for r in rs])
        lo = np.array([r.ident_orth_r2_lo for r in rs]); hi = np.array([r.ident_orth_r2_hi for r in rs])
        matched = pr == "matched"
        ax.errorbar(xs, y, yerr=[y - lo, hi - y], color=COL[a], marker=MARK[a], ms=5, lw=0, elinewidth=1.1,
                    capsize=2, mfc="white" if matched else COL[a], label=lab)
    ax.set_xticks(range(3)); ax.set_xticklabels(["uniform law", "Gaussian law", "Laplace law"])
    ax.set_ylabel("identifiability (orth. R²)"); ax.set_xlim(-0.5, 2.5)
    ax.legend(frameon=False, fontsize=6.3, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    save(fig, "fig5_h3_law")


def fig_h4(S):
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.2))
    al = [0.0, 0.5, 1.0]
    for a in ARMS6:
        r = [base(S, a, alpha=x) for x in al]
        x = np.array(al) + (ARMS6.index(a) - 2.5) * 0.015
        for ax, m in zip(axes, ["leak_r2", "leak_knn_r2", "content_r2"]):
            err(ax, x, r, m, a)
    for ax, t in zip(axes, ["linear leakage R² (↓)", "kNN leakage R² (↓)", "linear content R² (↑)"]):
        ax.set_xticks(al); ax.set_xlabel("nuisance strength α"); ax.set_title(t)
    fig.tight_layout(); bottom_legend(fig, axes[0])
    save(fig, "fig6_h4_alpha")


def fig_h5h6(S):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), gridspec_kw=dict(width_ratios=[5, 10]))
    ax = axes[0]
    tarms = ["ema", "ema+sksd", "ema+sigreg", "sksd", "vicreg"]
    for i, a in enumerate(tarms):
        r = base(S, a, tier="temporal")
        for m, off, mk in [("leak_r2", -0.15, "o"), ("leak_knn_r2", 0.15, "s")]:
            y = getattr(r, m)
            ax.errorbar([i + off], [y], yerr=[[y - getattr(r, m + "_lo")], [getattr(r, m + "_hi") - y]],
                        color="k" if m == "leak_r2" else "0.5", marker=mk, ms=4, capsize=2, lw=1,
                        label=("linear" if m == "leak_r2" else "kNN") if i == 0 else None)
    ax.set_xticks(range(len(tarms))); ax.set_xticklabels([TNAMES[a].replace(" + ", "+\n") for a in tarms], fontsize=6.3)
    ax.set_title("temporal tier: leakage R² (↓)"); ax.legend(frameon=False, fontsize=6.3); ax.set_ylim(0, 1)
    ax = axes[1]
    items = [(NAMES[a], base(S, a), COL[a]) for a in ARMS8] + \
            [(VNAMES[a], base(S, a, tier="vae"), "#444444") for a in ["betavae", "ksd", "mmd"]]
    for i, (lab, r, c) in enumerate(items):
        for m, off, mk in [("leak_r2", -0.15, "o"), ("leak_knn_r2", 0.15, "s")]:
            y = getattr(r, m)
            ax.errorbar([i + off], [y], yerr=[[y - getattr(r, m + "_lo")], [getattr(r, m + "_hi") - y]],
                        color=c, marker=mk, ms=4, capsize=2, lw=1, mfc=c if m == "leak_r2" else "white")
    ax.axvline(len(ARMS8) - 0.5, color="0.6", ls=":", lw=0.8)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([i[0] for i in items], fontsize=6.3, rotation=35, ha="right")
    ax.text(3, 0.97, "JEPA arms", ha="center", fontsize=7); ax.text(len(ARMS8) + 1, 0.2, "autoencoders", ha="center", fontsize=7)
    ax.set_title("3DShapes, d = 12: JEPA arms vs autoencoders, leakage R² (● linear, ○ kNN)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    save(fig, "fig7_h5_h6")


# ----------------------------------------------------------------------------- blog data

def blog_data(S, L, N):
    os.makedirs(BDATA, exist_ok=True)
    cols = ["tier", "arm", "d", "sigma", "alpha", "law", "prior", "lam", "ema_tau", "n_seeds", "collapsed_frac"]
    mets = ["content_r2", "content_knn_r2", "leak_r2", "leak_knn_r2", "ident_orth_r2", "class_acc", "eff_rank",
            "emb_std_mean"]
    out = S[cols + [m + s for m in mets for s in ("", "_lo", "_hi")]].copy()
    out["arm_name"] = [TNAMES.get(a, NAMES.get(a, a)) if t != "vae" else VNAMES[a] for t, a in zip(out.tier, out.arm)]
    out.round(4).to_csv(f"{BDATA}/summary.csv", index=False)
    lo = L[["arm", "lam"] + [m + s for m in ["leak_r2", "content_r2"] for s in ("", "_lo", "_hi")]].copy()
    lo["lam_rel"] = [r.lam / base(S, r.arm).lam for r in lo.itertuples()]
    lo["arm_name"] = lo.arm.map(NAMES)
    lo.round(4).to_csv(f"{BDATA}/lam_sens.csv", index=False)
    write_yaml(f"{BDATA}/numbers.yml", N)


def write_yaml(path, N):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("# generated by python -m stein.report; do not edit\nn:\n")
        for k in sorted(N):
            fh.write(f"  {k}: {json.dumps(N[k], ensure_ascii=False)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-views", action="store_true")
    args = ap.parse_args()
    S, C, L, H, LR, TU, SEL = load()
    N = numbers(S, C, L, H, LR, TU, SEL)
    os.makedirs(PGEN, exist_ok=True)
    write_yaml(f"{PGEN}/numbers.yml", N)
    for k, v in tables(S, C, L, SEL).items():
        open(f"{PGEN}/{k}.md", "w").write(v)
    if not args.no_views:
        fig_views()
    fig_h1(S); fig_lam(L, S); fig_h2(S); fig_h3(S); fig_h4(S); fig_h5h6(S)
    blog_data(S, L, N)
    print(f"numbers: {len(N)} keys; figures in {PFIG}; tables in {PGEN}; blog data in {BDATA}")


if __name__ == "__main__":
    main()
