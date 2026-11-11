"""Study-2 report builder: figures, tables and in-text numbers for paper2/paper.qmd.

    .venv/bin/python -W ignore -m stein.report2

Reads only results2/*.parquet, results2/verdicts.json, results2/theory_t*.json,
results2/validation2.json and (for the motivating study-1 numbers) results/lam_sens_summary.parquet.
Writes
    paper2/figures/fig*.pdf|svg      vector figures
    paper2/_generated/numbers.yml    every number quoted in the paper text ({{< meta n.KEY >}})
    paper2/_generated/tab_*.md       tables included by the paper
No number in the paper is typed by hand: they all pass through this file. The verdict words in
tab_verdicts.md are the P2 verdicts recorded in study2/gates/GATE_2.md; their evidence columns are
generated from results2/verdicts.json.
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R2 = "results2"
FIG, GEN = "paper2/figures", "paper2/_generated"
K, M = 6, 5
ARMS = ["cov", "vicreg", "sigreg", "sksd", "ksd", "mmd"]
KERN = ["sigreg", "sksd", "ksd", "mmd"]
NAMES = dict(cov="cov", vicreg="VICReg", sigreg="SIGReg", sksd="sliced KSD", ksd="KSD", mmd="MMD")
COL = dict(cov="#444444", vicreg="#1b9e77", sigreg="#d95f02", sksd="#7570b3", ksd="#e7298a", mmd="#66a61e")
MARK = dict(cov="P", vicreg="s", sigreg="^", sksd="v", ksd="o", mmd="D")
MINUS = "−"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


# ----------------------------------------------------------------------------- formatting

def f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    s = f"{x:.{nd}f}"
    if s.startswith("-"):
        s = MINUS + s[1:]
    if float(s.replace(MINUS, "-")) == 0:
        s = s.replace(MINUS, "")
    return s


def fs(x, nd=3):
    s = f(x, nd)
    return s if s.startswith(MINUS) or s == "n/a" or float(s) == 0 else "+" + s


def g(x, sig=2):
    """Significant-figure format (for lambdas and ratios spanning decades)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    x = float(x)
    if x == 0:
        return "0"
    nd = max(0, sig - 1 - int(np.floor(np.log10(abs(x)))))
    s = f"{x:.{nd}f}"
    return (MINUS + s[1:]) if s.startswith("-") else s


def sci(x, nd=0):
    """Scientific notation in pandoc markdown (superscript ^...^)."""
    m, e = f"{x:.{nd}e}".split("e")
    e = int(e)
    return f"{m}×10^{MINUS if e < 0 else ''}{abs(e)}^"


def ci(m, lo, hi, nd=3, signed=False, fmt=None):
    if fmt is not None:
        return f"{fmt(m)} [{fmt(lo)}, {fmt(hi)}]"
    return f"{(fs if signed else f)(m, nd)} [{f(lo, nd)}, {f(hi, nd)}]"


def lamstr(val, censor):
    return f"{censor} {g(val, 2)}" if censor else g(val, 3)


N = {}


def put(key, val):
    N[key] = val


# ----------------------------------------------------------------------------- loading

def load():
    C = pd.read_parquet(f"{R2}/cells2.parquet")
    T = pd.read_parquet(f"{R2}/thresholds.parquet")
    F = pd.read_parquet(f"{R2}/frontier.parquet")
    RUNS = pd.read_parquet(f"{R2}/runs2.parquet")
    V = json.load(open(f"{R2}/verdicts.json"))
    TH = {t: json.load(open(f"{R2}/theory_{t}.json")) for t in ("t1", "t2", "t3", "t4")}
    VAL = json.load(open(f"{R2}/validation2.json"))
    return C, T, F, RUNS, V, TH, VAL


def path(C, arm, d=12, alpha=1.0, encoder="conv"):
    on = C.blocks.str.contains(r"E1_|E2_|E6_", regex=True)
    s = C[on & (C.arm == arm) & (C.d == d) & (C.alpha == alpha) & (C.encoder == encoder) & (C.sigma == 0)
          & (C.law == "uniform") & (C.prior == "gaussian") & (C.steps == 2000)]
    return s.sort_values("lam")


def theory_lam(TH, arm, d=12, alpha=1.0):
    if arm == "vicreg":
        return 7.0 if (d == 12 and alpha == 1.0) else None
    t = TH["t2"]["thresholds"].get(f"{arm}_d{d}")
    if not t or not t.get("lam_star_code"):
        return None
    return t["lam_star_code"].get(str(float(alpha)))


def thr(T, arm, d=12, alpha=1.0, encoder="conv", metric="leak_r2", upward=False):
    s = T[(T.arm == arm) & (T.d == d) & (T.alpha == alpha) & (T.encoder == encoder) & (T.metric == metric)
          & (T.upward == upward) & (T.sigma == 0)]
    return None if s.empty else s.iloc[0]


# ----------------------------------------------------------------------------- numbers

def numbers(C, T, F, RUNS, V, TH, VAL):
    put("n_runs", str(len(RUNS)))
    put("n_cells", str(len(C)))
    put("n_seeds", str(int(C.n_seeds.max())))
    # study 1 motivation (results/, frozen)
    L = pd.read_parquet("results/lam_sens_summary.parquet")
    for lam in (0.2, 1.0, 5.0):
        r = L[(L.arm == "ksd") & (L.lam == lam) & (L.d == 12)].iloc[0]
        put(f"s1_ksd_leak_l{str(lam).replace('.', '')}", f(r.leak_r2))
    # theory constants
    t2 = TH["t2"]["thresholds"]
    for arm in KERN + ["cov"]:
        for d in (12, 24):
            lt = theory_lam(TH, arm, d, 1.0)
            put(f"th_{arm}_d{d}", g(lt, 3))
        for a in (0.5, 0.25):
            put(f"th_{arm}_a{str(a).replace('.', '')}", g(theory_lam(TH, arm, 12, a), 3))
        tt = t2[f"{arm}_d12"]
        put(f"tq_{arm}", g(tt["t_star"] / tt["t_quad"], 2))
        put(f"tloc_{arm}", f"{abs(tt['t_star'] / tt['t_loc'] - 1) * 100:.1f}")
    put("tloc_max", f"{max(abs(t2[f'{a}_d12']['t_star'] / t2[f'{a}_d12']['t_loc'] - 1) for a in KERN + ['cov']) * 100:.0f}")
    put("tq_ksd_d24", g(t2["ksd_d24"]["t_star"] / t2["ksd_d24"]["t_quad"], 2))
    put("cov_tstar", f(t2["cov_d12"]["t_star"], 3))
    kp = {p["lam_over_star"]: p["v_mean"] for p in t2["ksd_d12"]["path"]}
    put("th_ksd_v2", f(kp[2], 2))
    for arm in ("sigreg", "mmd"):
        pp = {p["lam_over_star"]: p["v_mean"] for p in t2[f"{arm}_d12"]["path"]}
        put(f"th_{arm}_v2", f(pp[2], 2))
    # T1 checks
    t1 = TH["t1"]
    b2 = t1["B2_joint_W"]
    put("traps_joint", f"{b2['n'] - b2['n_global']}/{b2['n']}")
    put("traps_fast", f"{t1['B_traps']}/{len(t1['B_learned_linear'])}")
    dn = t1["D_nonlinear"]
    put("nl_J_max", sci(max(r["J_nonlinear"] for r in dn)))
    put("nl_Jlin_min", g(min(r["J_linear_min"] for r in dn), 2))
    put("t1_tol", str(t1["tolerance"]["pop"]))
    # T3
    t3 = {(r["reg"], r["alpha"], r["mode"], r["sigma"]): r["lam_star"] for r in TH["t3"]["part1"]}
    for s in (0.0, 0.5, 1.0):
        put(f"t3_ksd_b{str(s).replace('.', '')}", g(t3[("ksd", 1.0, "batch", s)], 2))
    put("t3_ksd_f1", g(t3[("ksd", 1.0, "fixed", 1.0)], 1))
    put("t3_ksd_f05", g(t3[("ksd", 1.0, "fixed", 0.5)], 2))
    put("t3_cov_b1", g(t3[("cov", 1.0, "batch", 1.0)], 3))
    # T4 checks
    t4 = TH["t4"]
    inv = {r["reg"]: r for r in t4["invariance"]}
    put("t4_z_max", f(max(abs(inv[k]["z"]) for k in ("sksd_gauss", "sigreg", "mmd_gauss")), 1))
    put("t4_ksd_rel", sci(abs(inv["ksd_gauss"]["diff"] / inv["ksd_gauss"]["mean"])))
    put("t4_z_uni", f(inv["ksd_uniform_prior"]["z"], 1))
    mm = t4["matched_mcc"]
    put("t4_mcc_uni", f"{f(min(mm[f'indep_uniform{i}'] for i in range(3)), 2)}–{f(max(mm[f'indep_uniform{i}'] for i in range(3)), 2)}")
    put("t4_mcc_lap", "/".join(f(mm[f"indep_laplace{i}"], 2) for i in range(3)))
    gm = t4["summary"]["gauss_target_mcc"]
    put("t4_mcc_gauss", f"{f(min(gm), 2)}–{f(max(gm), 2)}")
    fu = [r for r in t4["fits"] if r["kind"] == "indep_uniform" and r["prior"] == "uniform"]
    put("t4_ksd_fit", f(np.mean([r["ksd_fit"] for r in fu]), 3))
    put("t4_ksd_id", f(np.mean([r["ksd_identity"] for r in fu]), 4))
    put("t4_orth_min", f(min(r["orth_r2"] for r in t4["fits"]), 2))
    cp = t4["summary"]["circle_pair_mcc"]
    put("t4_circle", f"{f(min(cp), 2)}–{f(max(cp), 2)}")
    vs = TH["t2"]["vicreg_scan"]
    put("vic_none_upto", g(max(r["lam"] for r in vs if r["nuis_share"] < 1e-6 and r["lam"] < 10), 2))
    put("vic_share10", f([r for r in vs if abs(r["lam"] - 10) < 1e-9][0]["nuis_share"], 2))
    # theory-meets-code (real training loop, synthetic views)
    A = VAL["A"]
    rows = A["rows"]
    for q in (2.0, 5.0):
        rr = [r for r in rows if abs(r["q"] - q) < 1e-9]
        put(f"val_v_q{int(q)}", "/".join(f(r["v"], 3) for r in rr))
        put(f"val_vref_q{int(q)}", f(rr[0]["v_ref"], 3))
    below = [r for r in rows if r["q"] < 1]
    put("val_leak_below_lo", f(min(r["leak_r2"] for r in below), 2))
    put("val_leak_below_hi", f(max(r["leak_r2"] for r in below), 2))
    put("val_v_below_max", sci(max(r["v"] for r in below), 1))
    put("val_nlearn", f"{A['n_pass_mean']}/{len(rows)}")
    # thresholds: E1 α = 1, d = 12
    P2 = V["P2"]
    for arm in ARMS:
        r = thr(T, arm)
        put(f"lh_{arm}", lamstr(r.lam_hat, r.censor))
        put(f"lhci_{arm}", "–" if r.censor else f"[{g(r.lo, 3)}, {g(r.hi, 3)}]")
        if P2[arm]["ratio"] is not None:
            put(f"ratio_{arm}", f(P2[arm]["ratio"], 2))
        for a in (0.5, 0.25):
            x = V["P4"][arm][str(a)]
            put(f"ar_{arm}_{str(a).replace('.', '')}",
                ci(x["ratio"], *x["ci"], fmt=lambda v: f(v, 2)) if x["ratio"] is not None else "censored")
        x = V["P6"][arm]
        put(f"d24_{arm}", (f"{f(x['ratio'], 2)} [{'censored' if x['ci'][0] <= 0 else f(x['ci'][0], 2)}, "
                           f"{f(x['ci'][1], 2)}]") if x["ratio"] is not None else "censored")
        r6 = thr(T, arm, d=6)
        put(f"lh6_{arm}", lamstr(r6.lam_hat, r6.censor))
        put(f"max6_{arm}", f(V["P5"][arm]["max_d6"], 2))
        p1 = V["P1"][arm]
        put(f"p1_low_{arm}", f(p1["leak_lowest"], 3))
        put(f"p1_max_{arm}", f(p1["max"], 2))
        if p1["decades"] is not None:
            put(f"p1_dec_{arm}", f(p1["decades"], 2))
        p7 = V["P7"][arm]
        put(f"p7_leak_{arm}", f(p7["width_leak"], 2))
        put(f"p7_share_{arm}", f(p7["width_share"], 2))
        rs = thr(T, arm, metric="nuis_share")
        put(f"lhs_{arm}", lamstr(rs.lam_hat, rs.censor))
    kb = pd.concat([path(C, a) for a in KERN])
    kb = kb[(kb.leak_r2 < 0.05) & (kb.leak_r2 > -0.5)]
    put("knn_below_lo", f(kb.leak_knn_r2.min(), 2))
    put("knn_below_hi", f(kb.leak_knn_r2.max(), 2))
    put("max6_lo", f(min(V["P5"][a]["max_d6"] for a in ARMS), 2))
    put("max6_hi", f(max(V["P5"][a]["max_d6"] for a in ARMS), 2))
    kr = [V["P2"][a]["ratio"] for a in KERN]
    put("ratio_lo", f(min(kr), 2))
    put("ratio_hi", f(max(kr), 2))
    a25 = [V["P4"][a]["0.25"] for a in KERN]
    put("ar025_lo", f(min(x["ratio"] for x in a25), 2))
    put("ar025_hi", f(max(x["ratio"] for x in a25), 2))
    put("ar025_cilo", f(min(x["ci"][0] for x in a25), 2))
    for key in ("ksd/sigreg", "sksd/ksd", "mmd/sksd"):
        x = V["P3"][key]
        put("p3_" + key.replace("/", "_"), ci(x["ratio"], *x["ci"], fmt=lambda v: f(v, 1)))
    # cov / VICReg collapse at the lowest λ
    for arm in ("cov", "vicreg"):
        p = path(C, arm).iloc[0]
        put(f"low_{arm}_lam", g(p.lam, 2))
        put(f"low_{arm}_std", f(p.emb_std_mean, 3))
        put(f"low_{arm}_content", f(p.content_r2, 2))
        put(f"low_{arm}_rank", f(p.eff_rank, 1))
    # P8, P10 linear rung
    P8 = V["P8"]
    for arm in ("cov", "ksd"):
        put(f"lin_{arm}", g(P8[arm]["lam_hat_a1"][0], 3))
        r = thr(T, arm, encoder="linear")
        put(f"linci_{arm}", f"[{g(r.lo, 3)}, {g(r.hi, 3)}]")
        put(f"lin_ratio_{arm}", f(P8[arm]["ratio_to_theory"], 2))
        x = P8[arm]["alpha_ratio"]
        put(f"lin_ar_{arm}", ci(x["ratio"], *x["ci"], fmt=lambda v: f(v, 2)))
    x = V["P10"]["linear"]
    put("p10_lin", ci(x["ratio"], *x["ci"], fmt=lambda v: f(v, 1)))
    x = V["P10"]["conv_share"]
    put("p10_share", ci(x["ratio"], *x["ci"], fmt=lambda v: f(v, 1)))
    # P9
    P9 = V["P9"]
    put("p9_conv_v2", f(P9["cov|conv"]["v_at_2x_share_est"], 2))
    put("p9_conv_v5", f(P9["cov|conv"]["v_at_5x_share_est"], 2))
    put("p9_conv_v064", f(P9["cov|conv"]["path"]["0.64"], 2))
    put("p9_conv_v128", f(P9["cov|conv"]["path"]["1.28"], 2))
    put("p9_lin_v2", f(P9["cov|linear"]["v_at_2x"], 2))
    put("p9_lin_v5", f(P9["cov|linear"]["v_at_5x"], 2))
    for arm in ("ksd", "sigreg", "mmd"):
        put(f"p9_{arm}_v2", f(P9[f"{arm}|conv"]["v_at_2x"], 2))
    # frontier
    for arm in ARMS:
        for ceil in (0.05, 0.1, 0.2):
            r = F[(F.arm == arm) & (F.d == 12) & (F.alpha == 1.0) & (F.encoder == "conv") & (F.sigma == 0)
                  & (F.ceiling == ceil)].iloc[0]
            put(f"fr_{arm}_{int(round(ceil * 100)):02d}",
                "none" if not np.isfinite(r.best_content) else ci(r.best_content, r.lo, r.hi, nd=2))
        fm = V["frontier_matched"][arm]
        put(f"fm_max_{arm}", f(fm["max_content"], 2))
        x = fm["leak_at_content_0.5"]
        put(f"fm05_{arm}", ci(x["value"], *x["ci"], nd=2) if x["value"] is not None else "never reached")
    # E3
    E3 = V["E3"]
    for cell, v in E3.items():
        arm, lam = cell.split("|")
        kk = f"{arm}_{lam.replace('.', '')}"
        x = v["leak_8000_minus_2000"]
        put(f"e3_dleak_{kk}", ci(x["diff"], *x["ci"], nd=2, signed=True))
        for met in ("leak_r2", "content_r2", "leak_knn_r2", "content_knn_r2", "unexpl_share", "unexpl_share_oh",
                    "nuis_share", "content_share"):
            put(f"e3_{met}_{kk}_2000", f(v[met]["2000"][0], 2))
            put(f"e3_{met}_{kk}_8000", f(v[met]["8000"][0], 2))
    # E4
    E4 = V["E4"]
    for cell, v in E4.items():
        arm, lam = cell.split("|")
        kk = f"{arm}_{lam.replace('.', '')}"
        put(f"e4_{kk}_s0", f(v["sigma0"]["leak_r2"][0], 3))
        for cond in ("batch0.5", "batch1.0", "fixed0.5", "fixed1.0"):
            c = cond.replace(".", "")
            put(f"e4_{kk}_{c}", f(v[cond]["leak_r2"][0], 2))
            x = v[cond]["leak_minus_sigma0"]
            put(f"e4_{kk}_{c}_d", ci(x["diff"], *x["ci"], nd=3, signed=True))
            put(f"e4_{kk}_{c}_std", f(v[cond]["emb_std_mean"][0], 2))
            put(f"e4_{kk}_{c}_cs", f(v[cond]["content_share"][0], 2))
        put(f"e4_{kk}_s0_std", f(v["sigma0"]["emb_std_mean"][0], 2))
        put(f"e4_{kk}_s0_cs", f(v["sigma0"]["content_share"][0], 2))
        for s in ("0.5", "1.0"):
            x = v[f"fixed_minus_batch{s}"]
            for met in ("leak_r2", "emb_std_mean"):
                put(f"e4_{kk}_fmb{s.replace('.', '')}_{met}",
                    ci(x[met]["diff"], *x[met]["ci"], nd=2, signed=True))
    # E5
    E5 = V["E5"]
    for cell, v in E5.items():
        if cell == "all_cells":
            continue
        law, gl, ml = cell.split("|")
        kk = f"{law}_{gl.replace('gauss', '').replace('.', '')}"
        for met in ("ident_orth_r2", "mcc", "mcc_hue", "mcc_scale", "leak_r2", "content_r2"):
            x = v[met]["matched_minus_gauss"]
            put(f"e5_{kk}_{met}", ci(x["diff"], *x["ci"], nd=2, signed=True))
    return N


# ----------------------------------------------------------------------------- tables

def md_table(header, rows, align=None):
    align = align or ["l"] + ["r"] * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def tables(C, T, V, TH):
    rows = []
    for arm in ARMS:
        lt = theory_lam(TH, arm)
        r = thr(T, arm)
        lam = lamstr(r.lam_hat, r.censor)
        cis = "" if r.censor else f" [{g(r.lo, 3)}, {g(r.hi, 3)}]"
        ratio = V["P2"][arm]["ratio"]
        a5, a25 = (V["P4"][arm][a] for a in ("0.5", "0.25"))
        d24 = V["P6"][arm]
        r6 = thr(T, arm, d=6)
        th = ("≈ " if arm == "vicreg" else "") + g(lt, 2)
        rows.append([NAMES[arm], th, lam + cis, f(ratio, 2) if ratio is not None else "–",
                     f(a5["ratio"], 2) if a5["ratio"] is not None else "–",
                     f(a25["ratio"], 2) if a25["ratio"] is not None else "–",
                     f(d24["ratio"], 2) if d24["ratio"] is not None else "–",
                     lamstr(r6.lam_hat, r6.censor)])
    hdr = ["arm", "linear model $\\lambda^*$", "$\\hat\\lambda^*$ [95% CI]", "ratio",
           "$\\alpha{=}0.5$", "$\\alpha{=}0.25$", "$d{=}24$", "$\\hat\\lambda^*$ at $d{=}6$"]
    with open(f"{GEN}/tab_thresholds.md", "w") as fh:
        fh.write(md_table(hdr, rows))

    # linear-model thresholds (T2, numerical)
    rows = []
    t2 = TH["t2"]["thresholds"]
    for arm in ["cov", "sigreg", "ksd", "sksd", "mmd"]:
        tt = t2[f"{arm}_d12"]
        rows.append([NAMES[arm]] + [g(theory_lam(TH, arm, 12, a), 3) for a in (1.0, 0.5, 0.25)]
                    + [g(theory_lam(TH, arm, 24, 1.0), 3), g(tt["t_star"] / tt["t_quad"], 2)])
    hdr = ["arm", "$\\lambda^*$, $\\alpha{=}1$", "$\\alpha{=}0.5$", "$\\alpha{=}0.25$", "$d{=}24$, $\\alpha{=}1$",
           "$t^*/t_\\text{quad}$"]
    with open(f"{GEN}/tab_theory.md", "w") as fh:
        fh.write(md_table(hdr, rows))

    # E5 contrasts (matched − Gaussian-target KSD)
    E5 = V["E5"]
    lab = {"uniform|gauss0.2|matched0.2": "uniform, λ 0.2 vs 0.2",
           "uniform|gauss0.04|matched0.04": "uniform, λ 0.04 vs 0.04",
           "laplace|gauss0.2|matched1.0": "Laplace, λ 0.2 vs 1.0",
           "laplace|gauss0.04|matched0.2": "Laplace, λ 0.04 vs 0.2"}
    rows = []
    for cell, name in lab.items():
        v = E5[cell]
        rows.append([name] + [ci(v[m]["matched_minus_gauss"]["diff"], *v[m]["matched_minus_gauss"]["ci"], nd=2,
                                 signed=True)
                              for m in ("ident_orth_r2", "mcc", "mcc_hue", "mcc_scale", "leak_r2", "content_r2")])
    hdr = ["law, λ (Gaussian vs matched)", "orth-fit $R^2$", "MCC", "MCC hue pair", "MCC scale", "leakage",
           "content $R^2$"]
    with open(f"{GEN}/tab_e5.md", "w") as fh:
        fh.write(md_table(hdr, rows))

    # verdicts (words from GATE_2; evidence from verdicts.json)
    P = V
    ev = {
        "P1": f"kernel arms rise within {f(min(P['P1'][a]['decades'] for a in KERN), 2)}–"
              f"{f(max(P['P1'][a]['decades'] for a in KERN), 2)} decades; cov, VICReg ≥ 0.10 at lowest λ",
        "P2": f"KSD {f(P['P2']['ksd']['ratio'], 2)}×, sliced KSD {f(P['P2']['sksd']['ratio'], 2)}×, "
              f"SIGReg {f(P['P2']['sigreg']['ratio'], 2)}×, MMD {f(P['P2']['mmd']['ratio'], 2)}×",
        "P3": "successive ratios " + ", ".join(f(P["P3"][k]["ratio"], 1) for k in ("ksd/sigreg", "sksd/ksd",
                                                                                  "mmd/sksd")) + "; CIs exclude 1",
        "P4": f"ratio at α = 0.25: {f(min(P['P4'][a]['0.25']['ratio'] for a in KERN), 2)}–"
              f"{f(max(P['P4'][a]['0.25']['ratio'] for a in KERN), 2)} (∝ α: 0.25)",
        "P5": f"all 6 arms reach ≥ 0.10 at d = 6 (max {f(min(P['P5'][a]['max_d6'] for a in ARMS), 2)}–"
              f"{f(max(P['P5'][a]['max_d6'] for a in ARMS), 2)})",
        "P6": f"KSD {f(P['P6']['ksd']['ratio'], 2)}, SIGReg {f(P['P6']['sigreg']['ratio'], 2)}, "
              f"MMD {f(P['P6']['mmd']['ratio'], 2)}",
        "P7": f"probe narrower in {sum(P['P7'][a]['leak_steeper'] for a in ARMS)}/6 arms",
        "P8": f"cov {f(P['P8']['cov']['ratio_to_theory'], 1)}×, KSD {f(P['P8']['ksd']['ratio_to_theory'], 2)}× theory",
        "P9": f"linear-rung cov v = {f(P['P9']['cov|linear']['v_at_2x'], 2)} / {f(P['P9']['cov|linear']['v_at_5x'], 2)}"
              f" at 2/5 λ̂*",
        "P10": f"linear rung {f(P['P10']['linear']['ratio'], 1)}",
        "P11": "8000 − 2000 steps: " + ", ".join(fs(P["E3"][c]["leak_8000_minus_2000"]["diff"], 2)
                                                for c in P["E3"]),
        "P12": "unexplained share rises; kNN content R² → " + f(P["E3"]["ksd|0.8"]["content_knn_r2"]["8000"][0], 2),
        "P13": f"KSD batch σ = 1: {fs(P['E4']['ksd|0.1']['batch1.0']['leak_minus_sigma0']['diff'], 2)}, "
               f"{fs(P['E4']['ksd|0.2']['batch1.0']['leak_minus_sigma0']['diff'], 2)}",
        "P14": f"λ = 0.2, σ = 1: {fs(P['E4']['ksd|0.2']['batch1.0']['leak_minus_sigma0']['diff'], 2)} (predicted 0)",
        "P15": f"fixed − batch at σ 0.5: {fs(P['E4']['ksd|0.1']['fixed_minus_batch0.5']['leak_r2']['diff'], 2)}, "
               f"{fs(P['E4']['ksd|0.2']['fixed_minus_batch0.5']['leak_r2']['diff'], 2)}",
        "P16": "uniform: " + ", ".join(fs(P["E5"][c]["ident_orth_r2"]["matched_minus_gauss"]["diff"], 2)
                                       for c in ("uniform|gauss0.2|matched0.2", "uniform|gauss0.04|matched0.04"))
               + " (CIs include 0)",
        "P17": f"hue-pair MCC {fs(P['E5']['uniform|gauss0.2|matched0.2']['mcc_hue']['matched_minus_gauss']['diff'], 2)}"
               " at uniform λ 0.2",
        "P18": "E6 cut",
    }
    pred = {
        "P1": "step-like onset", "P2": "location within ×3", "P3": "ordering SIGReg ≈ cov < KSD < sKSD < MMD",
        "P4": "α ratio in [α(2−α), 1]", "P5": "no threshold at d = 6", "P6": "d = 24 ratios",
        "P7": "probe steeper than share", "P8": "linear rung within ×3", "P9": "cov form 1 − λ*/λ",
        "P10": "KSD/cov > 5", "P11": "leakage persists at 8000 steps", "P12": "spare directions (descr.)",
        "P13": "sign of batch σ (KSD)", "P14": "size of σ effect (KSD)", "P15": "fixed vs batch σ",
        "P16": "no orth-fit $R^2$ gain from matched prior", "P17": "no hue-pair MCC gain", "P18": "top-k cov arm",
    }
    verdict = {
        "P1": "refuted (cov, VICReg); step holds for kernel arms", "P2": "refuted", "P3": "confirmed (cov censored)",
        "P4": "confirmed (4 kernel arms)", "P5": "refuted", "P6": "not refuted; point values miss",
        "P7": "confirmed (weak)", "P8": "refuted (cov)", "P9": "inconclusive (cov); kernel part confirmed",
        "P10": "refuted (linear rung)", "P11": "refuted", "P12": "–", "P13": "confirmed (leakage)",
        "P14": "refuted in size", "P15": "refuted (KSD)", "P16": "confirmed", "P17": "refuted",
        "P18": "not tested",
    }
    rows = [[k, pred[k], verdict[k], ev[k]] for k in pred]
    with open(f"{GEN}/tab_verdicts.md", "w") as fh:
        fh.write(md_table(["#", "prediction", "verdict", "key statistic"], rows, align=["l", "l", "l", "l"]))


# ----------------------------------------------------------------------------- figures

def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.svg", bbox_inches="tight")
    if os.environ.get("REPORT2_PNG"):
        fig.savefig(f"{os.environ['REPORT2_PNG']}/{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_theory(TH, VAL):
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.4))
    q = np.linspace(0.3, 10, 400)
    ax[0].plot(q, np.clip(1 - 1 / q, 0, None), color="k", lw=1.2, label="cov, Thm 1: $1-\\lambda^*/\\lambda$")
    t2 = TH["t2"]["thresholds"]
    for arm in ["cov"] + KERN:
        pth = t2[f"{arm}_d12"]["path"]
        x = [0.5, 0.8] + [p["lam_over_star"] for p in pth]
        y = [0, 0] + [p["v_mean"] for p in pth]
        ax[0].plot(x, y, marker=MARK[arm], ms=3, lw=0.8, ls="--" if arm == "cov" else "-", color=COL[arm],
                   label=f"{NAMES[arm]} (population, numerical)")
    rows = VAL["A"]["rows"]
    ax[0].scatter([r["q"] for r in rows], [r["v"] for r in rows], marker="x", s=18, color="red", zorder=5,
                  label="cov, real training loop")
    ax[0].set_xscale("log"); ax[0].set_xlabel("$\\lambda/\\lambda^*$"); ax[0].set_ylabel("nuisance variance per direction")
    ax[0].set_title("(a) nuisance variance above the threshold")
    ax[0].legend(frameon=False, fontsize=5.8, loc="upper left")
    v = np.clip(1 - 1 / q, 0, None)
    r = M
    ax[1].plot(q, np.where(v > 0, 1.0, 0.0), color="k", lw=1.2, label="probe $R^2$ (population)")
    ax[1].plot(q, r * v / (K + r * v), color="k", lw=1.2, ls=":", label="nuisance variance share")
    ax[1].scatter([r_["q"] for r_ in rows], [r_["leak_r2"] for r_ in rows], marker="x", s=18, color="red",
                  label="probe $R^2$, real loop")
    ax[1].scatter([r_["q"] for r_ in rows], [r_["nuis_share"] for r_ in rows], marker="+", s=22, color="tab:blue",
                  label="share, real loop")
    ax[1].set_xscale("log"); ax[1].set_xlabel("$\\lambda/\\lambda^*$"); ax[1].set_ylabel("value")
    ax[1].set_title("(b) cov arm: probe $R^2$ vs variance share")
    ax[1].legend(frameon=False, fontsize=5.8, loc="upper left", bbox_to_anchor=(0.45, 0.9))
    fig.tight_layout()
    save(fig, "fig1_theory")


def fig_paths(C, T, TH):
    fig, axs = plt.subplots(2, 3, figsize=(6.8, 3.8), sharey=True)
    for ax, arm in zip(axs.ravel(), ARMS):
        p = path(C, arm)
        for met, ls, lab, col in (("leak_r2", "-", "leakage $R^2$", COL[arm]),
                                  ("nuis_share", ":", "nuisance share", COL[arm]),
                                  ("content_r2", "--", "content $R^2$", "0.55")):
            y = p[met].clip(lower=-0.05)
            ax.errorbar(p.lam, y, yerr=[(p[met] - p[met + "_lo"]).clip(lower=0).where(p[met] > -0.05, 0),
                                        (p[met + "_hi"] - p[met]).clip(lower=0).where(p[met] > -0.05, 0)],
                        ls=ls, marker=MARK[arm] if met == "leak_r2" else None, ms=3, lw=1, color=col, capsize=1.5,
                        label=lab)
        lt = theory_lam(TH, arm)
        if lt:
            ax.axvline(lt, color="k", lw=0.8, ls="-.", label="linear-model $\\lambda^*$" + (" (heur.)" if arm == "vicreg" else ""))
        r = thr(T, arm)
        if not r.censor:
            ax.axvspan(r.lo, r.hi, color=COL[arm], alpha=0.15, lw=0)
            ax.axvline(r.lam_hat, color=COL[arm], lw=0.8)
        ax.axhline(0.10, color="0.7", lw=0.5)
        ax.set_xscale("log"); ax.set_ylim(-0.06, 0.8); ax.set_title(NAMES[arm])
        if arm == "cov":
            ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    for ax in axs[1]:
        ax.set_xlabel("λ")
    for ax in axs[:, 0]:
        ax.set_ylabel("value")
    fig.tight_layout()
    save(fig, "fig2_paths")


def fig_scaling(T, V, TH):
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))
    a = np.linspace(0.2, 1, 100)
    ax[0].plot(a, a * (2 - a), "k-", lw=1, label="$\\alpha(2-\\alpha)$ (learned predictor)")
    ax[0].plot(a, a, "k:", lw=1, label="$\\alpha$ (identity predictor)")
    ax[0].axhline(1, color="0.6", lw=0.8, ls="--", label="1 (trap escape)")
    for j, arm in enumerate(KERN):
        xs, ys, lo, hi = [1.0], [1.0], [0], [0]
        for al in (0.5, 0.25):
            x = V["P4"][arm][str(al)]
            xs.append(al); ys.append(x["ratio"]); lo.append(x["ratio"] - x["ci"][0]); hi.append(x["ci"][1] - x["ratio"])
        off = (j - 1.5) * 0.012
        ax[0].errorbar(np.array(xs) + off, ys, yerr=[lo, hi], marker=MARK[arm], ms=3.5, lw=0.8, color=COL[arm],
                       capsize=1.5, label=NAMES[arm])
    ax[0].set_xlabel("α"); ax[0].set_ylabel("$\\hat\\lambda^*(\\alpha)/\\hat\\lambda^*(1)$")
    ax[0].set_title("(a) threshold against α"); ax[0].legend(frameon=False, fontsize=5.6, loc="upper left")
    lims = (0.008, 40)
    xx = np.array(lims)
    ax[1].plot(xx, xx, "k-", lw=0.8); ax[1].fill_between(xx, xx / 3, xx * 3, color="0.9", lw=0)
    shapes = {("conv", 12, 1.0): "o", ("conv", 12, 0.5): "s", ("conv", 12, 0.25): "^", ("conv", 24, 1.0): "D",
              ("linear", 12, 1.0): "*", ("linear", 12, 0.5): "X"}
    labs = {("conv", 12, 1.0): "conv, d 12, α 1", ("conv", 12, 0.5): "conv, α 0.5", ("conv", 12, 0.25): "conv, α 0.25",
            ("conv", 24, 1.0): "conv, d 24", ("linear", 12, 1.0): "linear enc., α 1",
            ("linear", 12, 0.5): "linear enc., α 0.5"}
    used = set()
    for arm in ["cov"] + KERN:
        for (enc, d, al), mk in shapes.items():
            r = thr(T, arm, d=d, alpha=al, encoder=enc)
            lt = theory_lam(TH, arm, d, al)
            if r is None or lt is None or r.censor:
                continue
            ax[1].errorbar(lt, r.lam_hat, yerr=[[r.lam_hat - r.lo if np.isfinite(r.lo) else 0],
                                                [r.hi - r.lam_hat if np.isfinite(r.hi) else 0]],
                           marker=mk, ms=4, color=COL[arm], lw=0.8, capsize=1.5, mfc="none" if enc == "linear" else COL[arm])
    for (enc, d, al), mk in shapes.items():
        ax[1].plot([], [], marker=mk, ls="", color="0.3", ms=4, mfc="none" if enc == "linear" else "0.3", label=labs[(enc, d, al)])
    for arm in ["cov"] + KERN:
        ax[1].plot([], [], ls="-", color=COL[arm], label=NAMES[arm])
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlim(lims); ax[1].set_ylim(lims)
    ax[1].set_xlabel("linear-model $\\lambda^*$"); ax[1].set_ylabel("measured $\\hat\\lambda^*$")
    ax[1].set_title("(b) measured vs predicted (band: ×3)")
    ax[1].legend(frameon=False, fontsize=5.2, loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "fig3_scaling")


def fig_frontier(C):
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
    for arm in ARMS:
        p = path(C, arm)
        p = p[p.leak_r2 > -0.5]
        for a, met in zip(ax, ("leak_r2", "leak_knn_r2")):
            a.plot(p[met], p.content_r2, marker=MARK[arm], ms=3.5, lw=1, color=COL[arm], label=NAMES[arm])
            a.plot(p[met].iloc[:1], p.content_r2.iloc[:1], marker=MARK[arm], ms=6, color=COL[arm], mec="k",
                   mew=0.8, ls="")
    ax[0].axvline(0.05, color="0.7", lw=0.6); ax[0].axvline(0.10, color="0.7", lw=0.6, ls=":")
    ax[0].set_xlabel("linear leakage $R^2$"); ax[1].set_xlabel("kNN leakage $R^2$")
    ax[0].set_ylabel("linear content $R^2$")
    ax[0].set_title("(a) content vs linear leakage along the λ path"); ax[1].set_title("(b) content vs kNN leakage")
    h, l = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(h, l, frameon=False, fontsize=6.5, loc="lower center", ncol=6)
    save(fig, "fig4_frontier")


def fig_e3(V):
    E3 = V["E3"]
    cells = list(E3)
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))
    mets = [("leak_r2", "leakage\nlinear"), ("leak_knn_r2", "leakage\nkNN"), ("content_r2", "content\nlinear"),
            ("content_knn_r2", "content\nkNN")]
    w = 0.1
    x = np.arange(len(mets))
    for i, c in enumerate(cells):
        arm = c.split("|")[0]
        for j, steps in enumerate(("2000", "8000")):
            ys = [E3[c][m][steps][0] for m, _ in mets]
            lo = [E3[c][m][steps][0] - E3[c][m][steps][1][0] for m, _ in mets]
            hi = [E3[c][m][steps][1][1] - E3[c][m][steps][0] for m, _ in mets]
            pos = x + (i * 2 + j - 3.5) * w
            ax[0].bar(pos, ys, w, yerr=[lo, hi], color=COL[arm], alpha=0.45 if steps == "2000" else 1.0,
                      hatch="" if i % 2 == 0 else "//", edgecolor="white", lw=0.3, error_kw=dict(lw=0.5, capsize=1),
                      label=f"{NAMES[arm]} λ {c.split('|')[1]}, {steps}" )
    ax[0].set_xticks(x); ax[0].set_xticklabels([l for _, l in mets]); ax[0].set_ylabel("$R^2$")
    ax[0].set_title("(a) 2000 (light) vs 8000 (dark) steps"); ax[0].legend(frameon=False, fontsize=4.8, ncol=2, loc="upper left")
    ax[0].set_ylim(-0.02, 1.25)
    labels, base = [], []
    for c in cells:
        for steps in ("2000", "8000"):
            labels.append(f"{NAMES[c.split('|')[0]]}\nλ {c.split('|')[1]}\n{steps}")
    xs = np.arange(len(labels))
    cs = [max(E3[c]["content_share"][s][0], 0) for c in cells for s in ("2000", "8000")]
    ns = [max(E3[c]["nuis_share"][s][0], 0) for c in cells for s in ("2000", "8000")]
    us = [max(E3[c]["unexpl_share"][s][0], 0) for c in cells for s in ("2000", "8000")]
    ax[1].bar(xs, cs, color="#4477aa", label="content (linear code)")
    ax[1].bar(xs, ns, bottom=cs, color="#cc6677", label="nuisance (linear code)")
    ax[1].bar(xs, us, bottom=np.array(cs) + np.array(ns), color="#dddddd", label="unexplained")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(labels, fontsize=5)
    ax[1].set_ylabel("share of embedding variance"); ax[1].set_title("(b) what fills the embedding")
    ax[1].legend(frameon=False, fontsize=5.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.3))
    ax[1].set_ylim(0, 1.0)
    fig.tight_layout()
    save(fig, "fig5_e3")


def fig_e4(V):
    E4 = V["E4"]
    cells = ["ksd|0.1", "ksd|0.2", "vicreg|1.25", "vicreg|5.0"]
    fig, axs = plt.subplots(1, 4, figsize=(6.8, 1.9), sharey=True)
    for ax, c in zip(axs, cells):
        arm, lam = c.split("|")
        for mode, ls, mk in (("batch", "-", "o"), ("fixed", "--", "s")):
            ys = [E4[c]["sigma0"]["leak_r2"]] + [E4[c][f"{mode}{s}"]["leak_r2"] for s in ("0.5", "1.0")]
            m = [y[0] for y in ys]
            lo = [y[0] - y[1][0] for y in ys]
            hi = [y[1][1] - y[0] for y in ys]
            ax.errorbar([0, 0.5, 1.0], m, yerr=[lo, hi], ls=ls, marker=mk, ms=3.5, lw=1, capsize=1.5,
                        color=COL[arm], mfc="none" if mode == "fixed" else COL[arm], label=f"{mode}-scale σ")
        ax.set_title(f"{NAMES[arm]}, λ = {lam}"); ax.set_xlabel("σ"); ax.set_xticks([0, 0.5, 1])
    axs[0].set_ylabel("leakage $R^2$"); axs[0].legend(frameon=False, fontsize=5.8, loc="upper left")
    fig.tight_layout()
    save(fig, "fig6_e4")


def write_yaml(path, D):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Generated by stein/report2.py from results2/ (and results/ for study-1 numbers). Do not edit.\n")
        fh.write("n:\n")
        for k in sorted(D):
            v = str(D[k]).replace('"', '\\"')
            fh.write(f'  {k}: "{v}"\n')


def main():
    os.makedirs(GEN, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    C, T, F, RUNS, V, TH, VAL = load()
    numbers(C, T, F, RUNS, V, TH, VAL)
    tables(C, T, V, TH)
    fig_theory(TH, VAL); fig_paths(C, T, TH); fig_scaling(T, V, TH); fig_frontier(C); fig_e3(V); fig_e4(V)
    write_yaml(f"{GEN}/numbers.yml", N)
    print(f"{len(N)} numbers, 3 tables, 6 figures written")


if __name__ == "__main__":
    main()
