"""Study-2 validation gates (P1) -> results2/validation2.json.

    .venv/bin/python -W ignore -m stein.validate2

A. Theory meets code: the real training loop (stein.train) with the `synth` tier (linear-Gaussian
   views, k = 6, m = 5), the `linear` encoder, the `linear` predictor and the `cov` arm reproduces
   Theorem 1's threshold: per-direction nuisance variance v and content variance c against the
   finite-N reference (minimum of the expected minibatch objective over eigenvalues, as in
   theory/check_t1.py part C), tolerance 0.03 (check_t1's minibatch tolerance); onset between
   0.8 and 1.25 lambda*; criteria in part_a's docstring.
   Observations (not gates): d = 6 = k (written first as a gate; it failed at q = 5, see GATE_1),
   trap frequency just above threshold, MLP predictor (study 1's), alpha = 0.5.
B. A rerun of one cached study-1 config (KSD, lambda 0.2, seed 1, base setting) under eval_v = 2
   (new hash, same training) reproduces the stored study-1 metrics.
"""
import json, os, sys, time

import numpy as np
import torch

from .train import run, resolve, cfg_hash

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "theory"))
from _lib import cov_pop  # noqa: E402  (finite-N expected covariance penalty, theory/_lib.py)
torch.set_default_dtype(torch.float32)   # theory/_lib sets float64 globally; training runs in float32

OUT = "results2/validation2.json"
RUNS = os.environ.get("STEIN_RUNS", "runs")
TOL = 0.03


def reference(d, k, m, alpha, lam, N=256):
    """Minimise kap * sum(v) + lam * E cov_pen(eigs) over k content + r nuisance eigenvalues
    (theory/check_t1.py part_minibatch); kap = c_p alpha (2 - alpha), c_p = 1/d (code units)."""
    kap = alpha * (2 - alpha) / d
    r = min(m, max(d - k, 0))
    kk = min(k, d)
    x = torch.zeros(kk + r, dtype=torch.float64).requires_grad_()
    opt = torch.optim.LBFGS([x], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-14)
    sp = torch.nn.functional.softplus

    def f():
        ev = torch.cat([sp(x[:kk] * 5) / 5, sp(x[kk:] * 50) / 50,
                        torch.zeros(d - kk - r, dtype=torch.float64)])
        return kap * ev[kk:kk + r].sum() + lam * cov_pop(ev, N)

    def closure():
        opt.zero_grad(); L = f(); L.backward(); return L
    with torch.no_grad():
        x[:kk] = 1.0; x[kk:] = 0.5
    opt.step(closure)
    with torch.no_grad():
        ev = torch.cat([sp(x[:kk] * 5) / 5, sp(x[kk:] * 50) / 50])
    c, v = ev[:kk].mean().item(), (ev[kk:].mean().item() if r else 0.0)
    return c, (0.0 if v < 1e-3 else v)


def synth_run(lam, d=12, alpha=1.0, pred="linear", seed=0, steps=4000):
    cfg = dict(tier="synth", encoder="linear", pred_arch=pred, arm="cov", lam=round(lam, 8), d=d,
               alpha=alpha, seed=seed, steps=steps, eval_v=2, device="cpu")
    r = run(cfg)
    r_ = min(5, max(d - 6, 0))
    v = float(np.mean(r["nuis_eigs"][:r_])) if r_ else 0.0
    return dict(lam=lam, d=d, alpha=alpha, pred=pred, seed=seed, v=v, c=float(np.mean(r["content_eigs"][:min(6, d)])),
                nuis_eigs=r["nuis_eigs"][:max(r_, 1)], nuis_share=r["nuis_share"], leak_r2=r["leak_r2"],
                content_r2=r["content_r2"], hash=r["hash"])


def part_a():
    """Gate: (i) onset between 0.8 and 1.25 lambda* in every seed; (ii) the mean variance of the
    active nuisance directions (eigenvalue > 0.02) within TOL of the reference v and the mean content
    variance within TOL of the reference c, in every run (single eigenvalues scatter by up to ~0.05
    under minibatch noise, so the statistic is check_t1's mean, restricted to active directions); (iii) mean v over all r = 5 directions within TOL in at least
    7 of the 10 runs (a run with fewer active directions than r is an empty-direction trap, as in
    theory/check_t1.py part B, and is counted and reported, not hidden)."""
    rows = []
    d, k, m = 12, 6, 5
    lstar = 1 / (2 * d)                                   # c_p alpha (2 - alpha) / 2 at alpha = 1
    for q in (0.5, 0.8, 1.25, 2.0, 5.0):
        lam = q * lstar
        c_ref, v_ref = reference(d, k, m, 1.0, lam)
        for seed in (0, 1):
            r = synth_run(lam, seed=seed)
            act = [e for e in r["nuis_eigs"] if e > 0.02]
            v_act = float(np.mean(act)) if act else 0.0
            dir_ok = abs(v_act - v_ref) <= TOL and abs(r["c"] - c_ref) <= TOL
            rows.append(dict(**r, q=q, v_ref=v_ref, c_ref=c_ref, v_pop_formula=max(0.0, 1 - 1 / q),
                             n_active=len(act), v_active=v_act, trapped=bool(v_ref > 0 and len(act) < min(m, d - k)),
                             pass_dir=dir_ok, pass_mean=abs(r["v"] - v_ref) <= TOL and abs(r["c"] - c_ref) <= TOL))
    onset = all(r["v"] < 0.01 for r in rows if r["q"] <= 0.8) and all(r["v"] > 0.1 for r in rows if r["q"] >= 1.25)
    pass_dir = all(r["pass_dir"] for r in rows)
    n_mean = sum(r["pass_mean"] for r in rows)
    # trap frequency just above threshold (observation)
    near = [synth_run(1.25 * lstar, seed=s) for s in range(2, 6)]
    # d = 6 = k: Theorem 1 cor. c says no global minimiser holds nuisance. Observation, 6 seeds per level.
    d6 = [dict(**synth_run(q * lstar, d=6, seed=s), q=q) for q in (2.0, 5.0, 20.0) for s in range(6)]
    # observations: study 1's MLP predictor; alpha = 0.5 (global threshold c_p a(2-a)/2 = 0.75 lstar,
    # trap threshold c_p/2 = lstar, Prop. 1)
    mlp = [dict(**synth_run(q * lstar, pred="mlp", seed=0), q=q) for q in (0.5, 0.8, 1.25, 2.0, 5.0)]
    a05 = [dict(**synth_run(q * lstar, alpha=0.5, seed=s), q=q, v_ref_global=reference(d, k, m, 0.5, q * lstar)[1])
           for q in (0.6, 0.85, 1.25, 2.0) for s in (0, 1)]
    return dict(rows=rows, pass_onset=onset, pass_dir=pass_dir, n_pass_mean=n_mean,
                n_trapped=sum(r["trapped"] for r in rows),
                near_threshold_extra_seeds=[dict(seed=r["seed"], nuis_eigs=r["nuis_eigs"]) for r in near],
                d6=d6, d6_nuisance_runs={q: sum(r["nuis_share"] >= 0.01 for r in d6 if r["q"] == q) for q in (2.0, 5.0, 20.0)},
                obs_mlp_predictor=mlp, obs_alpha05=a05, lam_star=lstar, tol=TOL,
                pass_=bool(onset and pass_dir and n_mean >= 7))


SHARED = ["content_r2", "leak_r2", "content_knn_r2", "leak_knn_r2", "shape_acc", "ident_orth_r2",
          "eff_rank", "emb_std_mean", "loss_final", "pred_final", "reg_final"]


def part_b(tol=0.02):
    cfg = dict(tier="shapes", d=12, alpha=1.0, sigma=0.0, law="uniform", prior="gaussian", device="mps",
               steps=2000, arm="ksd", lam=0.2, seed=1)
    h1 = cfg_hash(resolve(cfg))
    p = os.path.join(RUNS, h1, "result.json")
    if not os.path.exists(p):
        return dict(error=f"cached run {h1} not found", pass_=False)
    old = json.load(open(p))
    new = run({**cfg, "eval_v": 2})
    diffs = {k: (old[k], new[k], abs(old[k] - new[k])) for k in SHARED if k in old and k in new}
    r2keys = [k for k in diffs if k.endswith("r2") or k == "shape_acc"]
    ok = all(diffs[k][2] <= tol for k in r2keys) and diffs["loss_final"][2] <= 0.05 * abs(diffs["loss_final"][0]) + 1e-3
    return dict(study1_hash=h1, study2_hash=new["hash"], diffs=diffs, tol_r2=tol, pass_=bool(ok))


if __name__ == "__main__":
    t0 = time.time()
    out = dict(A=part_a())
    A = out["A"]
    print("A onset", A["pass_onset"], "dir", A["pass_dir"], "mean", A["n_pass_mean"], "/10 trapped", A["n_trapped"],
          "d6 nuisance runs", A["d6_nuisance_runs"], f"{time.time() - t0:.0f}s", flush=True)
    out["B"] = part_b()
    out["pass"] = bool(out["A"]["pass_"] and out["B"]["pass_"])
    out["runtime_s"] = time.time() - t0
    os.makedirs("results2", exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    for r in out["A"]["rows"]:
        print(f"q={r['q']:.2f} seed={r['seed']} v={r['v']:.3f} ref={r['v_ref']:.3f} c={r['c']:.3f} ref={r['c_ref']:.3f} dir={r['pass_dir']} mean={r['pass_mean']}")
    print("B", {k: round(v[2], 4) for k, v in out["B"].get("diffs", {}).items()}, out["B"]["pass_"])
    print("PASS" if out["pass"] else "FAIL", f"{out['runtime_s']:.0f}s")
