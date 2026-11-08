"""Validation gates (numbers for gates/GATE_1.md). Writes results/validation.json.
Run: .venv/bin/python -m stein.validate
"""
import json
import numpy as np
import torch

from .losses import Regulariser, SIGReg, sigreg_ref, Prior, ksd_imq
from .data import Shapes, Cifar, content_code, nuisance_code, CONTENT, NUISANCE, SH
from . import metrics as M


def regulariser_table(d=12, n=256, reps=5):
    torch.manual_seed(0)
    scales = torch.ones(d); scales[:3] = 3.0; scales[-3:] = 0.1
    cases = {
        "N(0,I)": lambda: torch.randn(n, d),
        "collapsed (1 + 0.01 N)": lambda: 0.01 * torch.randn(n, d) + 1.0,
        "rank-1": lambda: torch.randn(n, 1) * torch.ones(1, d) / d ** 0.5 + 0.01 * torch.randn(n, d),
        "anisotropic (sd 3/1/0.1)": lambda: torch.randn(n, d) * scales,
        "shifted N(0.5,I)": lambda: torch.randn(n, d) + 0.5,
    }
    out = {}
    for name in ["vicreg", "sigreg", "sksd", "ksd", "mmd", "oracle"]:
        reg = Regulariser(name, d, k=6, oracle_eps=0.3)       # eps as in training
        row = {}
        for cname, fn in cases.items():
            vals = [float(reg(fn())) for _ in range(reps)]
            row[cname] = (float(np.mean(vals)), float(np.std(vals)))
        if name == "oracle":           # its own null: samples from the rank-k target
            p = Prior("rank_k", d=d, k=6, eps=0.3)
            vals = [float(reg(p.sample(n, d))) for _ in range(reps)]
            row["oracle target N(0, diag(1_6, 0.09))"] = (float(np.mean(vals)), float(np.std(vals)))
        out[name] = row
    # matched-prior KSD null
    for law in ["laplace", "uniform"]:
        p = Prior(law, d=d)
        reg = Regulariser("ksd", d, law=law)
        out[f"ksd[{law}]"] = {
            f"{law} sample (surrogate)": float(np.mean([float(reg(p.sample(n, d, exact=False))) for _ in range(reps)])),
            f"{law} sample (exact)": float(np.mean([float(reg(p.sample(n, d, exact=True))) for _ in range(reps)])),
            "N(0,I)": float(np.mean([float(reg(torch.randn(n, d))) for _ in range(reps)])),
        }
    return out


def sigreg_check():
    torch.manual_seed(0)
    res = {}
    for d, n in [(12, 256), (64, 512)]:
        z = torch.randn(n, d) * 1.3 + 0.2
        ref = float(SIGReg(num_slices=256)(z))
        g = torch.Generator().manual_seed(0)
        A = torch.randn(d, 256, generator=g); A = A / A.norm(p=2, dim=0)
        mine = float(sigreg_ref(z, A))
        res[f"d={d},n={n}"] = {"lejepa": ref, "ours": mine, "rel_diff": abs(ref - mine) / abs(ref)}
    return res


def sampler_check():
    S = Shapes(seed=0)
    out = {}
    for a in [0.0, 0.5, 1.0]:
        f1, f2 = S.sample_pair_factors(40000, a)
        n1, n2 = nuisance_code(f1), nuisance_code(f2)
        C = np.corrcoef(n1.T, n2.T)[:5, 5:]
        out[f"alpha={a}"] = {
            "content_identical": bool(np.array_equal(f1[:, CONTENT], f2[:, CONTENT])),
            "nuisance_cross_view_corr_diag": np.diag(C).round(4).tolist(),
            "max_abs_cross_corr": float(np.abs(C).max()),
            "test_split_touched": bool(S.is_test[S._idx(f1)].any() or S.is_test[S._idx(f2)].any()),
        }
    # pixel-level: images of the two views at alpha=0 differ only by view noise
    f1, f2 = S.sample_pair_factors(64, 0.0)
    x1, x2 = S.images_for(f1, noise=False), S.images_for(f2, noise=False)
    out["alpha=0 pixel max diff (no view noise)"] = float((x1 - x2).abs().max())
    f = S.sample_clip_factors(5000, 4, 1.0)
    out["clips: content constant"] = bool((f[:, :, CONTENT] == f[:, :1, CONTENT]).all())
    nc = nuisance_code(f.reshape(-1, 6)).reshape(5000, 4, 5)
    out["clips alpha=1: corr(frame0, frame3) per nuisance dim"] = [
        float(np.corrcoef(nc[:, 0, j], nc[:, 3, j])[0, 1]) for j in range(5)]
    C = Cifar(seed=0)
    x1, x2, p1, p2, _ = C.sample_pair(20000, 1.0)
    Cc = np.corrcoef(p1.T, p2.T)[:8, 8:]
    out["cifar alpha=1 aug-param cross-view max |corr|"] = float(np.abs(Cc).max())
    return out


def probe_check():
    S = Shapes(seed=1)
    ftr, fte = S.sample_factors(10000), S.sample_factors(5000, "test")
    Str, Ste, Ntr, Nte = content_code(ftr), content_code(fte), nuisance_code(ftr), nuisance_code(fte)
    rng = np.random.default_rng(0)
    R = np.linalg.qr(rng.normal(size=(11, 11)))[0]
    Ztr = np.concatenate([Str, Ntr], 1) @ R; Zte = np.concatenate([Ste, Nte], 1) @ R
    Sz = (Ste - Ste.mean(0)) / Ste.std(0)
    return {
        "given true factors (rotated content+nuisance): content R2": M.ridge_r2(Ztr, Str, Zte, Ste)[0],
        "given true factors: leakage R2": M.ridge_r2(Ztr, Ntr, Zte, Nte)[0],
        "given content only: leakage R2": M.ridge_r2(Str, Ntr, Ste, Nte)[0],
        "given content only: shape acc": M.logreg_acc(Str, ftr[:, SH], Ste, fte[:, SH]),
        "orth identifiability of rotated standardised content": M.orth_identifiability(
            Sz @ np.linalg.qr(rng.normal(size=(6, 6)))[0] * 2, Ste),
        "orth identifiability of random linear mix": M.orth_identifiability(Ste @ rng.normal(size=(6, 6)), Ste),
        "effective rank, N(0, I_12)": M.effective_rank(rng.normal(size=(5000, 12))),
        "NP-KSD p-value, N(0,I_6) vs gaussian prior": M.np_ksd(rng.normal(size=(500, 6)), "gaussian", n_null=30)[1],
        "NP-KSD p-value, 1.5*N(0,I_6) vs gaussian prior": M.np_ksd(1.5 * rng.normal(size=(500, 6)), "gaussian", n_null=30)[1],
        "NP-KSD p-value, uniform sample vs uniform prior": M.np_ksd(
            rng.uniform(-3 ** 0.5, 3 ** 0.5, size=(500, 6)), "uniform", n_null=30)[1],
        "NP-KSD p-value, N(0,I_6) vs uniform prior": M.np_ksd(rng.normal(size=(500, 6)), "uniform", n_null=30)[1],
    }


if __name__ == "__main__":
    res = {"regularisers_d12_n256": regulariser_table(), "sigreg_vs_lejepa": sigreg_check(),
           "sampler": sampler_check(), "probes": probe_check()}
    json.dump(res, open("results/validation.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
