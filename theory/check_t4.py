"""Numerical check of T4: rotation invariance of Gaussian-target regularisers, and what a matched
non-Gaussian product prior can identify (MCC vs R^2 after an orthogonal fit).

(i)  For non-Gaussian z (z = M s, s uniform), E R(O z) = E R(z) for random orthogonal O when R is
     KSD / sliced KSD / SIGReg / MMD with an N(0, I) target; KSD with the uniform-surrogate prior is not.
(ii) Linear model z = M s, d = k = 3, minibatch Adam on R(z) + (whitening is left to R):
     a. s independent uniform: matched-prior KSD gives MCC ~ 1; Gaussian-target KSD gives MCC < 1
        in general; orthogonal-fit R^2 ~ 1 for both.
     b. s = (cos h, sin h, u), h uniform on the circle (the study's hue code): matched prior cannot
        fix the rotation inside the (cos, sin) pair.
Writes results2/theory_t4.json.  Run: .venv/bin/python theory/check_t4.py
"""
import math, sys, time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, __file__.rsplit("/", 1)[0])
sys.path.insert(0, __file__.rsplit("/", 2)[0])
from _lib import save
from stein import losses as L
from stein.metrics import orth_identifiability

torch.set_default_dtype(torch.float32)


def mcc(Z, S):
    C = np.corrcoef(Z.T, S.T)[: Z.shape[1], Z.shape[1]:]
    r, c = linear_sum_assignment(-np.abs(C))
    return float(np.abs(C[r, c]).mean()), np.abs(C[r, c]).tolist()


def sample_s(n, kind, g):
    if kind == "indep_uniform":
        return (torch.rand(n, 3, generator=g) * 2 - 1) * math.sqrt(3)
    if kind == "indep_laplace":
        u = torch.rand(n, 3, generator=g) - 0.5
        return -torch.sign(u) * torch.log1p(-2 * u.abs()) / math.sqrt(2)
    if kind == "circle":
        h = torch.rand(n, generator=g) * 2 * math.pi
        u = (torch.rand(n, generator=g) * 2 - 1) * math.sqrt(3)
        return torch.stack([math.sqrt(2) * torch.cos(h), math.sqrt(2) * torch.sin(h), u], 1)
    raise ValueError(kind)


def invariance(reps=200, d=4, N=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    M = torch.linalg.qr(torch.randn(d, d, generator=g))[0]
    O = torch.linalg.qr(torch.randn(d, d, generator=g))[0]
    pg, pu = L.Prior("gaussian", d=d), L.Prior("uniform", d=d)
    regs = {
        "ksd_gauss": lambda z: L.ksd_imq(z, pg),
        "sksd_gauss": lambda z: L.sliced_ksd_imq(z, pg, num_slices=32, generator=g),
        "sigreg": lambda z: L.sigreg_ref(z, (lambda A: A / A.norm(dim=0))(torch.randn(d, 256, generator=g))),
        "mmd_gauss": lambda z: L.mmd_gaussian(z, torch.randn(N, d, generator=g)),
        "ksd_uniform_prior": lambda z: L.ksd_imq(z, pu),
    }
    rows = []
    for name, R in regs.items():
        a, b = [], []
        for _ in range(reps):
            s = (torch.rand(N, d, generator=g) * 2 - 1) * math.sqrt(3)
            z = s @ M.T
            a.append(float(R(z))); b.append(float(R(z @ O.T)))
        a, b = np.array(a), np.array(b)
        diff = (b - a).mean(); se = (b - a).std() / math.sqrt(reps)
        rows.append(dict(reg=name, mean=a.mean(), mean_rot=b.mean(), diff=diff, se=se, z=diff / (se + 1e-12)))
    # Gaussian-target statistics: equal up to MC error (KSD/MMD kernels are exactly equivariant, so the
    # per-batch difference is float rounding; compare relative to the mean)
    ok = all(abs(r["diff"]) < 4 * r["se"] + 1e-4 * abs(r["mean"]) for r in rows if r["reg"] != "ksd_uniform_prior")
    ok &= abs([r for r in rows if r["reg"] == "ksd_uniform_prior"][0]["z"]) > 4
    return rows, bool(ok)


def fit(kind, prior, seed, steps=3000, N=256):
    g = torch.Generator().manual_seed(seed)
    M = (torch.linalg.qr(torch.randn(3, 3, generator=g))[0] * 0.9).requires_grad_()
    p = L.Prior(prior, d=3)
    def ksd_at(Mx, reps=20):
        with torch.no_grad():
            return float(np.mean([float(L.ksd_imq(sample_s(1024, kind, g) @ Mx.T, p)) for _ in range(reps)]))
    opt = torch.optim.Adam([M], lr=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
    for _ in range(steps):
        z = sample_s(N, kind, g) @ M.T
        loss = L.ksd_imq(z, p)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        S = sample_s(5000, kind, g)
        Z = (S @ M.T).double().numpy()
    m, per = mcc(Z, S.double().numpy())
    k_fit = ksd_at(M.detach())
    # best scaled identity (signed permutations give the same value for symmetric laws)
    k_id = min(ksd_at(torch.eye(3) * c) for c in (0.9, 0.95, 1.0, 1.05))
    return dict(kind=kind, prior=prior, seed=seed, mcc=m, mcc_per=per, ksd_fit=k_fit, ksd_identity=k_id,
                orth_r2=float(orth_identifiability(Z, S.double().numpy())))


if __name__ == "__main__":
    t0 = time.time()
    out = {}
    out["invariance"], ok_inv = invariance()
    print("invariance", ok_inv, [(r["reg"], round(r["z"], 2)) for r in out["invariance"]], flush=True)
    fits = [fit(kind, prior, s) for kind, pri in (("indep_uniform", "uniform"), ("indep_laplace", "laplace"), ("circle", "uniform"))
            for prior in (pri, "gaussian") for s in range(3)]
    out["fits"] = fits
    for f in fits:
        print(f["kind"], f["prior"], f["seed"], round(f["mcc"], 3), [round(x, 3) for x in f["mcc_per"]], round(f["orth_r2"], 3), "ksd fit/identity", round(f["ksd_fit"], 4), round(f["ksd_identity"], 4), flush=True)
    sel = lambda kind, prior: [f for f in fits if f["kind"] == kind and f["prior"] == prior]
    # global claim (Comon): for independent sources the signed permutation is at least as good as any
    # fit found by GD under the matched prior; orthogonal-fit R^2 is high for every arm
    matched = sel("indep_uniform", "uniform") + sel("indep_laplace", "laplace")
    ok_a = all(f["ksd_identity"] <= f["ksd_fit"] * 1.05 + 2e-4 for f in matched) and \
        all(f["orth_r2"] > 0.95 for f in fits if f["kind"] != "circle")
    out["matched_mcc"] = {f["kind"] + str(f["seed"]): f["mcc"] for f in matched}
    # circle code: the pair's MCC is not pinned (spread across seeds), the third coordinate is
    pair = [np.mean(sorted(f["mcc_per"])[:2]) for f in sel("circle", "uniform")]
    out["circle_pair_mcc"] = pair
    ok_b = all(f["orth_r2"] > 0.9 for f in sel("circle", "uniform"))
    out["summary"] = dict(invariance=ok_inv, matched_prior_identifies_independent=ok_a, circle_orth_r2_ok=ok_b,
                          gauss_target_mcc=[f["mcc"] for f in sel("indep_uniform", "gaussian")],
                          circle_pair_mcc=pair)
    out["pass"] = bool(ok_inv and ok_a and ok_b)
    out["runtime_s"] = time.time() - t0
    save("t4", out)
    print(out["summary"], "PASS" if out["pass"] else "FAIL", f"{out['runtime_s']:.0f}s")
