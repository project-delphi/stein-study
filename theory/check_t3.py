"""Numerical check of T3: noise on the predictor input in the linear-Gaussian model.

Fast linear predictor, embedding principal axes = coordinate axes (assumption A-diag in theory.md).
Per unit of variance, the invariance cost of a direction is
  batch-scaled noise sigma * sd(z) * xi (scale detached, fixed points):
     content  kappa_s = (sigma^2 / (1 + sigma^2))^2
     nuisance kappa_n = w^2 - 2 w (1 - alpha) + 1,  w = (1 - alpha) / (1 + sigma^2)
  fixed-scale noise sigma * xi (a true objective):
     content  c sigma^2 / (c + sigma^2)            (per direction with variance c)
     nuisance v - (1 - alpha)^2 v^2 / (v + sigma^2)
Objective c_p * sum(costs) + lam * R(eigenvalues).  Part 1 solves it over eigenvalues for each
regulariser and reports the nuisance threshold lam*(sigma) and the variance shares at fixed lam.
Part 2 trains the linear model by minibatch SGD with real noise injection (cov and KSD, batch 256)
and compares with Part 1.  Writes results2/theory_t3.json.  Run: .venv/bin/python theory/check_t3.py
"""
import math, sys, time
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
sys.path.insert(0, __file__.rsplit("/", 2)[0])
from _lib import save
from check_t2 import make_regs
from stein import losses as L

F = torch.nn.functional


def costs(mode, alpha, sigma):
    s2 = sigma ** 2
    if mode == "batch":
        ks = (s2 / (1 + s2)) ** 2
        w = (1 - alpha) / (1 + s2)
        kn = w * w - 2 * w * (1 - alpha) + 1
        return (lambda c: ks * c.sum()), (lambda v: kn * v.sum())
    if mode == "fixed":
        if sigma == 0:
            return (lambda c: 0 * c.sum()), (lambda v: alpha * (2 - alpha) * v.sum())
        return (lambda c: (c * s2 / (c + s2)).sum()), (lambda v: (v - (1 - alpha) ** 2 * v * v / (v + s2)).sum())
    raise ValueError(mode)


def solve(R, d, k, r, lam, cp, alpha, sigma, mode):
    fc, fn = costs(mode, alpha, sigma)
    best = None
    for c0, v0 in ((1.0, 1e-3), (1.0, 0.3), (1.0, 1.0), (0.3, 0.3)):
        xc = torch.full((k,), math.log(math.expm1(c0))).requires_grad_()
        xv = torch.full((r,), math.log(math.expm1(v0))).requires_grad_()
        opt = torch.optim.LBFGS([xc, xv], max_iter=400, tolerance_grad=1e-11, tolerance_change=1e-14,
                                line_search_fn="strong_wolfe")
        def f():
            c, v = F.softplus(xc), F.softplus(xv)
            ev = torch.cat([c, v, torch.zeros(d - k - r)])
            return cp * (fc(c) + fn(v)) / lam + R(ev)
        def closure():
            opt.zero_grad(); val = f(); val.backward(); return val
        for _ in range(3):
            opt.step(closure)
        with torch.no_grad():
            val = float(f()); c = F.softplus(xc).numpy().copy(); v = F.softplus(xv).numpy().copy()
        if best is None or val < best[0] - 1e-12:
            best = (val, c, v)
    _, c, v = best
    m = 5
    vm = float(v.mean()) if v.max() > 1e-3 else 0.0
    share = float(v.sum() / (c.sum() + v.sum())) if vm > 0 else 0.0
    return dict(c_mean=float(c.mean()), v_mean=vm, nuis_share=share, leak_r2_pop=(r / m if vm > 0 else 0.0))


def lam_threshold(R, d, k, r, cp, alpha, sigma, mode, lo=1e-3, hi=1e3, it=26):
    has = lambda lam: solve(R, d, k, r, lam, cp, alpha, sigma, mode)["v_mean"] > 0
    if has(lo):
        return 0.0
    if not has(hi):
        return float("inf")
    for _ in range(it):
        mid = math.sqrt(lo * hi)
        lo, hi = (lo, mid) if has(mid) else (mid, hi)
    return math.sqrt(lo * hi)


def sgd(reg, d, k, m, alpha, sigma, mode, lam, steps=5000, N=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    prior = L.Prior("gaussian", d=d)
    A = (0.5 * torch.randn(d, k, generator=g)).requires_grad_()
    B = (0.5 * torch.randn(d, m, generator=g)).requires_grad_()
    W = (torch.eye(d) + 0.1 * torch.randn(d, d, generator=g)).requires_grad_()
    opt = torch.optim.Adam([{"params": [A, B], "lr": 3e-3}, {"params": [W], "lr": 3e-2}])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-6)
    def R(z):
        if reg == "cov":
            zc = z - z.mean(0); C = zc.T @ zc / (N - 1)
            return ((C - torch.eye(d)) ** 2).sum()
        return L.ksd_imq(z, prior)
    def noisy(z):
        if sigma == 0:
            return z
        sc = z.detach().std(0, keepdim=True) if mode == "batch" else 1.0
        return z + sigma * sc * torch.randn(z.shape, generator=g)
    for t in range(steps):
        s = torch.randn(N, k, generator=g)
        n1 = torch.randn(N, m, generator=g)
        n2 = torch.where(torch.rand(N, m, generator=g) < alpha, torch.randn(N, m, generator=g), n1)
        z1, z2 = s @ A.T + n1 @ B.T, s @ A.T + n2 @ B.T
        pl = 0.5 * (F.mse_loss(noisy(z1) @ W.T, z2) + F.mse_loss(noisy(z2) @ W.T, z1))
        loss = pl + lam * 0.5 * (R(z1) + R(z2))
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        P, Q = A @ A.T, B @ B.T
        pe = torch.linalg.eigvalsh(P).flip(0)[:k]
        qe = torch.linalg.eigvalsh(Q).flip(0)[:min(m, d - k)]
        S = P + Q
        offdiag = float((S - torch.diag(torch.diagonal(S))).norm() / S.norm())
    vm = float(qe.mean()) if qe.max() > 1e-2 else 0.0
    return dict(c_mean=float(pe.mean()), v_mean=vm, nuis_share=float(torch.trace(Q) / torch.trace(S)),
                offdiag_rel=offdiag)


if __name__ == "__main__":
    t0 = time.time()
    d, k, m = 12, 6, 5
    r, cp = min(m, d - k), 1.0 / d
    regs = make_regs(d)
    out = {"part1": [], "part2": []}
    base = {}
    for name, R in regs.items():
        for alpha in (1.0, 0.5):
            for mode in ("batch", "fixed"):
                for sigma in (0.0, 0.5, 1.0):
                    if mode == "fixed" and sigma == 0:
                        continue
                    ls = lam_threshold(R, d, k, r, cp, alpha, sigma, mode)
                    if sigma == 0:
                        base[(name, alpha)] = ls
                    l0 = base[(name, alpha)]
                    at = {str(q): solve(R, d, k, r, q * l0, cp, alpha, sigma, mode) for q in (0.5, 1.5, 3.0)}
                    row = dict(reg=name, alpha=alpha, mode=mode, sigma=sigma, lam_star=ls, lam_star_sigma0=l0,
                               at_lam_over_star0=at)
                    out["part1"].append(row)
                    print(name, alpha, mode, sigma, "lam*", round(ls, 4),
                          {q: (round(a["c_mean"], 3), round(a["v_mean"], 3), round(a["nuis_share"], 3)) for q, a in at.items()},
                          f"{time.time() - t0:.0f}s", flush=True)
    ok = True
    for reg in ("cov", "ksd"):
        l0 = base[(reg, 1.0)]
        for mode, sigma in (("batch", 0.0), ("batch", 1.0), ("fixed", 1.0)):
            lam = 2.0 * l0
            ref = solve(regs[reg], d, k, r, lam, cp, 1.0, sigma, mode)
            got = sgd(reg, d, k, m, 1.0, sigma, mode, lam)
            good = abs(got["c_mean"] - ref["c_mean"]) < 0.05 * max(1, ref["c_mean"]) and abs(got["v_mean"] - ref["v_mean"]) < 0.05
            ok &= good
            out["part2"].append(dict(reg=reg, mode=mode, sigma=sigma, lam=lam, ref=ref, sgd=got, pass_=bool(good)))
            print("SGD", reg, mode, sigma, {k_: round(v, 3) for k_, v in ref.items()}, {k_: round(v, 3) for k_, v in got.items()},
                  good, f"{time.time() - t0:.0f}s", flush=True)
    out["pass"] = bool(ok)
    out["tolerance"] = "SGD vs eigenvalue solution: content mean within 5%, nuisance mean within 0.05"
    out["runtime_s"] = time.time() - t0
    save("t3", out)
    print("PASS" if ok else "FAIL", f"{out['runtime_s']:.0f}s")
