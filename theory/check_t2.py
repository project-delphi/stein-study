"""Numerical check of T2: thresholds of the study's regularisers in the linear-Gaussian model.

1. Each population formula in _lib.py is checked against the batch estimator in stein/losses.py by
   Monte Carlo (Gaussian batches of N = 256).
2. Scale reduction: with the fast (optimal linear) predictor the objective is
   lam * [ t * tr(Q) + R(Sigma) ],  t = c_p alpha (2 - alpha) / lam,
   so the minimiser depends on t only and lam* = c_p alpha (2 - alpha) / t*. For orthogonally invariant
   R (Gaussian targets) it reduces to the eigenvalues: k content + r = min(m, d - k) nuisance
   eigenvalues free, the rest 0. t* is found by bisection on the global eigenvalue minimum; t_loc is
   the local (slope) criterion t_loc = -dR/dv at v = 0; t_quad is the second-order (curvature) proxy.
3. Gradient descent on (A, B) with the fast predictor and the population regulariser, several starts,
   compared with the eigenvalue solution (nuisance variance and objective).
4. VICReg (not orthogonally invariant): threshold located by GD on full (A, B) only (heuristic).
Writes results2/theory_t2.json.  Run: .venv/bin/python theory/check_t2.py
"""
import math, sys, time
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
sys.path.insert(0, __file__.rsplit("/", 2)[0])
from _lib import (cov_pop, ksd_pop, sksd_pop, sigreg_pop, mmd_pop, vicreg_pop, ep_expected, ksd1_interp,
                  sphere_dirs, pred_loss, save)
from stein import losses as L

N = 256
F = torch.nn.functional


def make_regs(d):
    dirs = sphere_dirs(d, 4096, seed=1)
    return {
        "cov": lambda ev: cov_pop(ev, N),
        "ksd": ksd_pop,
        "sksd": lambda ev: sksd_pop(ev, dirs),
        "sigreg": lambda ev: sigreg_pop(ev, dirs, N),
        "mmd": mmd_pop,
    }


# ----------------------------------------------------------------------------- 1. MC validation

def mc_validate(d=12, reps=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    prior = L.Prior("gaussian", d=d)
    regs = make_regs(d)
    est = {
        "cov": lambda z: ((torch.cov(z.T) - torch.eye(d)) ** 2).sum(),
        "ksd": lambda z: L.ksd_imq(z, prior),
        "sksd": lambda z: L.sliced_ksd_imq(z, prior, num_slices=32, generator=g),
        "sigreg": lambda z: L.sigreg_ref(z, (lambda A: A / A.norm(dim=0))(torch.randn(d, 256, generator=g))),
        "mmd": lambda z: L.mmd_gaussian(z, torch.randn(N, d, generator=g)),
    }
    cases = {"I": torch.ones(d), "aniso": torch.tensor([1.0] * 6 + [0.3] * 5 + [0.0] * (d - 11)),
             "big": torch.tensor([1.6] * 6 + [0.7] * (d - 6))}
    rows, ok = [], True
    for cname, ev in cases.items():
        for name in regs:
            vals = []
            for _ in range(reps):
                z = torch.randn(N, d, generator=g) * ev.sqrt()
                vals.append(float(est[name](z)))
            vals = np.array(vals)
            mc, se = vals.mean(), vals.std() / math.sqrt(reps)
            pop = float(regs[name](ev))
            good = abs(mc - pop) <= 4 * se + 0.02 * abs(pop) + 1e-4
            ok &= good
            rows.append(dict(case=cname, reg=name, mc=mc, se=se, pop=pop, pass_=bool(good)))
    return rows, ok


# ----------------------------------------------------------------------------- 2. eigenvalue problem

def eig_min(R, d, k, r, t, starts=((1.0, 1e-3), (1.0, 0.3), (1.0, 1.0))):
    """min over content eigs c (k) and nuisance eigs v (r) of t * sum(v) + R(c, v, 0...)."""
    best = None
    for c0, v0 in starts:
        xc = torch.full((k,), math.log(math.expm1(c0))).requires_grad_()
        xv = torch.full((r,), math.log(math.expm1(v0))).requires_grad_() if r else torch.zeros(0)
        params = [xc] + ([xv] if r else [])
        opt = torch.optim.LBFGS(params, max_iter=400, tolerance_grad=1e-11, tolerance_change=1e-14,
                                line_search_fn="strong_wolfe")
        def f():
            ev = torch.cat([F.softplus(xc), F.softplus(xv) if r else torch.zeros(0), torch.zeros(d - k - r)])
            return t * ev[k:k + r].sum() + R(ev)
        def closure():
            opt.zero_grad(); val = f(); val.backward(); return val
        for _ in range(3):
            opt.step(closure)
        with torch.no_grad():
            val = float(f())
            c = F.softplus(xc).tolist(); v = F.softplus(xv).tolist() if r else []
        if best is None or val < best[0] - 1e-12:
            best = (val, c, v)
    return dict(f=best[0], c=best[1], v=best[2])


def threshold(R, d, k, r, lo=1e-5, hi=1e3, it=28, vthr=1e-3):
    """t* = sup{t : best minimiser has nuisance}; bisection in log t."""
    has = lambda t: r > 0 and max(eig_min(R, d, k, r, t)["v"]) > vthr
    if not has(lo):
        return None
    if has(hi):
        return float("inf")
    for _ in range(it):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if has(mid) else (lo, mid)
    return math.sqrt(lo * hi)


def slope_and_curv(R, d, k, r):
    """t_loc = -dR/dv at v = 0 (one nuisance direction, content at its R-only optimum, other
    nuisance directions 0); t_quad = 2 * curvature of R along one eigenvalue at Sigma = I (the
    second-order proxy: R ~ kappa (v - 1)^2 gives t = 2 kappa)."""
    base = eig_min(R, d, k, 0, 0.0)   # r = 0: content only
    c = torch.tensor(base["c"])
    v = torch.zeros(1, requires_grad=True)
    ev = torch.cat([c, v, torch.zeros(d - k - 1)])
    gv, = torch.autograd.grad(R(ev), v)
    t_loc = float(-gv)
    with torch.no_grad():               # central finite difference (the sliced-KSD table is piecewise linear)
        hstep = 0.05
        e0 = torch.ones(d); ep = e0.clone(); ep[0] += hstep; em = e0.clone(); em[0] -= hstep
        h = float((R(ep) + R(em) - 2 * R(e0)) / hstep ** 2)
    t_quad = h                        # R ~ (h/2)(v-1)^2  =>  kappa = h/2, t = 2 kappa = h
    return t_loc, t_quad


# ----------------------------------------------------------------------------- 3. GD on (A, B)

def gd_fast(Rmat, d, k, m, alpha, lam, cp, seed, steps=2500, lr=2e-2):
    g = torch.Generator().manual_seed(seed)
    A = (0.6 * torch.randn(d, k, generator=g)).requires_grad_()
    B = (0.6 * torch.randn(d, m, generator=g)).requires_grad_()
    opt = torch.optim.Adam([A, B], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 1e-3)
    def obj():
        P, Q = A @ A.T, B @ B.T
        S = P + Q
        with torch.no_grad():
            W = (P + (1 - alpha) * Q) @ torch.linalg.inv(S + 1e-10 * torch.eye(d))
        return pred_loss(A, B, alpha, "param", cp, W) + lam * Rmat(S)
    for _ in range(steps):
        J = obj()
        opt.zero_grad(); J.backward(); opt.step(); sch.step()
    with torch.no_grad():
        Q, P = B @ B.T, A @ A.T
        S = P + Q
        W = (P + (1 - alpha) * Q) @ torch.linalg.inv(S + 1e-10 * torch.eye(d))
        J = float(pred_loss(A, B, alpha, "param", cp, W) + lam * Rmat(S))
        qe = torch.linalg.eigvalsh(Q).flip(0)[:m].tolist()
        pe = torch.linalg.eigvalsh(P).flip(0)[:k].tolist()
        share = float(torch.trace(Q) / torch.trace(S))
        diag = torch.diagonal(S).tolist()
    return dict(J=J, nuis_eigs=qe, content_eigs=pe, nuis_share=share, diag=diag)


def eig_to_mat(R):
    return lambda S: R(torch.linalg.eigvalsh(0.5 * (S + S.T)).clamp_min(0))


if __name__ == "__main__":
    t0 = time.time()
    out = {"N": N}
    out["mc_validation"], ok_mc = mc_validate()
    print("MC validation:", ok_mc, f"{time.time() - t0:.0f}s", flush=True)

    k, m = 6, 5
    thr, ok_thr = {}, True
    for d in (6, 12, 24):
        r = min(m, max(d - k, 0))
        regs = make_regs(d)
        for name, R in regs.items():
            ts = threshold(R, d, k, r)
            t_loc, t_quad = slope_and_curv(R, d, k, r) if r else (None, None)
            row = dict(d=d, r=r, t_star=ts, t_loc=t_loc, t_quad=t_quad)
            cp = 1.0 / d
            if ts:
                row["lam_star_code"] = {str(a): cp * a * (2 - a) / ts for a in (0.25, 0.5, 1.0)}
                row["lam_trap_code"] = cp / t_loc if t_loc else None   # empty direction, W u = 0: c(u) = 1
                row["first_order"] = bool(abs(ts - t_loc) / t_loc > 0.02)
                # nuisance variance along the path lam / lam* in a grid (alpha = 1)
                path = []
                for q in (1.25, 1.5, 2, 3, 5, 10):
                    sol = eig_min(R, d, k, r, ts / q)
                    path.append(dict(lam_over_star=q, v_mean=float(np.mean(sol["v"])), c_mean=float(np.mean(sol["c"])),
                                     one_minus_inv=1 - 1 / q))
                row["path"] = path
            thr[f"{name}_d{d}"] = row
            print(name, d, {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in row.items() if k_ not in ("path", "lam_star_code")},
                  f"{time.time() - t0:.0f}s", flush=True)
    # closed form for cov in population is t* = 2 (finite-N shifts it slightly)
    out["thresholds"] = thr

    # 3. GD check at d = 12, alpha = 1 and alpha = 0.5
    d, cp = 12, 1.0 / 12
    regs = make_regs(d)
    gd_rows, ok_gd = [], True
    for name, R in regs.items():
        ts = thr[f"{name}_d{d}"]["t_star"]
        for a in (1.0, 0.5):
            for q in (0.6, 2.0, 5.0):
                lam = cp * a * (2 - a) / ts * q
                sol = eig_min(R, d, k, min(m, d - k), ts / q)
                runs = [gd_fast(eig_to_mat(R), d, k, m, a, lam, cp, s) for s in range(2)]
                best = min(runs, key=lambda x: x["J"])
                Jref = lam * sol["f"]
                v_ref = sol["v"] if max(sol["v"]) > 1e-3 else [0.0] * len(sol["v"])
                ev_err = max(abs(x - y) for x, y in zip(sorted(best["nuis_eigs"], reverse=True), sorted(v_ref, reverse=True)))
                J_rel = (best["J"] - Jref) / max(abs(Jref), 1e-9)
                good = ev_err < 0.02 and abs(J_rel) < 5e-3 and all(x["J"] >= Jref * (1 - 5e-3) for x in runs)
                ok_gd &= good
                gd_rows.append(dict(reg=name, alpha=a, lam=lam, lam_over_star=q, v_ref=v_ref, gd_best=best,
                                    J_ref=Jref, ev_err=ev_err, J_rel=J_rel, pass_=bool(good)))
                print("GD", name, a, q, round(ev_err, 4), round(J_rel, 5), good, f"{time.time() - t0:.0f}s", flush=True)
    out["gd_check"] = gd_rows

    # 4. VICReg: GD only (not orthogonally invariant; heuristic threshold)
    vic = []
    for a in (1.0,):
        for lam in np.geomspace(0.01, 1000.0, 16):
            runs = [gd_fast(vicreg_pop, d, k, m, a, float(lam), cp, s, steps=3000) for s in range(3)]
            best = min(runs, key=lambda x: x["J"])
            vic.append(dict(alpha=a, lam=float(lam), J=best["J"], nuis_share=best["nuis_share"],
                            nuis_eigs=best["nuis_eigs"], content_eigs=best["content_eigs"], diag=best["diag"],
                            spread_J=max(x["J"] for x in runs) - best["J"]))
            print("VIC", round(float(lam), 4), round(best["nuis_share"], 4), [round(x, 3) for x in best["nuis_eigs"]],
                  f"{time.time() - t0:.0f}s", flush=True)
    out["vicreg_scan"] = vic

    out["summary"] = dict(mc_validation=ok_mc, gd_check=ok_gd)
    out["pass"] = ok_mc and ok_gd
    out["runtime_s"] = time.time() - t0
    save("t2", out)
    print(out["summary"], "PASS" if out["pass"] else "FAIL", f"{out['runtime_s']:.0f}s")
