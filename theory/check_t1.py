"""Numerical check of T1 (covariance penalty, linear encoder) and its corollaries.

Gradient descent (Adam, cosine-decayed) on the linear model z = A s + B n from several random
starts, compared with the closed form in theory.md:
  A. identity predictor, population objective  2 a c_p ||B||^2 + lam ||AA'+BB'-I||^2
  B. learned linear predictor W trained jointly (population objective)
  C. the real minibatch setting: Gaussian s, n, n' sampled, batch N = 256, code units c_p = 1/d,
     symmetric MSE loss and batch-covariance penalty as in stein/train.py; compared with the
     finite-N expected objective minimised over eigenvalues
  D. corollary (d): give the encoder d-k extra unit-variance functions of s uncorrelated with s
     (pairwise products) and check that the minimum is 0 with B = 0 above the threshold
Writes results2/theory_t1.json.  Run: .venv/bin/python theory/check_t1.py
"""
import math, sys, time
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _lib import pred_loss, cov_pop, save

ZERO = 1e-6     # a direction with variance below this is "empty"


def closed_form(d, k, m, alpha, lam, kappa_per_alpha="identity", cp=1.0):
    """Population minimiser for the covariance penalty. kappa = invariance cost per unit nuisance
    variance: 2 alpha c_p (identity predictor) or alpha (2 - alpha) c_p (optimal linear predictor)."""
    kap = (2 * alpha if kappa_per_alpha == "identity" else alpha * (2 - alpha)) * cp
    kc = min(k, d)
    r = min(m, max(d - k, 0))
    v = max(0.0, 1 - kap / (2 * lam)) if (lam > 0 and r > 0) else 0.0
    J = r * (kap * v + lam * (v - 1) ** 2) + lam * max(d - k - r, 0)
    leak = (r / m) if v > 0 else 0.0
    share = r * v / (kc + r * v)
    return dict(lam_star=kap / 2, v=v, r=r, J=J, leak_r2=leak, nuis_share=share,
                content_eigs=[1.0] * kc)


def gd(d, k, m, alpha, lam, predictor, seed, steps=4000, lr=2e-2, cp=1.0, extra=0):
    g = torch.Generator().manual_seed(seed)
    A = (0.5 * torch.randn(d, k + extra, generator=g)).requires_grad_()
    B = (0.5 * torch.randn(d, m, generator=g)).requires_grad_()
    params = [A, B]
    W = None
    if predictor == "param":
        W = (torch.eye(d) + 0.3 * torch.randn(d, d, generator=g)).requires_grad_()
        params.append(W)
    opt = torch.optim.Adam(params, lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 1e-4)
    for _ in range(steps):
        S = A @ A.T + B @ B.T
        if predictor == "optimal":       # fast-predictor limit: W set to its optimum each step (envelope theorem)
            with torch.no_grad():
                C = A @ A.T + (1 - alpha) * (B @ B.T)
                Wopt = C @ torch.linalg.inv(S + 1e-10 * torch.eye(d))
            J = pred_loss(A, B, alpha, "param", cp, Wopt) + lam * ((S - torch.eye(d)) ** 2).sum()
        else:
            J = pred_loss(A, B, alpha, predictor, cp, W) + lam * ((S - torch.eye(d)) ** 2).sum()
        opt.zero_grad(); J.backward(); opt.step(); sch.step()
    with torch.no_grad():
        P, Q = A @ A.T, B @ B.T
        S = P + Q
        if predictor == "optimal":
            W = (P + (1 - alpha) * Q) @ torch.linalg.inv(S + 1e-10 * torch.eye(d))
        J = float(pred_loss(A, B, alpha, "param" if W is not None else predictor, cp, W) + lam * ((S - torch.eye(d)) ** 2).sum())
        qe = torch.linalg.eigvalsh(Q).flip(0)
        pe = torch.linalg.eigvalsh(P).flip(0)
        r_emp = int((qe > ZERO).sum())
        # probe R^2 restricted to non-empty directions (scale invariance makes any residual count)
        ev, U = torch.linalg.eigh(S)
        keep = ev > ZERO
        Sp = (U[:, keep] / ev[keep]) @ U[:, keep].T
        leak = float(torch.trace(B.T @ Sp @ B) / m)
        share = float(torch.trace(Q) / torch.trace(S))
        # content and nuisance subspaces orthogonal?
        cross = float((A.T @ B).norm() / (A.norm() * B.norm() + 1e-12))
        # metastability: invariance cost per unit variance in the empty subspace, c(u) = u'Mu
        cmin = None
        if W is not None and (~keep).any():
            U0 = U[:, ~keep]
            Mx = W.T @ W - (1 - alpha) * (W + W.T) + torch.eye(d)
            cmin = float(torch.linalg.eigvalsh(U0.T @ Mx @ U0).min()) * cp
    return dict(c_empty_min=cmin, J=J, nuis_eigs=qe[:max(r_emp, 1)].tolist(), r=r_emp, content_eigs=pe[:min(k, d)].tolist(),
                leak_r2=leak, nuis_share=share, AtB_rel=cross)


def construct_J(d, k, m, alpha, lam, cf, predictor):
    """Objective at the explicit minimiser of theory.md: content on the first min(k, d) axes,
    nuisance variance v on the next r axes."""
    A = torch.zeros(d, k); B = torch.zeros(d, m)
    kc = min(k, d)
    A[:kc, :kc] = torch.eye(kc)
    for j in range(cf["r"]):
        B[kc + j, j] = math.sqrt(cf["v"])
    S = A @ A.T + B @ B.T
    if predictor == "optimal":
        W = (A @ A.T + (1 - alpha) * B @ B.T) @ torch.linalg.pinv(S, hermitian=True)
        L = pred_loss(A, B, alpha, "param", 1.0, W)
    else:
        L = pred_loss(A, B, alpha, predictor, 1.0)
    return float(L + lam * ((S - torch.eye(d)) ** 2).sum())


def compare(cf, res, tol):
    errs = dict(
        J=abs(res["J"] - cf["J"]) / max(1.0, abs(cf["J"])),
        v=max(abs(x - cf["v"]) for x in res["nuis_eigs"]) if cf["v"] > 0 else max(res["nuis_eigs"]),
        r=0.0 if (cf["v"] == 0 and res["r"] == 0) or (cf["v"] > 0 and res["r"] == cf["r"]) else 1.0,
        content=max(abs(x - 1) for x in res["content_eigs"]),
        share=abs(res["nuis_share"] - cf["nuis_share"]),
        leak=abs(res["leak_r2"] - cf["leak_r2"]))
    return errs, all(e <= tol for e in errs.values())


def part_pop(predictor, configs, alphas, rel, starts, tol=2e-3, best_of=False):
    rows, ok = [], True
    for (d, k, m) in configs:
        for a in alphas:
            for q in rel:
                kind = "identity" if predictor == "identity" else "optimal"
                lam_star = closed_form(d, k, m, a, 1.0, kind)["lam_star"]
                lam = q * lam_star
                cf = closed_form(d, k, m, a, lam, kind)
                grp = []
                for s in range(starts):
                    res = gd(d, k, m, a, lam, predictor, seed=s)
                    errs, good = compare(cf, res, tol)
                    grp.append(dict(d=d, k=k, m=m, alpha=a, lam=lam, lam_over_star=q, seed=s,
                                    closed=cf, gd=res, err=errs, pass_=good))
                if best_of:
                    # global claim: the explicit closed-form minimiser attains J*, no start goes below it,
                    # and every start either reaches it or sits in an empty-direction trap (c(u) >= 2 lam)
                    Jc = construct_J(d, k, m, a, lam, cf, predictor)
                    good = abs(Jc - cf["J"]) <= tol * max(1, cf["J"]) and \
                        all(r["gd"]["J"] >= cf["J"] - tol * max(1, cf["J"]) for r in grp) and \
                        all(r["pass_"] or (r["gd"]["c_empty_min"] is not None and r["gd"]["c_empty_min"] >= 2 * lam - tol)
                            for r in grp)
                    for r in grp:
                        r["group_pass"] = good
                else:
                    good = all(r["pass_"] for r in grp)
                ok &= good
                rows += grp
    return rows, ok


def part_minibatch(d=12, k=6, m=5, alpha=1.0, rels=(0.5, 2.0, 5.0), N=256, steps=6000, seed=0):
    """Real minibatch training with code units; reference = finite-N expected objective minimised
    over eigenvalues (k content + r nuisance + rest zero)."""
    cp = 1.0 / d
    kap = alpha * (2 - alpha) * cp
    rows, ok = [], True
    for q in rels:
        lam = q * kap / 2
        # reference: minimise kap * sum(v) + lam * E cov(ev, N) over eigenvalues
        r = min(m, d - k)
        x = torch.zeros(k + r).requires_grad_()
        opt = torch.optim.LBFGS([x], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-14)
        def f():
            ev = torch.cat([torch.nn.functional.softplus(x[:k] * 5) / 5, torch.nn.functional.softplus(x[k:] * 50) / 50,
                            torch.zeros(d - k - r)])
            return kap * ev[k:k + r].sum() + lam * cov_pop(ev, N)
        def closure():
            opt.zero_grad(); L = f(); L.backward(); return L
        x.data[:k] = 1.0; x.data[k:] = 0.5
        opt.step(closure)
        with torch.no_grad():
            ev_ref = torch.cat([torch.nn.functional.softplus(x[:k] * 5) / 5, torch.nn.functional.softplus(x[k:] * 50) / 50])
        c_ref, v_ref = ev_ref[:k].mean().item(), ev_ref[k:].mean().item()
        if v_ref < 1e-3:
            v_ref = 0.0
        # stochastic training
        g = torch.Generator().manual_seed(seed)
        A = (0.5 * torch.randn(d, k, generator=g)).requires_grad_()
        B = (0.5 * torch.randn(d, m, generator=g)).requires_grad_()
        W = (torch.eye(d) + 0.1 * torch.randn(d, d, generator=g)).requires_grad_()
        opt2 = torch.optim.Adam([A, B, W], lr=5e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, steps, eta_min=5e-6)
        for t in range(steps):
            s = torch.randn(N, k, generator=g)
            n1 = torch.randn(N, m, generator=g)
            res = torch.rand(N, m, generator=g) < alpha
            n2 = torch.where(res, torch.randn(N, m, generator=g), n1)
            z1, z2 = s @ A.T + n1 @ B.T, s @ A.T + n2 @ B.T
            pl = 0.5 * (torch.nn.functional.mse_loss(z1 @ W.T, z2) + torch.nn.functional.mse_loss(z2 @ W.T, z1))
            def covpen(z):
                zc = z - z.mean(0)
                C = zc.T @ zc / (N - 1)
                return ((C - torch.eye(d)) ** 2).sum()
            L = pl + lam * 0.5 * (covpen(z1) + covpen(z2))
            opt2.zero_grad(); L.backward(); opt2.step(); sch.step()
        with torch.no_grad():
            qe = torch.linalg.eigvalsh(B @ B.T).flip(0)[:r]
            pe = torch.linalg.eigvalsh(A @ A.T).flip(0)[:k]
        v_gd = qe.mean().item()
        good = abs(v_gd - v_ref) < 0.03 and abs(pe.mean().item() - c_ref) < 0.03
        ok &= good
        rows.append(dict(lam=lam, lam_over_star_pop=q, v_ref=v_ref, c_ref=c_ref, v_pop_formula=max(0, 1 - 1 / q),
                         v_gd=v_gd, v_gd_all=qe.tolist(), c_gd=pe.mean().item(), pass_=good))
    return rows, ok


def part_nonlinear(d=12, k=6, m=5, alpha=1.0, rels=(2.0, 5.0), starts=2):
    """Encoder sees [s, g(s)] with g = d - k pairwise products s_i s_j (unit variance, uncorrelated
    with s and with each other for Gaussian s), so A is d x d. Minimum should be J = 0, B = 0."""
    rows, ok = [], True
    for q in rels:
        lam = q * alpha
        for s in range(starts):
            res = gd(d, k, m, alpha, lam, "identity", seed=s, extra=d - k)
            lin = closed_form(d, k, m, alpha, lam)["J"]
            good = res["J"] < 1e-4 and res["nuis_share"] < 1e-4
            ok &= good
            rows.append(dict(lam=lam, J_nonlinear=res["J"], J_linear_min=lin, nuis_share=res["nuis_share"],
                             pass_=good))
    return rows, ok


if __name__ == "__main__":
    t0 = time.time()
    rel = (0.5, 0.8, 1.25, 2.0, 5.0)
    out = {}
    out["A_identity"], okA = part_pop("identity", [(12, 6, 5), (12, 6, 8)], (0.25, 0.5, 1.0), rel, 3)
    rA2, okA2 = part_pop("identity", [(4, 6, 5), (6, 6, 5), (24, 6, 5)], (1.0,), rel, 2)
    out["A_identity"] += rA2; okA &= okA2
    # alpha = 0: nuisance is free, v = 1 at any lam > 0
    cf0 = closed_form(12, 6, 8, 0.0, 0.1)
    r0 = gd(12, 6, 8, 0.0, 0.1, "identity", 0)
    # alpha = 0: P + Q = I is optimal with any split (minimiser not unique); check J = 0, Sigma = I, B != 0
    okA0 = r0["J"] < 1e-6 and r0["leak_r2"] > 0.5 and r0["nuis_share"] > 0.1
    out["A_alpha0"] = dict(closed=cf0, gd=r0, pass_=okA0)
    # B1: predictor profiled out (W = C Sigma^+, the fast-predictor limit): must match the closed form
    out["B_learned_linear"], okB = part_pop("optimal", [(12, 6, 5), (12, 6, 8), (4, 6, 5)], (0.25, 0.5, 1.0), rel, 3, best_of=True)
    out["B_traps"] = sum(not r["pass_"] for r in out["B_learned_linear"])
    # B2: W trained jointly by GD from W ~ I: an observation, not a pass/fail test. Runs that miss the
    # global minimum must sit at a local minimum with an empty direction whose cost c(u) >= 2 lam.
    rowsB2, _ = part_pop("param", [(12, 6, 5), (12, 6, 8)], (0.25, 0.5, 1.0), rel, 2)
    n_glob = sum(r["pass_"] for r in rowsB2)
    meta_ok = all(r["pass_"] or (r["gd"]["c_empty_min"] is not None and r["gd"]["c_empty_min"] >= 2 * r["lam"] - 2e-3)
                  for r in rowsB2)
    out["B2_joint_W"] = dict(rows=rowsB2, n=len(rowsB2), n_global=n_glob, misses_are_local_minima=meta_ok)
    okB &= meta_ok
    out["C_minibatch"], okC = part_minibatch()
    out["D_nonlinear"], okD = part_nonlinear()
    summary = dict(A_identity=okA, A_alpha0=okA0, B_learned_linear=okB, C_minibatch=okC, D_nonlinear=okD)
    out["summary"] = summary
    out["pass"] = all(summary.values())
    out["tolerance"] = dict(pop=2e-3, minibatch=0.03, zero_variance=ZERO)
    out["runtime_s"] = time.time() - t0
    save("t1", out)
    print(summary, "PASS" if out["pass"] else "FAIL", f"{out['runtime_s']:.0f}s")
    print("B traps (starts not at global min):", out["B_traps"], "/", len(out["B_learned_linear"]),
          "| B2 joint W: global", out["B2_joint_W"]["n_global"], "/", out["B2_joint_W"]["n"],
          "misses are local minima:", out["B2_joint_W"]["misses_are_local_minima"])
    for name in ("A_identity", "B_learned_linear"):
        bad = [r for r in out[name] if not r.get("group_pass", r["pass_"])]
        for r in bad[:8]:
            print(name, r["d"], r["m"], r["alpha"], r["lam_over_star"], r["seed"], {k: round(v, 4) for k, v in r["err"].items()})
    for r in out["C_minibatch"]:
        print("C", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "v_gd_all"})
    for r in out["D_nonlinear"]:
        print("D", r)
