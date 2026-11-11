"""Independent checks of Lemma 1, Theorem 1, Corollary 1 (paper2 / theory.md).
Population objective of the linear model, written from scratch (no use of theory/_lib.py)."""
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(0)
out = {}

def pinv(S):
    return np.linalg.pinv(S, rcond=1e-10, hermitian=True)

# ------------------------------------------------------------- Lemma 1
def learned_cost_direct(P, Q, a):
    S = P + Q; C = P + (1 - a) * Q
    # min_W tr(W S W^T) - 2 tr(W C) + tr S, solved by least squares on W S = C
    W = C @ pinv(S)
    return np.trace(W @ S @ W.T) - 2 * np.trace(W @ C) + np.trace(S)

viol = 0; eq_cases = []
for trial in range(2000):
    d = rng.integers(2, 7); k = rng.integers(1, 4); m = rng.integers(1, 4); a = rng.uniform(0.05, 1)
    A = rng.normal(size=(d, k)); B = rng.normal(size=(d, m))
    if trial % 3 == 0:  # overlapping ranges
        B[:, 0] = A[:, 0]
    P, Q = A @ A.T, B @ B.T
    S = P + Q
    val = learned_cost_direct(P, Q, a)
    formula = 2 * a * np.trace(Q) - a * a * np.trace(Q @ pinv(S) @ Q)
    bound = a * (2 - a) * np.trace(Q)
    if abs(val - formula) > 1e-6 * (1 + abs(val)) or val < bound - 1e-8:
        viol += 1
    gap = val - bound
    ind = np.linalg.norm(P @ pinv(S) @ Q)
    eq_cases.append((gap, ind))
eq_cases = np.array(eq_cases)
# equality iff P S^+ Q = 0
lin_indep = eq_cases[:, 1] < 1e-8
out["lemma1_violations"] = viol
out["lemma1_eq_when_PSQ0_maxgap"] = float(eq_cases[lin_indep, 0].max()) if lin_indep.any() else None
out["lemma1_strict_when_PSQ_ne0_mingap"] = float(eq_cases[~lin_indep, 0].min())
# oblique but independent ranges: P, Q not orthogonal but range P ∩ range Q = {0}
d, k, m = 4, 2, 2
A = rng.normal(size=(d, k)); B = rng.normal(size=(d, m))
P, Q = A @ A.T, B @ B.T
out["lemma1_oblique_gap"] = float(learned_cost_direct(P, Q, 0.5) - 0.75 * np.trace(Q))
out["lemma1_oblique_PSQ"] = float(np.linalg.norm(P @ pinv(P + Q) @ Q))

# ------------------------------------------------------------- Theorem 1 by joint optimisation over (A,B,W)
def make_J(d, k, m, a, lam, cp, pred):
    def J(x):
        A = x[:d * k].reshape(d, k); B = x[d * k:d * k + d * m].reshape(d, m)
        P, Q = A @ A.T, B @ B.T; S = P + Q; C = P + (1 - a) * Q
        R = ((S - np.eye(d)) ** 2).sum()
        if pred == "id":
            L = 2 * a * np.trace(Q)
            gS = 2 * lam * (S - np.eye(d)); gQ = cp * 2 * a * np.eye(d) + gS; gP = gS
            gW = None
        else:
            W = x[d * k + d * m:].reshape(d, d)
            L = np.trace(W @ S @ W.T) - 2 * np.trace(W @ C) + np.trace(S)
            # dL/dS (via P and Q)
            dLdS = W.T @ W + np.eye(d)
            dLdC = -2 * W.T  # d(-2 tr(WC))/dC = -2 W^T
            dLdC = 0.5 * (dLdC + dLdC.T)
            gS = 2 * lam * (S - np.eye(d))
            gP = cp * (dLdS + dLdC) + gS
            gQ = cp * (dLdS + (1 - a) * dLdC) + gS
            gW = cp * (2 * W @ S - 2 * C.T)
        f = cp * L + lam * R
        gA = 2 * 0.5 * (gP + gP.T) @ A; gB = 2 * 0.5 * (gQ + gQ.T) @ B
        g = np.concatenate([gA.ravel(), gB.ravel()] + ([gW.ravel()] if gW is not None else []))
        return f, g
    return J

def closed_form(d, k, m, a, lam, cp, pred):
    kap = 2 * a if pred == "id" else a * (2 - a)
    r = min(m, max(d - k, 0))
    if d <= k:
        return 0.0, 0.0, r
    v = max(0.0, 1 - cp * kap / (2 * lam))
    return r * (cp * kap * v + lam * (v - 1) ** 2) + lam * (d - k - r), v, r

rows = []
worst_below = 0.0; worst_above = 0.0
for (d, k, m) in [(3, 4, 2), (6, 6, 5), (12, 6, 5), (8, 6, 5), (5, 2, 1), (4, 1, 3)]:
    for a in (0.25, 1.0):
        for pred in ("id", "lin"):
            cp = 1.0 / d
            kap = 2 * a if pred == "id" else a * (2 - a)
            lstar = cp * kap / 2
            for q in (0.5, 0.9, 1.1, 2.0, 5.0):
                lam = q * lstar
                Jf = make_J(d, k, m, a, lam, cp, pred)
                best = np.inf; bestx = None
                n = d * k + d * m + (d * d if pred == "lin" else 0)
                for s in range(12):
                    x0 = rng.normal(scale=rng.choice([0.1, 0.5, 1.0]), size=n)
                    res = minimize(Jf, x0, jac=True, method="L-BFGS-B",
                                   options=dict(maxiter=20000, gtol=1e-12, ftol=1e-15))
                    if res.fun < best:
                        best, bestx = res.fun, res.x
                cf, v, r = closed_form(d, k, m, a, lam, cp, pred)
                A = bestx[:d * k].reshape(d, k); B = bestx[d * k:d * k + d * m].reshape(d, m)
                P, Q = A @ A.T, B @ B.T; S = P + Q
                evS = np.sort(np.linalg.eigvalsh(S))[::-1]
                evP = np.sort(np.linalg.eigvalsh(P))[::-1]
                evQ = np.sort(np.linalg.eigvalsh(Q))[::-1]
                # probe R2 of n from z (population)
                probe = np.trace(B.T @ pinv(S) @ B) / m if np.trace(Q) > 1e-6 else 0.0
                rel = (best - cf) / max(1e-12, abs(cf) + 1e-6)
                worst_below = min(worst_below, best - cf)
                worst_above = max(worst_above, best - cf)
                rows.append(dict(d=d, k=k, m=m, a=a, pred=pred, q=q, Jgd=best, Jcf=cf,
                                 v_star=v, Q_top=evQ[:max(r, 1)].round(4).tolist(),
                                 P_top=evP[:min(k, d)].round(4).tolist(), probe=round(probe, 4),
                                 PQcross=float(np.linalg.norm(P @ Q))))
out["thm1_worst_gd_minus_closed_form_below"] = worst_below
out["thm1_worst_gd_minus_closed_form_above"] = worst_above
bad = [r for r in rows if abs(r["Jgd"] - r["Jcf"]) > 1e-5]
out["thm1_n_cases"] = len(rows)
out["thm1_n_mismatch_gt_1e-5"] = len(bad)
out["thm1_mismatch_examples"] = bad[:6]
# structural check on matches
struct_bad = []
for r in rows:
    if abs(r["Jgd"] - r["Jcf"]) > 1e-5:
        continue
    vq = np.array(r["Q_top"]); rr = min(r["m"], max(r["d"] - r["k"], 0))
    if rr > 0 and r["v_star"] > 0:
        if np.abs(vq[:rr] - r["v_star"]).max() > 1e-3: struct_bad.append(r)
        if abs(r["probe"] - rr / r["m"]) > 1e-3: struct_bad.append(r)
    else:
        if vq.max() > 1e-3: struct_bad.append(r)
    if r["PQcross"] > 1e-3: struct_bad.append(r)
out["thm1_structure_violations"] = len(struct_bad)
out["thm1_structure_examples"] = struct_bad[:4]

# ------------------------------------------------------------- random (P,Q) never beat closed form
below = 0
for t in range(20000):
    d = int(rng.integers(2, 9)); k = int(rng.integers(1, 5)); m = int(rng.integers(1, 5))
    a = float(rng.uniform(0.05, 1)); cp = 1.0 / d; pred = "lin"
    lam = float(rng.uniform(0.2, 5)) * cp * a * (2 - a) / 2
    A = rng.normal(scale=rng.uniform(0.1, 1.2), size=(d, k)); B = rng.normal(scale=rng.uniform(0.0, 1.2), size=(d, m))
    P, Q = A @ A.T, B @ B.T; S = P + Q
    J = cp * (2 * a * np.trace(Q) - a * a * np.trace(Q @ pinv(S) @ Q)) + lam * ((S - np.eye(d)) ** 2).sum()
    cf = closed_form(d, k, m, a, lam, cp, pred)[0]
    if J < cf - 1e-9:
        below += 1
out["thm1_random_PQ_below_closed_form"] = below

# ------------------------------------------------------------- alpha = 0 edge case, d > k + m
d, k, m = 12, 6, 5
a = 0.0; lam = 1.0; cp = 1 / 12
Jf = make_J(d, k, m, a, lam, cp, "lin")
best = np.inf
for s in range(10):
    x0 = rng.normal(size=d * k + d * m + d * d)
    res = minimize(Jf, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=20000, gtol=1e-12))
    best = min(best, res.fun)
out["alpha0_d12_k6_m5_minJ_gd"] = best
out["alpha0_claimed_minJ_if_P_plus_Q_eq_I"] = 0.0
out["alpha0_true_minJ_lambda_times_(d-k-m)"] = lam * (d - k - m)

import json
print(json.dumps(out, indent=1, default=float))
json.dump(out, open("/tmp/proof_referee/check_lemma_thm1.json", "w"), indent=1, default=float)
