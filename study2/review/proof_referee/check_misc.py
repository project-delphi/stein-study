"""Independent checks: finite-N cov expectation and t*, Heuristic 1 with duplicated content,
Theorem 1 (a) formula sign for d<=k, Lemma-1 remark on j-factor features, Cor 1(d) vs kernel arms,
Prop 3 invariance and its failure under batch-scaled noise."""
import json, math, sys
import numpy as np
import torch
sys.path.insert(0, "/Users/ravikalia/Code/github.com/stein-study")
from stein.losses import ksd_imq, Prior, cov_reg  # read-only use of the estimators under test

rng = np.random.default_rng(1)
out = {}

# ---- 1. Wishart finite-N term
d, N = 5, 40
S = np.diag([2.0, 1.0, 0.5, 0.3, 0.0]); S[0, 1] = S[1, 0] = 0.4
L = np.linalg.cholesky(S + 1e-12 * np.eye(d))
vals = []
for _ in range(40000):
    z = rng.normal(size=(N, d)) @ L.T
    zc = z - z.mean(0); C = zc.T @ zc / (N - 1)
    vals.append(((C - np.eye(d)) ** 2).sum())
vals = np.array(vals)
pred = ((S - np.eye(d)) ** 2).sum() + ((S ** 2).sum() + np.trace(S) ** 2) / (N - 1)
out["wishart_mc_mean"] = float(vals.mean()); out["wishart_mc_se"] = float(vals.std() / math.sqrt(len(vals)))
out["wishart_formula"] = float(pred)

# ---- 2. t* for cov with finite-N term (spectral problem, k content, m nuisance, d)
def tstar_cov(d, k, m, N):
    from scipy.optimize import minimize
    def F(x, t):
        s = np.concatenate([x[:k], x[k:]]) ** 2
        full = np.concatenate([s, np.zeros(d - len(s))])
        R = ((full - 1) ** 2).sum() + ((full ** 2).sum() + full.sum() ** 2) / (N - 1)
        return t * (s[k:].sum()) + R
    def nuis(t):
        best = (np.inf, None)
        for st in range(6):
            x0 = np.concatenate([np.ones(k), rng.uniform(0, 1, size=min(m, d - k))])
            res = minimize(F, x0, args=(t,), method="L-BFGS-B")
            if res.fun < best[0]:
                best = (res.fun, res.x)
        return (best[1][k:] ** 2).sum()
    lo, hi = 0.5, 4.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if nuis(mid) > 1e-6: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)
out["cov_tstar_d12_k6_m5_N256"] = tstar_cov(12, 6, 5, 256)

# ---- 3. Heuristic 1: adding content that is already encoded (correlated) into an empty direction
d, k = 4, 2
U = np.eye(d)[:, :k]; u = np.eye(d)[:, 3]
Sigma = U @ U.T  # content at unit variance, d-k empty
a_vec = np.array([1.0, 0.0])  # duplicate content coordinate s_1
w = U @ a_vec
for v in (1e-3, 1e-2):
    E = math.sqrt(v) * (np.outer(u, w) + np.outer(w, u)) + v * np.outer(u, u)
    dR = ((Sigma + E - np.eye(d)) ** 2).sum() - ((Sigma - np.eye(d)) ** 2).sum()
    out[f"heur1_dup_content_dR_over_v_v={v}"] = dR / v
out["heur1_fresh_signal_dR_over_v_formula"] = "-2 + v (from the paper)"

# ---- 4. Lemma-1 remark: j-factor features cost 1-(1-alpha)^{2j}? additive 2-factor feature
a = 0.5
# g = (h(n1)+h(n2))/sqrt2 with unit-variance zero-mean h; E[g(n')|n] = (1-a) g(n)
cost_additive = 1 - (1 - a) ** 2
cost_claim = 1 - (1 - a) ** 4
out["lemma1_remark_additive_2factor_true_cost"] = cost_additive
out["lemma1_remark_claim_1-(1-a)^{2j}"] = cost_claim

# ---- 5. Cor 1(d) and E3: KSD of z = (s, g(s)) with discrete content is not zero
torch.manual_seed(0)
prior = Prior("gaussian")
# discrete content: 320 atoms, whitened 6-dim code + 6 whitened nonlinear features
atoms = torch.randn(320, 6, dtype=torch.float64)
feats = torch.cat([atoms, torch.sin(3 * atoms) + atoms ** 2], 1)
feats = feats - feats.mean(0)
Lw = torch.linalg.cholesky(feats.T @ feats / 320)
white = feats @ torch.linalg.inv(Lw).T  # population covariance exactly I over the 320 atoms
ks_disc, ks_gauss = [], []
for rep in range(20):
    idx = torch.randint(0, 320, (1024,))
    ks_disc.append(float(ksd_imq(white[idx], prior)))
    ks_gauss.append(float(ksd_imq(torch.randn(1024, 12, dtype=torch.float64), prior)))
out["cor1d_ksd_discrete_whitened_content_mean"] = float(np.mean(ks_disc))
out["cor1d_ksd_gaussian_reference_mean"] = float(np.mean(ks_gauss))
out["cor1d_cov_reg_discrete_whitened_content(N=1024)"] = float(cov_reg(white[torch.randint(0, 320, (1024,))]))

# ---- 6. Prop 3: KSD rotation invariance, and batch-scaled noise breaks O(d) invariance
z = torch.randn(256, 6, dtype=torch.float64) @ torch.diag(torch.tensor([2., 1, 1, .5, .5, .2], dtype=torch.float64))
O, _ = torch.linalg.qr(torch.randn(6, 6, dtype=torch.float64))
out["prop3_ksd_rel_change_under_rotation"] = float(abs(ksd_imq(z @ O.T, prior) - ksd_imq(z, prior)) / ksd_imq(z, prior))
Sig = np.diag([4., 1, 1, .25, .25, .04]); Om = O.numpy(); sigma = 1.0
# predictor-input noise covariance sigma^2 diag(Sigma): optimal linear prediction cost of z from z+noise
def pred_cost(Sig, Nn):
    W = Sig @ np.linalg.inv(Sig + Nn)
    return np.trace(Sig) - np.trace(W @ Sig)
c0 = pred_cost(Sig, sigma ** 2 * np.diag(np.diag(Sig)))
SigR = Om @ Sig @ Om.T
c1 = pred_cost(SigR, sigma ** 2 * np.diag(np.diag(SigR)))
out["prop3_batch_noise_pred_cost_axis_aligned"] = float(c0)
out["prop3_batch_noise_pred_cost_rotated"] = float(c1)

# ---- 7. Theorem 1 min-J formula at d <= k (theory.md quotes it only under (b))
d, k, m, lam = 3, 4, 2, 1.0
r = min(m, max(d - k, 0))
out["thm1_formula_lambda(d-k-r)_at_d<k"] = lam * (d - k - r)

print(json.dumps(out, indent=1))
json.dump(out, open("/tmp/proof_referee/check_misc.json", "w"), indent=1)
