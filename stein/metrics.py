"""Evaluation metrics on frozen embeddings."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge

from .losses import Prior, ksd_imq, epps_pulley


def _std(Y, ref):
    mu, sd = ref.mean(0), ref.std(0) + 1e-8
    return (Y - mu) / sd


def ridge_r2(Ztr, Ytr, Zte, Yte, alpha=1e-3):
    """Linear probe: per-target R^2 on held-out data (averaged); Z standardised on the fit set."""
    Ztr_s, Zte_s = _std(Ztr, Ztr), _std(Zte, Ztr)
    m = Ridge(alpha=alpha).fit(Ztr_s, Ytr)
    P = m.predict(Zte_s)
    ss_res = ((Yte - P) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12
    r2 = 1 - ss_res / ss_tot
    return float(np.mean(r2)), r2


def knn_r2(Ztr, Ytr, Zte, Yte, k=10):
    """Nonlinear (kNN regression) probe: per-target R^2 averaged. Measures information present
    regardless of linear accessibility."""
    from sklearn.neighbors import KNeighborsRegressor
    m = KNeighborsRegressor(n_neighbors=k).fit(_std(Ztr, Ztr), Ytr)
    P = m.predict(_std(Zte, Ztr))
    r2 = 1 - ((Yte - P) ** 2).sum(0) / (((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12)
    return float(np.mean(r2))


def logreg_acc(Ztr, ytr, Zte, yte):
    m = LogisticRegression(max_iter=2000, C=1.0).fit(_std(Ztr, Ztr), ytr)
    return float((m.predict(_std(Zte, Ztr)) == yte).mean())


def orth_identifiability(Z, S):
    """R^2 of the best scaled semi-orthogonal map from centred Z (n,d) to standardised S (n,k):
    min_{c, W: W^T W = I (or W W^T = I if d<k)} |c Z W - S|^2 / |S|^2."""
    Z = Z - Z.mean(0)
    S = (S - S.mean(0)) / (S.std(0) + 1e-8)
    U, sv, Vt = np.linalg.svd(Z.T @ S, full_matrices=False)
    W = U @ Vt                                         # (d, k) semi-orthogonal
    ZW = Z @ W
    c = (ZW * S).sum() / ((ZW * ZW).sum() + 1e-12)
    return float(1 - ((c * ZW - S) ** 2).sum() / (S ** 2).sum())


def effective_rank(Z):
    """RankMe (Garrido et al. 2023): exp entropy of normalised singular values."""
    s = np.linalg.svd(Z, compute_uv=False)
    p = s / (s.sum() + 1e-12) + 1e-12
    return float(np.exp(-(p * np.log(p)).sum()))


def gaussianity(Z, num_slices=256, seed=0):
    """Sliced Epps-Pulley statistic divided by N (sample-size free), on raw Z vs N(0, I),
    and on per-slice standardised projections (shape only)."""
    g = torch.Generator().manual_seed(seed)
    z = torch.from_numpy(Z).double()
    A = torch.randn(z.shape[1], num_slices, generator=g, dtype=torch.float64)
    A = A / A.norm(dim=0, keepdim=True)
    P = z @ A
    raw = (epps_pulley(P) / P.shape[0]).mean().item()
    Ps = (P - P.mean(0)) / (P.std(0) + 1e-8)
    return raw, (epps_pulley(Ps) / P.shape[0]).mean().item()


def ksd_to_prior(Z, law, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    z = torch.from_numpy(Z[rng.choice(len(Z), min(n, len(Z)), replace=False)]).double()
    return float(ksd_imq(z, Prior(law, d=z.shape[1])).item())


def _kde_score_1d(x, ref, h):
    """Score of a 1-D Gaussian KDE fitted to ref, evaluated at x."""
    D = (x[:, None] - ref[None, :]) / h
    w = np.exp(-0.5 * D ** 2)
    return (-(D / h) * w).sum(1) / (w.sum(1) + 1e-300)


def np_ksd(Z, law, n=500, n_gen=4000, n_null=50, n_coords=None, seed=0):
    """NP-KSD (Xu & Reinert 2022) of embeddings against the intended prior, seen as an implicit
    generator: the score is estimated from generator samples (prior is a product law, so the
    coordinate conditionals equal the marginals; 1-D Gaussian-KDE score, Silverman bandwidth),
    Stein kernel with a Gaussian base kernel (bandwidth: median heuristic on the generator
    sample), V-statistic averaged over coordinates, null simulated by fresh generator draws.
    Returns (statistic, p-value)."""
    rng = np.random.default_rng(seed)
    d = Z.shape[1]
    prior = Prior(law, d=d)
    torch.manual_seed(seed)
    G = prior.sample(n_gen, d, dtype=torch.float64).numpy()
    coords = np.arange(d) if n_coords is None else rng.choice(d, n_coords, replace=False)
    hs = 1.06 * G.std(0) * n_gen ** (-0.2)
    sub = G[rng.choice(n_gen, 300, replace=False)]
    bw2 = np.median(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)) + 1e-12

    def stat(X):
        diff = X[:, None, :] - X[None, :, :]
        r2 = (diff ** 2).sum(-1)
        K = np.exp(-r2 / bw2)
        tot = 0.0
        for i in coords:
            s = _kde_score_1d(X[:, i], G[:, i], hs[i])
            di = diff[:, :, i]
            dk_dxj = K * (2 * di / bw2)                 # d/dy_i k(x, y)
            dk_dxi = -dk_dxj                            # d/dx_i k(x, y)
            d2 = K * (2 / bw2 - 4 * di ** 2 / bw2 ** 2)
            H = s[:, None] * s[None, :] * K + s[:, None] * dk_dxj + s[None, :] * dk_dxi + d2
            tot += H.mean()
        return tot / len(coords)

    X = Z[rng.choice(len(Z), min(n, len(Z)), replace=False)].astype(np.float64)
    t = stat(X)
    null = [stat(prior.sample(len(X), d, dtype=torch.float64).numpy()) for _ in range(n_null)]
    p = (1 + np.sum(np.array(null) >= t)) / (1 + n_null)
    return float(t), float(p)


# ----------------------------------------------------------------------------- study 2 metrics

def ols_predict(Xtr, Ytr, Xte):
    """Least-squares fit (with intercept) of Y on X on the fit set; predictions on the eval set."""
    A = np.c_[Xtr, np.ones(len(Xtr))]
    coef, *_ = np.linalg.lstsq(A, Ytr, rcond=None)
    return np.c_[Xte, np.ones(len(Xte))] @ coef


def variance_share(Zte, Zhat):
    """Fraction of the embedding's total variance explained: 1 - sum|Z - Zhat|^2 / sum|Z - mean Z|^2
    (raw embedding units, all coordinates pooled). Zhat = held-out OLS prediction of Z from a code."""
    tot = ((Zte - Zte.mean(0)) ** 2).sum()
    return float(1 - ((Zte - Zhat) ** 2).sum() / (tot + 1e-300))


def explained_cov_eigs(Zhat):
    """Eigenvalues (descending) of the covariance of the explained part Zhat: the per-direction
    variance a code puts into the embedding (theory: eigenvalues of B B^T for nuisance, A A^T for content)."""
    return np.linalg.eigvalsh(np.cov(Zhat.T)).clip(min=0)[::-1]


def pc_decomposition(Zte, parts):
    """Principal axes of Z (descending variance); for each axis u, var(u^T Z) and var(u^T P) for each
    explained part P in `parts` (dict name -> Zhat)."""
    C = np.cov(Zte.T)
    ev, U = np.linalg.eigh(C)
    U = U[:, ::-1]
    out = {"pc_var": ev[::-1].clip(min=0)}
    for k, P in parts.items():
        out[f"pc_{k}_var"] = ((P - P.mean(0)) @ U).var(0, ddof=1)
    return out


def mcc(Z, S):
    """Mean correlation coefficient: absolute Pearson correlations between embedding coordinates
    (n, d) and content factors (n, k), best one-to-one matching (Hungarian); returns the mean over
    matched pairs and the per-factor matched |corr| (NaN for a factor left unmatched when d < k)."""
    from scipy.optimize import linear_sum_assignment
    Zc = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    Sc = (S - S.mean(0)) / (S.std(0) + 1e-12)
    R = np.abs(Zc.T @ Sc / len(Z))                      # (d, k)
    ri, ci = linear_sum_assignment(-R)
    per = np.full(S.shape[1], np.nan)
    per[ci] = R[ri, ci]
    return float(R[ri, ci].mean()), per


def onehot(f, cols, sizes):
    """One-hot coding of integer factor columns, first level dropped (additive factor model)."""
    blocks = [np.eye(sizes[j])[f[:, j]][:, 1:] for j in cols]
    return np.concatenate(blocks, 1)


def study2_metrics(Ztr, Zte, Str, Ste, Ntr, Nte, OHtr=None, OHte=None, d=None, k=6, m=5):
    """Study-2 metrics (eval_v = 2). S, N: linear content / nuisance codes; OH: optional pair of
    (content one-hot, nuisance one-hot) arrays for the additive per-factor (nonlinear-in-code) shares."""
    out = {}
    Zs = ols_predict(Str, Ztr, Ste)
    Zn = ols_predict(Ntr, Ztr, Nte)
    Zsn = ols_predict(np.c_[Str, Ntr], Ztr, np.c_[Ste, Nte])
    out["content_share"] = variance_share(Zte, Zs)
    out["nuis_share"] = variance_share(Zte, Zn)
    out["joint_share"] = variance_share(Zte, Zsn)
    out["unexpl_share"] = 1 - out["joint_share"]
    if OHtr is not None:
        (Ctr, Ntr_oh), (Cte, Nte_oh) = OHtr, OHte
        out["content_share_oh"] = variance_share(Zte, ols_predict(Ctr, Ztr, Cte))
        out["nuis_share_oh"] = variance_share(Zte, ols_predict(Ntr_oh, Ztr, Nte_oh))
        out["joint_share_oh"] = variance_share(Zte, ols_predict(np.c_[Ctr, Ntr_oh], Ztr, np.c_[Cte, Nte_oh]))
        out["unexpl_share_oh"] = 1 - out["joint_share_oh"]
    ne, ce = explained_cov_eigs(Zn), explained_cov_eigs(Zs)
    out["nuis_eigs"], out["content_eigs"] = ne.tolist(), ce.tolist()
    d = Zte.shape[1] if d is None else d
    r = min(m, max(d - k, 0))
    # theory's v = 1 - lam*/lam: mean variance per nuisance direction (top r) relative to the mean
    # variance per content direction (top min(k, d)); undefined when d <= k
    out["nuis_dir_ratio"] = float(ne[:r].mean() / (ce[:min(k, d)].mean() + 1e-300)) if r > 0 else float("nan")
    out["total_var"] = float(np.trace(np.cov(Zte.T)))
    for key, v in pc_decomposition(Zte, {"nuis": Zn, "content": Zs}).items():
        out[key] = v.tolist()
    out["mcc"], per = mcc(Zte, Ste)
    out["mcc_per"] = per.tolist()
    return out
