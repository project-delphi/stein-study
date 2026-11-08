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
