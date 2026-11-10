"""Population (expected-value) versions of the study's regularisers for a Gaussian embedding
z ~ N(0, Sigma), written in torch so they can be differentiated, plus helpers shared by the
theory checks. Every function here mirrors a batch estimator in stein/losses.py; check_t2.py
verifies each against that estimator by Monte Carlo before using it.

Conventions: `ev` is a 1-D tensor of the eigenvalues of Sigma (all regularisers with an N(0, I)
target are orthogonally invariant, so for Gaussian z they depend on Sigma only through ev).
"""
import json, math, os
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results2")


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"theory_{name}.json")
    json.dump(obj, open(p, "w"), indent=1, default=float)
    return p


# ----------------------------------------------------------------------------- covariance penalty

def cov_pop(ev, N=None):
    """E ||Sigma_hat - I||_F^2. N=None: population ||Sigma - I||^2. With a batch of N Gaussian
    samples and the unbiased centred covariance, (N-1) Sigma_hat ~ Wishart(Sigma, N-1), so
    Var(Sigma_hat_ij) = (Sigma_ij^2 + Sigma_ii Sigma_jj)/(N-1) and the expectation gains
    (||Sigma||_F^2 + (tr Sigma)^2)/(N-1)."""
    v = ((ev - 1) ** 2).sum()
    if N is not None:
        v = v + ((ev ** 2).sum() + ev.sum() ** 2) / (N - 1)
    return v


# ----------------------------------------------------------------------------- KSD (IMQ, c=1, beta=-1/2)

_U = torch.linspace(-40.0, 8.0, 4001)          # t = exp(u) quadrature grid for Gamma-integral identities
_DU = float(_U[1] - _U[0])
_T = torch.exp(_U)


def _gamma_moments(ev2, a_list):
    """For R = sum_i ev2_i g_i^2 (g ~ N(0, I)), return dict a -> (E[(1+R)^-a], E[R (1+R)^-a]) using
    (1+R)^-a = Gamma(a)^-1 int_0^inf t^(a-1) e^-t e^(-tR) dt, E e^(-tR) = prod (1+2 t ev2)^(-1/2),
    E R e^(-tR) = E e^(-tR) * sum ev2/(1+2 t ev2). Integral in u = log t by the trapezoid rule."""
    t = _T[:, None]
    one = 1 + 2 * t * ev2[None, :]
    L = torch.exp(-0.5 * torch.log(one).sum(1))
    LR = L * (ev2[None, :] / one).sum(1)
    out = {}
    for a in a_list:
        w = torch.exp(a * _U - _T) / math.gamma(a) * _DU     # t^(a-1) e^-t dt with dt = t du
        w = w.clone(); w[0] *= 0.5; w[-1] *= 0.5
        out[a] = ((w * L).sum(), (w * LR).sum())
    return out


def ksd_pop(ev):
    """Population KSD^2 of N(0, diag(ev)) against N(0, I), IMQ kernel (1+|r|^2)^(-1/2), as in
    stein.losses.imq_stein_kernel_matrix. With r = x - x' ~ N(0, 2 Sigma), m = (x+x')/2 independent
    of r, x.x' = |m|^2 - |r|^2/4, E|m|^2 = tr Sigma / 2:
    E u = E[q^-1/2 (tr Sigma/2 - R/4)] - E[q^-3/2 R] + d E[q^-3/2] - 3 E[q^-5/2 R],  q = 1 + R."""
    d = ev.shape[0]
    M = _gamma_moments(2 * ev, [0.5, 1.5, 2.5])
    E12, E12R = M[0.5]
    E32, E32R = M[1.5]
    E52, E52R = M[2.5]
    return E12 * ev.sum() / 2 - E12R / 4 - E32R + d * E32 - 3 * E52R


def ksd1_pop(s2):
    """1-D population KSD^2 of N(0, s2) vs N(0, 1) (vectorised over s2), d = 1."""
    out = []
    for v in s2.reshape(-1):
        out.append(ksd_pop(v.reshape(1)))
    return torch.stack(out).reshape(s2.shape)


# ----------------------------------------------------------------------------- slicing

def sphere_dirs(d, K=4096, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(K, d, generator=g)
    return u / u.norm(dim=1, keepdim=True)


def sliced(fn1, ev, dirs):
    """E_u fn1(u' Sigma u) over unit directions u (Monte Carlo with fixed dirs)."""
    ev = torch.sort(ev, descending=True).values      # canonical order: exactly permutation invariant
    s2 = (dirs ** 2) @ ev
    return fn1(s2).mean()


# 1-D KSD as a smooth function of s2, tabulated once and interpolated (fast and differentiable)
_S2GRID = torch.linspace(0.0, 4.0, 801)
_K1TAB = None


def ksd1_interp(s2):
    global _K1TAB
    if _K1TAB is None:
        _K1TAB = torch.stack([ksd_pop(v.reshape(1)) for v in _S2GRID])
    x = s2.clamp(0, 4.0) / 4.0 * (len(_S2GRID) - 1)
    i0 = x.floor().long().clamp(0, len(_S2GRID) - 2)
    f = x - i0
    return _K1TAB[i0] * (1 - f) + _K1TAB[i0 + 1] * f


def sksd_pop(ev, dirs):
    return sliced(ksd1_interp, ev, dirs)


# ----------------------------------------------------------------------------- SIGReg (Epps-Pulley, expected batch statistic)

def ep_expected(s2, N=256, t_max=3.0, n_points=17):
    """E[EppsPulley] for a batch of N draws of N(0, s2), as computed by stein.losses.epps_pulley:
    sum_t w_t phi_t [ N (psi_t - phi_t)^2 + 1 - psi_t^2 ],  psi_t = exp(-t^2 s2 / 2), phi_t = exp(-t^2/2).
    (E|mean e^{itX} - phi|^2 = (psi - phi)^2 + (1 - psi^2)/N.)"""
    t = torch.linspace(0, t_max, n_points)
    dt = t_max / (n_points - 1)
    w = torch.full((n_points,), 2 * dt); w[0] = dt; w[-1] = dt
    phi = torch.exp(-0.5 * t * t)
    psi = torch.exp(-0.5 * t[None, :] ** 2 * s2.reshape(-1, 1))
    val = (N * (psi - phi) ** 2 + 1 - psi ** 2) @ (w * phi)
    return val.reshape(s2.shape)


def sigreg_pop(ev, dirs, N=256):
    return sliced(lambda s2: ep_expected(s2, N), ev, dirs)


# ----------------------------------------------------------------------------- MMD (Gaussian kernels)

def mmd_pop(ev, bandwidths=(0.5, 1.0, 2.0)):
    """Population MMD^2 between N(0, diag ev) and N(0, I) for the kernel sum in stein.losses.mmd_gaussian,
    k(x, y) = exp(-|x-y|^2 / g), g = 2 h^2 d:  E k(x, y) for x ~ N(0, S1), y ~ N(0, S2) independent is
    prod (1 + 2 (s1 + s2)/g)^(-1/2)."""
    d = ev.shape[0]
    one = torch.ones_like(ev)
    out = 0.0
    for h in bandwidths:
        g = 2 * h * h * d
        f = lambda a, b: torch.exp(-0.5 * torch.log(1 + 2 * (a + b) / g).sum())
        out = out + f(ev, ev) + f(one, one) - 2 * f(ev, one)
    return out / len(bandwidths)


# ----------------------------------------------------------------------------- VICReg (not orthogonally invariant)

def vicreg_pop(S, eps=1e-4, cov_coeff=0.04):
    """Population version of stein.losses.vicreg_reg for covariance matrix S (d x d):
    mean_j relu(1 - sqrt(S_jj + eps)) + cov_coeff * sum_{i != j} S_ij^2 / d.
    (Batch fluctuations of the per-dimension std are ignored: an approximation, see theory.md.)"""
    d = S.shape[0]
    var = torch.relu(1 - torch.sqrt(torch.diagonal(S) + eps)).mean()
    off = S - torch.diag(torch.diagonal(S))
    return var + cov_coeff * off.pow(2).sum() / d


# ----------------------------------------------------------------------------- linear model

def pred_loss(A, B, alpha, predictor="identity", cp=1.0, W=None):
    """c_p E||W z - z'||^2 for z = A s + B n, z' = A s + B n', Cov(n, n') = (1 - alpha) I.
    predictor: 'identity' (W = I), 'param' (W given, trained jointly), 'optimal' (W = C Sigma^+)."""
    P, Q = A @ A.T, B @ B.T
    S = P + Q
    C = P + (1 - alpha) * Q                       # Cov(z', z)
    if predictor == "identity":
        return cp * 2 * alpha * torch.trace(Q)
    if predictor == "param":
        return cp * (torch.trace(W @ S @ W.T) - 2 * torch.trace(W @ C) + torch.trace(S))
    if predictor == "optimal":
        Si = torch.linalg.inv(S + 1e-9 * torch.eye(S.shape[0]))
        return cp * (torch.trace(S) - torch.trace(C @ Si @ C.T))
    raise ValueError(predictor)


def leak_r2(A, B):
    """Population linear-probe R^2 of n (identity covariance) from z, averaged over the m
    coordinates: tr(B^T Sigma^+ B)/m. Scale invariant."""
    S = A @ A.T + B @ B.T
    Sp = torch.linalg.pinv(S, hermitian=True, rtol=1e-8)
    return float(torch.trace(B.T @ Sp @ B) / B.shape[1])


def nuis_share(A, B):
    """Share of embedding variance explained by n: tr(BB^T)/tr(Sigma)."""
    Q = B @ B.T
    return float(torch.trace(Q) / torch.trace(A @ A.T + Q))
