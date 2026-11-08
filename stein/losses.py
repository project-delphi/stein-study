"""Distributional regularisers and target-law scores.

All regularisers take an embedding batch z of shape (N, d) and return a scalar.
U-statistics (KSD, MMD) have expectation 0 when z follows the target law.
"""
import math
import torch

# ----------------------------------------------------------------------------- priors

class Prior:
    """Product prior on R^d with unit-variance coordinates. Provides score and sampler.

    law='gaussian'  : N(0,1) per coordinate.
    law='laplace'   : Laplace(0, b=1/sqrt 2); score -sign(x)/b smoothed as -tanh(x/delta)/b.
    law='uniform'   : smooth surrogate = generalised normal, shape p=8, scaled to unit variance
                      (score -p/a |x/a|^(p-1) sign x); sampled from the true GN(p=8).
    law='rank_k'    : N(0, diag(1_k, eps^2 1_{d-k})) -- oracle target (near-singular rank-k Gaussian).
    """

    def __init__(self, law="gaussian", d=None, k=None, delta=0.1, p=8.0, eps=0.1):
        self.law, self.d, self.k = law, d, k
        self.delta, self.p, self.eps = delta, p, eps
        self.b = 1 / math.sqrt(2)
        # GN scale a with Var = a^2 Gamma(3/p)/Gamma(1/p) = 1
        self.a = math.sqrt(math.exp(math.lgamma(1 / p) - math.lgamma(3 / p)))

    def scale_vec(self, d, device, dtype):
        if self.law != "rank_k" or self.k is None or self.k >= d:
            return None
        v = torch.full((d,), self.eps, device=device, dtype=dtype)
        v[: self.k] = 1.0
        return v

    def score(self, x):
        if self.law == "gaussian":
            return -x
        if self.law == "laplace":
            return -torch.tanh(x / self.delta) / self.b
        if self.law == "uniform":
            u = x / self.a
            return -(self.p / self.a) * u.abs().pow(self.p - 1) * torch.sign(u)
        if self.law == "rank_k":
            s = self.scale_vec(x.shape[-1], x.device, x.dtype)
            return -x if s is None else -x / s.pow(2)
        raise ValueError(self.law)

    def sample(self, n, d, device="cpu", dtype=torch.float32, generator=None, exact=True):
        """exact=True draws from the intended law (uniform = U(-sqrt3, sqrt3)); exact=False
        draws from the smooth surrogate whose score is used (only differs for uniform)."""
        if self.law == "uniform" and exact:
            u = torch.rand(n, d, device=device, dtype=dtype, generator=generator)
            return (2 * u - 1) * math.sqrt(3)
        if self.law == "gaussian":
            return torch.randn(n, d, device=device, dtype=dtype, generator=generator)
        if self.law == "laplace":
            u = torch.rand(n, d, device=device, dtype=dtype, generator=generator) - 0.5
            return -self.b * torch.sign(u) * torch.log1p(-2 * u.abs())
        if self.law == "uniform":
            # GN(p): |x/a|^p ~ Gamma(1/p, 1)
            g = torch.distributions.Gamma(torch.tensor(1 / self.p), torch.tensor(1.0))
            mag = g.sample((n, d)).pow(1 / self.p) * self.a
            sgn = torch.where(torch.rand(n, d, generator=generator) < 0.5, -1.0, 1.0)
            return (mag * sgn).to(device=device, dtype=dtype)
        if self.law == "rank_k":
            z = torch.randn(n, d, device=device, dtype=dtype, generator=generator)
            s = self.scale_vec(d, device, dtype)
            return z if s is None else z * s
        raise ValueError(self.law)


# ----------------------------------------------------------------------------- KSD

def imq_stein_kernel_matrix(x, s, c=1.0, beta=-0.5):
    """Langevin-Stein kernel u_p(x_i, x_j) for the IMQ base kernel k = (c^2 + |r|^2)^beta.
    x, s: (N, d) samples and target scores. Returns (N, N)."""
    d = x.shape[1]
    r = x[:, None, :] - x[None, :, :]                       # x_i - x_j
    r2 = (r * r).sum(-1)
    q = c * c + r2
    k = q.pow(beta)
    qb1 = q.pow(beta - 1)
    t1 = k * (s @ s.T)
    # s(x_i)^T grad_{x_j} k = -2 beta q^{b-1} s_i^T r_ij ; s(x_j)^T grad_{x_i} k = 2 beta q^{b-1} s_j^T r_ij
    si_r = (s[:, None, :] * r).sum(-1)
    sj_r = (s[None, :, :] * r).sum(-1)
    t2 = -2 * beta * qb1 * si_r + 2 * beta * qb1 * sj_r
    t3 = -2 * beta * d * qb1 - 4 * beta * (beta - 1) * q.pow(beta - 2) * r2
    return t1 + t2 + t3


def ksd_imq(z, prior, c=1.0, beta=-0.5):
    """Full KSD^2 U-statistic with IMQ kernel (Liu et al. 2016; Gorham & Mackey 2017)."""
    n = z.shape[0]
    H = imq_stein_kernel_matrix(z, prior.score(z), c, beta)
    return (H.sum() - H.diagonal().sum()) / (n * (n - 1))


def sliced_ksd_imq(z, prior, num_slices=64, c=1.0, beta=-0.5, generator=None, dirs=None):
    """Sliced KSD^2 (Gong et al. 2021, SKSD with r = g, random unit directions resampled per call).
    For each direction g: 1-D Stein kernel on t = g^T z with projected score g^T s(z). U-statistic,
    averaged over slices."""
    n, d = z.shape
    if dirs is None:
        dirs = torch.randn(d, num_slices, generator=generator)
        dirs = (dirs / dirs.norm(dim=0, keepdim=True)).to(z.device, z.dtype)
    t = z @ dirs                       # (N, K)
    sp = prior.score(z) @ dirs         # (N, K)
    r = t[:, None, :] - t[None, :, :]  # (N, N, K)
    r2 = r * r
    q = c * c + r2
    if beta == -0.5:                   # fast path: q^b = rsqrt(q), q^(b-1) = k^3, q^(b-2) = k^5
        k = torch.rsqrt(q)
        k3 = k * k * k
        H = (k * sp[:, None, :] * sp[None, :, :]
             - k3 * r * (sp[None, :, :] - sp[:, None, :])
             + k3 - 3 * k3 * k * k * r2)
    else:
        k = q.pow(beta)
        qb1 = q.pow(beta - 1)
        H = (k * sp[:, None, :] * sp[None, :, :]
             - 2 * beta * qb1 * r * sp[:, None, :]
             + 2 * beta * qb1 * r * sp[None, :, :]
             - 2 * beta * qb1 - 4 * beta * (beta - 1) * q.pow(beta - 2) * r2)
    # diagonal (r = 0): k(0) sp^2 - 2 beta c^(2b-2)
    diag = (c * c) ** beta * sp * sp - 2 * beta * (c * c) ** (beta - 1)
    off = H.sum((0, 1)) - diag.sum(0)
    return (off / (n * (n - 1))).mean()


# ----------------------------------------------------------------------------- MMD

def mmd_gaussian(z, y, bandwidths=(0.5, 1.0, 2.0)):
    """Unbiased MMD^2 between samples z and y with a sum of Gaussian kernels
    exp(-|x-y|^2 / (2 h^2 d)), bandwidths relative to sqrt(d)."""
    n, d = z.shape
    m = y.shape[0]
    xx = torch.cdist(z, z).pow(2); yy = torch.cdist(y, y).pow(2); xy = torch.cdist(z, y).pow(2)
    out = 0.0
    for h in bandwidths:
        g = 2 * h * h * d
        kxx = torch.exp(-xx / g); kyy = torch.exp(-yy / g); kxy = torch.exp(-xy / g)
        out = out + (kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1)) \
            + (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1)) - 2 * kxy.mean()
    return out / len(bandwidths)


# ----------------------------------------------------------------------------- VICReg

def vicreg_reg(z, gamma=1.0, eps=1e-4, cov_coeff=0.04):
    """VICReg variance + covariance terms (Bardes et al. 2022), with the paper's ratio
    mu:nu = 25:1 expressed relative to the variance term (invariance = prediction loss)."""
    n, d = z.shape
    zc = z - z.mean(0)
    std = torch.sqrt(zc.var(0) + eps)
    var = torch.relu(gamma - std).mean()
    cov = (zc.T @ zc) / (n - 1)
    off = cov - torch.diag(torch.diagonal(cov))
    return var + cov_coeff * off.pow(2).sum() / d


# ----------------------------------------------------------------------------- SIGReg (re-implementation, validated against lejepa)

def epps_pulley(x, t_max=3.0, n_points=17):
    """Epps-Pulley statistic per column of x (N, K), as in lejepa.univariate.EppsPulley."""
    N = x.shape[-2]
    t = torch.linspace(0, t_max, n_points, dtype=x.dtype, device=x.device)
    dt = t_max / (n_points - 1)
    w = torch.full((n_points,), 2 * dt, dtype=x.dtype, device=x.device)
    w[0] = dt; w[-1] = dt
    phi = torch.exp(-0.5 * t * t)
    xt = x.unsqueeze(-1) * t
    err = (torch.cos(xt).mean(-3) - phi).pow(2) + torch.sin(xt).mean(-3).pow(2)
    return (err @ (w * phi)) * N


def sigreg_ref(z, dirs):
    """SIGReg = mean over slices of Epps-Pulley(z @ dirs)."""
    return epps_pulley(z @ dirs).mean()


class SIGReg(torch.nn.Module):
    """SIGReg via the official lejepa package (reference implementation used in training)."""

    def __init__(self, num_slices=256, t_max=3.0, n_points=17):
        super().__init__()
        import lejepa
        self.test = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=lejepa.univariate.EppsPulley(t_max=t_max, n_points=n_points),
            num_slices=num_slices)

    def forward(self, z):
        return self.test(z)


# ----------------------------------------------------------------------------- dispatcher

class Regulariser(torch.nn.Module):
    """name in {none, vicreg, sigreg, sksd, ksd, mmd, oracle, ema}; law = target law for KSD/MMD."""

    def __init__(self, name, d, law="gaussian", k=None, num_slices=64, oracle_eps=0.1):
        super().__init__()
        self.name, self.d = name, d
        self.prior = Prior("rank_k", d=d, k=k, eps=oracle_eps) if name == "oracle" else Prior(law, d=d)
        self.num_slices = num_slices
        if name == "sigreg":
            self.sig = SIGReg(num_slices=256)
        self.sigreg_impl = "lejepa"
        self._step = 0

    def forward(self, z):
        n = self.name
        if n in ("none", "ema"):
            return z.new_zeros(())
        if n == "vicreg":
            return vicreg_reg(z)
        if n == "sigreg":
            if self.sigreg_impl == "lejepa":
                return self.sig(z)
            # identical statistic (validated against lejepa, rel. diff 0) without lejepa's per-call
            # .item() host sync: directions drawn on CPU from a counter-seeded generator, as lejepa does
            g = torch.Generator().manual_seed(self._step)
            self._step += 1
            A = torch.randn(z.shape[1], 256, generator=g)
            A = (A / A.norm(p=2, dim=0)).to(z.device, z.dtype)
            return sigreg_ref(z, A)
        if n == "sksd":
            return sliced_ksd_imq(z, self.prior, self.num_slices)
        if n in ("ksd", "oracle"):
            return ksd_imq(z, self.prior)
        if n == "mmd":
            y = self.prior.sample(z.shape[0], z.shape[1], device=z.device, dtype=z.dtype)
            return mmd_gaussian(z, y)
        raise ValueError(n)
