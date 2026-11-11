"""Independent spot check of Theorem 2's numerical constant for MMD (closed-form Gaussian MMD^2 between
N(0, Sigma) and N(0, I) with the code's kernel sum exp(-|x-y|^2/(2 h^2 d)), h in {0.5, 1, 2})."""
import numpy as np
from scipy.optimize import minimize

def mmd2(sig, d, hs=(0.5, 1.0, 2.0)):
    out = 0.0
    for h in hs:
        g = h * h * d
        # E exp(-|x-y|^2/(2g)), x~N(0,A), y~N(0,B) indep, diagonal: prod (1 + (a+b)/g)^(-1/2)
        e_ss = np.prod((1 + 2 * sig / g) ** -0.5)
        e_ii = (1 + 2 / g) ** (-d / 2)
        e_si = np.prod((1 + (sig + 1) / g) ** -0.5)
        out += e_ss + e_ii - 2 * e_si
    return out / len(hs)

def solve(t, d, k, m, rng):
    nfree = min(d, k + m)
    def F(x):
        s = np.zeros(d); s[:nfree] = x ** 2
        srt = np.sort(s[:nfree])[::-1]
        return t * srt[k:].sum() + mmd2(s, d)
    best = (np.inf, None)
    for _ in range(8):
        x0 = np.concatenate([np.ones(k), rng.uniform(0, 1.2, nfree - k)]) * rng.uniform(0.7, 1.2)
        r = minimize(F, x0, method="Nelder-Mead", options=dict(maxiter=40000, xatol=1e-10, fatol=1e-16))
        r = minimize(F, r.x, method="L-BFGS-B")
        if r.fun < best[0]: best = (r.fun, r.x)
    s = np.sort(best[1] ** 2)[::-1]
    return s[k:].sum(), s

rng = np.random.default_rng(0)
for d in (12, 24):
    k, m = 6, 5
    lo, hi = 1e-4, 0.05
    for _ in range(30):
        mid = np.sqrt(lo * hi)
        N, _ = solve(mid, d, k, m, rng)
        if N > 1e-5: lo = mid
        else: hi = mid
    tstar = np.sqrt(lo * hi)
    print(f"d={d}: t*={tstar:.6g}  lambda*(alpha=1, c_p=1/d)={1/d/tstar:.4g}")
