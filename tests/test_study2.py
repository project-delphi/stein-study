"""Study-2 validation tests: run cache unchanged, new metrics on known answers, cov arms, noise mode."""
import glob, json, math, os

import numpy as np
import pytest
import torch

from stein.train import resolve, cfg_hash, bottleneck, HASH_OPTIONAL
from stein import metrics as M
from stein.losses import cov_reg, cov_topk_reg, Regulariser


# ----------------------------------------------------------------------------- run cache

def test_cached_run_hashes_unchanged():
    """Every cached run's stored config re-hashes to its directory name under the current code."""
    paths = sorted(glob.glob("runs/*/result.json"))
    if not paths:
        pytest.skip("no runs/ cache")
    import hashlib
    from stein.train import DEFAULTS
    study1_defaults = {k: v for k, v in DEFAULTS.items() if k not in HASH_OPTIONAL}

    def study1_hash(cfg):          # study 1's resolve + cfg_hash (commit before study 2)
        c = {**study1_defaults, **cfg}
        keep = {k: v for k, v in c.items() if k not in ("device", "threads")}
        return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:12]

    bad, legacy = [], 0
    for p in paths:
        r = json.load(open(p))
        h = cfg_hash(resolve(r["config"]))
        s2 = any(k in r["config"] and r["config"][k] != v for k, v in HASH_OPTIONAL.items())
        if s2:                     # a study-2 run: its directory is its hash
            if h != os.path.basename(os.path.dirname(p)):
                bad.append(p)
            continue
        if h != study1_hash({k: v for k, v in r["config"].items() if k not in HASH_OPTIONAL}):
            bad.append(p)
        if set(study1_defaults) <= set(r["config"]):
            if h != os.path.basename(os.path.dirname(p)) or h != r["hash"]:
                bad.append(p)
        else:
            legacy += 1      # early study-1 pilots stored before some keys existed (already orphaned in study 1)
    assert not bad, f"{len(bad)}/{len(paths)} cached runs changed hash, e.g. {bad[:3]}"
    assert legacy <= 27


def test_new_keys_hash_only_off_default():
    base = dict(arm="ksd", lam=1.0, seed=3)
    h0 = cfg_hash(resolve(base))
    assert all(cfg_hash(resolve({**base, k: v})) == h0 for k, v in HASH_OPTIONAL.items())
    for k, v in dict(encoder="linear", pred_arch="linear", noise_mode="fixed", eval_v=2).items():
        assert cfg_hash(resolve({**base, k: v})) != h0


# ----------------------------------------------------------------------------- MCC

def test_mcc_signed_permutation_is_one():
    rng = np.random.default_rng(0)
    S = rng.standard_normal((5000, 6))
    P = np.eye(6)[rng.permutation(6)] * rng.choice([-1, 1], 6)
    Z = np.c_[S @ P * rng.uniform(0.5, 3, 6), rng.standard_normal((5000, 6))]   # scaled + 6 noise coords
    m, per = M.mcc(Z, S)
    assert m == pytest.approx(1.0, abs=1e-9) and np.allclose(per, 1.0)


def test_mcc_rotation_below_one():
    rng = np.random.default_rng(1)
    S = rng.standard_normal((5000, 6))
    Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    m, _ = M.mcc(S @ Q, S)
    assert m < 0.9
    # a 45-degree rotation of one pair: those two factors are matched at |corr| = 1/sqrt 2
    R = np.eye(6); c = 1 / math.sqrt(2)
    R[:2, :2] = [[c, -c], [c, c]]
    m2, per2 = M.mcc(S @ R, S)
    assert per2[:2] == pytest.approx([c, c], abs=0.03) and np.allclose(per2[2:], 1.0)


# ----------------------------------------------------------------------------- variance shares

def test_variance_share_constructed_mixture():
    rng = np.random.default_rng(2)
    n = 20000
    S, N = rng.standard_normal((n, 6)), rng.standard_normal((n, 5))
    E = rng.standard_normal((n, 3))
    # embedding: content variance 6 x 1, nuisance 2 directions of variance 4 and 1, noise 3 x 1 -> total 14
    B = np.zeros((5, 2)); B[0, 0] = 2.0; B[1, 1] = 1.0
    Z = np.c_[S, N @ B, E]
    tr, te = slice(0, 10000), slice(10000, None)
    out = M.study2_metrics(Z[tr], Z[te], S[tr], S[te], N[tr], N[te], d=11)
    assert out["content_share"] == pytest.approx(6 / 14, abs=0.02)
    assert out["nuis_share"] == pytest.approx(5 / 14, abs=0.02)
    assert out["unexpl_share"] == pytest.approx(3 / 14, abs=0.02)
    assert out["nuis_eigs"][:2] == pytest.approx([4.0, 1.0], rel=0.05)
    assert out["content_eigs"][:6] == pytest.approx([1.0] * 6, abs=0.06)
    # r = min(m, d - k) = 5 nuisance directions: (4 + 1 + 0 + 0 + 0)/5 relative to content 1
    assert out["nuis_dir_ratio"] == pytest.approx(1.0, abs=0.06)
    # principal axes: the top axis is the nuisance direction of variance 4
    assert out["pc_var"][0] == pytest.approx(4.0, rel=0.05)
    assert out["pc_nuis_var"][0] == pytest.approx(4.0, rel=0.05)
    assert out["pc_content_var"][0] == pytest.approx(0.0, abs=0.01)


def test_onehot_share_sees_nonlinear_code():
    """A coordinate equal to cos(2 h) of a 10-level factor is invisible to the (cos h, sin h) code
    but fully explained by the one-hot code."""
    rng = np.random.default_rng(3)
    f = rng.integers(0, 10, (8000, 1))
    h = 2 * np.pi * f[:, 0] / 10
    lin = np.c_[np.cos(h), np.sin(h)]
    oh = M.onehot(f, [0], [10])
    Z = np.c_[np.cos(2 * h)]
    tr, te = slice(0, 4000), slice(4000, None)
    lin_share = M.variance_share(Z[te], M.ols_predict(lin[tr], Z[tr], lin[te]))
    oh_share = M.variance_share(Z[te], M.ols_predict(oh[tr], Z[tr], oh[te]))
    assert abs(lin_share) < 0.01 and oh_share > 0.999


# ----------------------------------------------------------------------------- cov arms

def _expected_cov_pen(ev, N):
    ev = np.asarray(ev, float)
    return ((ev - 1) ** 2).sum() + ((ev ** 2).sum() + ev.sum() ** 2) / (N - 1)


def test_cov_reg_isotropic_vs_collapsed():
    torch.manual_seed(0)
    d, N = 12, 256
    iso = torch.stack([cov_reg(torch.randn(N, d, dtype=torch.float64)) for _ in range(200)]).mean().item()
    assert iso == pytest.approx(_expected_cov_pen(np.ones(d), N), rel=0.05)     # = finite-N bias only
    assert iso < 0.7
    scale = torch.tensor([1.0] * 6 + [0.0] * 6, dtype=torch.float64)            # rank-6 collapse
    col = torch.stack([cov_reg(torch.randn(N, d, dtype=torch.float64) * scale) for _ in range(200)]).mean().item()
    assert col == pytest.approx(_expected_cov_pen(scale ** 2, N), rel=0.05) and col > 6
    ani = torch.tensor([3.0] + [1.0] * 11, dtype=torch.float64)
    a = torch.stack([cov_reg(torch.randn(N, d, dtype=torch.float64) * ani) for _ in range(200)]).mean().item()
    assert a > 60


def test_cov_reg_gradcheck():
    z = torch.randn(32, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(cov_reg, (z,))
    z2 = (torch.randn(64, 6, dtype=torch.float64) * torch.tensor([3, 2.5, 2, 1.5, 0.2, 0.1], dtype=torch.float64)).requires_grad_()
    assert torch.autograd.gradcheck(lambda t: cov_topk_reg(t, 3), (z2,))


def test_cov_topk_ignores_spare_directions():
    torch.manual_seed(1)
    N = 4096
    z = torch.randn(N, 12, dtype=torch.float64)
    z[:, 6:] *= 0.01                                    # six spare directions empty
    assert cov_topk_reg(z, 6).item() < 0.05
    assert cov_reg(z).item() > 5.9
    r = Regulariser("cov", 12, k=6)
    assert r(z).item() == pytest.approx(cov_reg(z).item())


# ----------------------------------------------------------------------------- noise mode

def test_bottleneck_modes():
    torch.manual_seed(2)
    z = 5.0 * torch.randn(20000, 4)
    zb = bottleneck(z, 0.5, "batch") - z
    zf = bottleneck(z, 0.5, "fixed") - z
    assert zb.std().item() == pytest.approx(2.5, rel=0.03)
    assert zf.std().item() == pytest.approx(0.5, rel=0.03)
    assert torch.equal(bottleneck(z, 0.0, "fixed"), z)


# ----------------------------------------------------------------------------- analysis (threshold estimator)

def _fake_path(leak_by_lam, seeds=5, noise=0.0):
    import pandas as pd
    rng = np.random.default_rng(0)
    rows = []
    for lam, y in leak_by_lam.items():
        for s in range(1, seeds + 1):
            rows.append(dict(tier="shapes", encoder="conv", pred_arch="mlp", arm="cov", d=12, alpha=1.0, sigma=0.0,
                             noise_mode="batch", law="uniform", prior="gaussian", steps=2000, lam=lam, seed=s,
                             leak_r2=y + noise * rng.standard_normal(), content_r2=0.5 + lam / 10))
    return pd.DataFrame(rows)


def test_threshold_estimator_log_interpolation():
    from stein.analyze2 import thresholds, crossing
    lams = np.array([0.1, 0.2, 0.4, 0.8])
    # 0.05 at 0.2 and 0.15 at 0.4: tau = 0.10 is crossed halfway in log lambda -> sqrt(0.2 * 0.4)
    v, c = crossing(lams, np.array([0.0, 0.05, 0.15, 0.5]), 0.10)
    assert c == "" and v == pytest.approx(math.sqrt(0.08))
    assert crossing(lams, np.array([0.2, 0.3, 0.4, 0.5]), 0.10)[1] == "<"
    assert crossing(lams, np.array([0.0, 0.0, 0.0, 0.05]), 0.10)[1] == ">"
    # upward search from the minimum skips a high low-lambda start (collapse regime)
    y = np.array([0.3, 0.02, 0.05, 0.15])
    assert crossing(lams, y, 0.10, start=1)[0] == pytest.approx(math.sqrt(0.32))
    T = thresholds(_fake_path(dict(zip(lams, [0.0, 0.05, 0.15, 0.5]))), np.random.default_rng(0))
    assert len(T) == 1 and T.lam_hat[0] == pytest.approx(math.sqrt(0.08)) and T.lo[0] <= T.lam_hat[0] <= T.hi[0]


def test_frontier_respects_ceiling():
    from stein.analyze2 import frontier
    F = frontier(_fake_path({0.1: 0.0, 0.2: 0.05, 0.4: 0.15, 0.8: 0.5}), np.random.default_rng(0), ceilings=(0.10,))
    assert F.best_content[0] == pytest.approx(0.5 + 0.2 / 10)
