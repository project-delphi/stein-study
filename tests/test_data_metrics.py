import numpy as np
import torch

from stein.data import Shapes, Cifar, content_code, nuisance_code, CONTENT, NUISANCE, law_weights, OH, SC
from stein import metrics as M

FACT = np.load("data/processed/shapes_factors.npy", mmap_mode="r")


def test_image_index_matches_factors():
    S = Shapes(seed=1)
    f = S.sample_factors(500)
    assert np.array_equal(FACT[S._idx(f)], f)


def test_pairs_share_content_and_alpha0_shares_nuisance():
    S = Shapes(seed=2)
    f1, f2 = S.sample_pair_factors(5000, alpha=0.0)
    assert np.array_equal(f1, f2)
    f1, f2 = S.sample_pair_factors(5000, alpha=0.5)
    assert np.array_equal(f1[:, CONTENT], f2[:, CONTENT])
    frac_same = (f1[:, NUISANCE] == f2[:, NUISANCE]).mean()
    assert 0.45 < frac_same < 0.6          # 0.5 kept + 0.5 * 1/n resampled to same value


def test_alpha1_nuisance_uncorrelated():
    S = Shapes(seed=3)
    f1, f2 = S.sample_pair_factors(40000, alpha=1.0)
    assert np.array_equal(f1[:, CONTENT], f2[:, CONTENT])
    n1, n2 = nuisance_code(f1), nuisance_code(f2)
    C = np.corrcoef(n1.T, n2.T)[:5, 5:]
    assert np.abs(C).max() < 0.03, C


def test_pairs_avoid_test_split():
    S = Shapes(seed=4)
    f1, f2 = S.sample_pair_factors(20000, 1.0)
    assert not S.is_test[S._idx(f1)].any() and not S.is_test[S._idx(f2)].any()
    ft = S.sample_factors(2000, "test")
    assert S.is_test[S._idx(ft)].all()


def test_latent_law_frequencies():
    S = Shapes("laplace", seed=5)
    f = S.sample_factors(60000)
    emp = np.bincount(f[:, SC], minlength=8) / len(f)
    assert np.abs(emp - law_weights(8, "laplace")).max() < 0.01


def test_clips_fixed_content_moving_nuisance():
    S = Shapes(seed=6)
    f = S.sample_clip_factors(2000, 4, 1.0)
    assert (f[:, :, CONTENT] == f[:, :1, CONTENT]).all()
    steps = np.abs(np.diff(f[:, :, list(NUISANCE)], axis=1))
    assert ((steps == 1) | (steps == 9) | (steps == 2)).all()   # +-1, hue wrap 9, reflect at ends 2? no



def test_cifar_alpha0_identity():
    C = Cifar(seed=0, noise=0.0)
    x = C.raw(np.arange(16))
    y, p = C.augment(x, 0.0)
    assert torch.allclose(y + 0.5, x, atol=1e-4)
    assert np.allclose(p[:, 5:], 1.0) and np.allclose(p[:, :5], 0.0)


def test_probes_recover_given_factors():
    S = Shapes(seed=7)
    ftr, fte = S.sample_factors(5000), S.sample_factors(2000, "test")
    Str, Ste = content_code(ftr), content_code(fte)
    Ntr, Nte = nuisance_code(ftr), nuisance_code(fte)
    assert M.ridge_r2(Str, Str, Ste, Ste)[0] > 0.999
    assert M.ridge_r2(Ntr, Ntr, Nte, Nte)[0] > 0.999
    assert M.ridge_r2(Ntr, Str, Nte, Ste)[0] < 0.01       # nuisance carries no content
    assert M.logreg_acc(Str, ftr[:, 4], Ste, fte[:, 4]) > 0.99
    R = np.linalg.qr(np.random.default_rng(0).normal(size=(6, 6)))[0]
    Sz = (Ste - Ste.mean(0)) / Ste.std(0)          # metric is defined on standardised content
    assert M.orth_identifiability(Sz @ R * 3.0, Ste) > 0.999
    # nonlinear (non-orthogonal) map lowers it
    A = np.random.default_rng(1).normal(size=(6, 6))
    assert M.orth_identifiability(Ste @ A, Ste) < 0.9


def test_effective_rank():
    Z = np.random.default_rng(0).normal(size=(5000, 8))
    assert 7.5 < M.effective_rank(Z) <= 8.0
    Z[:, 1:] *= 1e-6
    assert M.effective_rank(Z) < 1.1
