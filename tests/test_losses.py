import math
import torch
import pytest

from stein.losses import (Prior, imq_stein_kernel_matrix, ksd_imq, sliced_ksd_imq, mmd_gaussian,
                          vicreg_reg, epps_pulley, sigreg_ref, SIGReg, Regulariser)

@pytest.fixture(autouse=True)
def float64_default():
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)


def autograd_stein_kernel(x, prior, c=1.0, beta=-0.5):
    """Reference Langevin-Stein kernel by autograd, O(n^2) loops (small n)."""
    n, d = x.shape
    H = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            xi = x[i].clone().requires_grad_(True)
            xj = x[j].clone().requires_grad_(True)
            k = (c * c + ((xi - xj) ** 2).sum()) ** beta
            gi, gj = torch.autograd.grad(k, (xi, xj), create_graph=True)
            tr = sum(torch.autograd.grad(gi[a], xj, retain_graph=True)[0][a] for a in range(d))
            si, sj = prior.score(x[i]), prior.score(x[j])
            H[i, j] = k * (si @ sj) + si @ gj + sj @ gi + tr
    return H.detach()


@pytest.mark.parametrize("law", ["gaussian", "laplace", "uniform"])
def test_stein_kernel_matches_autograd(law):
    torch.manual_seed(0)
    x = torch.randn(6, 3)
    p = Prior(law, d=3)
    assert torch.allclose(imq_stein_kernel_matrix(x, p.score(x)), autograd_stein_kernel(x, p), atol=1e-8)


def test_sliced_ksd_1d_equals_full():
    torch.manual_seed(0)
    x = torch.randn(50, 1) * 1.3 + 0.2
    p = Prior("gaussian", d=1)
    full = ksd_imq(x, p)
    sl = sliced_ksd_imq(x, p, dirs=torch.ones(1, 1))
    assert torch.allclose(full, sl, atol=1e-10)


def test_ksd_gradcheck():
    torch.manual_seed(0)
    x = torch.randn(8, 3, requires_grad=True)
    for law in ["gaussian", "laplace", "uniform"]:
        p = Prior(law, d=3)
        assert torch.autograd.gradcheck(lambda z: ksd_imq(z, p), (x,))
    p = Prior("rank_k", d=3, k=2)
    assert torch.autograd.gradcheck(lambda z: ksd_imq(z, p), (x,))


def test_sliced_ksd_gradcheck():
    torch.manual_seed(0)
    x = torch.randn(8, 3, requires_grad=True)
    dirs = torch.randn(3, 5); dirs = dirs / dirs.norm(dim=0)
    for law in ["gaussian", "laplace"]:
        p = Prior(law, d=3)
        assert torch.autograd.gradcheck(lambda z: sliced_ksd_imq(z, p, dirs=dirs), (x,))


def test_mmd_gradcheck():
    torch.manual_seed(0)
    x = torch.randn(8, 3, requires_grad=True)
    y = torch.randn(9, 3)
    assert torch.autograd.gradcheck(lambda z: mmd_gaussian(z, y), (x,))


def test_vicreg_gradcheck():
    torch.manual_seed(0)
    x = (0.5 * torch.randn(10, 4)).requires_grad_(True)
    assert torch.autograd.gradcheck(vicreg_reg, (x,))


def test_sigreg_gradcheck():
    torch.manual_seed(0)
    x = torch.randn(10, 4, requires_grad=True)
    dirs = torch.randn(4, 6); dirs = dirs / dirs.norm(dim=0)
    assert torch.autograd.gradcheck(lambda z: sigreg_ref(z, dirs), (x,))


def test_sigreg_matches_lejepa():
    torch.set_default_dtype(torch.float32)
    try:
        torch.manual_seed(0)
        z = torch.randn(256, 16) * 1.5 + 0.3
        mod = SIGReg(num_slices=128)
        ref = mod(z)                                  # lejepa: generator seeded with global_step = 0
        g = torch.Generator().manual_seed(0)
        A = torch.randn(16, 128, generator=g); A = A / A.norm(p=2, dim=0)
        mine = sigreg_ref(z, A)
        assert torch.allclose(ref, mine, rtol=1e-5), (ref, mine)
        # gradients agree too
        z1 = z.clone().requires_grad_(True); z2 = z.clone().requires_grad_(True)
        mod2 = SIGReg(num_slices=128)
        mod2(z1).backward(); sigreg_ref(z2, A).backward()
        assert torch.allclose(z1.grad, z2.grad, rtol=1e-4, atol=1e-7)
    finally:
        torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize("name", ["vicreg", "sigreg", "sksd", "ksd", "mmd", "oracle"])
def test_regulariser_null_vs_alternatives(name):
    """~0 (relative) on N(0,I); clearly larger on collapsed and anisotropic samples."""
    torch.set_default_dtype(torch.float32)
    try:
        torch.manual_seed(0)
        d, n = 8, 512
        reg = Regulariser(name, d, k=d)              # oracle with k = d targets N(0, I)
        vals = {}
        for case, z in {
            "gauss": torch.randn(n, d),
            "collapsed": 0.01 * torch.randn(n, d) + 1.0,
            "aniso": torch.randn(n, d) * torch.tensor([3.0, 3, 1, 1, 1, 1, 0.1, 0.1]),
        }.items():
            vals[case] = float(torch.stack([reg(z) for _ in range(3)]).mean())
        assert vals["collapsed"] > 10 * abs(vals["gauss"]) + 1e-3, vals
        assert vals["aniso"] > 5 * abs(vals["gauss"]) + 1e-3, vals
    finally:
        torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize("law", ["laplace", "uniform"])
def test_matched_prior_ksd_near_zero(law):
    torch.manual_seed(0)
    p = Prior(law, d=4)
    zs = p.sample(800, 4, exact=False)
    zg = torch.randn(800, 4)
    assert ksd_imq(zs, p) < 0.2 * ksd_imq(zg, p), (ksd_imq(zs, p), ksd_imq(zg, p))


def test_prior_samples_unit_variance():
    torch.manual_seed(0)
    for law in ["gaussian", "laplace", "uniform"]:
        for exact in (True, False):
            v = Prior(law, d=3).sample(200000, 3, exact=exact).var(0).double()
            assert torch.allclose(v, torch.ones(3), atol=0.02), (law, exact, v)
