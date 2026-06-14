import pytest
import torch

from multigrate.distributions._jeffreys import Jeffreys, _check_params


def test_check_params_raises_on_shape_mismatch():
    mu = torch.zeros(4, 2)
    sigma = torch.ones(4, 3)
    with pytest.raises(ValueError, match=r"must have the same shape"):
        _check_params(mu, sigma, "params1")


def test_check_params_raises_on_scalar_inputs():
    mu = torch.tensor(0.0)
    sigma = torch.tensor(1.0)
    with pytest.raises(ValueError, match=r"expected at least 1D"):
        _check_params(mu, sigma, "params1")


def test_sym_kld_zero_for_identical_gaussians():
    j = Jeffreys()
    mu = torch.zeros(8, 3)
    var = torch.ones(8, 3)  # variance must be positive
    out = j.sym_kld(mu, var, mu, var)
    assert torch.isfinite(out)
    assert torch.abs(out) < 1e-6


def test_sym_kld_is_symmetric():
    j = Jeffreys()
    mu1 = torch.zeros(10, 4)
    var1 = torch.ones(10, 4)

    mu2 = torch.ones(10, 4) * 0.3
    var2 = torch.ones(10, 4) * 2.0

    a = j.sym_kld(mu1, var1, mu2, var2)
    b = j.sym_kld(mu2, var2, mu1, var1)

    assert torch.isfinite(a) and torch.isfinite(b)
    assert torch.allclose(a, b, atol=1e-6)


def test_forward_returns_finite_nonnegative():
    j = Jeffreys()
    mu1 = torch.randn(6, 2)
    var1 = torch.ones(6, 2) * 0.5
    mu2 = torch.randn(6, 2)
    var2 = torch.ones(6, 2) * 1.5

    out = j((mu1, var1), (mu2, var2))
    assert torch.isfinite(out)
    # Jeffreys divergence should be >= 0 (allow tiny numerical jitter)
    assert out >= -1e-8


def test_forward_backprop_works():
    j = Jeffreys()
    mu1 = torch.randn(5, 3, requires_grad=True)
    var1 = torch.ones(5, 3, requires_grad=True)
    mu2 = torch.randn(5, 3)
    var2 = torch.ones(5, 3)

    out = j((mu1, var1), (mu2, var2))
    out.backward()

    assert mu1.grad is not None
    assert var1.grad is not None
    assert torch.isfinite(mu1.grad).all()
    assert torch.isfinite(var1.grad).all()


def test_forward_raises_on_mu_sigma_shape_mismatch_params1():
    j = Jeffreys()
    mu1 = torch.zeros(4, 2)
    var1 = torch.ones(4, 3)  # mismatch
    mu2 = torch.zeros(4, 2)
    var2 = torch.ones(4, 2)

    with pytest.raises(ValueError, match=r"params1: `mu` and `sigma` must have the same shape"):
        _ = j((mu1, var1), (mu2, var2))


def test_forward_raises_on_mu_sigma_shape_mismatch_params2():
    j = Jeffreys()
    mu1 = torch.zeros(4, 2)
    var1 = torch.ones(4, 2)
    mu2 = torch.zeros(4, 2)
    var2 = torch.ones(4, 3)  # mismatch

    with pytest.raises(ValueError, match=r"params2: `mu` and `sigma` must have the same shape"):
        _ = j((mu1, var1), (mu2, var2))


def test_forward_raises_on_nonpositive_sigma_params1():
    j = Jeffreys()
    mu1 = torch.zeros(4, 2)
    var1 = torch.tensor([[1.0, 0.0]]).repeat(4, 1)  # contains 0 -> invalid
    mu2 = torch.zeros(4, 2)
    var2 = torch.ones(4, 2)

    with pytest.raises(ValueError, match=r"params1: all elements of `sigma` must be positive"):
        _ = j((mu1, var1), (mu2, var2))


def test_forward_raises_on_nonpositive_sigma_params2():
    j = Jeffreys()
    mu1 = torch.zeros(4, 2)
    var1 = torch.ones(4, 2)
    mu2 = torch.zeros(4, 2)
    var2 = torch.tensor([[-1.0, 2.0]]).repeat(4, 1)  # contains negative -> invalid

    with pytest.raises(ValueError, match=r"params2: all elements of `sigma` must be positive"):
        _ = j((mu1, var1), (mu2, var2))
