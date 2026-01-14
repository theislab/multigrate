import pytest
import torch

from multigrate.distributions._mmd import MMD


def test_gaussian_kernel_shape_and_symmetry():
    mmd = MMD(kernel_type="gaussian")
    x = torch.randn(5, 3)
    y = torch.randn(7, 3)

    Kxy = mmd.gaussian_kernel(x, y, gamma=[1.0, 10.0])
    assert Kxy.shape == (5, 7)

    # Kxx should be symmetric
    Kxx = mmd.gaussian_kernel(x, x, gamma=[1.0, 10.0])
    assert Kxx.shape == (5, 5)
    assert torch.allclose(Kxx, Kxx.T, atol=1e-6)


def test_forward_gaussian_nonnegative_and_zero_when_identical():
    mmd = MMD(kernel_type="gaussian")
    x = torch.randn(10, 4)

    # identical distributions => MMD ~ 0 (numerical tolerance)
    val_same = mmd(x, x)
    assert torch.isfinite(val_same)
    assert val_same >= -1e-6  # allow tiny negative from floating error
    assert torch.abs(val_same) < 1e-5

    # different samples => typically > 0
    y = torch.randn(10, 4)
    val_diff = mmd(x, y)
    assert torch.isfinite(val_diff)
    assert val_diff >= -1e-6


def test_forward_not_gaussian_zero_when_identical():
    mmd = MMD(kernel_type="not gaussian")
    x = torch.randn(10, 4)
    val = mmd(x, x)
    assert torch.isfinite(val)
    assert torch.abs(val) < 1e-6


def test_skip_batch_len_1_returns_zero_scalar():
    mmd = MMD(kernel_type="gaussian")
    x = torch.randn(1, 4)
    y = torch.randn(5, 4)

    out = mmd(x, y)
    assert out.shape == torch.Size([])  # scalar
    assert float(out) == 0.0


def test_skip_batch_len_1_matches_device_and_dtype():
    mmd = MMD(kernel_type="gaussian")
    x = torch.randn(1, 4, dtype=torch.float64)
    y = torch.randn(3, 4, dtype=torch.float64)

    out = mmd(x, y)
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_gamma_validation_empty_list_raises():
    mmd = MMD(kernel_type="gaussian")
    x = torch.randn(3, 2)
    y = torch.randn(4, 2)

    with pytest.raises(ValueError, match="gamma.*non-empty"):
        _ = mmd.gaussian_kernel(x, y, gamma=[])


def test_gamma_validation_nonpositive_raises():
    mmd = MMD(kernel_type="gaussian")
    x = torch.randn(3, 2)
    y = torch.randn(4, 2)

    with pytest.raises(ValueError, match="positive"):
        _ = mmd.gaussian_kernel(x, y, gamma=[1.0, 0.0])


def test_forward_feature_dim_mismatch_raises():
    mmd = MMD(kernel_type="gaussian")
    x = torch.randn(3, 2)
    y = torch.randn(3, 5)

    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        _ = mmd(x, y)


def test_forward_requires_2d_inputs():
    mmd = MMD(kernel_type="gaussian")
    x = torch.randn(3)      # 1D
    y = torch.randn(3, 2)

    with pytest.raises(ValueError, match="must be 2D"):
        _ = mmd(x, y)


def test_invalid_kernel_type_raises():
    with pytest.raises(ValueError, match="kernel_type"):
        _ = MMD(kernel_type="typo")
