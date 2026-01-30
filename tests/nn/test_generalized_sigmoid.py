import pytest
import torch

from multigrate.nn._base_components import GeneralizedSigmoid


@pytest.mark.parametrize("nonlin", ["logsigm", "sigm", None])
def test_generalized_sigmoid_output_shape(nonlin):
    torch.manual_seed(0)
    dim = 6
    gs = GeneralizedSigmoid(dim=dim, nonlin=nonlin)

    if nonlin == "logsigm":
        # log1p(x) requires x > -1 to avoid NaNs/Infs; choose safe inputs
        x = torch.rand(5, dim, requires_grad=True)  # in (0,1)
    else:
        x = torch.randn(5, dim, requires_grad=True)

    y = gs(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_generalized_sigmoid_none_is_identity():
    torch.manual_seed(0)
    dim = 4
    gs = GeneralizedSigmoid(dim=dim, nonlin=None)

    x = torch.randn(3, dim)
    y = gs(x)
    assert torch.allclose(x, y)


def test_generalized_sigmoid_logsigtm_outputs_in_unit_interval():
    torch.manual_seed(0)
    dim = 4
    gs = GeneralizedSigmoid(dim=dim, nonlin="logsigm")

    x = torch.rand(10, dim)  # >=0 safe for log1p
    y = gs(x)

    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)


def test_generalized_sigmoid_sigm_outputs_in_unit_interval():
    torch.manual_seed(0)
    dim = 4
    gs = GeneralizedSigmoid(dim=dim, nonlin="sigm")

    x = torch.randn(10, dim)
    y = gs(x)

    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)


def test_generalized_sigmoid_has_trainable_parameters_with_correct_shape():
    dim = 7
    gs = GeneralizedSigmoid(dim=dim, nonlin="sigm")
    assert gs.beta.shape == (1, dim)
    assert gs.bias.shape == (1, dim)
    assert gs.beta.requires_grad
    assert gs.bias.requires_grad


@pytest.mark.parametrize("nonlin", ["logsigm", "sigm"])
def test_generalized_sigmoid_backward_updates_beta_and_bias(nonlin):
    torch.manual_seed(0)
    dim = 5
    gs = GeneralizedSigmoid(dim=dim, nonlin=nonlin)

    if nonlin == "logsigm":
        x = torch.rand(8, dim, requires_grad=True)
    else:
        x = torch.randn(8, dim, requires_grad=True)

    y = gs(x)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    assert gs.beta.grad is not None
    assert gs.bias.grad is not None
    assert torch.isfinite(gs.beta.grad).all()
    assert torch.isfinite(gs.bias.grad).all()


def test_generalized_sigmoid_logsigm_raises_on_non_positive_input():
    dim = 3
    gs = GeneralizedSigmoid(dim=dim, nonlin="logsigm")

    # includes zero and negative values
    x = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [1.0, -0.5, 3.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"All continuous covariates must be positive",
    ):
        _ = gs(x)


def test_generalized_sigmoid_logsigm_accepts_strictly_positive_input():
    dim = 3
    gs = GeneralizedSigmoid(dim=dim, nonlin="logsigm")

    x = torch.tensor(
        [
            [0.1, 1.0, 10.0],
            [0.01, 2.5, 3.3],
        ]
    )

    y = gs(x)

    assert torch.isfinite(y).all()
    assert y.shape == x.shape
    assert torch.all((y > 0) & (y < 1))
