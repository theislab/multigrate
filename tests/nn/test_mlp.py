import pytest
import torch
from torch import nn

from multigrate.nn._base_components import MLP


def _count_modules(model: nn.Module, cls: type[nn.Module]) -> int:
    return sum(1 for m in model.modules() if isinstance(m, cls))


@pytest.mark.parametrize(
    "normalization, expect_layer_norm, expect_batch_norm",
    [
        ("layer", True, False),
        ("batch", False, True),
        (None, False, False),
    ],
)
def test_mlp_normalization_wiring(normalization, expect_layer_norm, expect_batch_norm):
    mlp = MLP(
        n_input=5,
        n_output=7,
        n_layers=2,
        n_hidden=16,
        dropout_rate=0.0,
        normalization=normalization,
        activation=nn.ReLU,
    )

    n_ln = _count_modules(mlp, nn.LayerNorm)
    n_bn = _count_modules(mlp, nn.BatchNorm1d)

    assert (n_ln > 0) == expect_layer_norm
    assert (n_bn > 0) == expect_batch_norm


@pytest.mark.parametrize("activation_cls", [nn.LeakyReLU, nn.Tanh])
def test_mlp_contains_requested_activation_modules(activation_cls):
    mlp = MLP(
        n_input=4,
        n_output=3,
        n_layers=2,
        n_hidden=8,
        dropout_rate=0.0,
        normalization=None,
        activation=activation_cls,
    )

    # For n_layers>0, we expect at least one activation module of the requested type.
    n_act = _count_modules(mlp, activation_cls)
    assert n_act >= 1

    # Also check forward works
    x = torch.randn(6, 4)
    y = mlp(x)
    assert y.shape == (6, 3)
    assert torch.isfinite(y).all()


def test_mlp_forward_shape_and_grad():
    torch.manual_seed(0)
    mlp = MLP(
        n_input=5,
        n_output=7,
        n_layers=2,
        n_hidden=16,
        dropout_rate=0.1,
        normalization="layer",
        activation=nn.LeakyReLU,
    )

    x = torch.randn(4, 5, requires_grad=True)
    y = mlp(x)

    assert y.shape == (4, 7)
    assert torch.isfinite(y).all()

    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert any(p.grad is not None for p in mlp.parameters())


def test_mlp_dropout_modules_present_when_dropout_rate_positive():
    mlp = MLP(
        n_input=5,
        n_output=7,
        n_layers=2,
        n_hidden=16,
        dropout_rate=0.2,
        normalization=None,
        activation=nn.ReLU,
    )
    n_do = _count_modules(mlp, nn.Dropout)
    assert n_do >= 1


def test_mlp_dropout_modules_absent_when_dropout_rate_zero():
    mlp = MLP(
        n_input=5,
        n_output=7,
        n_layers=2,
        n_hidden=16,
        dropout_rate=0.0,
        normalization=None,
        activation=nn.ReLU,
    )
    n_do = _count_modules(mlp, nn.Dropout)
    assert n_do == 0
