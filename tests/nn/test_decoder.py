import pytest
import torch
from torch import nn

from multigrate.nn._base_components import Decoder


@pytest.mark.parametrize("loss", ["mse", "bce", "nb"])
def test_decoder_forward_returns_tensor_with_correct_shape(loss):
    torch.manual_seed(0)
    dec = Decoder(
        n_input=6,
        n_output=10,
        n_layers=2,
        n_hidden=12,
        dropout_rate=0.0,
        normalization="layer",
        activation=nn.LeakyReLU,
        loss=loss,
    )

    x = torch.randn(4, 6)
    out = dec(x)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 10)
    assert torch.isfinite(out).all()


def test_decoder_forward_zinb_returns_tuple_of_tensors_with_correct_shape():
    torch.manual_seed(0)
    dec = Decoder(
        n_input=6,
        n_output=10,
        n_layers=2,
        n_hidden=12,
        dropout_rate=0.0,
        normalization="layer",
        activation=nn.LeakyReLU,
        loss="zinb",
    )

    x = torch.randn(4, 6)
    mean, dropout = dec(x)

    assert isinstance(mean, torch.Tensor)
    assert isinstance(dropout, torch.Tensor)
    assert mean.shape == (4, 10)
    assert dropout.shape == (4, 10)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(dropout).all()


def test_decoder_bce_output_in_unit_interval():
    torch.manual_seed(0)
    dec = Decoder(
        n_input=5,
        n_output=7,
        n_layers=1,
        n_hidden=9,
        dropout_rate=0.0,
        normalization=None,
        activation=nn.ReLU,
        loss="bce",
    )

    x = torch.randn(6, 5)
    out = dec(x)

    # Because activation_fn=Sigmoid in the BCE recon decoder, output should be probabilities. :contentReference[oaicite:1]{index=1}
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


@pytest.mark.parametrize("loss", ["mse", "bce", "nb", "zinb"])
def test_decoder_backward_pass(loss):
    torch.manual_seed(0)
    dec = Decoder(
        n_input=6,
        n_output=10,
        n_layers=2,
        n_hidden=12,
        dropout_rate=0.1,
        normalization="layer",
        activation=nn.LeakyReLU,
        loss=loss,
    )

    x = torch.randn(4, 6, requires_grad=True)
    out = dec(x)

    if loss == "zinb":
        mean, dropout = out
        # Any scalar function will do; we just want backward to work.
        loss_val = mean.pow(2).mean() + dropout.pow(2).mean()
    else:
        loss_val = out.pow(2).mean()

    loss_val.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert any(p.grad is not None for p in dec.parameters())


def test_decoder_invalid_loss_raises():
    with pytest.raises(ValueError, match=r"`loss` must be one of {'mse', 'nb', 'zinb', 'bce'}."):
        _ = Decoder(n_input=4, n_output=5, loss="not_a_loss")


def test_decoder_invalid_normalization_raises():
    with pytest.raises(ValueError, match=r"`normalization` must be one of {'layer', 'batch', None}."):
        _ = Decoder(n_input=4, n_output=5, normalization="invalid_norm")
