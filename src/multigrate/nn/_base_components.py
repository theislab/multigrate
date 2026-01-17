from typing import Literal

import torch
from scvi.nn import FCLayers
from torch import nn


class MLP(nn.Module):
    """A helper class to build blocks of fully-connected, normalization, dropout and activation layers.

    Parameters
    ----------
    n_input
        Number of input features.
    n_output
        Number of output features.
    n_layers
        Number of hidden layers.
    n_hidden
        Number of hidden units.
    dropout_rate
        Dropout rate.
    normalization
        Type of normalization to use. Can be one of ["layer", "batch", None].
    activation
        Activation function to use. Can be one of ["leaky_relu", "tanh"].

    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        normalization: str | None = "layer",
        activation: nn.Module = nn.LeakyReLU,
    ):
        super().__init__()

        if n_input <= 0 or n_output <= 0:
            raise ValueError("`n_input` and `n_output` must be positive.")
        if n_layers < 0:
            raise ValueError("`n_layers` must be >= 0.")
        if n_hidden <= 0:
            raise ValueError("`n_hidden` must be positive.")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError("`dropout_rate` must be in [0, 1].")

        if normalization not in {"layer", "batch", None}:
            raise ValueError("`normalization` must be one of {'layer', 'batch', None}.")

        # if none, both are False
        use_layer_norm = normalization == "layer"
        use_batch_norm = normalization == "batch"

        self.mlp = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            activation_fn=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            Tensor of values with shape ``(batch_size, n_input)``.

        Returns
        -------
        Tensor of values with shape ``(batch_size, n_output)``.
        """
        return self.mlp(x)


class Decoder(nn.Module):
    """A helper class to build custom decoders depending on which loss was passed.

    Parameters
    ----------
    n_input
        Number of input features.
    n_output
        Number of output features.
    n_layers
        Number of hidden layers.
    n_hidden
        Number of hidden units.
    dropout_rate
        Dropout rate.
    normalization
        Type of normalization to use. Can be one of ["layer", "batch", None].
    activation
        Activation function to use. Can be one of ["leaky_relu", "tanh"].
    loss
        Loss function to use. Can be one of ["mse", "nb", "zinb", "bce"].
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        normalization: str | None = "layer",
        activation: nn.Module = nn.LeakyReLU,
        loss: str = "mse",
    ):
        super().__init__()

        if n_input <= 0 or n_output <= 0:
            raise ValueError("`n_input` and `n_output` must be positive.")
        if n_layers < 0:
            raise ValueError("`n_layers` must be >= 0.")
        if n_hidden <= 0:
            raise ValueError("`n_hidden` must be positive.")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError("`dropout_rate` must be in [0, 1].")

        if loss not in ["mse", "nb", "zinb", "bce"]:
            raise ValueError("`loss` must be one of {'mse', 'nb', 'zinb', 'bce'}.")
        else:
            self.loss = loss

        if normalization not in {"layer", "batch", None}:
            raise ValueError("`normalization` must be one of {'layer', 'batch', None}.")

        self.decoder = MLP(
            n_input=n_input,
            n_output=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            normalization=normalization,
            activation=activation,
        )
        if loss == "mse":
            self.recon_decoder = nn.Linear(n_hidden, n_output)
        elif loss == "nb":
            self.mean_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1))

        elif loss == "zinb":
            self.mean_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1))
            self.dropout_decoder = nn.Linear(n_hidden, n_output)

        elif loss == "bce":
            self.recon_decoder = FCLayers(
                n_in=n_hidden,
                n_out=n_output,
                n_layers=0,
                dropout_rate=0,
                use_layer_norm=False,
                use_batch_norm=False,
                activation_fn=nn.Sigmoid,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            Tensor of values with shape ``(batch_size, n_input)``.

        Returns
        -------
        If ``loss == "mse"``:
            torch.Tensor
                Reconstructed values with shape ``(batch_size, n_output)``.

        If ``loss == "bce"``:
            torch.Tensor
                Bernoulli probabilities with shape ``(batch_size, n_output)``.

        If ``loss == "nb"``:
            torch.Tensor
                Mean parameter of the Negative Binomial distribution with shape
                ``(batch_size, n_output)``.

        If ``loss == "zinb"``:
            Tuple[torch.Tensor, torch.Tensor]
                * Mean parameter of the ZINB distribution with shape ``(batch_size, n_output)``
                * Dropout logits with shape ``(batch_size, n_output)``
        """
        x = self.decoder(x)
        if self.loss == "mse" or self.loss == "bce":
            return self.recon_decoder(x)
        elif self.loss == "nb":
            return self.mean_decoder(x)
        elif self.loss == "zinb":
            return self.mean_decoder(x), self.dropout_decoder(x)


class GeneralizedSigmoid(nn.Module):
    """Sigmoid, log-sigmoid or linear functions for encoding continuous covariates.

    Adapted from
    Title: CPA (c) Facebook, Inc.
    Date: 26.01.2022
    Link to the used code:
    https://github.com/facebookresearch/CPA/blob/382ff641c588820a453d801e5d0e5bb56642f282/compert/model.py#L109

    Parameters
    ----------
    dim
        Number of input features.
    nonlin
        Type of non-linearity to use. Can be one of ["logsigm", "sigm", None]. Default is "logsigm".
    """

    def __init__(self, dim: int, nonlin: Literal["logsigm", "sigm"] | None = "logsigm"):
        super().__init__()
        if dim <= 0:
            raise ValueError("`dim` must be positive.")
        if nonlin not in {"logsigm", "sigm", None}:
            raise ValueError("`nonlin` must be one of {'logsigm', 'sigm', None}.")
        self.nonlin = nonlin
        self.beta = torch.nn.Parameter(torch.ones(1, dim), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros(1, dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            Tensor of values.

        Returns
        -------
        Tensor of values with the same shape as ``x``.
        """
        if self.nonlin == "logsigm":
            return (torch.log1p(x) * self.beta + self.bias).sigmoid()
        elif self.nonlin == "sigm":
            return (x * self.beta + self.bias).sigmoid()
        else:
            return x
