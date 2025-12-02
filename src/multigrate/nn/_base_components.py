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
        Type of normalization to use. Can be one of ["layer", "batch", "none"].
    activation
        Activation function to use.

    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        normalization: str = "layer",
        activation=nn.LeakyReLU,
    ):
        super().__init__()
        use_layer_norm = False
        use_batch_norm = True
        if normalization == "layer":
            use_layer_norm = True
            use_batch_norm = False
        elif normalization == "none":
            use_batch_norm = False

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
            Tensor of values with shape ``(n_input,)``.

        Returns
        -------
        Tensor of values with shape ``(n_output,)``.
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
        Type of normalization to use. Can be one of ["layer", "batch", "none"].
    activation
        Activation function to use.
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
        normalization: str = "layer",
        activation=nn.LeakyReLU,
        loss="mse",
    ):
        super().__init__()

        if loss not in ["mse", "nb", "zinb", "bce"]:
            raise NotImplementedError(f"Loss function {loss} is not implemented.")
        else:
            self.loss = loss

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            Tensor of values with shape ``(n_input,)``.

        Returns
        -------
        Tensor of values with shape ``(n_output,)``.
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
        Type of non-linearity to use. Can be one of ["logsigm", "sigm"]. Default is "logsigm".
    """

    def __init__(self, dim, nonlin: Literal["logsigm", "sigm"] | None = "logsigm"):
        super().__init__()
        self.nonlin = nonlin
        self.beta = torch.nn.Parameter(torch.ones(1, dim), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros(1, dim), requires_grad=True)

    def forward(self, x) -> torch.Tensor:
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
