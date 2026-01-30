# code adapted from https://github.com/scverse/scvi-tools/blob/c53efe06379c866e36e549afbb8158a120b82d14/src/scvi/module/_multivae.py#L900C1-L923C56
# last accessed on 16th October 2024
import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as kld


def _check_params(mu, sigma, name: str):
    if mu.shape != sigma.shape:
        raise ValueError(f"{name}: `mu` and `sigma` must have the same shape, got {mu.shape} vs {sigma.shape}.")
    if mu.ndim < 1:
        raise ValueError(f"{name}: expected at least 1D tensors for `mu`/`sigma`.")


class Jeffreys(torch.nn.Module):
    """Jeffreys divergence (Symmetric KL divergence) using torch distributions.

    Parameters
    ----------
    None
    """

    def __init__(self):
        super().__init__()

    def sym_kld(
        self,
        mu1: torch.Tensor,
        sigma1: torch.Tensor,
        mu2: torch.Tensor,
        sigma2: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric KL divergence between two Gaussians using torch.distributions.

        Parameters
        ----------
        mu1
            Mean of the first distribution.
        sigma1
            Variance of the first distribution (note: this will be square-rooted to get std dev).
        mu2
            Mean of the second distribution.
        sigma2
            Variance of the second distribution (note: this will be square-rooted to get std dev).

        Returns
        -------
        Symmetric KL divergence between the two distributions.
        """
        rv1 = Normal(mu1, sigma1.sqrt())
        rv2 = Normal(mu2, sigma2.sqrt())

        out = kld(rv1, rv2).mean() + kld(rv2, rv1).mean()
        return out

    def forward(
        self,
        params1: tuple[torch.Tensor, torch.Tensor],
        params2: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Forward computation for Jeffreys divergence.

        Parameters
        ----------
        params1
            A tuple of mean and variance of the first distribution.
        params2
            A tuple of mean and variance of the second distribution.

        Returns
        -------
        Jeffreys divergence between the two distributions.
        """
        mu1, sigma1 = params1
        mu2, sigma2 = params2

        _check_params(mu1, sigma1, "params1")
        _check_params(mu2, sigma2, "params2")

        # sigmas should be positive
        if torch.any(sigma1 <= 0):
            raise ValueError("params1: all elements of `sigma` must be positive.")
        if torch.any(sigma2 <= 0):
            raise ValueError("params2: all elements of `sigma` must be positive.")

        return self.sym_kld(mu1, sigma1, mu2, sigma2)
