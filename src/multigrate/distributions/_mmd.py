import torch


class MMD(torch.nn.Module):
    """Maximum mean discrepancy.

    Parameters
    ----------
    kernel_type
        Indicates if to use Gaussian kernel. One of
        * ``'gaussian'`` - use Gaussian kernel
        * ``'not gaussian'`` - do not use Gaussian kernel.
    """

    def __init__(self, kernel_type="gaussian"):
        super().__init__()
        if kernel_type not in {"gaussian", "not gaussian"}:
            raise ValueError("`kernel_type` must be one of {'gaussian', 'not gaussian'}.")
        self.kernel_type = kernel_type

    def gaussian_kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        gamma: list[float] | None = None,
    ) -> torch.Tensor:
        """Apply Guassian kernel.

        Parameters
        ----------
        x
            Tensor from the first distribution.
        y
            Tensor from the second distribution.
        gamma
            List of gamma parameters.

        Returns
        -------
        Gaussian kernel between ``x`` and ``y``.
        """
        if gamma is not None:
            if len(gamma) == 0:
                raise ValueError("`gamma` must be a non-empty list of floats.")
            if any(g <= 0 for g in gamma):
                raise ValueError("All entries in `gamma` must be positive.")
        else:
            gamma = [
                1e-6,
                1e-5,
                1e-4,
                1e-3,
                1e-2,
                1e-1,
                1,
                5,
                10,
                15,
                20,
                25,
                30,
                35,
                100,
                1e3,
                1e4,
                1e5,
                1e6,
            ]

        D = torch.cdist(x, y).pow(2)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K / len(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Adapted from
        Title: scarches
        Date: 9th Octover 2021
        Code version: 0.4.0
        Availability: https://github.com/theislab/scarches/blob/63a7c2b35a01e55fe7e1dd871add459a86cd27fb/scarches/models/trvae/losses.py
        Citation: Gretton, Arthur, et al. "A Kernel Two-Sample Test", 2012.

        Parameters
        ----------
        x
            Tensor with shape ``(batch_size, z_dim)``.
        y
            Tensor with shape ``(batch_size, z_dim)``.

        Returns
        -------
        MMD between ``x`` and ``y``.
        """
        if len(x) == 0 or len(y) == 0:
            raise ValueError("`x` and `y` must have at least one sample.")

        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("`x` and `y` must be 2D tensors of shape (batch, dim).")
        if x.shape[1] != y.shape[1]:
            raise ValueError(f"Feature dimension mismatch: x has {x.shape[1]}, y has {y.shape[1]}.")

        # in case there is only one sample in a batch belonging to one of the groups, then skip the batch
        if len(x) == 1 or len(y) == 1:
            return torch.zeros((), device=x.device, dtype=x.dtype)

        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff
