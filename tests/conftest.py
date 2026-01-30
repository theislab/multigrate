import anndata as ad
import numpy as np
import pandas as pd
import pytest

from multigrate.model._multivae import MultiVAE


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def modality_lengths_2mod():
    return (8, 5)


@pytest.fixture
def multivae_adata_2mod(rng, modality_lengths_2mod):
    """
    Standard synthetic 2-modality AnnData for MultiVAE tests.
    Uses .X as continuous values (works well with MSE losses).
    """
    n_cells = 32
    n0, n1 = modality_lengths_2mod
    x = rng.normal(size=(n_cells, n0 + n1)).astype(np.float32)

    adata = ad.AnnData(X=x)
    adata.obs = pd.DataFrame(
        {
            "donor": rng.choice(["d0", "d1", "d2"], size=n_cells),
            "cont1": rng.normal(size=n_cells).astype(np.float32),
            "size_factor": np.ones(n_cells, dtype=np.float32),
        },
        index=adata.obs_names,
    )
    adata.obs.loc[adata.obs["donor"].isin(["d0", "d1"]), "batch"] = "b0"
    adata.obs.loc[adata.obs["donor"].isin(["d2"]), "batch"] = "b1"
    adata.uns["modality_lengths"] = [n0, n1]
    return adata


@pytest.fixture
def setup_multivae_anndata():
    """
    Callable fixture: registers AnnData with scvi for MultiVAE.
    """

    def _setup(adata: ad.AnnData):
        MultiVAE.setup_anndata(
            adata,
            size_factor_key="size_factor",
            categorical_covariate_keys=["batch", "donor"],
            continuous_covariate_keys=["cont1"],
        )
        return adata

    return _setup


@pytest.fixture
def assert_imputed():
    def _assert(adata: ad.AnnData):
        m0, m1 = adata.uns["modality_lengths"]
        assert "imputed_modality_0" in adata.obsm
        assert "imputed_modality_1" in adata.obsm
        assert adata.obsm["imputed_modality_0"].shape == (adata.n_obs, m0)
        assert adata.obsm["imputed_modality_1"].shape == (adata.n_obs, m1)

    return _assert
