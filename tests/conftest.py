# tests/conftest.py
import anndata as ad
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def tiny_multimodal_adata(rng):
    n_cells = 32
    n_rna = 8
    n_other = 5
    x = rng.poisson(lam=1.0, size=(n_cells, n_rna + n_other)).astype(np.float32)

    adata = ad.AnnData(X=x)
    adata.obs = pd.DataFrame(
        {
            "batch": rng.choice(["b1", "b2"], size=n_cells),
        },
        index=adata.obs_names,
    )

    adata.var["modality"] = (["rna"] * n_rna) + (["other"] * n_other)
    adata.uns["modality_lengths"] = [n_rna, n_other]

    return adata
