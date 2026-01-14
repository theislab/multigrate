import pytest

from multigrate.model import MultiVAE


def _setup(adata, n_rna: int):
    MultiVAE.setup_anndata(
        adata,
        categorical_covariate_keys=["batch"],
        rna_indices_end=n_rna,
    )


@pytest.mark.parametrize(
    "model_kwargs",
    [
        dict(mix="product", z_dim=4, cond_dim=4, losses=["nb", "nb"]),
        dict(mix="product", z_dim=4, cond_dim=4, losses=["nb", "nb"], modality_alignment="MMD", alignment_type="latent"),
    ],
)
def test_multivae_smoke_train_and_latent(tiny_multimodal_adata, model_kwargs):
    adata = tiny_multimodal_adata.copy()
    n_rna = adata.uns["modality_lengths"][0]

    _setup(adata, n_rna=n_rna)

    model = MultiVAE(adata, **model_kwargs)

    model.train(
        max_epochs=1,
        check_val_every_n_epoch=1,
        batch_size=2,
    )

    model.get_model_output()
    assert "X_multigrate" in adata.obsm
    assert adata.obsm["X_multigrate"].shape == (adata.n_obs, model_kwargs.get("z_dim", 16))


def test_multivae_raises_on_wrong_losses_length(tiny_multimodal_adata):
    adata = tiny_multimodal_adata.copy()
    n_rna = adata.uns["modality_lengths"][0]
    _setup(adata, n_rna=n_rna)

    with pytest.raises(Exception):
        _ = MultiVAE(adata, losses=["nb"])  # wrong length
