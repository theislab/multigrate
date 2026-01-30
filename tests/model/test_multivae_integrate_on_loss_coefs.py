import pytest

from multigrate.model._multivae import MultiVAE


def test_integrate_on_sets_default_integ_with_warning(multivae_adata_2mod, setup_multivae_anndata):
    adata = setup_multivae_anndata(multivae_adata_2mod.copy())

    with pytest.warns(UserWarning, match=r"integ=1\.0"):
        model = MultiVAE(
            adata,
            integrate_on="batch",
            loss_coefs=None,
            z_dim=4,
            n_layers_encoders=[1, 1],
            n_layers_decoders=[1, 1],
            n_hidden_encoders=[8, 8],
            n_hidden_decoders=[8, 8],
            dropout=0.0,
        )

    assert model.module.loss_coefs["integ"] == 1.0


def test_integrate_on_with_zero_integ_raises(multivae_adata_2mod, setup_multivae_anndata):
    adata = setup_multivae_anndata(multivae_adata_2mod.copy())

    with pytest.raises(ValueError, match=r"loss_coefs\['integ'\]"):
        _ = MultiVAE(
            adata,
            integrate_on="batch",
            loss_coefs={"integ": 0.0},
            z_dim=4,
            n_layers_encoders=[1, 1],
            n_layers_decoders=[1, 1],
            n_hidden_encoders=[8, 8],
            n_hidden_decoders=[8, 8],
            dropout=0.0,
        )


def test_integrate_on_with_positive_integ_is_kept(multivae_adata_2mod, setup_multivae_anndata):
    adata = setup_multivae_anndata(multivae_adata_2mod.copy())

    model = MultiVAE(
        adata,
        integrate_on="batch",
        loss_coefs={"integ": 2.5},
        z_dim=4,
        n_layers_encoders=[1, 1],
        n_layers_decoders=[1, 1],
        n_hidden_encoders=[8, 8],
        n_hidden_decoders=[8, 8],
        dropout=0.0,
    )

    assert model.module.loss_coefs["integ"] == 2.5
