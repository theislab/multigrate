import pytest

from multigrate.model._multivae import MultiVAE


@pytest.fixture
def adata(setup_multivae_anndata, multivae_adata_2mod):
    return setup_multivae_anndata(multivae_adata_2mod.copy())


@pytest.fixture
def model(adata):
    # Small model for fast unit tests (no training)
    m = MultiVAE(
        adata,
        z_dim=8,
        cond_dim=4,
        n_layers_encoders=[1, 1],
        n_layers_decoders=[1, 1],
        n_hidden_encoders=[16, 16],
        n_hidden_decoders=[16, 16],
        dropout=0.0,
        losses=["mse", "mse"],
    )
    return m


# --------------------------
# Constructor / validation
# --------------------------


def test_init_requires_modality_lengths_key(setup_multivae_anndata, multivae_adata_2mod):
    adata = setup_multivae_anndata(multivae_adata_2mod.copy())
    del adata.uns["modality_lengths"]

    with pytest.raises(ValueError, match=r"Missing `adata\.uns\['modality_lengths'\]`"):
        _ = MultiVAE(adata)


def test_init_requires_modality_lengths_is_list(setup_multivae_anndata, multivae_adata_2mod):
    adata = setup_multivae_anndata(multivae_adata_2mod.copy())
    adata.uns["modality_lengths"] = "not a list"

    with pytest.raises(ValueError, match=r"must be a list"):
        _ = MultiVAE(adata)


def test_init_requires_modality_lengths_sum_matches_n_vars(setup_multivae_anndata, multivae_adata_2mod):
    adata = setup_multivae_anndata(multivae_adata_2mod.copy())
    # wrong sum: n_vars won’t match
    adata.uns["modality_lengths"] = [adata.n_vars, 1]

    with pytest.raises(ValueError, match=r"Sum of `modality_lengths`"):
        _ = MultiVAE(adata)


def test_init_losses_length_must_match_n_mods(adata):
    # n_mod = 2 but losses has length 1
    with pytest.raises(ValueError, match=r"`losses` must have length 2"):
        _ = MultiVAE(adata, losses=["mse"])


def test_init_integrate_on_must_be_registered(adata):
    with pytest.raises(ValueError, match=r"Cannot integrate on"):
        _ = MultiVAE(adata, integrate_on="not_registered_cov")


def test_init_integrate_on_valid_key(adata):
    # "batch" is registered in setup_multivae_anndata fixture
    m = MultiVAE(adata, integrate_on="batch", z_dim=8, dropout=0.0, loss_coefs={"integ": 1.0})
    assert m.integrate_on_idx is not None


# --------------------------
# Helper-method gatekeeping
# --------------------------


def test_get_model_output_requires_trained(model, adata):
    assert model.is_trained_ is False
    with pytest.raises(RuntimeError, match=r"train the model first"):
        model.get_model_output(adata, batch_size=8)


def test_impute_requires_trained(model, adata):
    assert model.is_trained_ is False
    with pytest.raises(RuntimeError, match=r"train the model first"):
        model.impute(adata, batch_size=8)


# --------------------------
# Helper-method behavior (without full training)
# These tests intentionally bypass training to keep them fast and stable.
# --------------------------


def test_get_model_output_writes_joint_latent(model, adata):
    model.is_trained_ = True
    model.module.eval()

    model.get_model_output(adata, batch_size=8)

    assert "X_multigrate" in adata.obsm
    Xz = adata.obsm["X_multigrate"]
    assert Xz.shape == (adata.n_obs, model.module.z_dim)


def test_get_model_output_writes_unimodal_latents_when_requested(model, adata):
    model.is_trained_ = True
    model.module.eval()

    model.get_model_output(adata, batch_size=8, save_unimodal_latent=True)

    assert "X_unimodal_0" in adata.obsm
    assert "X_unimodal_1" in adata.obsm
    assert adata.obsm["X_unimodal_0"].shape == (adata.n_obs, model.module.z_dim)
    assert adata.obsm["X_unimodal_1"].shape == (adata.n_obs, model.module.z_dim)


def test_get_model_output_writes_unimodal_params_when_requested(model, adata):
    model.is_trained_ = True
    model.module.eval()

    model.get_model_output(adata, batch_size=8, save_unimodal_params=True)

    for i in (0, 1):
        assert f"mu_unimodal_{i}" in adata.obsm
        assert f"logvar_unimodal_{i}" in adata.obsm
        assert adata.obsm[f"mu_unimodal_{i}"].shape == (adata.n_obs, model.module.z_dim)
        assert adata.obsm[f"logvar_unimodal_{i}"].shape == (adata.n_obs, model.module.z_dim)


def test_impute_writes_imputed_modalities(model, adata):
    model.is_trained_ = True
    model.module.eval()

    model.impute(adata, batch_size=8)

    m0, m1 = adata.uns["modality_lengths"]
    assert "imputed_modality_0" in adata.obsm
    assert "imputed_modality_1" in adata.obsm
    assert adata.obsm["imputed_modality_0"].shape == (adata.n_obs, m0)
    assert adata.obsm["imputed_modality_1"].shape == (adata.n_obs, m1)
