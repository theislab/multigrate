import anndata as ad
import numpy as np
import pytest

from multigrate.model._multivae import MultiVAE


def _make_minimal_multimodal_adata(
    n_obs: int = 16,
    modality_lengths=(4, 3),
    with_covariates: bool = True,
) -> ad.AnnData:
    n_vars = int(sum(modality_lengths))
    X = np.random.poisson(lam=1.0, size=(n_obs, n_vars)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.uns["modality_lengths"] = list(modality_lengths)

    # scvi usually expects some size factor if you register it
    adata.obs["size_factor"] = np.ones(n_obs, dtype=np.float32)

    if with_covariates:
        adata.obs["batch"] = np.random.choice(["b0", "b1"], size=n_obs)
        adata.obs["cat2"] = np.random.choice(["c0", "c1", "c2"], size=n_obs)
        adata.obs["cont1"] = np.random.randn(n_obs).astype(np.float32)

    return adata


def _setup_anndata(adata: ad.AnnData) -> None:
    MultiVAE.setup_anndata(
        adata,
        size_factor_key="size_factor",
        categorical_covariate_keys=["batch", "cat2"],
        continuous_covariate_keys=["cont1"],
    )


@pytest.fixture()
def adata():
    adata = _make_minimal_multimodal_adata()
    _setup_anndata(adata)
    return adata


@pytest.fixture()
def model(adata):
    # Keep it small so CPU tests are fast
    m = MultiVAE(
        adata,
        z_dim=8,
        n_layers_encoders=[1, 1],
        n_layers_decoders=[1, 1],
        n_hidden_encoders=[16, 16],
        n_hidden_decoders=[16, 16],
        dropout=0.0,
        # leaving losses=None -> defaults to mse for all modalities in module
    )
    return m


def _setup(adata, n_rna: int):
    MultiVAE.setup_anndata(
        adata,
        categorical_covariate_keys=["batch"],
        rna_indices_end=n_rna,
    )


@pytest.mark.parametrize(
    "model_kwargs",
    [
        {"mix": "product", "z_dim": 4, "cond_dim": 4, "losses": ["nb", "nb"]},
        {
            "mix": "product",
            "z_dim": 4,
            "cond_dim": 4,
            "losses": ["nb", "nb"],
            "modality_alignment": "MMD",
            "alignment_type": "latent",
        },
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

    with pytest.raises(ValueError, match=r"`losses` must have length 2"):
        _ = MultiVAE(adata, losses=["nb"])  # wrong length


# --------------------------
# Constructor / validation tests
# --------------------------


def test_init_requires_modality_lengths_key():
    adata = _make_minimal_multimodal_adata()
    _setup_anndata(adata)
    del adata.uns["modality_lengths"]

    with pytest.raises(ValueError, match=r"Missing `adata\.uns\['modality_lengths'\]`."):
        _ = MultiVAE(adata)


def test_init_requires_modality_lengths_is_list():
    adata = _make_minimal_multimodal_adata()
    _setup_anndata(adata)
    adata.uns["modality_lengths"] = "not a list"

    with pytest.raises(ValueError, match=r"must be a list"):
        _ = MultiVAE(adata)


def test_init_requires_modality_lengths_sum_matches_n_vars():
    adata = _make_minimal_multimodal_adata(modality_lengths=(4, 3))
    _setup_anndata(adata)
    adata.uns["modality_lengths"] = [4, 999]  # wrong sum

    with pytest.raises(ValueError, match=r"Sum of `modality_lengths`"):
        _ = MultiVAE(adata)


def test_init_losses_length_must_match_n_mods(adata):
    # n_mod = 2, but losses has len 1
    with pytest.raises(ValueError, match=r"`losses` must have length 2"):
        _ = MultiVAE(adata, losses=["mse"])


def test_init_integrate_on_must_be_registered(adata):
    with pytest.raises(ValueError, match=r"Cannot integrate on"):
        _ = MultiVAE(adata, integrate_on="not_registered_cov")


def test_init_integrate_on_valid_key(adata):
    # should succeed because "batch" was registered as a categorical covariate
    m = MultiVAE(adata, integrate_on="batch", z_dim=8, dropout=0.0)
    assert m.integrate_on_idx is not None


# --------------------------
# Inference helper tests (no training loop)
# --------------------------


def test_get_model_output_requires_trained(model, adata):
    # default is not trained
    assert model.is_trained_ is False
    with pytest.raises(RuntimeError, match=r"train the model first"):
        model.get_model_output(adata, batch_size=8)


def test_impute_requires_trained(model, adata):
    assert model.is_trained_ is False
    with pytest.raises(RuntimeError, match=r"train the model first"):
        model.impute(adata, batch_size=8)


def test_get_model_output_writes_joint_latent(model, adata):
    # Bypass training for a unit test: we only want to test forward + storage
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

    # for 2 modalities
    assert "X_unimodal_0" in adata.obsm
    assert "X_unimodal_1" in adata.obsm
    assert adata.obsm["X_unimodal_0"].shape == (adata.n_obs, model.module.z_dim)
    assert adata.obsm["X_unimodal_1"].shape == (adata.n_obs, model.module.z_dim)


def test_get_model_output_writes_unimodal_params_when_requested(model, adata):
    model.is_trained_ = True
    model.module.eval()

    model.get_model_output(adata, batch_size=8, save_unimodal_params=True)

    assert "mu_unimodal_0" in adata.obsm
    assert "logvar_unimodal_0" in adata.obsm
    assert "mu_unimodal_1" in adata.obsm
    assert "logvar_unimodal_1" in adata.obsm

    assert adata.obsm["mu_unimodal_0"].shape == (adata.n_obs, model.module.z_dim)
    assert adata.obsm["logvar_unimodal_0"].shape == (adata.n_obs, model.module.z_dim)


def test_impute_writes_imputed_modalities(model, adata):
    model.is_trained_ = True
    model.module.eval()

    model.impute(adata, batch_size=8)

    # should create imputed_modality_0 and _1 with correct feature dims per modality
    assert "imputed_modality_0" in adata.obsm
    assert "imputed_modality_1" in adata.obsm

    m0, m1 = adata.uns["modality_lengths"]
    assert adata.obsm["imputed_modality_0"].shape == (adata.n_obs, m0)
    assert adata.obsm["imputed_modality_1"].shape == (adata.n_obs, m1)
