import pytest
import torch

from multigrate.module._multivae_torch import MultiVAETorch


def _make_model(
    *,
    modality_lengths=(3, 2),
    z_dim=5,
    losses=None,
    num_groups=1,
    condition_encoders=False,
    condition_decoders=True,
    cat_covariate_dims=(3,),
    cont_covariate_dims=(1,),
    cat_covs_idx=None,
    cont_covs_idx=None,
    mix="product",
    alignment_type="latent",
    modality_alignment=None,
):
    return MultiVAETorch(
        modality_lengths=list(modality_lengths),
        z_dim=z_dim,
        losses=None if losses is None else list(losses),
        num_groups=num_groups,
        condition_encoders=condition_encoders,
        condition_decoders=condition_decoders,
        normalization="layer",
        dropout=0.0,
        cond_dim=4,
        kernel_type="gaussian",
        loss_coefs=None,
        integrate_on_idx=None,
        cat_covariate_dims=list(cat_covariate_dims),
        cont_covariate_dims=list(cont_covariate_dims),
        cat_covs_idx=cat_covs_idx,
        cont_covs_idx=cont_covs_idx,
        cont_cov_type="logsigm",
        n_layers_cont_embed=1,
        n_hidden_cont_embed=8,
        n_layers_encoders=None,
        n_layers_decoders=None,
        n_hidden_encoders=None,
        n_hidden_decoders=None,
        modality_alignment=modality_alignment,
        alignment_type=alignment_type,
        activation="leaky_relu",
        initialization=None,
        mix=mix,
    )


# -------------------------
# __init__ validation tests
# -------------------------

def test_init_requires_cat_covariate_dims():
    with pytest.raises(ValueError, match=r"cat_covariate_dims = None"):
        _ = MultiVAETorch(
            modality_lengths=[3, 2],
            cat_covariate_dims=None,
            cont_covariate_dims=[1],
        )


def test_init_requires_cont_covariate_dims():
    with pytest.raises(ValueError, match=r"cont_covariate_dims = None"):
        _ = MultiVAETorch(
            modality_lengths=[3, 2],
            cat_covariate_dims=[3],
            cont_covariate_dims=None,
        )


@pytest.mark.parametrize("mix", ["bad", "poe", ""])
def test_init_invalid_mix_raises(mix):
    with pytest.raises(ValueError, match=r"mix should be one of"):
        _ = _make_model(mix=mix)


@pytest.mark.parametrize("alignment_type", ["bad", "", "latent_only"])
def test_init_invalid_alignment_type_raises(alignment_type):
    with pytest.raises(ValueError, match=r"alignment_type should be one of"):
        _ = _make_model(alignment_type=alignment_type)


@pytest.mark.parametrize("modality_alignment", ["bad", "mmd", "JEFFREYS"])
def test_init_invalid_modality_alignment_raises(modality_alignment):
    with pytest.raises(ValueError, match=r"modality_alignment should be one of"):
        _ = _make_model(modality_alignment=modality_alignment)


def test_init_default_losses_length_matches_n_modality():
    m = _make_model(losses=None)
    assert m.losses == ["mse"] * len(m.input_dims)


def test_theta_parameterlist_shapes_and_grad_flags():
    # 2 modalities: mse, nb
    m = _make_model(losses=("mse", "nb"), modality_lengths=(4, 7), num_groups=3)

    # modality 0 mse -> empty param (requires_grad False)
    assert isinstance(m.theta[0], torch.nn.Parameter)
    assert m.theta[0].numel() == 0
    assert m.theta[0].requires_grad is False

    # modality 1 nb -> learnable theta shape (n_features, num_groups)
    assert m.theta[1].shape == (7, 3)
    assert m.theta[1].requires_grad is True


# -------------------------
# Core forward-path shape tests
# -------------------------

def test_inference_outputs_have_expected_shapes_product():
    # Use explicit empty indices (works even before you implement the "None->empty" fix)
    cat_idx = torch.tensor([], dtype=torch.long)
    cont_idx = torch.tensor([], dtype=torch.long)
    m = _make_model(
        condition_encoders=False,
        cat_covs_idx=cat_idx,
        cont_covs_idx=cont_idx,
        mix="product",
        modality_lengths=(3, 2),
        z_dim=6,
    )

    batch = 5
    # make inputs positive so x.sum(dim=1) > 0 masks are True
    x = torch.rand(batch, sum(m.input_dims))

    out = m.inference(x)

    assert out["z_joint"].shape == (batch, 6)
    assert out["mu"].shape == (batch, 6)
    assert out["logvar"].shape == (batch, 6)

    assert out["z_marginal"].shape == (batch, m.n_modality, 6)
    assert out["mu_marginal"].shape == (batch, m.n_modality, 6)
    assert out["logvar_marginal"].shape == (batch, m.n_modality, 6)

    # compatibility alias
    assert out["z"].shape == (batch, 6)


def test_inference_outputs_have_expected_shapes_mixture():
    cat_idx = torch.tensor([], dtype=torch.long)
    cont_idx = torch.tensor([], dtype=torch.long)
    m = _make_model(
        condition_encoders=False,
        cat_covs_idx=cat_idx,
        cont_covs_idx=cont_idx,
        mix="mixture",
        modality_lengths=(3, 2),
        z_dim=4,
    )

    batch = 4
    x = torch.rand(batch, sum(m.input_dims))
    out = m.inference(x)

    assert out["z_joint"].shape == (batch, 4)
    assert out["z_marginal"].shape == (batch, m.n_modality, 4)


def test_generative_returns_list_of_reconstructions_per_modality():
    cat_idx = torch.tensor([], dtype=torch.long)
    cont_idx = torch.tensor([], dtype=torch.long)
    m = _make_model(
        condition_decoders=False,
        cat_covs_idx=cat_idx,
        cont_covs_idx=cont_idx,
        modality_lengths=(3, 2),
        losses=("mse", "mse"),
        z_dim=5,
    )

    batch = 6
    z = torch.randn(batch, 5)
    gen = m.generative(z)

    assert "rs" in gen
    rs = gen["rs"]
    assert isinstance(rs, list)
    assert len(rs) == m.n_modality

    assert rs[0].shape == (batch, 3)
    assert rs[1].shape == (batch, 2)


def test_inference_and_generative_with_conditioning_shapes():
    # Use one cat and one cont covariate; select both via idx
    cat_idx = torch.tensor([0], dtype=torch.long)
    cont_idx = torch.tensor([0], dtype=torch.long)

    m = _make_model(
        condition_encoders=True,
        condition_decoders=True,
        cat_covs_idx=cat_idx,
        cont_covs_idx=cont_idx,
        modality_lengths=(3, 2),
        losses=("mse", "mse"),
        z_dim=5,
        cat_covariate_dims=(4,),
        cont_covariate_dims=(1,),
    )

    batch = 5
    x = torch.rand(batch, sum(m.input_dims))

    # cat_covs: shape (batch, n_cat_total). values in [0, n_classes-1]
    cat_covs = torch.randint(low=0, high=4, size=(batch, 1))
    # cont_covs: shape (batch, n_cont_total)
    cont_covs = torch.rand(batch, 1)

    inf = m.inference(x, cat_covs=cat_covs, cont_covs=cont_covs)
    assert inf["z_joint"].shape == (batch, 5)

    gen = m.generative(inf["z_joint"], cat_covs=cat_covs, cont_covs=cont_covs)
    rs = gen["rs"]
    assert rs[0].shape == (batch, 3)
    assert rs[1].shape == (batch, 2)
