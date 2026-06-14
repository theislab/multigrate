import pytest
from tests._helpers import add_new_categories_for_query

from multigrate.model._multivae import MultiVAE


@pytest.mark.parametrize(
    "cfg",
    [
        {
            "mix": "product",
            "condition_encoders": False,
            "condition_decoders": True,
            "modality_alignment": None,
            "alignment_type": "latent",  # ignored if no alignment
            "integrate_on": None,
        },
        {
            "mix": "product",
            "condition_encoders": True,
            "condition_decoders": True,
            "modality_alignment": None,
            "alignment_type": "latent",  # ignored if no alignment
            "integrate_on": None,
        },
        {
            "mix": "product",
            "condition_encoders": True,
            "condition_decoders": True,
            "modality_alignment": "MMD",
            "alignment_type": "both",
            "integrate_on": "batch",
        },
        {
            "mix": "product",
            "condition_encoders": True,
            "condition_decoders": True,
            "modality_alignment": "Jeffreys",
            "alignment_type": "both",
            "integrate_on": "batch",
        },
    ],
)
def test_multivae_ref_train_q2r_train_and_impute(
    cfg,
    rng,
    multivae_adata_2mod,
    setup_multivae_anndata,
    assert_imputed,
):
    # --------------------
    # Reference model
    # --------------------
    ref = setup_multivae_anndata(multivae_adata_2mod.copy())

    loss_coefs = None
    if cfg["integrate_on"] is not None:
        # use your preferred convention (full dict is safest)
        loss_coefs = {"recon": 1.0, "kl": 1.0, "integ": 1.0}

    ref_model = MultiVAE(
        ref,
        z_dim=4,
        cond_dim=4,
        mix=cfg["mix"],
        condition_encoders=cfg["condition_encoders"],
        condition_decoders=cfg["condition_decoders"],
        modality_alignment=cfg["modality_alignment"],
        alignment_type=cfg["alignment_type"],
        integrate_on=cfg["integrate_on"],
        loss_coefs=loss_coefs,
        kernel_type="gaussian",
        n_layers_encoders=[1, 1],
        n_layers_decoders=[1, 1],
        n_hidden_encoders=[8, 8],
        n_hidden_decoders=[8, 8],
        dropout=0.0,
        losses=["mse", "mse"],
    )

    ref_model.train(
        max_epochs=1,
        batch_size=8,
        train_size=1.0,
        early_stopping=False,
    )

    ref_model.impute(ref, batch_size=8)
    assert_imputed(ref)

    # --------------------
    # Query model
    # --------------------
    query = setup_multivae_anndata(multivae_adata_2mod.copy())
    add_new_categories_for_query(query, rng)

    q_model = MultiVAE.load_query_data(
        adata=query,
        reference_model=ref_model,
    )

    q_model.train(
        max_epochs=1,
        batch_size=8,
        train_size=1.0,
        early_stopping=False,
    )

    q_model.impute(query, batch_size=8)
    assert_imputed(query)
