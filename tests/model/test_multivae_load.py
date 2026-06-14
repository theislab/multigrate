import shutil
from pathlib import Path

from multigrate.model._multivae import MultiVAE


def test_multivae_save_and_load_roundtrip(
    tmp_path: Path,
    setup_multivae_anndata,
    multivae_adata_2mod,
    assert_imputed,
):
    # Fresh registered adata
    adata = setup_multivae_anndata(multivae_adata_2mod.copy())

    model = MultiVAE(
        adata,
        z_dim=4,
        cond_dim=4,
        n_layers_encoders=[1, 1],
        n_layers_decoders=[1, 1],
        n_hidden_encoders=[8, 8],
        n_hidden_decoders=[8, 8],
        dropout=0.0,
        losses=["mse", "mse"],
    )

    model.train(
        max_epochs=1,
        batch_size=8,
        train_size=1.0,
        early_stopping=False,
    )

    save_dir = tmp_path / "multivae_saved"

    model.save(save_dir, overwrite=True)

    assert save_dir.exists()
    assert any(save_dir.iterdir()), "save directory is empty; save() likely failed"

    loaded = MultiVAE.load(save_dir, adata=adata)

    assert isinstance(loaded, MultiVAE)
    assert loaded.module is not None
    assert loaded.module.z_dim == model.module.z_dim
    assert loaded.is_trained is True

    # prove the loaded model can do something useful
    loaded.impute(adata, batch_size=8)
    assert_imputed(adata)

    # explicit cleanup (even though tmp_path is ephemeral)
    shutil.rmtree(save_dir, ignore_errors=False)
    assert not save_dir.exists()
