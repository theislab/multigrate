import numpy as np
import pandas as pd

import multigrate
from multigrate.data import organize_multimodal_anndatas
from multigrate.model import MultiVAE


def test_package_has_version():
    assert multigrate.__version__ is not None


def _make_adata(n_obs, n_vars, prefix="cell", var_prefix="gene"):
    import anndata as ad

    X = np.random.rand(n_obs, n_vars).astype(np.float32)
    obs = pd.DataFrame(index=[f"{prefix}{i}" for i in range(n_obs)])
    var = pd.DataFrame(
        {"mean": np.random.rand(n_vars), "highly_variable": True},
        index=[f"{var_prefix}{i}" for i in range(n_vars)],
    )
    return ad.AnnData(X, obs=obs, var=var)


def test_var_preservation():
    """organize_multimodal_anndatas must preserve .var columns from each modality (issue #4)."""
    rna = _make_adata(100, 50, prefix="cell", var_prefix="gene")
    prot = _make_adata(100, 10, prefix="cell", var_prefix="prot")
    mdata = organize_multimodal_anndatas([[rna], [prot]])
    # 'mean' appears in both modalities → suffixed
    assert "mean_0" in mdata.var.columns
    assert "mean_1" in mdata.var.columns
    assert mdata.var.shape[0] == 60


def test_end_to_end_default_losses():
    """Full pipeline with losses=None (documented default) must not crash (bugs 1 & 2)."""
    rna = _make_adata(120, 50, prefix="cell", var_prefix="gene")
    prot = _make_adata(120, 10, prefix="cell", var_prefix="prot")
    mdata = organize_multimodal_anndatas([[rna], [prot]])

    MultiVAE.setup_anndata(mdata, rna_indices_end=50)
    model = MultiVAE(mdata)  # losses=None → must not crash (bug 1)
    model.train(max_epochs=2, accelerator="cpu", batch_size=32)
    model.get_model_output()
    assert "X_multigrate" in mdata.obsm

    # impute must work (stores results in adata.obsm, returns None)
    model.impute()
    assert "imputed_modality_0" in mdata.obsm

    # load_query_data with a reference trained with integrate_on → theta guard (bug 2)
    rna_a = _make_adata(60, 50, prefix="a", var_prefix="gene")
    prot_a = _make_adata(60, 10, prefix="a", var_prefix="prot")
    rna_b = _make_adata(60, 50, prefix="b", var_prefix="gene")
    prot_b = _make_adata(60, 10, prefix="b", var_prefix="prot")
    mref = organize_multimodal_anndatas([[rna_a, rna_b], [prot_a, prot_b]])
    MultiVAE.setup_anndata(mref, rna_indices_end=50, categorical_covariate_keys=["group"])
    model_io = MultiVAE(mref, integrate_on="group")
    model_io.train(max_epochs=2, accelerator="cpu", batch_size=32)
    # With MSE losses only, theta ParameterList exists but contains only empty parameters
    assert all(p.numel() == 0 for p in model_io.module.theta)  # mse losses → empty theta

    rna_q = _make_adata(30, 50, prefix="q", var_prefix="gene")
    prot_q = _make_adata(30, 10, prefix="q", var_prefix="prot")
    mquery = organize_multimodal_anndatas([[rna_q], [prot_q]])
    MultiVAE.setup_anndata(mquery, rna_indices_end=50, categorical_covariate_keys=["group"])
    # freeze=False: not a real scArches use case, but tests that the theta guard doesn't crash
    qmodel = MultiVAE.load_query_data(mquery, model_io, freeze=False)  # must not crash (bug 2)
    qmodel.train(max_epochs=2, accelerator="cpu", batch_size=16)


def test_integrate_on_small_data():
    """StratifiedSampler must not silently produce 0 batches on small data (bug 3)."""
    rna = _make_adata(60, 50, prefix="x", var_prefix="gene")
    prot = _make_adata(60, 10, prefix="x", var_prefix="prot")
    mdata = organize_multimodal_anndatas([[rna], [prot]])
    MultiVAE.setup_anndata(mdata, rna_indices_end=50, categorical_covariate_keys=["group"])
    model = MultiVAE(mdata, integrate_on="group")
    model.train(max_epochs=2, accelerator="cpu", batch_size=32)
    # if bug 3 were present, training would silently no-op with 0 batches;
    # check that history contains at least one loss entry
    assert len(model.history["train_loss"]) > 0
