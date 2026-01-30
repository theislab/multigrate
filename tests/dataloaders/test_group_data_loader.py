import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from multigrate.dataloaders._ann_dataloader import GroupAnnDataLoader, StratifiedSampler

# -------------------------
# StratifiedSampler tests
# -------------------------


def test_stratified_sampler_raises_on_empty_indices():
    with pytest.raises(ValueError, match=r"`indices` must be non-empty"):
        _ = StratifiedSampler(
            indices=np.array([], dtype=int),
            group_labels=np.array([], dtype=int),
            batch_size=4,
            min_size_per_class=2,
        )


def test_stratified_sampler_raises_on_group_labels_length_mismatch():
    with pytest.raises(ValueError, match=r"`group_labels` must have the same length as `indices`"):
        _ = StratifiedSampler(
            indices=np.arange(5),
            group_labels=np.array([0, 1, 0]),  # wrong length
            batch_size=4,
            min_size_per_class=2,
        )


def test_stratified_sampler_raises_on_nonpositive_min_size_per_class():
    with pytest.raises(ValueError, match=r"`min_size_per_class` must be positive"):
        _ = StratifiedSampler(
            indices=np.arange(5),
            group_labels=np.array([0, 0, 1, 1, 1]),
            batch_size=4,
            min_size_per_class=0,
        )


def test_stratified_sampler_raises_if_min_size_per_class_gt_batch_size():
    with pytest.raises(ValueError, match=r"`min_size_per_class` must be <= `batch_size`"):
        _ = StratifiedSampler(
            indices=np.arange(6),
            group_labels=np.array([0, 0, 0, 1, 1, 1]),
            batch_size=4,
            min_size_per_class=5,
        )


def test_stratified_sampler_raises_if_drop_last_gt_batch_size():
    indices = np.arange(10)
    labels = np.array([0] * 5 + [1] * 5)

    with pytest.raises(ValueError, match=r"drop_last can't be greater than batch_size"):
        _ = StratifiedSampler(
            indices=indices,
            group_labels=labels,
            batch_size=8,
            min_size_per_class=2,
            drop_last=9,
        )


def test_stratified_sampler_raises_if_batch_size_not_divisible_by_min_size():
    indices = np.arange(10)
    labels = np.array([0] * 5 + [1] * 5)

    with pytest.raises(ValueError, match=r"min_size_per_class has to be a divisor of batch_size"):
        _ = StratifiedSampler(
            indices=indices,
            group_labels=labels,
            batch_size=10,
            min_size_per_class=6,
            drop_last=True,
        )


def test_stratified_sampler_raises_if_drop_last_int_negative():
    indices = np.arange(6)
    labels = np.array([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match=r"must be >= 0"):
        _ = StratifiedSampler(
            indices=indices,
            group_labels=labels,
            batch_size=4,
            min_size_per_class=2,
            drop_last=-1,
        )


def test_stratified_sampler_batches_balanced_no_shuffle():
    # 2 classes, equal counts; min_size=2; batch=4; no shuffle -> perfect class separation in batches
    indices = np.arange(8)
    labels = np.array([0] * 4 + [1] * 4)

    sampler = StratifiedSampler(
        indices=indices,
        group_labels=labels,
        batch_size=4,
        min_size_per_class=2,
        shuffle=False,
        shuffle_classes=False,
        drop_last=False,
    )

    batches = list(iter(sampler))
    assert len(batches) == 2
    assert all(len(b) == 4 for b in batches)

    b0, b1 = batches
    b0 = np.array(b0)
    b1 = np.array(b1)
    cls0 = labels[b0]
    cls1 = labels[b1]
    assert (cls0 == 0).sum() == 4
    assert (cls1 == 1).sum() == 4


def test_stratified_sampler_drop_last_true_drops_remainder_within_class():
    # class0 has 3 samples, class1 has 4; min_size=2
    # remainder 1 sample in class0 should be dropped when drop_last True
    indices = np.arange(7)
    labels = np.array([0] * 3 + [1] * 4)

    sampler = StratifiedSampler(
        indices=indices,
        group_labels=labels,
        batch_size=4,
        min_size_per_class=2,
        shuffle=False,
        shuffle_classes=False,
        drop_last=True,
    )

    flat = np.array([i for batch in sampler for i in batch])
    # In non-shuffle mode, class0 indices are [0,1,2]; with min_size=2 and drop_last True,
    # the leftover element would be index 2 -> should not appear.
    assert 2 not in flat


# -------------------------
# GroupAnnDataLoader tests
# -------------------------


class DummyAnnTorchDataset:
    """Stand-in for scvi.dataloaders.AnnTorchDataset."""

    def __init__(self, adata_manager, getitem_tensors=None):
        self.adata_manager = adata_manager
        self.getitem_tensors = getitem_tensors

    def __len__(self):
        return self.adata_manager.adata.n_obs

    def __getitem__(self, idx):
        return {"idx": int(idx)}


class CaptureSampler(torch.utils.data.Sampler):
    """Sampler that captures kwargs; yields a single batch (batched indices)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._batch = [kwargs["indices"].tolist()]

    def __iter__(self):
        return iter(self._batch)

    def __len__(self):
        return 1


class DummyAnnDataManager:
    """Minimal stub of AnnDataManager for unit testing GroupAnnDataLoader."""

    def __init__(self, adata, categorical_covariate_keys, data_registry_keys=None):
        self.adata = adata
        self.data_registry = dict.fromkeys(data_registry_keys or [])
        self.registry = {"setup_args": {"categorical_covariate_keys": categorical_covariate_keys}}


def _patch_anntorchdataset(monkeypatch):
    # Patch AnnTorchDataset imported in the module under test, so we avoid scvi internals.
    import multigrate.dataloaders._ann_dataloader as mod

    monkeypatch.setattr(mod, "AnnTorchDataset", DummyAnnTorchDataset)


def test_group_dataloader_raises_if_missing_scvi_extra_covs(monkeypatch):
    _patch_anntorchdataset(monkeypatch)

    adata = ad.AnnData(X=np.ones((5, 3), dtype=np.float32))
    # intentionally no adata.obsm["_scvi_extra_categorical_covs"]
    mgr = DummyAnnDataManager(adata=adata, categorical_covariate_keys=["group"])

    with pytest.raises(ValueError, match=r"missing.*_scvi_extra_categorical_covs"):
        _ = GroupAnnDataLoader(mgr, group_column="group", batch_size=128, sampler=CaptureSampler)


def test_group_dataloader_raises_if_data_and_attributes_key_not_registered(monkeypatch):
    _patch_anntorchdataset(monkeypatch)

    adata = ad.AnnData(X=np.ones((5, 3), dtype=np.float32))
    adata.obsm["_scvi_extra_categorical_covs"] = pd.DataFrame({"group": ["a"] * 5}, index=adata.obs_names)
    mgr = DummyAnnDataManager(adata=adata, categorical_covariate_keys=["group"], data_registry_keys=["X"])

    with pytest.raises(ValueError, match=r"required for model but not registered"):
        _ = GroupAnnDataLoader(
            mgr,
            group_column="group",
            batch_size=128,
            sampler=CaptureSampler,
            data_and_attributes={"NOT_REGISTERED": np.float32},
        )


def test_group_dataloader_raises_if_group_column_not_registered_as_categorical_covariate(monkeypatch):
    _patch_anntorchdataset(monkeypatch)

    adata = ad.AnnData(X=np.ones((5, 3), dtype=np.float32))
    adata.obsm["_scvi_extra_categorical_covs"] = pd.DataFrame({"group": ["a"] * 5}, index=adata.obs_names)
    mgr = DummyAnnDataManager(adata=adata, categorical_covariate_keys=["other"])

    with pytest.raises(ValueError, match=r"has to be one of the registered categorical covariates"):
        _ = GroupAnnDataLoader(mgr, group_column="group", batch_size=128, sampler=CaptureSampler)


def test_group_dataloader_boolean_indices_are_converted_and_group_labels_subset(monkeypatch):
    _patch_anntorchdataset(monkeypatch)

    adata = ad.AnnData(X=np.ones((5, 3), dtype=np.float32))
    adata.obsm["_scvi_extra_categorical_covs"] = pd.DataFrame(
        {"group": ["a", "a", "b", "b", "a"]},
        index=adata.obs_names,
    )
    mgr = DummyAnnDataManager(adata=adata, categorical_covariate_keys=["group"])

    loader = GroupAnnDataLoader(
        mgr,
        group_column="group",
        indices=np.array([True, False, True, False, True]),  # selects 0,2,4
        batch_size=4,
        min_size_per_class=2,
        sampler=CaptureSampler,  # avoid StratifiedSampler complexity in this unit test
        shuffle=False,
        shuffle_classes=False,
        drop_last=False,
    )

    # Check indices conversion
    assert loader.indices.tolist() == [0, 2, 4]
    assert loader.sampler_kwargs["indices"].tolist() == [0, 2, 4]

    # Check group_labels subset order corresponds to selected indices
    assert loader.sampler_kwargs["group_labels"].tolist() == ["a", "b", "a"]

    # DataLoader configured for batched sampler
    assert loader.data_loader_kwargs["batch_size"] is None
    assert "sampler" in loader.data_loader_kwargs
