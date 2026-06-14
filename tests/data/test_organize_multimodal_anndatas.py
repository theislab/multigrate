import anndata as ad
import numpy as np
import pytest

from multigrate.data._preprocessing import organize_multimodal_anndatas


def make_adata(n_obs: int, var_names, cell_prefix, *, layer_name: str | None = None, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_obs, len(var_names))).astype(np.float32)
    a = ad.AnnData(X=X)
    a.obs_names = [f"{cell_prefix}{i}" for i in range(n_obs)]
    a.var_names = list(var_names)
    a.obs["foo"] = np.arange(n_obs)
    if layer_name is not None:
        a.layers[layer_name] = (X + 10).copy()
    return a


def test_basic_concat_with_missing_modality_filled_with_zeros():
    # 2 modalities, 2 groups. One missing entry replaced with zeros.
    m0_g0 = make_adata(3, ["g1", "g2"], "cell-g0", seed=1)
    m0_g1 = make_adata(3, ["g1", "g2"], "cell-g1", seed=2)

    m1_g0 = make_adata(3, ["p1", "p2", "p3"], "cell-g0", seed=3)
    m1_g1 = None  # missing modality in group1 -> should be filled with zeros

    out = organize_multimodal_anndatas(
        adatas=[
            [m0_g0, m0_g1],  # modality 0
            [m1_g0, m1_g1],  # modality 1
        ],
        layers=None,
    )

    # Expect 6 cells total (3+3), features = 2 + 3
    assert out.shape == (6, 5)

    assert "group" in out.obs.columns
    assert set(out.obs["group"].unique()) == {0, 1}

    assert "modality_lengths" in out.uns
    assert out.uns["modality_lengths"] == [2, 3]

    # The missing modality block (modality 1, group 1) should be zeros in X for those rows & those cols.
    idx_group1 = np.where(out.obs["group"].to_numpy() == 1)[0]
    # modality 1 is last 3 features after concat
    block = out.X[idx_group1, 2:5]
    assert np.allclose(block, 0.0)


def test_warns_if_group_column_exists():
    m0_g0 = make_adata(2, ["g1", "g2"], "cell-g0")
    m0_g0.obs["group"] = 999  # triggers warning
    m0_g1 = make_adata(2, ["g1", "g2"], "cell-g1", seed=1)

    m1_g0 = make_adata(2, ["p1"], "cell-g0", seed=2)
    m1_g1 = make_adata(2, ["p1"], "cell-g1", seed=3)

    with pytest.warns(UserWarning, match=r"Column `\.obs\['group'\]` will be overwritten"):
        _ = organize_multimodal_anndatas([[m0_g0, m0_g1], [m1_g0, m1_g1]])


def test_raises_on_non_rectangular_adatas():
    # modality 0 has 2 groups, modality 1 has 1 group -> should raise
    m0_g0 = make_adata(2, ["g1"], "cell-g0")
    m0_g1 = make_adata(2, ["g1"], "cell-g1", seed=1)
    m1_g0 = make_adata(2, ["p1"], "cell-g0", seed=2)

    with pytest.raises(ValueError, match=r"same length"):
        _ = organize_multimodal_anndatas([[m0_g0, m0_g1], [m1_g0]])


def test_raises_on_layers_shape_mismatch_modalities():
    m0_g0 = make_adata(2, ["g1"], "cell-g0", layer_name="counts")
    m0_g1 = make_adata(2, ["g1"], "cell-g1", layer_name="counts", seed=1)
    m1_g0 = make_adata(2, ["p1"], "cell-g0", layer_name="counts", seed=2)
    m1_g1 = make_adata(2, ["p1"], "cell-g1", layer_name="counts", seed=3)

    with pytest.raises(ValueError, match=r"`layers` must have the same number of modalities"):
        _ = organize_multimodal_anndatas(
            [[m0_g0, m0_g1], [m1_g0, m1_g1]],
            layers=[["counts", "counts"]],  # only 1 modality layer list
        )


def test_raises_on_layers_shape_mismatch_groups():
    m0_g0 = make_adata(2, ["g1"], "cell-g0", layer_name="counts")
    m0_g1 = make_adata(2, ["g1"], "cell-g1", layer_name="counts", seed=1)
    m1_g0 = make_adata(2, ["p1"], "cell-g0", layer_name="counts", seed=2)
    m1_g1 = make_adata(2, ["p1"], "cell-g1", layer_name="counts", seed=3)

    with pytest.raises(ValueError, match=r"must match the number of groups"):
        _ = organize_multimodal_anndatas(
            [[m0_g0, m0_g1], [m1_g0, m1_g1]],
            layers=[["counts"], ["counts", "counts"]],  # modality 0 layers length wrong
        )


def test_raises_on_wrong_element_type():
    m0_g0 = make_adata(2, ["g1"], "cell-g0")
    m0_g1 = "not anndata"  # wrong type
    m1_g0 = make_adata(2, ["p1"], "cell-g0", seed=1)
    m1_g1 = make_adata(2, ["p1"], "cell-g1", seed=2)

    with pytest.raises(TypeError, match=r"must be an AnnData or None"):
        _ = organize_multimodal_anndatas([[m0_g0, m0_g1], [m1_g0, m1_g1]])


def test_raises_on_var_names_mismatch_within_modality():
    m0_g0 = make_adata(2, ["g1", "g2"], "cell-g0")
    m0_g1 = make_adata(2, ["g1", "DIFFERENT"], "cell-g1")  # var_names mismatch

    m1_g0 = make_adata(2, ["p1"], "cell-g0")
    m1_g1 = make_adata(2, ["p1"], "cell-g1", seed=1)

    with pytest.raises(ValueError, match=r"different `\.var_names` within modality 0"):
        _ = organize_multimodal_anndatas([[m0_g0, m0_g1], [m1_g0, m1_g1]])


def test_raises_on_different_obs_counts_for_paired_groups():
    m0_g0 = make_adata(2, ["g1", "g2"], "cell-g0")
    m0_g1 = make_adata(2, ["g1", "g2"], "cell-g1", seed=1)

    m1_g0 = make_adata(3, ["p1"], "cell-g0")  # mismatch obs count for group0
    m1_g1 = make_adata(2, ["p1"], "cell-g1", seed=2)

    with pytest.raises(ValueError, match=r"different number of observations for group 0"):
        _ = organize_multimodal_anndatas([[m0_g0, m0_g1], [m1_g0, m1_g1]])


def test_raises_on_different_obs_names_for_paired_groups():
    m0_g0 = make_adata(2, ["g1", "g2"], "cell-g0")
    m0_g1 = make_adata(2, ["g1", "g2"], "cell-g1", seed=1)

    m1_g0 = make_adata(2, ["p1"], "cell-g0")
    m1_g0.obs_names = ["cellX", "cellY"]  # same length, different names -> should raise
    m1_g1 = make_adata(2, ["p1"], "cell-g1", seed=2)

    with pytest.raises(ValueError, match=r"`\.obs_names` are not the same for group 0"):
        _ = organize_multimodal_anndatas([[m0_g0, m0_g1], [m1_g0, m1_g1]])


def test_raises_if_a_group_has_all_none():
    # group 1 has None in all modalities -> should raise group_has_any check
    m0_g0 = make_adata(2, ["g1"], "cell-g0")
    m0_g1 = None
    m1_g0 = make_adata(2, ["p1"], "cell-g0", seed=1)
    m1_g1 = None

    with pytest.raises(ValueError, match=r"must have at least one non-None"):
        _ = organize_multimodal_anndatas([[m0_g0, m0_g1], [m1_g0, m1_g1]])


def test_raises_if_a_modality_is_all_none():
    # all groups have data somewhere (so group_has_any passes),
    # but modality 1 is entirely None -> modality_lengths[1] stays -1 and should raise.
    m0_g0 = make_adata(2, ["g1"], "cell-g0")
    m0_g1 = make_adata(2, ["g1"], "cell-g1", seed=1)

    m1_g0 = None
    m1_g1 = None

    with pytest.raises(ValueError, match=r"Modality 1 length could not be determined"):
        _ = organize_multimodal_anndatas([[m0_g0, m0_g1], [m1_g0, m1_g1]])


def test_layers_argument_uses_specified_layer_values():
    m0_g0 = make_adata(2, ["g1", "g2"], "cell-g0", layer_name="counts", seed=0)
    m0_g1 = make_adata(2, ["g1", "g2"], "cell-g1", layer_name="counts", seed=1)

    m1_g0 = make_adata(2, ["p1"], "cell-g0", layer_name="counts", seed=2)
    m1_g1 = make_adata(2, ["p1"], "cell-g1", layer_name="counts", seed=3)

    out = organize_multimodal_anndatas(
        [[m0_g0, m0_g1], [m1_g0, m1_g1]],
        layers=[["counts", "counts"], ["counts", "counts"]],
    )

    # We set counts layer = X + 10 in make_adata, so the mean should be noticeably higher than ~0
    assert float(np.mean(out.X)) > 5.0


def test_layers_missing_layer_raises_keyerror():
    m0_g0 = make_adata(2, ["g1", "g2"], "cell-g0", layer_name="counts", seed=0)
    m0_g1 = make_adata(2, ["g1", "g2"], "cell-g1", layer_name="counts", seed=1)

    m1_g0 = make_adata(2, ["p1"], "cell-g0", layer_name="counts", seed=2)
    m1_g1 = make_adata(2, ["p1"], "cell-g1", layer_name="counts", seed=3)

    with pytest.raises(KeyError, match=r"Layer 'NOT_A_LAYER' not found"):
        _ = organize_multimodal_anndatas(
            [[m0_g0, m0_g1], [m1_g0, m1_g1]],
            layers=[["NOT_A_LAYER", "counts"], ["counts", "counts"]],
        )


def test_warns_and_makes_unique_if_var_names_duplicate_across_modalities():
    # Make duplicate feature names across modalities (e.g. both have "dup")
    m0_g0 = make_adata(2, ["dup", "g2"], "cell-g0", seed=0)
    m0_g1 = make_adata(2, ["dup", "g2"], "cell-g1", seed=1)

    m1_g0 = make_adata(2, ["dup", "p2"], "cell-g0", seed=2)
    m1_g1 = make_adata(2, ["dup", "p2"], "cell-g1", seed=3)

    with pytest.warns(UserWarning, match=r"var_names are not unique; making them unique."):
        out = organize_multimodal_anndatas([[m0_g0, m0_g1], [m1_g0, m1_g1]])

    assert out.var_names.is_unique


def test_raises_if_obs_names_duplicate_across_groups():
    # Make duplicate obs names across groups (e.g. both have "cell0", "cell1")
    m0_g0 = make_adata(2, ["g1", "g2"], "cell-g0", seed=0)
    m0_g1 = make_adata(2, ["g1", "g2"], "cell-g1", seed=1)
    m0_g1.obs_names = m0_g0.obs_names.copy()  # duplicate obs names

    m1_g0 = make_adata(2, ["p1"], "cell-g0", seed=2)
    m1_g1 = make_adata(2, ["p1"], "cell-g1", seed=3)
    m1_g1.obs_names = m1_g0.obs_names.copy()  # duplicate obs names

    with pytest.raises(ValueError, match=r"`\.obs_names` are not unique across datasets;"):
        _ = organize_multimodal_anndatas([[m0_g0, m0_g1], [m1_g0, m1_g1]])
