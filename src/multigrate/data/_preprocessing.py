import warnings

import anndata as ad
import numpy as np
import pandas as pd


def organize_multimodal_anndatas(
    adatas: list[list[ad.AnnData | None]],
    layers: list[list[str | None]] | None = None,
) -> ad.AnnData:
    """Concatenate all the input anndata objects.

    These anndata objects should already have been preprocessed so that all single-modality
    objects use a subset of the features used in the multimodal object. The feature names (index of
    `.var`) should match between the objects for vertical integration and cell names (index of
    `.obs`) should match between the objects for horizontal integration.

    Parameters
    ----------
    adatas
        List of Lists with AnnData objects or None where each sublist corresponds to a modality.
    layers
        List of Lists of the same lengths as `adatas` specifying which `.layer` to use for each AnnData. Default is None which means using `.X`.

    Returns
    -------
    Concatenated AnnData object across modalities and datasets.
    """
    if not isinstance(adatas, list) or len(adatas) == 0:
        raise ValueError("`adatas` must be a non-empty list of modality lists.")

    n_groups = None
    for m, modality_list in enumerate(adatas):
        if not isinstance(modality_list, list) or len(modality_list) == 0:
            raise ValueError(f"`adatas[{m}]` must be a non-empty list (datasets/groups).")
        if n_groups is None:
            n_groups = len(modality_list)
        elif len(modality_list) != n_groups:
            raise ValueError(
                "All modality lists in `adatas` must have the same length (same number of datasets/groups)."
            )

    if layers is not None:
        if len(layers) != len(adatas):
            raise ValueError("`layers` must have the same number of modalities as `adatas`.")
        for mod in range(len(adatas)):
            if len(layers[mod]) != len(adatas[mod]):
                raise ValueError("Each `layers[modality]` must match the number of groups in `adatas[modality]`.")

    # needed for scArches operation setup
    datasets_lengths = {}
    datasets_obs_names = {}
    datasets_obs = {}
    modality_lengths = [-1] * len(adatas)
    modality_var_names = {}
    group_has_any = [False] * n_groups

    # sanity checks and preparing data for concat
    for mod, modality_adatas in enumerate(adatas):
        for i, adata in enumerate(modality_adatas):
            if adata is not None and not isinstance(adata, ad.AnnData):
                raise TypeError(f"`adatas[{mod}][{i}]` must be an AnnData or None, got {type(adata)}.")

            if adata is not None:
                group_has_any[i] = True
                # will create .obs['group'] later, so throw a warning here if the column already exists
                if "group" in adata.obs.columns:
                    warnings.warn(
                        "Column `.obs['group']` will be overwritten. Please save the original data in another column if needed.",
                        stacklevel=2,
                    )
                # check that all adatas in the same modality have the same features
                if (mod_var := modality_var_names.get(mod, None)) is None:
                    modality_var_names[mod] = adata.var_names
                    modality_lengths[mod] = adata.shape[1]
                else:
                    if not adata.var_names.equals(mod_var):
                        raise ValueError(f"Adatas have different `.var_names` within modality {mod}.")
                # check that there is the same number of observations for paired data
                if (dataset_length := datasets_lengths.get(i, None)) is None:
                    datasets_lengths[i] = adata.shape[0]
                else:
                    if adata.shape[0] != dataset_length:
                        raise ValueError(
                            f"Paired adatas have different number of observations for group {i}, namely {dataset_length} and {adata.shape[0]}."
                        )
                # check that .obs_names are the same for paired data
                if (dataset_obs_names := datasets_obs_names.get(i, None)) is None:
                    datasets_obs_names[i] = adata.obs_names
                else:
                    if np.sum(adata.obs_names != dataset_obs_names):
                        raise ValueError(f"`.obs_names` are not the same for group {i}.")
                # keep all the .obs
                if datasets_obs.get(i, None) is None:
                    datasets_obs[i] = adata.obs.copy()
                    datasets_obs[i]["group"] = i
                else:
                    cols_to_use = adata.obs.columns.difference(datasets_obs[i].columns)
                    datasets_obs[i] = datasets_obs[i].join(adata.obs[cols_to_use])

    if not all(group_has_any):
        bad = [i for i, ok in enumerate(group_has_any) if not ok]
        raise ValueError(
            f"Each dataset/group must have at least one non-None AnnData across modalities. Missing for groups: {bad}"
        )

    # check that modality lengths are not -1
    for m, length in enumerate(modality_lengths):
        if length == -1:
            raise ValueError(f"Modality {m} length could not be determined.")

    for mod, modality_adatas in enumerate(adatas):
        for i, adata in enumerate(modality_adatas):
            if adata is None:
                X_zeros = np.zeros((datasets_lengths[i], modality_lengths[mod]))
                adatas[mod][i] = ad.AnnData(X_zeros, dtype=X_zeros.dtype)
                adatas[mod][i].obs_names = datasets_obs_names[i]
                adatas[mod][i].var_names = modality_var_names[mod]
                adatas[mod][i] = adatas[mod][i].copy()
            if layers is not None:
                if layers[mod][i] is not None:
                    layer = layers[mod][i]
                    if layer not in adatas[mod][i].layers:
                        raise KeyError(f"Layer '{layer}' not found in adatas[{mod}][{i}].layers")
                    adatas[mod][i] = adatas[mod][i].copy()
                    adatas[mod][i].X = adatas[mod][i].layers[layer].copy()

    # concat adatas within each modality first
    mod_adatas = []
    for modality_adatas in adatas:
        mod_adatas.append(mod_adata := ad.concat(modality_adatas, join="outer"))
        if not mod_adata.obs_names.is_unique:
            raise ValueError(
                "`.obs_names` are not unique across datasets; please make them unique before proceeding by e.g. adding a prefix."
            )

    # concat modality adatas along the feature axis
    multimodal_anndata = ad.concat(mod_adatas, axis=1, label="modality")

    # add .obs back
    multimodal_anndata.obs = pd.concat(datasets_obs.values())

    # we will need modality_length later for the model init
    multimodal_anndata.uns["modality_lengths"] = modality_lengths

    # check if var_names are unique and make them so if not with a warning
    if not multimodal_anndata.var_names.is_unique:
        warnings.warn("Concatenated var_names are not unique; making them unique.", stacklevel=2)
        multimodal_anndata.var_names_make_unique()

    return multimodal_anndata
