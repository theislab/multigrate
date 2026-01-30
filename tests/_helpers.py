import numpy as np
import anndata as ad


def add_new_categories_for_query(adata: ad.AnnData, rng: np.random.Generator) -> ad.AnnData:
    """
    Mutates adata.obs to include new categories in categorical covariates.
    """
    adata.obs["donor"] = rng.choice(["d0", "d1", "d2", "d3_new"], size=adata.n_obs)
    adata.obs.loc[adata.obs['donor'].isin(['d0', 'd1']), 'batch'] = 'b0'
    adata.obs.loc[adata.obs['donor'].isin(['d2']), 'batch'] = 'b1'
    adata.obs.loc[adata.obs['donor'].isin(['d3_new']), 'batch'] = 'b2_new'
    return adata
