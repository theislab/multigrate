import anndata

# taken from scIB
# on 11 November 2021
# https://github.com/theislab/scib/blob/985d8155391fdfbddec024de428308b5a57ee280/scib/metrics/utils.py

# checker functions for data sanity
def check_adata(adata):
    if type(adata) is not anndata.AnnData:
        raise TypeError("Input is not a valid AnnData object")


def check_batch(batch, obs, verbose=False):
    if batch not in obs:
        raise ValueError(f"column {batch} is not in obs")
    elif verbose:
        print(f"Object contains {obs[batch].nunique()} batches.")
