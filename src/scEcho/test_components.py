import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy.sparse as sparse
from .Echo_features import embeddings_predict_layer



def sweep_diffusion_components(
    ad,
    layer,
    obsm_key,
    ls=None,
    ls_factor=1,
    sigma=0.1,
    gp_type=None,
    min_components=2,
):
    """Run embeddings_predict_layer across a sweep of diffusion component counts.

    For each n from min_components to the total number of components in obsm_key,
    slices the first n components and runs embeddings_predict_layer, saving results
    to uniquely named layers encoding the number of components used.

    Parameters
    ----------
    ad : anndata.AnnData
        The annotated data matrix.
    layer : str
        Key in ad.layers to impute.
    obsm_key : str
        Key in ad.obsm for the diffusion map to sweep over.
    ls : float, optional
        Length scale for the GP.
    ls_factor : float
        Length scale factor.
    sigma : float
        Prior on the standard deviation of features.
    gp_type : optional
        Gaussian process type passed to mellon.FunctionEstimator.
    min_components : int
        Number of components to start the sweep from (default 2).
    """

    # ── Validate inputs ───────────────────────────────────────────────────────

    assert obsm_key in ad.obsm, (
        f"'{obsm_key}' not found in ad.obsm. Available: {list(ad.obsm.keys())}"
    )
    assert layer in ad.layers, (
        f"Layer '{layer}' not found in ad.layers. Available: {list(ad.layers.keys())}"
    )

    n_components = ad.obsm[obsm_key].shape[1]

    assert min_components >= 2, "min_components must be at least 2."
    assert min_components <= n_components, (
        f"min_components ({min_components}) exceeds the number of available "
        f"components ({n_components}) in '{obsm_key}'."
    )

    # ── Temporarily store full embedding ──────────────────────────────────────

    full_embedding  = ad.obsm[obsm_key]
    tmp_key         = f"{obsm_key}_sweep_tmp"

    # ── Sweep over component counts ───────────────────────────────────────────

    for n in tqdm(range(min_components, n_components + 1), desc=f"Sweeping {obsm_key}"):

        # slice to first n components and store under a temporary key
        ad.obsm[tmp_key] = full_embedding[:, :n]

        embeddings_predict_layer(
            ad,
            ls=ls,
            ls_factor=ls_factor,
            sigma=sigma,
            embeddings=tmp_key,
            layer=layer,
            gp_type=gp_type,
        )

        # rename outputs to include component count
        for suffix in ["", "_residuals", "_uncertainty"]:
            src = f"predicted_{layer}_{tmp_key}_space{suffix}"
            dst = f"predicted_{layer}_{obsm_key}_{n}dims_space{suffix}"

            store = ad.obsp if suffix == "_uncertainty" else ad.layers
            if src in store:
                store[dst] = store.pop(src)

    # ── Clean up temporary key ────────────────────────────────────────────────

    del ad.obsm[tmp_key]
    
    
def collect_sweep_residual_means(
    ad,
    layer,
    obsm_key,
    min_components=2,
):
    """Collect mean residuals per gene across the diffusion component sweep.

    For each component count in the sweep output of sweep_diffusion_components,
    computes the mean per-gene residual across cells and returns as a DataFrame.

    Parameters
    ----------
    ad : anndata.AnnData
        The annotated data matrix.
    layer : str
        Layer name used in the sweep.
    obsm_key : str
        Embedding key used in the sweep.
    min_components : int
        Minimum number of components used in the sweep (default 2).

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (n_genes, n_component_counts) with mean residuals,
        indexed by gene name and columns labeled by number of components used.
    """

    n_components = ad.obsm[obsm_key].shape[1]
    res          = pd.DataFrame(index=ad.var_names)

    for n in tqdm(range(min_components, n_components + 1), desc="Collecting residual means"):
        residuals_key = f"predicted_{layer}_{obsm_key}_{n}dims_space_residuals"

        assert residuals_key in ad.layers, (
            f"'{residuals_key}' not found in ad.layers. "
            f"Run sweep_diffusion_components first."
        )

        vals = ad.layers[residuals_key]
        if sparse.issparse(vals):
            vals = vals.toarray()

        res[n] = vals.mean(axis=0)  # column is number of components used

    return res