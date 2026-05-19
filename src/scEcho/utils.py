from __future__ import annotations

import warnings
from typing import Optional, Union

import anndata
import matplotlib.pyplot as plt
import numpy as np
import palantir
import pandas as pd
import scanpy as sc
import sklearn
from matplotlib.figure import Figure
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

__all__ = [
    "run_and_store_pr_res",
    "plot_branch_comp",
    "regress_embedding",
    "calc_corr",
    "df_to_adata_layer",
]



def run_and_store_pr_res(
    ad: anndata.AnnData,
    term_cells_series: pd.Series,
    start_celltype: str,
    DM_comp_multiscaled: str,
    num_waypoints: int = 500,
    modality: Optional[str] = None,
    **kwargs,
) -> None:
    pr_res = palantir.core.run_palantir(
        ad,
        term_cells_series.loc[term_cells_series == start_celltype].index[0],
        num_waypoints=num_waypoints,
        terminal_states=term_cells_series.loc[term_cells_series != start_celltype],
        eigvec_key=DM_comp_multiscaled,
        use_early_cell_as_start=True,
        **kwargs,
    )

    if modality is None:
        modality_suffix = ""
    else:
        modality_suffix = f"_{modality}"
        
        
    
    
    ad.obs[f"pseudotime{modality_suffix}"] = pr_res.pseudotime

    pr_res.branch_probs.columns = [
        f"{col}_prob{modality_suffix}" for col in pr_res.branch_probs.columns
    ]

    masks = palantir.presults.select_branch_cells(ad, q=.01, eps=.01)
    masks = pd.DataFrame(
        masks,
        columns=[f"{col}_mask{modality_suffix}" for col in pr_res.branch_probs.columns],
        index=ad.obs_names,
    )

    # drop existing columns before concat to avoid duplicates on re-run
    new_cols    = pr_res.branch_probs.columns.tolist() + masks.columns.tolist()
    existing    = [c for c in new_cols if c in ad.obs.columns]
    if len(existing) > 0:
        warnings.warn(f"Overwriting existing columns in ad.obs: {existing}")
        ad.obs.drop(columns=existing, inplace=True)

    # column-wise assignment preserves existing categorical dtypes; whole-row
    # `ad.obs = pd.concat(...)` has historically dropped them via the
    # AnnData setter
    for col in pr_res.branch_probs.columns:
        ad.obs[col] = pr_res.branch_probs[col]
    for col in masks.columns:
        ad.obs[col] = masks[col]
    
    
    
    
    
    
    
def plot_branch_comp(
    ad: anndata.AnnData,
    branch_name: str,
    pseudotime_col: Optional[str] = None,
    modality1: str = "RNA",
    modality2: str = "ATAC",
    vline_locs: Optional[Union[float, list]] = None,
    color_by: Optional[str] = None,
    layer: Optional[str] = None,
    sort_order: bool = False,
    figsize: Optional[tuple] = None,
) -> Figure:
    """Plot branch probabilities over pseudotime for two modalities.

    Creates a two-panel figure showing branch probabilities for modality2
    (top) and modality1 (bottom) as a function of pseudotime.

    Parameters
    ----------
    ad : anndata.AnnData
        The annotated data matrix.
    branch_name : str
        Name of the branch to plot. Used to look up f"{branch_name}_prob_{modality}"
        columns in ad.obs.
    pseudotime_col : str, optional
        Column in ad.obs containing pseudotime values. Defaults to
        f"pseudotime_{modality2}" if not specified.
    modality1 : str
        Name of the first modality (bottom panel). Default 'RNA'.
    modality2 : str
        Name of the second modality (top panel). Default 'ATAC'.
    vline_locs : float, int, or list of float/int, optional
        Pseudotime location(s) at which to draw vertical lines.
    color_by : str, optional
        Column in ad.obs or feature in ad.var_names to color points by.
        If in ad.var_names, layer must be specified.
    layer : str, optional
        Key in ad.layers to pull feature values from when color_by is in
        ad.var_names.
    sort_order : bool
        Whether to sort points by color value. Passed to sc.pl.scatter.
    figsize : tuple, optional
        Figure size (width, height).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    # ── Validate inputs ───────────────────────────────────────────────────────

    if pseudotime_col is None:
        pseudotime_col = f"pseudotime_{modality2}"
        warnings.warn(f"No pseudotime column set, defaulting to '{pseudotime_col}'.")

    if pseudotime_col not in ad.obs.columns:
        raise KeyError(
            f"Pseudotime column '{pseudotime_col}' not found in ad.obs. "
            f"Run Palantir first or specify a valid pseudotime_col."
        )

    for modality in [modality1, modality2]:
        branch_col = f"{branch_name}_prob_{modality}"
        if branch_col not in ad.obs.columns:
            raise KeyError(
                f"Branch probability column '{branch_col}' not found in ad.obs. "
                f"Run run_and_store_pr_res with modality='{modality}' first."
            )

    if color_by is not None:
        in_obs = color_by in ad.obs.columns
        in_var = color_by in ad.var_names

        if in_obs and in_var:
            raise ValueError(
                f"'{color_by}' is present in both ad.obs.columns and ad.var_names — "
                f"cannot determine how to color. Rename one to disambiguate."
            )
        elif not in_obs and not in_var:
            raise ValueError(
                f"'{color_by}' not found in ad.obs.columns or ad.var_names."
            )
        elif in_var:
            if layer is not None:
                if layer not in ad.layers:
                    raise KeyError(
                        f"Layer '{layer}' not found in ad.layers. "
                        f"Available layers: {list(ad.layers.keys())}"
                    )

    # ── Build figure ──────────────────────────────────────────────────────────

    fig, axs = plt.subplots(2, 1, layout="constrained", sharex=True, figsize=figsize)

    sc.pl.scatter(
        ad,
        x=pseudotime_col,
        y=f"{branch_name}_prob_{modality2}",
        ax=axs[0],
        show=False,
        color=color_by,
        sort_order=sort_order,
        title=f"{modality2}\n{branch_name} branch",
    )
    sc.pl.scatter(
        ad,
        x=pseudotime_col,
        y=f"{branch_name}_prob_{modality1}",
        ax=axs[1],
        show=False,
        color=color_by,
        sort_order=sort_order,
        title=f"{modality1}\n{branch_name} branch",
    )

    # ── Add vertical lines ────────────────────────────────────────────────────

    if vline_locs is not None:
        if isinstance(vline_locs, (int, float)):
            vline_locs = [vline_locs]
        for ax in axs:
            for loc in vline_locs:
                ax.axvline(loc, color="grey", linestyle="--")

    return fig
                    




    
                    

def regress_embedding(
    ad: anndata.AnnData,
    emb_key: str,
    depth_col: str,
) -> None:
    """
    Calculates the correlation between and embedding in ad.obsm and a read deth column (such as total_counts, nFrags) in obs.

    Parameters
    ----------
    ad : anndata.AnnData
        The annotated data matrix.
    emb_key : string
        The obsm key for the embedding
    depth_col : str
        The column of obsm storing read depth information

    Returns
    -------
    Modified ad so it has the new embedding ad.obsm[f"{emb_key}_{depth_col}_regressed"]
    """
    
    
    
    # adapted from Liza's code
    emb = np.array(ad.obsm[emb_key])
    to_reg = np.array(ad.obs[depth_col]).reshape(-1, 1)
    
    
    model = sklearn.linear_model.Ridge(alpha = 0.0) # learn reg model (alpha doesn't matter much here)
    model.fit(to_reg, emb) # fit LSI_comp to regression model as a function of depth
    emb_predict_by_depth = model.predict(to_reg) # predict values of LSI_comp based on depth
    emb_comp_residual = emb - emb_predict_by_depth # remove components of data that are described by depth
    
    ad.obsm[f"{emb_key}_{depth_col}_regressed"] = emb_comp_residual
    

    
    
    
def calc_corr(
    ad: anndata.AnnData,
    emb: str,
    depth_col: str,
) -> pd.DataFrame:
    """
    Calculates the correlation between and embedding in ad.obsm and a read deth column (such as total_counts, nFrags) in obs.

    Parameters
    ----------
    ad : anndata.AnnData
        The annotated data matrix.
    emb : string
        The obsm key for the embedding
    depth_col : str
        The column of obsm storing read depth information

    Returns
    -------
    A dataframe containing the columns dim (dimension of embedding) and corr (spearman correlation between) depth and ebedding coordinates
    """
    
    
    
    corr_dict = {"dim" : [],
                 "corr" : []}


    for i in range(ad.obsm[emb].shape[1]):

        sres = spearmanr(ad.obsm[emb][:, i],
                                    ad.obs[depth_col])
        corr_dict["dim"].append(i)
        corr_dict["corr"].append(sres.statistic)
        
    
    return pd.DataFrame(corr_dict)
    
    

    
def df_to_adata_layer(
    adata: anndata.AnnData,
    df: pd.DataFrame,
    layer_name: str,
    sparse: bool = False,
) -> None:
    """Add a dataframe as a new layer in an AnnData object.

    Aligns df to adata's obs/var indices, filling missing cells/features with NaN.

    Parameters
    ----------
    adata : anndata.AnnData
        Target AnnData object.
    df : pd.DataFrame
        DataFrame with cells as index and features as columns.
    layer_name : str
        Name of the layer to create in adata.layers.
    sparse : bool
        If True, store as a sparse matrix with NaN filled as 0.
    """
    aligned = df.reindex(index=adata.obs_names, columns=adata.var_names)

    if sparse:
        adata.layers[layer_name] = csr_matrix(aligned.fillna(0).values)
    else:
        adata.layers[layer_name] = aligned.values