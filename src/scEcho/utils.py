from __future__ import annotations

import logging
import warnings
from typing import Optional, Sequence, Union

import anndata
import matplotlib.pyplot as plt
import mellon
import numpy as np
import palantir
import pandas as pd
import scanpy as sc
import seaborn as sns
import sklearn
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from .echo_features import embeddings_predict_layer

__all__ = [
    # existing utils
    "run_and_store_pr_res",
    "regress_embedding",
    "calc_corr",
    "df_to_adata_layer",
    "plot_branch_comp",
    # from try_models
    "try_models",
    "read_test_results",
    "plot_model_heatmap",
    # from test_components
    "sweep_diffusion_components",
    "collect_sweep_residual_means",
]

logger = logging.getLogger(__name__)



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



def try_models(
    ad: anndata.AnnData,
    ls_vals: Sequence[float],
    sigmas: Sequence[float],
    layer: str,
    embeddings: Sequence[str] = ("DM_EigenVectors_RNA", "DM_EigenVectors_ATAC"),
    name_dict: Optional[dict] = None,
    loo_residuals: bool = True,
    groups: Optional[str] = None,
    test_frac: float = 0.2,
    random_state: int = 0,
) -> None:
    """Test combinations of length scale and sigma hyperparameters for GP imputation.

    Fits a mellon FunctionEstimator for each combination of ls, sigma, and embedding,
    using a train/test split stored in ad.obs['tr_tst']. If groups is specified, the
    split is stratified to ensure equal representation across groups. If ad.obs['tr_tst']
    already exists it will be used as-is and the split will not be recomputed.

    Parameters
    ----------
    ad : anndata.AnnData
        The annotated data matrix.
    ls_vals : array-like
        Length scale values to try.
    sigmas : array-like
        Sigma values to try.
    layer : str
        Key in ad.layers to impute.
    embeddings : tuple of str
        Keys in ad.obsm for the embeddings to impute over.
    name_dict : dict, optional
        Mapping from embedding key to display name used in output layer/column names.
        Defaults to identity mapping if None.
    loo_residuals : bool
        If True, compute LOO residuals via fest.predict.loo_residuals_squared.
        If False, compute naive squared residuals (y - y_pred)^2.
    groups : str, optional
        Column in ad.obs to stratify the train/test split over. If None, split is
        random across all cells. Ignored if ad.obs['tr_tst'] already exists.
    test_frac : float
        Fraction of cells to hold out as test set. Default 0.2.
    random_state : int
        Random seed for reproducibility of the train/test split.

    Returns
    -------
    Updates ad in place with the following per (ls, sigma, embedding) combination:
        ad.obs['tr_tst']                                                        : Train/test split labels.
        ad.layers[f"predicted_{layer}_{emb_name}_ls{ls}_sigma{sigma}"]         : Predictions for all cells.
        ad.var[f"predicted_{layer}_{emb_name}_ls{ls}_sigma{sigma}_MSE_train"]  : MSE on train set.
        ad.var[f"predicted_{layer}_{emb_name}_ls{ls}_sigma{sigma}_MSE_test"]   : MSE on test set.
        ad.var[f"predicted_{layer}_{emb_name}_ls{ls}_sigma{sigma}_residuals"]  : Per-feature residuals.
    """

    # ── Validate inputs ───────────────────────────────────────────────────────

    if layer not in ad.layers:
        raise KeyError(
            f"Layer '{layer}' not found in ad.layers. "
            f"Available layers: {list(ad.layers.keys())}"
        )

    if groups is not None:
        if groups not in ad.obs.columns:
            raise KeyError(
                f"groups column '{groups}' not found in ad.obs. "
                f"Available columns: {list(ad.obs.columns)}"
            )

    missing_embeddings = [e for e in embeddings if e not in ad.obsm]
    if len(missing_embeddings) != 0:
        raise KeyError(
            f"The following embeddings are missing from ad.obsm:\n\t{missing_embeddings}"
        )

    if isinstance(embeddings, str):
        embeddings = (embeddings,)

    # ── Train/test split ──────────────────────────────────────────────────────

    if "tr_tst" in ad.obs.columns:
        warnings.warn(
            "ad.obs['tr_tst'] already exists — using existing split. "
            "Delete ad.obs['tr_tst'] to recompute."
        )
    else:
        rng        = np.random.default_rng(random_state)
        split      = np.full(ad.n_obs, "train", dtype=object)

        if groups is not None:
            # stratified split — sample test_frac from each group independently
            for group in ad.obs[groups].unique():
                group_idx = np.where(ad.obs[groups] == group)[0]
                n_test    = max(1, int(len(group_idx) * test_frac))
                test_idx  = rng.choice(group_idx, size=n_test, replace=False)
                split[test_idx] = "test"
        else:
            n_test           = max(1, int(ad.n_obs * test_frac))
            test_idx         = rng.choice(ad.n_obs, size=n_test, replace=False)
            split[test_idx]  = "test"

        ad.obs["tr_tst"] = split
        logger.info(
            "Train/test split: %d train, %d test cells.",
            (ad.obs["tr_tst"] == "train").sum(),
            (ad.obs["tr_tst"] == "test").sum(),
        )

    # ── Resolve name dict ─────────────────────────────────────────────────────

    if name_dict is None:
        name_dict = {"DM_EigenVectors_RNA": "RNA", "DM_EigenVectors_ATAC": "ATAC"}
    for e in embeddings:
        if e not in name_dict:
            name_dict[e] = e

    # ── Train/test masks ──────────────────────────────────────────────────────

    train_mask = ad.obs["tr_tst"] == "train"
    test_mask  = ad.obs["tr_tst"] != "train"

    # ── Grid search ───────────────────────────────────────────────────────────

    def dense_var(arr):
                    if sparse.issparse(arr):
                        arr = arr.toarray()
                    return np.array(arr.var(axis=0)).flatten()

    train_var = dense_var(ad[train_mask].layers[layer])
    test_var  = dense_var(ad[test_mask].layers[layer])

    # Sometimes we don't have a feature in a group
    # in this case we need to drop it

    invalid_features = (train_var == 0) | (test_var==0)
    train_var[invalid_features] = np.nan
    test_var[invalid_features] = np.nan


    for ls in (pbar_ls := tqdm(ls_vals)):
        pbar_ls.set_description(f"Length scale: {ls}")

        for sigma in (pbar_sigma := tqdm(sigmas, leave=False)):
            pbar_sigma.set_description(f"Sigma: {sigma}")

            for e in (pbar_emb := tqdm(embeddings, leave=False)):
                emb_name = name_dict[e]
                pbar_emb.set_description(f"Embedding: {emb_name}")

                key_suffix = f"{layer}_{emb_name}_ls{ls}_sigma{sigma}"

                # ── Fit model ─────────────────────────────────────────────────

                fest = mellon.FunctionEstimator(sigma=sigma, ls=ls)
                fest.fit(
                    ad[train_mask].obsm[e],
                    ad[train_mask].layers[layer],
                )

                # ── Predictions for all cells ─────────────────────────────────

                pred_key = f"predicted_{key_suffix}"
                ad.layers[pred_key] = np.asarray(fest.predict(ad.obsm[e]))

                # ── Train / test variance explained ───────────────────────────

                y_train      = ad[train_mask].layers[layer]
                y_pred_train = ad[train_mask].layers[pred_key]
                y_test       = ad[test_mask].layers[layer]
                y_pred_test  = ad[test_mask].layers[pred_key]


                # ── Residuals ─────────────────────────────────────────────────


                ad.layers[f"{pred_key}_residuals"] = np.square(
                    ad.layers[layer] - ad.layers[pred_key]
                )

                ad.var = ad.var.copy()


                # compute per-feature variance for each split
                train_mse = np.array(ad[train_mask].layers[f"{pred_key}_residuals"].mean(axis=0)).flatten()
                test_mse  = np.array(ad[test_mask].layers[f"{pred_key}_residuals"].mean(axis=0)).flatten()

                ad.var[f"{pred_key}_error_var_ratio_train"] = train_mse / train_var
                ad.var[f"{pred_key}_error_var_ratio_test"]  = test_mse  / test_var


def read_test_results(ad: anndata.AnnData, layer: str) -> pd.DataFrame:
    """Parse results from try_models into a tidy DataFrame.

    Parameters
    ----------
    ad : anndata.AnnData
        The annotated data matrix after running try_models.
    layer : str
        The layer key used in try_models.

    Returns
    -------
    pd.DataFrame with columns: variable, MSE, embedding, ls, sigma, tr_tst.
    """

    mse_cols = [
        col for col in ad.var.columns
        if col.startswith(f"predicted_{layer}_") and ("_error_var_ratio_train" in col or "_error_var_ratio_test" in col)
    ]

    if len(mse_cols) == 0:
        raise KeyError(
            f"No variance explained columns found for layer '{layer}'. "
            f"Run try_models with layer='{layer}' first."
        )

    res = ad.var[mse_cols].melt()
    res.columns = ["variable", "error_var_ratio"]

    # format: predicted_{layer}_{emb_name}_ls{ls}_sigma{sigma}_error_var_ratio_{train/test}
    res["tr_tst"]    = res["variable"].str.split("_error_var_ratio_").str[1]
    res["sigma"]     = res["variable"].str.split("_sigma").str[1].str.split("_error_var_ratio").str[0].astype(float)
    res["ls"]        = res["variable"].str.split("_ls").str[1].str.split("_sigma").str[0].astype(float)

    prefix           = f"predicted_{layer}_"
    res["embedding"] = res["variable"].str.removeprefix(prefix).str.split("_ls").str[0]

    # ── Print best parameters ─────────────────────────────────────────────────

    test_res = res[res["tr_tst"] == "test"]
    best     = (
        test_res.groupby(["embedding", "ls", "sigma"])["error_var_ratio"]
        .mean()
        .reset_index()
    )

    logger.info("Best hyperparameters per embedding (lowest mean test error/variance ratio):")
    for embedding in best["embedding"].unique():
        emb_best = best[best["embedding"] == embedding].sort_values("error_var_ratio", ascending=True).iloc[0]
        logger.info(
            "  %s: ls=%s, sigma=%s (mean test var explained=%.4f)",
            embedding,
            emb_best["ls"],
            emb_best["sigma"],
            emb_best["error_var_ratio"],
        )

    return res


def plot_model_heatmap(
    res: pd.DataFrame,
    embedding: str,
    tr_tst: str = "test",
    ax: Optional[Axes] = None,
    figsize: tuple = (8, 6),
) -> Axes:
    """Plot a heatmap of mean variance explained across ls and sigma values for a given embedding.

    Parameters
    ----------
    res : pd.DataFrame
        Output from read_test_results.
    embedding : str
        Embedding name to plot.
    tr_tst : str
        'test' or 'train'.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    figsize : tuple
        Figure size.
    """


    available_embeddings = res["embedding"].unique()

    if embedding not in available_embeddings:
        raise ValueError(
            f"'{embedding}' not found in res['embedding']. "
            f"Available embeddings: {available_embeddings.tolist()}"
        )


    available_tr_tst = res["tr_tst"].unique()
    if tr_tst not in available_tr_tst:
        raise ValueError(
            f"'{tr_tst}' not found in res['tr_tst']. "
            f"Available values: {available_tr_tst.tolist()}"
        )



    plot_df = (
        res[(res["embedding"] == embedding) & (res["tr_tst"] == tr_tst)]
        .groupby(["ls", "sigma"])["error_var_ratio"]
        .mean()
        .unstack("sigma")
    )


    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        plot_df,
        ax=ax,
        cmap="Spectral_r",
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Mean error_var_ratio"},
    )

    ax.set_xlabel("sigma")
    ax.set_ylabel("ls")
    ax.set_title(f"{embedding} — {tr_tst} error_var_ratio")

    return ax


def sweep_diffusion_components(
    ad: anndata.AnnData,
    layer: str,
    obsm_key: str,
    ls: Optional[float] = None,
    ls_factor: float = 1,
    sigma: float = 0.1,
    gp_type: Optional[str] = None,
    min_components: int = 2,
) -> None:
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

    if obsm_key not in ad.obsm:
        raise KeyError(
            f"'{obsm_key}' not found in ad.obsm. Available: {list(ad.obsm.keys())}"
        )
    if layer not in ad.layers:
        raise KeyError(
            f"Layer '{layer}' not found in ad.layers. Available: {list(ad.layers.keys())}"
        )

    n_components = ad.obsm[obsm_key].shape[1]

    if min_components < 2:
        raise ValueError("min_components must be at least 2.")
    if min_components > n_components:
        raise ValueError(
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
    ad: anndata.AnnData,
    layer: str,
    obsm_key: str,
    min_components: int = 2,
) -> pd.DataFrame:
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

        if residuals_key not in ad.layers:
            raise KeyError(
                f"'{residuals_key}' not found in ad.layers. "
                f"Run sweep_diffusion_components first."
            )

        vals = ad.layers[residuals_key]
        if sparse.issparse(vals):
            vals = vals.toarray()

        res[n] = vals.mean(axis=0)  # column is number of components used

    return res