from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import mellon
import warnings
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns



def try_models(
    ad,
    ls_vals,
    sigmas,
    layer,
    embeddings=("DM_EigenVectors_RNA", "DM_EigenVectors_ATAC"),
    name_dict={"DM_EigenVectors_RNA":"RNA", "DM_EigenVectors_ATAC":"ATAC"},
    loo_residuals=True,
    groups=None,
    test_frac=0.2,
    random_state=0,
):
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
        print(f"Train/test split: {(ad.obs['tr_tst'] == 'train').sum()} train, "
              f"{(ad.obs['tr_tst'] == 'test').sum()} test cells.")

    # ── Resolve name dict ─────────────────────────────────────────────────────

    if name_dict is None:
        name_dict = {}
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







                


def read_test_results(ad, layer):
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

    print("Best hyperparameters per embedding (lowest error_var_ratio):")
    for embedding in best["embedding"].unique():
        emb_best = best[best["embedding"] == embedding].sort_values("error_var_ratio", ascending=True).iloc[0]
        print("Best hyperparameters per embedding (lowest mean test error/variance ratio):")
        print(
            f"  {embedding}: ls={emb_best['ls']}, sigma={emb_best['sigma']} "
            f"(mean test var explained={emb_best['error_var_ratio']:.4f})"
        )

    return res






def plot_model_heatmap(res, embedding, tr_tst="test", ax=None, figsize=(8, 6)):
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