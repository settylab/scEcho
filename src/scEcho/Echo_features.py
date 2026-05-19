import warnings
import mellon
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from tqdm.auto import tqdm
import scipy
from kompot.utils import compute_mahalanobis_distances
from statsmodels.stats.multitest import multipletests
from pandas.api.types import CategoricalDtype
from scipy.stats import norm as normal
import re




def compute_ncells(ad_sub, layer_key, col_name, res):
    
    vals = ad_sub.layers[layer_key]
    has_neg = vals.min() < 0

    if has_neg:
        warnings.warn(
            f"Negative values detected in layer '{layer_key}' — "
            f"ncells is not meaningful and will be filled with NaN. "
            f"Column: '{col_name}'"
        )
        res[col_name] = np.nan
    else:
        res[col_name] = np.array((vals > 0).sum(axis=0)).flatten()




def embeddings_predict_layer(
    ad,
    ls=None,
    ls_factor=10,
    sigma=0.1,
    embeddings=("DM_EigenVectors_RNA", "DM_EigenVectors_ATAC"),
    layer=None,
    gp_type=None,
    # loo_residuals = True,
    save_obs_variance=True
):
    """Impute a layer in ad.layers over a given space (or spaces) using a Gaussian process.

    Parameters
    ----------
    ad : anndata.AnnData
        The annotated data matrix.
    ls : float, optional
        Length scale of the estimator.
    ls_factor : float
        Length scale factor.
    sigma : float
        Prior on the standard deviation of the features in the layer.
    embeddings : str, tuple, or list
        Keys in ad.obsm representing the embedding spaces to impute over.
    layer : str or None
        Key in ad.layers containing the values to impute. Pass None to use ad.X.
    gp_type : optional
        Gaussian process type passed to mellon.FunctionEstimator.

    Returns
    -------
    Updates ad in place with the following keys per modality:
        ad.layers[f"predicted_{layer_name}_{m}_space"]           : Imputed layer values.
        ad.obsp[f"predicted_{layer_name}_{m}_space_uncertainty"] : Full covariance matrix.
        ad.layers[f"predicted_{layer_name}_{m}_space_residuals"] : LOO squared residuals.
    """

    
    # ── Resolve input data and output key ─────────────────────────────────────

    if layer is None:
        warnings.warn("layer is None — using ad.X as input.")
        data       = ad.X
        layer_name = "X"
    else:
        if layer not in ad.layers:
            raise KeyError(
                f"Layer '{layer}' not found in ad.layers. "
                f"Available layers: {list(ad.layers.keys())}"
            )
        data       = ad.layers[layer]
        layer_name = layer

    
    # ── Validate embeddings ───────────────────────────────────────────────────

    if isinstance(embeddings, str):
        embeddings = (embeddings,)
    
    missing_embeddings = [m for m in embeddings if m not in ad.obsm]
    if len(missing_embeddings) != 0:
        raise KeyError(
            f"The following embeddings are missing from ad.obsm:\n\t{missing_embeddings}\n"
            f"Available keys: {list(ad.obsm.keys())}"
        )

    
    # ── Fit and predict per modality ──────────────────────────────────────────=
    for m in (pbar := tqdm(embeddings)):
        pbar.set_description(f"Imputing '{layer_name}' over '{m}' space")

        fest = mellon.FunctionEstimator(
            sigma=sigma,
            ls=ls,
            ls_factor=ls_factor,
            predictor_with_uncertainty=True,
            gp_type=gp_type,
                                obs_variance=True
        )

        fest.fit(ad.obsm[m], data)

        ad.layers[f"predicted_{layer_name}_{m}_space"] = np.asarray(
            fest.predict(ad.obsm[m])
        )
        ad.obsp[f"predicted_{layer_name}_{m}_space_uncertainty"] = np.asarray(
            fest.predict.covariance(ad.obsm[m], diag=False)
        )
        
        
        
        #this is preferred
        # if loo_residuals:
        ad.layers[f"predicted_{layer_name}_{m}_space_residuals"] = (
            np.asarray(fest.predict.loo_residuals_squared(ad.obsm[m], data))
        )
        # else:
        #     ad.layers[f"predicted_{layer_name}_{m}_space_residuals"] = (
        #         np.asarray(np.square(ad.layers[layer] - ad.layers[f"predicted_{layer_name}_{m}_space"]))
        #     )
        
        
        if save_obs_variance:
            ov = fest.predict.obs_variance(ad.obsm[m])
            ad.layers[f"predicted_{layer_name}_{m}_space_smoothed_residuals"] = (
                np.asarray(np.maximum(ov, 1e-8))
            )
    
    

    


    
def get_desynch_stats(
    ad,
    obs_col,
    layer,
    modality1="RNA",
    modality2="ATAC",
    embedding1="DM_EigenVectors_RNA",
    embedding2="DM_EigenVectors_ATAC",
    extra_ncells_layers=[],
    eps=1e-16,
):
    """Compute per-feature desynchronization statistics between two modalities, grouped by a cell annotation.

    For each group in obs_col, computes mean squared error of each embedding's reconstruction,
    the LFC between predicted values across modalities, a Mahalanobis distance capturing 
    multivariate divergence between predictions, and the variance in layer values explained
    by each modality.
    
    Results are stored in ad.varm[f"reconstruction_results_{layer}"].

    Parameters
    ----------
    ad : anndata.AnnData
        The annotated data matrix.
    obs_col : str
        Column in ad.obs used to group cells (e.g. cell type, condition).
    layer : str
        Key in ad.layers for the base layer to evaluate.
    modality1 : str
        Display name for the first modality (e.g. 'RNA').
    modality2 : str
        Display name for the second modality (e.g. 'ATAC').
    embedding1 : str
        Key in ad.obsm for the first embedding space.
    embedding2 : str
        Key in ad.obsm for the second embedding space.
    eps : float
        Small constant added for numerical stability.

    Returns
    -------
    Updates ad.varm[f"reconstruction_results_{layer}"] with the following columns per group c:
        MSE_{obs_col}_{c}_{modality1}          : Mean squared error of modality1 reconstruction.
        MSE_{obs_col}_{c}_{modality2}          : Mean squared error of modality2 reconstruction.
        MSE_{obs_col}_{c}_diff                 : Difference in MSE between modalities.
        {obs_col}_{c}_LFC                      : Log fold change between modality reconstructions.
        MHD_{obs_col}_{c}_{modality1}_vs_{modality2} : Mahalanobis distance between modalities.
        ncells_{obs_col}_{c}                   : Number of cells with non-zero expression (if layer is non-negative).
        mean_val_{obs_col}_{c}                 : Mean value of ad.X per feature in group.
        {layer}_var_{c}                        : Variance of layer values in group.
        var_explained_diff_{layer}_{c}         : MSE difference normalized by layer variance.
    """

    # ── Compute total variance across the layer ────────────────────────────────

    arr = ad.layers[layer]
    if sparse.issparse(arr):
        arr = arr.toarray()
    ad.var[f"{layer}_var"] = arr.var(axis=0)
    
    
    # ── Get fold changes in prediction ────────────────────────────────────────
    ## TODO: Give the option to be between errors later
    
    ad.layers[f"predicted_{layer}_LFC_{embedding1}_v_{embedding2}"] = (
        ad.layers[f"predicted_{layer}_{embedding1}_space"] 
        - ad.layers[f"predicted_{layer}_{embedding2}_space"]
    )

    
    
    # ── Initialize results dataframe ──────────────────────────────────────────

    res = pd.DataFrame({"feature": ad.var_names}, index=ad.var_names)

    # ── Compute per-group statistics ──────────────────────────────────────────

    groups = ad.obs[obs_col].unique()

    for c in (pbar := tqdm(groups)):
        pbar.set_description(f"{obs_col}: {c}")
        
        # get the group index
        ind = ad.obs[obs_col] == c

        # Mean squared errors per modality
        res[f"MSE_{obs_col}_{c}_{modality1}"] = (
            ad[ind].layers[f"predicted_{layer}_{embedding1}_space_residuals"].mean(axis=0)
        )
        res[f"MSE_{obs_col}_{c}_{modality2}"] = (
            ad[ind].layers[f"predicted_{layer}_{embedding2}_space_residuals"].mean(axis=0)
        )
        
        # Diffrence in mean squared errors across modalities
        res[f"MSE_{obs_col}_{c}_diff"] = (
            res[f"MSE_{obs_col}_{c}_{modality2}"] - res[f"MSE_{obs_col}_{c}_{modality1}"]
        )
        
        # Diffrence in mean of prediction across modalities
        res[f"mean_LFC_{obs_col}_{c}"] = (
            ad[ind].layers[f"predicted_{layer}_LFC_{embedding1}_v_{embedding2}"].mean(axis=0)
        )
        
        
        
        # ── Calculating uncertanty ────────────────────────────────────────────
        
        # Diagonal variance (observed variance)
        smoothed_key1 = f"predicted_{layer}_{embedding1}_space_smoothed_residuals"
        smoothed_key2 = f"predicted_{layer}_{embedding2}_space_smoothed_residuals"

        if (smoothed_key1 in ad.layers) and (smoothed_key2 in ad.layers):
            ov1 = np.maximum(ad[ind].layers[smoothed_key1], eps)
            ov2 = np.maximum(ad[ind].layers[smoothed_key2], eps)  
            diagonal_variance = (ov1 + ov2).T
            
            
        else:
            missing_keys = [k for k in [smoothed_key1, smoothed_key2] if k not in ad.layers]
            warnings.warn(
                f"Smoothed residual keys not found in ad.layers — diagonal variance not used.\n"
                f"To use diagonal variance rerun embeddings_predict_layer with save_obs_variance = True.\n"
                f"\tMissing: {missing_keys}"
            )
            diagonal_variance = None
        
        # Model uncertainty

        unc1 = ad[ind].obsp[f"predicted_{layer}_{embedding1}_space_uncertainty"]
        unc2 = ad[ind].obsp[f"predicted_{layer}_{embedding2}_space_uncertainty"]

        
        
        # ── Mahalanobis distance ───────────────────────────────────────────────
        res[f"MHD_{obs_col}_{c}_{modality1}_vs_{modality2}"] = compute_mahalanobis_distances(
            diff_values=ad[ind].layers[f"predicted_{layer}_LFC_{embedding1}_v_{embedding2}"].T,
            covariance=unc1 + unc2 + 1e-16,
            diagonal_variance=diagonal_variance,
        )
        
        

        # ── Additional per-group layer statistics ──────────────────────────────

        # ncells for base layer and any extra layers

        compute_ncells(ad[ind], layer, f"ncells_{layer}_{obs_col}_{c}", res)
        
        if isinstance(extra_ncells_layers, str):
            extra_ncells_layers = [extra_ncells_layers]
            
        for extra_layer in extra_ncells_layers:
            compute_ncells(ad[ind], extra_layer, f"ncells_{extra_layer}_{obs_col}_{c}", res)
            
            
            
        # mean feature values for base layer 
        res[f"mean_val_{obs_col}_{c}"] = ad[ind].layers[layer].mean(axis=0).T

        if sparse.issparse(ad[ind].layers[layer]):
            layer_vals = ad[ind].layers[layer].toarray()
        else:
            layer_vals = ad[ind].layers[layer]

        res[f"{layer}_var_{obs_col}_{c}"] = layer_vals.var(axis=0)
        
        res[f"{layer}_var_explained_{modality1}_{obs_col}_{c}"] = (
            res[f"MSE_{obs_col}_{c}_{modality1}"] / res[f"{layer}_var_{obs_col}_{c}"])
        
        res[f"{layer}_var_explained_{modality2}_{obs_col}_{c}"] = (
            res[f"MSE_{obs_col}_{c}_{modality2}"] / res[f"{layer}_var_{obs_col}_{c}"])
        
        res[f"var_explained_diff_{layer}_{obs_col}_{c}"] = (
            res[f"MSE_{obs_col}_{c}_diff"] / res[f"{layer}_var_{obs_col}_{c}"]
        )

        res = res.copy()

    # ── Store results in varm ─────────────────────────────────────────────────

    varm_key = f"reconstruction_results_{layer}"

    if varm_key in ad.varm:
        shared_cols = res.columns.intersection(ad.varm[varm_key].columns)
        warnings.warn(
            f"Duplicate columns detected: {list(shared_cols)}. "
            f"These columns will not be updated — drop them manually before re-running if needed."
        )
        res.drop(shared_cols, axis=1, inplace=True)
        ad.varm[varm_key] = pd.concat((ad.varm[varm_key], res), axis=1)
    else:
        ad.varm[varm_key] = res
        
        

        


        
        
def make_null_layer(ad, layer, random_state=0):
    """Create a null layer by randomly shuffling layer values across cells per feature.

    Parameters
    ----------
    ad : anndata.AnnData
        The annotated data matrix.
    layer : str
        Key in ad.layers to shuffle.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Adds ad.layers[f"{layer}_null"] containing the shuffled values.
    """

    if layer not in ad.layers:
        raise KeyError(
            f"Layer '{layer}' not found in ad.layers. "
            f"Available layers: {list(ad.layers.keys())}"
        )

    vals = ad.layers[layer]
    is_sparse = sparse.issparse(vals)
    if is_sparse:
        vals = vals.toarray()
    else:
        vals = vals.copy()

    
    
    rng = np.random.default_rng(random_state)
    # shuffle cell indices independently per feature
    shuffled_idx = rng.permuted(np.arange(vals.shape[0])) 
    vals = vals[shuffled_idx, :]
    
    # print(vals.shape)

    ad.layers[f"{layer}_null"] = sparse.csr_matrix(vals) if is_sparse else vals



    





def run_null_desynch_test(
    ad,
    obs_col,
    layer,
    ls=None,
    ls_factor=10,
    sigma=0.1,
    modality1="RNA",
    modality2="ATAC",
    embedding1="DM_EigenVectors_RNA",
    embedding2="DM_EigenVectors_ATAC",
    p_val_threshold=0.05,
    random_state=0,
    min_cells=50,
    eps=1e-16
):
    """Run a null model test for desynchronization statistics.

    Creates a null layer by shuffling feature values across cells, runs the
    embedding prediction pipeline on the null layer, and uses the null
    distribution of var_explained_diff to identify features that are
    significantly desynchronized beyond what is expected by chance.

    Parameters
    ----------
    ad : anndata.AnnData
        The annotated data matrix.
    obs_col : str
        Column in ad.obs used to group cells.
    layer : str
        Key in ad.layers for the base layer to evaluate.
    embedding1 : str
        Key in ad.obsm for the first embedding space.
    embedding2 : str
        Key in ad.obsm for the second embedding space.
    p_val_threshold : float
        Two-sided p-value threshold for significance.
    random_state : int
        Random seed for null layer shuffling.

    Returns
    -------
    Updates ad.varm[f"reconstruction_results_{layer}"] with per-group columns:
        var_explained_diff_{layer}_null_{obs_col}_{c}      : Null var explained diff per group.
        var_explained_diff_{layer}_null_{obs_col}_{c}_mean : Mean of null distribution.
        var_explained_diff_{layer}_null_{obs_col}_{c}_sd   : SD of null distribution.
        MSE_null_diff_{obs_col}_{c}                        : Raw null MSE difference per group.
        var_explained_diff_{layer}_{obs_col}_{c}_pval      : Two-sided p-value vs null.
        var_explained_diff_{layer}_{obs_col}_{c}_sig       : Boolean significance call.
    """

    # ── Validate inputs ───────────────────────────────────────────────────────

    if layer not in ad.layers:
        raise KeyError(
            f"Layer '{layer}' not found in ad.layers. "
            f"Available layers: {list(ad.layers.keys())}"
        )

    varm_key = f"reconstruction_results_{layer}"
    if varm_key not in ad.varm:
        raise KeyError(
            f"'{varm_key}' not found in ad.varm. Run get_desynch_stats first."
        )

    if obs_col not in ad.obs.columns:
        raise KeyError(
            f"'{obs_col}' not found in ad.obs. "
            f"Available columns: {list(ad.obs.columns)}"
        )

    # ── Validate observed columns exist before running null pipeline ───────────

    groups          = ad.obs[obs_col].unique()
    obs_col_pattern = f"var_explained_diff_{layer}_{obs_col}_"
    matching_cols   = [col for col in ad.varm[varm_key].columns if col.startswith(obs_col_pattern)]

    if len(matching_cols) == 0:
        warnings.warn(
            f"No 'var_explained_diff_{layer}_{obs_col}_*' columns found in reconstruction_results_{layer}. "
            f"Running get_desynch_stats with obs_col='{obs_col}' automatically."
        )
        get_desynch_stats(ad, obs_col=obs_col, layer=layer)
    else:
        missing_groups = [c for c in groups if f"{obs_col_pattern}{c}" not in matching_cols]
        if len(missing_groups) > 0:
            found_groups = [col.replace(obs_col_pattern, "") for col in matching_cols]
            raise AssertionError(
                f"Groups {missing_groups} not found in reconstruction_results_{layer} for obs_col='{obs_col}'. "
                f"Found results for groups: {found_groups}. "
                f"Re-run get_desynch_stats with obs_col='{obs_col}' to include these groups."
            )

    # ── Create and predict null layer ─────────────────────────────────────────

    null_layer = f"{layer}_null"

    if null_layer not in ad.layers:
        make_null_layer(ad, layer, random_state=random_state)

        #TODO: this is functionally ok now, but needs to be updated later
        embeddings_predict_layer(
            ad,
            embeddings=(embedding1, embedding2),
            layer=null_layer,
            ls=ls,
            ls_factor=ls_factor,
            sigma=sigma
        )

    # ── Compute null var_explained_diff per group ─────────────────────────────

    res = ad.varm[varm_key]

    for c in (pbar := tqdm(groups)):
        pbar.set_description(f"Computing null distribution: {obs_col}: {c}")

        ind = ad.obs[obs_col] == c
        n_cells  = ind.sum()

        if n_cells < min_cells:
            warnings.warn(
                f"Group '{c}' has {n_cells} cells which is below min_cells={min_cells} — skipping."
            )
            continue

        # MSE of null residuals per modality
        mse_null_1 = ad[ind].layers[f"predicted_{null_layer}_{embedding1}_space_residuals"].mean(axis=0)
        mse_null_2 = ad[ind].layers[f"predicted_{null_layer}_{embedding2}_space_residuals"].mean(axis=0)
        mse_null_diff = mse_null_2 - mse_null_1

        # save raw null MSE difference — column name includes group identifier
        res[f"MSE_null_diff_{obs_col}_{c}"] = mse_null_diff

        # null layer variance in group
        if sparse.issparse(ad[ind].layers[null_layer]):
            null_vals = ad[ind].layers[null_layer].toarray()
        else:
            null_vals = ad[ind].layers[null_layer]

        null_var = null_vals.var(axis=0)

        # null var explained diff — column name includes group identifier
        null_col = f"var_explained_diff_{layer}_null_{obs_col}_{c}"
        res[null_col] = mse_null_diff / null_var

        # ── Null distribution summary statistics ──────────────────────────────

        expressed_mask = null_var > 0
        n_excluded     = (~expressed_mask).sum()
        if n_excluded > 0:
            warnings.warn(
                f"{n_excluded} features have zero variance in group '{c}' of layer '{null_layer}' "
                f"and will be excluded from null mean/SD calculation."
            )

        null_vals_expressed = res.loc[expressed_mask, null_col]
        null_mean           = null_vals_expressed.mean()
        null_sd             = null_vals_expressed.std()
        
        # column names include group identifier
        res[f"{null_col}_{obs_col}_{c}_mean"] = null_mean
        res[f"{null_col}_{obs_col}_{c}_sd"]   = null_sd

        # ── Two-sided test against null distribution ──────────────────────────

        observed_col = f"var_explained_diff_{layer}_{obs_col}_{c}"
        Z    = (res[observed_col] - null_mean) / null_sd
        pval = 2 * normal.sf(np.abs(Z))

        # BH correction — exclude NaN pvals from correction then reinsert
        pval_col = f"var_explained_diff_{layer}_{obs_col}_{c}_pval"
        fdr_col  = f"var_explained_diff_{layer}_{obs_col}_{c}_fdr"
        sig_col  = f"var_explained_diff_{layer}_{obs_col}_{c}_sig"

        valid_mask        = ~np.isnan(pval)
        fdr               = np.full(len(pval), np.nan)
        _, fdr[valid_mask], _, _ = multipletests(pval[valid_mask], method="fdr_bh")

        # column names include group identifier
        res[pval_col] = pval
        res[fdr_col]  = fdr
        res[sig_col]  = fdr < p_val_threshold
        
        
        direction_col = f"var_explained_diff_{layer}_{obs_col}_{c}_direction"

        direction = np.where(
            ~res[sig_col],
            "not significant",
            np.where(
                res[observed_col] < 0,
                f"associated with {modality2} structure",  # modality2 MSE > modality1 MSE so modality1 explains more
                f"associated with {modality1} structure",
            )
        )
        
        cat_type = CategoricalDtype(
            categories=[
                f"associated with {modality2} structure",
                f"associated with {modality1} structure",
                "not significant",
            ],
            ordered=True,
        )
        res[direction_col] = pd.Categorical(direction).astype(cat_type)


        
        # ── Calculating uncertanty ────────────────────────────────────────────
        
        # Diagonal variance (observed variance)
        smoothed_key1 = f"predicted_{null_layer}_{embedding1}_space_smoothed_residuals"
        smoothed_key2 = f"predicted_{null_layer}_{embedding2}_space_smoothed_residuals"

        if (smoothed_key1 in ad.layers) and (smoothed_key2 in ad.layers):
            ov1 = np.maximum(ad[ind].layers[smoothed_key1], eps)
            ov2 = np.maximum(ad[ind].layers[smoothed_key2], eps)  
            diagonal_variance = (ov1 + ov2).T
            
            
        else:
            missing_keys = [k for k in [smoothed_key1, smoothed_key2] if k not in ad.layers]
            warnings.warn(
                f"Smoothed residual keys not found in ad.layers — diagonal variance not used.\n"
                f"To use diagonal variance rerun embeddings_predict_layer with save_obs_variance = True.\n"
                f"\tMissing: {missing_keys}"
            )
            diagonal_variance = None
        
        # Model uncertainty

        unc1 = ad[ind].obsp[f"predicted_{null_layer}_{embedding1}_space_uncertainty"]
        unc2 = ad[ind].obsp[f"predicted_{null_layer}_{embedding2}_space_uncertainty"]

        
        diff = ad[ind].layers[f"predicted_{null_layer}_{embedding1}_space_residuals"] - ad[ind].layers[f"predicted_{null_layer}_{embedding2}_space_residuals"]
        # ── Mahalanobis distance ───────────────────────────────────────────────
        res[f"MHD_null_{obs_col}_{c}_{modality1}_vs_{modality2}"] = compute_mahalanobis_distances(
            diff_values=ad[ind].layers[f"predicted_{layer}_LFC_{embedding1}_v_{embedding2}"].T,
            covariance=unc1 + unc2 + 1e-16,
            diagonal_variance=diagonal_variance,
        )
        
        

    # ── Store direction colors in uns ─────────────────────────────────────────

    ad.uns[f"desynch_direction_{layer}_{modality1}_v_{modality2}_colors"] = [
        "#ff7f0e", "#1f77b4", "lightgrey"
    ]







def run_echo_features(
    ad,
    obs_col,
    layers,
    embedding1="DM_EigenVectors_RNA",
    embedding2="DM_EigenVectors_ATAC",
    modality1="RNA",
    modality2="ATAC",
    sigma=0.1,
    ls=None,
    ls_factor=10,
    p_val_threshold=0.05,
    random_state=0,
    min_cells=50,
    eps=1e-16,
    verbose=True,
):
    """Run the full desynchronization pipeline across one or more layers.

    For each layer in `layers`, this runs the three pipeline steps in order:
      1. embeddings_predict_layer  -- impute the layer over both embeddings
      2. get_desynch_stats         -- compute per-feature desynch statistics
      3. run_null_desynch_test     -- test significance against a null model

    

    Parameters
    ----------
    ad : anndata.AnnData
        The annotated data matrix. Modified in place.
    obs_col : str
        Column in ad.obs used to group cells (passed as obs_col / "combo_type").
    layers : str or sequence of str
        Layer key, or a list/tuple of layer keys, in ad.layers to process.
    embedding1, embedding2 : str
        Keys in ad.obsm for the two embedding spaces.
    modality1, modality2 : str
        Display names for the two modalities.
    sigma : float
        Prior std used for both embeddings_predict_layer and the null test.
    ls : float or None
        Length scale used for the GP fit. None lets the estimator choose.
    ls_factor : float
        Length scale factor.
    p_val_threshold : float
        Two-sided p-value threshold for the null test.
    random_state : int
        Random seed for the null layer shuffling.
    min_cells : int
        Minimum cells per group for the null test.
    eps : float
        Small constant for numerical stability.
    verbose : bool
        Print progress per layer.

    Returns
    -------
    None. Updates `ad` in place; see the individual functions for the keys
    written to ad.layers, ad.obsp, and ad.varm.
    """
    # Accept a single layer string or a collection of them.
    if isinstance(layers, str):
        layers = [layers]
    else:
        layers = list(layers)

    embeddings = (embedding1, embedding2)

    for layer in layers:
        if layer not in ad.layers:
            raise KeyError(
                f"Layer '{layer}' not found in ad.layers. "
                f"Available layers: {list(ad.layers.keys())}"
            )

        if verbose:
            print(f"[run_desynch_full] Processing layer: '{layer}'")

        # 1. Impute the layer over both embedding spaces.
        if verbose:
            print("  - embeddings_predict_layer")
        embeddings_predict_layer(
            ad,
            embeddings=embeddings,
            sigma=sigma,
            ls=ls,
            ls_factor=ls_factor,
            layer=layer,
        )

        # 2. Per-feature desynchronization statistics.
        if verbose:
            print("  - get_desynch_stats")
        get_desynch_stats(
            ad,
            obs_col=obs_col,
            layer=layer,
            modality1=modality1,
            modality2=modality2,
            embedding1=embedding1,
            embedding2=embedding2,
            eps=eps,
        )

        # 3. Null model significance test.
        if verbose:
            print("  - run_null_desynch_test")
        run_null_desynch_test(
            ad,
            obs_col,
            layer,
            sigma=sigma,
            ls=ls,
            ls_factor=ls_factor,
            modality1=modality1,
            modality2=modality2,
            embedding1=embedding1,
            embedding2=embedding2,
            p_val_threshold=p_val_threshold,
            random_state=random_state,
            min_cells=min_cells,
            eps=eps,
        )

        if verbose:
            print(f"[run_desynch_full] Done with layer: '{layer}'")






def get_reconstruction_results(ad, layer, grouping, group, min_cells=None):
    """Pull reconstruction results for a specific group from ad.varm.

    Parameters
    ----------
    ad : anndata.AnnData
        The annotated data matrix.
    layer : str
        Key used when running get_desynch_stats.
    grouping : str
        Column in ad.obs used to group cells.
    group : str
        Group value to retrieve results for.
    min_cells : int, optional
        Minimum number of cells with non-zero expression required to retain a feature.
        Filters using ncells_{layer}_{grouping}_{group} if present. If the ncells column
        is not found, a warning is issued and no filtering is applied.

    Returns
    -------
    pd.DataFrame subset of ad.varm[f"reconstruction_results_{layer}"]
    containing only columns relevant to the specified grouping and group.
    """

    varm_key = f"reconstruction_results_{layer}"
    if varm_key not in ad.varm:
        raise KeyError(
            f"'{varm_key}' not found in ad.varm. Run get_desynch_stats first."
        )

    res = ad.varm[varm_key]

    # exact match: column must contain _{grouping}_{group} followed by end or underscore
    # this prevents e.g. "RGC" matching "RGCpre" columns
    pattern    = re.compile(rf"_{re.escape(grouping)}_{re.escape(group)}(_|$)")
    group_cols = [col for col in res.columns if pattern.search(col)]

    if len(group_cols) == 0:
        raise KeyError(
            f"No columns found for grouping='{grouping}', group='{group}' in {varm_key}. "
            f"Available groups: {ad.obs[grouping].unique().tolist()}"
        )

    res = res[group_cols]

    # ── Apply min_cells filter ────────────────────────────────────────────────

    if min_cells is not None:
        ncells_col = f"ncells_{layer}_{grouping}_{group}"
        if ncells_col not in res.columns:
            warnings.warn(
                f"'{ncells_col}' not found in reconstruction_results_{layer} — "
                f"min_cells filter will not be applied. "
                f"Run get_desynch_stats with layer='{layer}' to enable filtering."
            )
        elif res[ncells_col].isna().all():
            warnings.warn(
                f"'{ncells_col}' is all NaN — likely because layer '{layer}' contains "
                f"negative values. min_cells filter will not be applied."
            )
        else:
            res = res[res[ncells_col] > min_cells]

    return res