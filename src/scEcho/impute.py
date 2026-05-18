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
        assert layer in ad.layers, (
            f"Layer '{layer}' not found in ad.layers. "
            f"Available layers: {list(ad.layers.keys())}"
        )
        data       = ad.layers[layer]
        layer_name = layer

    
    # ── Validate embeddings ───────────────────────────────────────────────────

    if isinstance(embeddings, str):
        embeddings = (embeddings,)
    
    missing_embeddings = [m for m in embeddings if m not in ad.obsm]
    assert len(missing_embeddings) == 0, (
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
    

        
        
        
        
        
        
def min_max_scaler(ad, layer, quantile=0.99):
    """Scale a layer to [0, 1] (assuming quantile is 1) using min and quantile normalization.

    Subtracts the per-feature minimum then divides by the per-feature quantile.
    Scale factors are stored in ad.varm["uncertainty_scale_factors"] for use
    in downstream uncertainty comparisons across layers.

    Parameters
    ----------
    ad : anndata.AnnData
        The annotated data matrix.
    layer : str
        Key in ad.layers to scale.
    quantile : float
        Upper quantile used as the scale factor (default 0.99).

    Returns
    -------
    Adds ad.layers[f"{layer}_scaled"] with values scaled to [0, 1] (assuming quantile is 1).
    Updates ad.varm["uncertainty_scale_factors"] with per-feature scale factors.
    """

    # ── Validate inputs ───────────────────────────────────────────────────────

    assert layer in ad.layers, (
        f"Layer '{layer}' not found in ad.layers. "
        f"Available layers: {list(ad.layers.keys())}"
    )
    assert not sparse.issparse(ad.layers[layer]), (
        f"Layer '{layer}' is sparse — min_max_scaler expects dense input. "
        f"Run embeddings_predict_layer on layer '{layer}' and pass the imputed layer instead."
    )

    # ── Scale ─────────────────────────────────────────────────────────────────

    scaled_layer = f"{layer}_scaled"
    vals = ad.layers[layer] - ad.layers[layer].min(axis=0)
    scale_factors = np.quantile(vals, quantile, axis=0)

    ad.layers[scaled_layer] = vals / scale_factors

    # ── Store scale factors ───────────────────────────────────────────────────

    varm_key = "uncertainty_scale_factors"
    if varm_key not in ad.varm:
        ad.varm[varm_key] = pd.DataFrame(
            scale_factors,
            index=ad.var_names,
            columns=[layer],
        )
    else:
        ad.varm[varm_key][layer] = scale_factors
        
        

    
        

        
# TODO fix naming, make saclaing optional
def compare_layers_cellwise(ad,
                 layer1 = "mellon_imputed_expression_rna_scaled",
                 layer2 = "mellon_imputed_SEAscore_atac_scaled",
                  new_layer_name = "priming",
                 eps = 1e-12):
    
    """
     Given two layers that have been imputed by embeddings_predict_layer over a
    shared embedding space, computes a per-cell, per-feature comparison of the
    two modalities. The comparison accounts for the uncertainty in each GP
    prediction, allowing cells where the two modalities are significantly
    discordant to be distinguished from cells where the difference is within
    the expected noise.

    Parameters
    ----------
    ad : anndata.AnnData
        The annotated data matrix.
    layer1 : str
        The name of the layer where imputed gene expression data is stored.
    gene_score_layer : str
        The name of the layer where imputed gene scores are stored.
    new_layer_name : str
        The name of the layer where the diffrence between the expression layer and gene score layer is stored
    eps : float
        Small constant to stabilize numerical operations, default is 1e-12.

    Returns
    -------
    
    """
    
    exp_prefix = layer1.split("_scaled")[0]
    gs_prefix = layer2.split("_scaled")[0]
    
    # Make sure layers are scaled
    assert "_scaled" in layer1, f"The expression layer does not appear to be scaled plase run min_max_scaler on {layer1}"
    assert "_scaled" in layer2, f"The gene score layer does not appear to be scaled plase run min_max_scaler on {layer2}"

    assert "uncertainty_scale_factors" in ad.varm.keys(), "gene-wise and feature set wise scale factors not found in ad.varm[\'uncertainty_scale_factors\']. Please use min_max_scaler or store your scale factors in ad.varm[\'uncertainty_scale_factors\']"

    # TODO: make these explanations better
    assert exp_prefix in ad.varm["uncertainty_scale_factors"].columns, f"no scale factors found for {exp_prefix} in ad.varm[\'uncertainty_scale_factors\']"
    assert gs_prefix in ad.varm["uncertainty_scale_factors"].columns, f"no scale factors found for {gs_prefix} in ad.varm[\'uncertainty_scale_factors\']"
    
    
    
    # first get uncertanty
    exp_unc = ((ad.obsp[f"{exp_prefix}_uncertainty"].diagonal() * np.ones((ad.shape[1], ad.shape[0]))).T * np.square(ad.varm["uncertainty_scale_factors"][exp_prefix].values))
    atac_unc = ((ad.obsp[f"{gs_prefix}_uncertainty"].diagonal() * np.ones((ad.shape[1], ad.shape[0]))).T * np.square(ad.varm["uncertainty_scale_factors"][gs_prefix].values))
    
    sd = np.sqrt(exp_unc + atac_unc + eps)
    
    ad.layers[new_layer_name] = ad.layers[layer1] - ad.layers[layer2]
    ad.layers[f"{new_layer_name}_deviations"] = ad.layers[new_layer_name]/sd
    
    ad.layers[f"{new_layer_name}_pval"] = np.zeros(ad.shape)
    for i in tqdm(range(ad.shape[1])):
        ad.layers[f"{new_layer_name}_pval"][:,i] = np.minimum(normal.logcdf(ad.layers[f"{new_layer_name}_deviations"][:,i]), 
                                                    normal.logcdf(-ad.layers[f"{new_layer_name}_deviations"][:,i])) + np.log(2)
    
    ad.layers[f"{new_layer_name}_ml10pval"] = -ad.layers[f"{new_layer_name}_pval"] / np.log(10)
    
    

    
    

    
# # TODO: clean this up
# def run_desynch_full(ad, 
#                      ls, 
#                      sigma,
#                      refrence_space,
#                      other_space,
#                      layer = "logcounts", 
#                      gp_type = None,
#                     ls_var_scaling_factor = 1.5):
    
    
#     #TODO: do this once
#     # ref_landmarks = mellon.parameters.compute_landmarks(ad.obsm[refrence_space])
#     # other_landmarks =  mellon.parameters.compute_landmarks(ad.obsm[other_space])
    
    
#     modality_predict_layer(ad,
#                            ls = ls, 
#                            sigma = sigma, 
#                            modalities = (refrence_space, 
#                                              other_space),
#                            layer =layer, 
#                            gp_type = gp_type)
    
    
#     ls_var = ls * ls_var_scaling_factor
#     est_desynch_features(ad,
#                          refrence_space = refrence_space,
#                          other_space = other_space,
#                          ls = ls_var,
#                          sigma = sigma, #TODO: adjust this
#                          gp_type=gp_type, 
#                         orig_layer = layer)
    
    
    
    
    

    
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

    assert layer in ad.layers, (
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

    assert layer in ad.layers, (
        f"Layer '{layer}' not found in ad.layers. "
        f"Available layers: {list(ad.layers.keys())}"
    )

    varm_key = f"reconstruction_results_{layer}"
    assert varm_key in ad.varm, (
        f"'{varm_key}' not found in ad.varm. Run get_desynch_stats first."
    )

    assert obs_col in ad.obs.columns, (
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
    assert varm_key in ad.varm, (
        f"'{varm_key}' not found in ad.varm. Run get_desynch_stats first."
    )

    res = ad.varm[varm_key]

    # exact match: column must contain _{grouping}_{group} followed by end or underscore
    # this prevents e.g. "RGC" matching "RGCpre" columns
    pattern    = re.compile(rf"_{re.escape(grouping)}_{re.escape(group)}(_|$)")
    group_cols = [col for col in res.columns if pattern.search(col)]

    assert len(group_cols) > 0, (
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