from __future__ import annotations

import logging
import warnings
from typing import Optional

import anndata
import kompot
import mellon
import numpy as np
from pandas.api.types import CategoricalDtype
from scipy.stats import norm as normal
from tqdm.auto import tqdm

__all__ = ["dn_comp_obsm"]

logger = logging.getLogger(__name__)


def dn_comp_obsm(
    ad: anndata.AnnData,
    obsm_key1: str = "DM_EigenVectors_RNA",
    obsm_key2: str = "DM_EigenVectors_ATAC",
    modality1_name: str = "RNA",
    modality2_name: str = "ATAC",
    ndims_embedding1: Optional[int] = None,
    ndims_embedding2: Optional[int] = None,
    pval_threshold: float = 0.05,
    log_fold_change_threshold: float = 2,
    ls_factor: float = 2,
    optimizer: str = "advi",
    sample_grouping_col: Optional[str] = None,
    sv_min_cells: int = 200,
) -> None:
    """Compare density between two embeddings in separate spaces.

    Parameters
    ----------
    ad : anndata.AnnData
        The annotated data matrix.
    obsm_key1 : str
        The .obsm key for the first embedding space.
    obsm_key2 : str
        The .obsm key for the second embedding space.
    modality1_name : str
        Name for the first modality/space.
    modality2_name : str
        Name for the second modality/space.
    ndims_embedding1 : int, optional
        Number of dimensions to use for the first space.
    ndims_embedding2 : int, optional
        Number of dimensions to use for the second space.
    pval_threshold : float
        P-value threshold for density difference.
    log_fold_change_threshold : float
        Log fold change threshold for density comparison.
    ls_factor : float
        Length scale factor for mellon DensityEstimator.
    sample_grouping_col : str, optional
        Column for sample groupings. If specified, includes sample variance.

    Returns
    -------
    Adds the following columns to ad.obs:
        log_density_{modality}                          : Log density from mellon for each modality.
        log_density_{modality}_uncertainty              : Uncertainty of log density estimates.
        density_lfc_{modality1}_vs_{modality2}          : Log fold change between densities.
        density_lfc_pval_{modality1}_vs_{modality2}     : P-value for density difference.
        density_lfc_ml10pval_{modality1}_vs_{modality2} : Negative log10 p-value.
        direction_{modality1}_v_{modality2}             : Which modality has higher density per cell.
    """


    # ── Slice embedding spaces ────────────────────────────────────────────────

    mod1_space = (
        ad.obsm[obsm_key1]
        if ndims_embedding1 is None
        else ad.obsm[obsm_key1][:, :ndims_embedding1]
    )

    mod2_space = (
        ad.obsm[obsm_key2]
        if ndims_embedding2 is None
        else ad.obsm[obsm_key2][:, :ndims_embedding2]
    )

    if ndims_embedding1 != ndims_embedding2:
        warnings.warn(
            "Using different number of dimensions for each modality — this can lead to odd results."
        )


    # ── Compute shared fractal dimensionality ─────────────────────────────────

    d_rna  = mellon.parameters.compute_d_factal(mod1_space)
    d_atac = mellon.parameters.compute_d_factal(mod2_space)
    d_use  = max(d_rna, d_atac)

    logger.info("RNA fractal dimensionality:  %s", d_rna)
    logger.info("ATAC fractal dimensionality: %s", d_atac)
    logger.info("Using dimensionality:        %s", d_use)



    # ── Fit density models ────────────────────────────────────────────────────

    modalities  = [modality1_name, modality2_name]
    spaces      = [mod1_space, mod2_space]

    for modality, space in (pbar := tqdm(zip(modalities, spaces), total=2)):
        pbar.set_description(f"Fitting density for space {modality}")

        model = mellon.DensityEstimator(
            predictor_with_uncertainty=True,
            optimizer=optimizer,
            d=d_use,
            ls_factor=ls_factor,
        )

        ad.obs[f"log_density_{modality}"]             = model.fit_predict(space)
        ad.obs[f"log_density_{modality}_uncertainty"] = model.predict.uncertainty(space)



    # ── Compute density log fold change ───────────────────────────────────────
    lfc_key = f"density_lfc_{modality1_name}_vs_{modality2_name}"
    ad.obs[lfc_key] = (
        ad.obs[f"log_density_{modality2_name}"] - ad.obs[f"log_density_{modality1_name}"]
    )
    lfc = ad.obs[lfc_key]



    # ── Compute combined uncertainty (with or without sample variance) ────────
    sd_key = f"density_lfc_sd_{modality1_name}_vs_{modality2_name}"

    variance_model = ad.obs[f"log_density_{modality1_name}_uncertainty"] + ad.obs[f"log_density_{modality2_name}_uncertainty"]


    if sample_grouping_col is not None:
        logger.info("Computing sample variance...")

        for modality, space in (pbar := tqdm(zip(modalities, spaces), total=2)):
            pbar.set_description(f"Fitting sample variance for modality: {modality}")

            model = kompot.SampleVarianceEstimator(estimator_type="density")
            model.fit(
                space,
                grouping_vector=ad.obs[sample_grouping_col],
                ls_factor=ls_factor,
                min_cells = sv_min_cells,
                estimator_kwargs={"d": d_use},
            )
            ad.obs[f"log_density_{modality}_sample_var"] = model.predict(space, diag=True)

        variance_model = (variance_model +
                          ad.obs[f"log_density_{modality1_name}_sample_var"] +
                          ad.obs[f"log_density_{modality2_name}_sample_var"]
        )


    ad.obs[sd_key] = np.sqrt(variance_model + 1e-16)


    # ── Compute Z-scores and p-values ─────────────────────────────────────────

    z_key = f"density_lfcZ_{modality1_name}_vs_{modality2_name}"
    ad.obs[z_key] = ad.obs[lfc_key] / ad.obs[sd_key]

    pval_key = f"density_lfc_pval_{modality1_name}_vs_{modality2_name}"
    ad.obs[pval_key] = (
        np.minimum(
            normal.logcdf( ad.obs[z_key]),
            normal.logcdf(-ad.obs[z_key]),
        )
        + np.log(2)
    )

    ml10pval_key = f"density_lfc_ml10pval_{modality1_name}_vs_{modality2_name}"
    ad.obs[ml10pval_key] = -ad.obs[pval_key] / np.log(10)
    ml10pval = ad.obs[ml10pval_key]


    # ── Assign labels ───────────────────────────────────────────────

    direction_key = f"direction_{modality1_name}_v_{modality2_name}"
    significant   = (np.abs(lfc) > log_fold_change_threshold) & (ml10pval > -np.log10(pval_threshold))
    direction     = np.where(lfc < 0, f"{modality2_name} variability higher", f"{modality1_name} variability higher")

    ad.obs[direction_key] = np.where(significant, direction, "neutral")

    cat_type = CategoricalDtype(
        categories=[
            f"{modality2_name} variability higher",
            f"{modality1_name} variability higher",
            "neutral",
        ],
        ordered=True,
    )
    ad.obs[direction_key] = ad.obs[direction_key].astype(cat_type)
    ad.uns[f"{direction_key}_colors"] = ["#ff7f0e", "#1f77b4", "lightgrey"]
