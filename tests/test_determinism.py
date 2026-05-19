"""Determinism regression: dn_comp_obsm must be bit-stable across runs on the
same input. This guards against silent reproducibility regressions from
either the scEcho code or its Mellon/Kompot deps.
"""
import anndata as ad
import numpy as np
import numpy.testing as npt
import pytest

import scEcho


def _build_adata(seed=0):
    np.random.seed(seed)
    n = 100
    X = np.random.poisson(0.5, size=(n, 30)).astype(np.float32)
    a = ad.AnnData(X=X)
    a.layers["L"] = X.copy()
    a.obsm["DM_EigenVectors_RNA"]  = np.random.randn(n, 6).astype(np.float32)
    a.obsm["DM_EigenVectors_ATAC"] = np.random.randn(n, 6).astype(np.float32)
    a.obs["combo_type"] = np.random.choice(["A", "B"], size=n)
    return a


@pytest.mark.slow
def test_dn_comp_obsm_is_deterministic():
    a1 = _build_adata(seed=0)
    a2 = _build_adata(seed=0)

    scEcho.Echo_states.dn_comp_obsm(a1, ls_factor=2, log_fold_change_threshold=0.5)
    scEcho.Echo_states.dn_comp_obsm(a2, ls_factor=2, log_fold_change_threshold=0.5)

    numeric_cols = [
        "log_density_RNA",
        "log_density_ATAC",
        "log_density_RNA_uncertainty",
        "log_density_ATAC_uncertainty",
        "density_lfc_RNA_vs_ATAC",
        "density_lfc_pval_RNA_vs_ATAC",
        "density_lfc_ml10pval_RNA_vs_ATAC",
    ]
    for col in numeric_cols:
        npt.assert_allclose(
            a1.obs[col].to_numpy(),
            a2.obs[col].to_numpy(),
            rtol=1e-10,
            atol=0,
            err_msg=f"column '{col}' diverged between repeat runs",
        )

    # categorical direction column must also match
    assert (a1.obs["direction_RNA_v_ATAC"].astype(str)
            == a2.obs["direction_RNA_v_ATAC"].astype(str)).all()
