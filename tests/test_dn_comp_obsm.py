import scEcho


def test_dn_comp_obsm_writes_expected_obs_columns(synthetic_adata):
    scEcho.Echo_states.dn_comp_obsm(
        synthetic_adata,
        ls_factor=2,
        log_fold_change_threshold=0.5,
    )

    expected_cols = {
        "log_density_RNA",
        "log_density_ATAC",
        "log_density_RNA_uncertainty",
        "log_density_ATAC_uncertainty",
        "density_lfc_RNA_vs_ATAC",
        "density_lfc_pval_RNA_vs_ATAC",
        "density_lfc_ml10pval_RNA_vs_ATAC",
        "direction_RNA_v_ATAC",
    }
    missing = expected_cols - set(synthetic_adata.obs.columns)
    assert not missing, f"Missing expected obs columns: {missing}"

    # direction is categorical with the documented three levels
    direction = synthetic_adata.obs["direction_RNA_v_ATAC"]
    assert set(direction.cat.categories) == {
        "RNA variability higher",
        "ATAC variability higher",
        "neutral",
    }
