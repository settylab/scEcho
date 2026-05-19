import scEcho


def test_run_echo_features_writes_expected_layers_and_varm(synthetic_adata):
    scEcho.Echo_features.run_echo_features(
        synthetic_adata,
        obs_col="combo_type",
        layers=["L"],
        sigma=0.1,
        ls=1.0,
        min_cells=10,
        verbose=False,
    )

    # GP-imputed layer + residuals + smoothed residuals per modality
    expected_layers = {
        "predicted_L_DM_EigenVectors_RNA_space",
        "predicted_L_DM_EigenVectors_ATAC_space",
        "predicted_L_DM_EigenVectors_RNA_space_residuals",
        "predicted_L_DM_EigenVectors_ATAC_space_residuals",
        "predicted_L_LFC_DM_EigenVectors_RNA_v_DM_EigenVectors_ATAC",
    }
    missing_layers = expected_layers - set(synthetic_adata.layers.keys())
    assert not missing_layers, f"Missing expected layers: {missing_layers}"

    # varm entry exists and has the per-group columns
    assert "reconstruction_results_L" in synthetic_adata.varm
    res = synthetic_adata.varm["reconstruction_results_L"]
    for c in synthetic_adata.obs["combo_type"].unique():
        assert f"MSE_combo_type_{c}_RNA" in res.columns
        assert f"MSE_combo_type_{c}_ATAC" in res.columns
        assert f"var_explained_diff_L_combo_type_{c}" in res.columns
