"""Tests for `scEcho.echo_features` — covers the six entries in
`echo_features.__all__` plus the un-exposed `compute_ncells` helper.

Smoke tests assert function-exit and expected output keys; correctness
tests assert hardcoded baseline values captured on the first green run
with seed-0 synthetic input.
"""
import numpy as np
import numpy.testing as npt
import pytest

import scEcho
from scEcho.echo_features import compute_ncells


# ── embeddings_predict_layer ─────────────────────────────────────────────────

def test_smoke_embeddings_predict_layer(synthetic_adata):
    scEcho.echo_features.embeddings_predict_layer(
        synthetic_adata, ls=1.0, sigma=0.1, layer="L",
    )
    assert "predicted_L_DM_EigenVectors_RNA_space" in synthetic_adata.layers
    assert "predicted_L_DM_EigenVectors_ATAC_space" in synthetic_adata.layers
    assert "predicted_L_DM_EigenVectors_RNA_space_residuals" in synthetic_adata.layers
    assert "predicted_L_DM_EigenVectors_RNA_space_uncertainty" in synthetic_adata.obsp


def test_correctness_embeddings_predict_layer_values(synthetic_adata):
    scEcho.echo_features.embeddings_predict_layer(
        synthetic_adata, ls=1.0, sigma=0.1, layer="L",
    )
    pred = np.asarray(synthetic_adata.layers["predicted_L_DM_EigenVectors_RNA_space"])
    expected_row0 = np.array([
        0.005927815805978053, 0.990390696403116, 0.002905172298006257,
        0.004740474523739172, 0.9975311500463615,
    ])
    npt.assert_allclose(pred[0, :5], expected_row0, rtol=1e-4, atol=1e-6)

    # uncertainty stored in obsp as a per-modality covariance
    cov = synthetic_adata.obsp["predicted_L_DM_EigenVectors_RNA_space_uncertainty"]
    assert cov.shape == (synthetic_adata.n_obs, synthetic_adata.n_obs)
    # diagonal must be non-negative — it's a variance
    diag = np.diag(np.asarray(cov))
    assert (diag >= -1e-8).all()


# ── get_desynch_stats ────────────────────────────────────────────────────────

def test_smoke_get_desynch_stats(synthetic_adata):
    scEcho.echo_features.embeddings_predict_layer(
        synthetic_adata, ls=1.0, sigma=0.1, layer="L",
    )
    scEcho.echo_features.get_desynch_stats(
        synthetic_adata, obs_col="combo_type", layer="L",
    )
    res = synthetic_adata.varm["reconstruction_results_L"]
    for c in synthetic_adata.obs["combo_type"].unique():
        assert f"MSE_combo_type_{c}_RNA" in res.columns
        assert f"MSE_combo_type_{c}_ATAC" in res.columns
        assert f"MHD_combo_type_{c}_RNA_vs_ATAC" in res.columns
        assert f"var_explained_diff_L_combo_type_{c}" in res.columns


def test_correctness_get_desynch_stats_mse_values(synthetic_adata):
    scEcho.echo_features.embeddings_predict_layer(
        synthetic_adata, ls=1.0, sigma=0.1, layer="L",
    )
    scEcho.echo_features.get_desynch_stats(
        synthetic_adata, obs_col="combo_type", layer="L",
    )
    res = synthetic_adata.varm["reconstruction_results_L"]
    expected_mse_a = np.array([
        0.518297160290036, 1.0067171090460476, 0.4362323008815807,
        0.45400894105947737, 0.532451712148036,
    ])
    npt.assert_allclose(
        res["MSE_combo_type_A_RNA"].to_numpy()[:5],
        expected_mse_a, rtol=1e-4, atol=1e-6,
    )

    # var_explained_diff matches the MSE diff divided by per-group variance
    expected_var_diff_a = np.array([
        0.34861899356791554, 0.20619575797483955, 0.028539860970168026,
        -0.10922566532519834, -0.11066619430608766,
    ])
    npt.assert_allclose(
        res["var_explained_diff_L_combo_type_A"].to_numpy()[:5],
        expected_var_diff_a, rtol=1e-4, atol=1e-6,
    )


# ── make_null_layer ──────────────────────────────────────────────────────────

def test_smoke_make_null_layer(synthetic_adata):
    scEcho.echo_features.make_null_layer(synthetic_adata, layer="L", random_state=0)
    assert "L_null" in synthetic_adata.layers
    assert synthetic_adata.layers["L_null"].shape == synthetic_adata.layers["L"].shape


def test_correctness_make_null_layer_preserves_per_feature_distribution(synthetic_adata):
    """Shuffling cells within a feature must preserve the per-feature value
    multiset — the audit notes the per-feature shuffle is the *point* of
    make_null_layer (breaks cell-cell correlations, keeps marginals).
    """
    original = synthetic_adata.layers["L"].copy()
    scEcho.echo_features.make_null_layer(synthetic_adata, layer="L", random_state=0)
    null = np.asarray(synthetic_adata.layers["L_null"])

    # the function uses one shuffle index applied to all features (audit #25)
    # so the entire row-permutation should match across features. Verify the
    # sorted per-feature values match the original.
    npt.assert_array_equal(
        np.sort(original, axis=0),
        np.sort(null, axis=0),
    )


# ── run_null_desynch_test ────────────────────────────────────────────────────

def test_smoke_run_null_desynch_test(synthetic_adata):
    scEcho.echo_features.embeddings_predict_layer(
        synthetic_adata, ls=1.0, sigma=0.1, layer="L",
    )
    scEcho.echo_features.get_desynch_stats(
        synthetic_adata, obs_col="combo_type", layer="L",
    )
    scEcho.echo_features.run_null_desynch_test(
        synthetic_adata, obs_col="combo_type", layer="L",
        ls=1.0, sigma=0.1, min_cells=10,
    )
    res = synthetic_adata.varm["reconstruction_results_L"]
    for c in synthetic_adata.obs["combo_type"].unique():
        assert f"MSE_null_diff_combo_type_{c}" in res.columns
        assert f"var_explained_diff_L_null_combo_type_{c}" in res.columns
        assert f"var_explained_diff_L_combo_type_{c}_pval" in res.columns
        assert f"var_explained_diff_L_combo_type_{c}_fdr" in res.columns
        assert f"var_explained_diff_L_combo_type_{c}_sig" in res.columns
        assert f"var_explained_diff_L_combo_type_{c}_direction" in res.columns


def test_correctness_run_null_desynch_test_values(synthetic_adata):
    scEcho.echo_features.embeddings_predict_layer(
        synthetic_adata, ls=1.0, sigma=0.1, layer="L",
    )
    scEcho.echo_features.get_desynch_stats(
        synthetic_adata, obs_col="combo_type", layer="L",
    )
    scEcho.echo_features.run_null_desynch_test(
        synthetic_adata, obs_col="combo_type", layer="L",
        ls=1.0, sigma=0.1, min_cells=10,
    )
    res = synthetic_adata.varm["reconstruction_results_L"]
    expected_null_diff_a = np.array([
        0.015021325883813885, 0.22540213079243265, 0.04017872146175672,
        -0.10674224827699619, 0.08714758614026352,
    ])
    npt.assert_allclose(
        res["MSE_null_diff_combo_type_A"].to_numpy()[:5],
        expected_null_diff_a, rtol=1e-4, atol=1e-6,
    )

    expected_var_diff_null_a = np.array([
        0.021397868564746594, 0.30026554454360205, 0.10612923226250778,
        -0.1734327134063327, 0.22442324497229924,
    ])
    npt.assert_allclose(
        res["var_explained_diff_L_null_combo_type_A"].to_numpy()[:5],
        expected_var_diff_null_a, rtol=1e-4, atol=1e-6,
    )

    # p-values bounded in [0, 1]
    pvals = res["var_explained_diff_L_combo_type_A_pval"].to_numpy()
    assert ((pvals >= 0) & (pvals <= 1)).all()


# ── run_echo_features (orchestrator) ─────────────────────────────────────────

def test_smoke_run_echo_features(synthetic_adata):
    scEcho.echo_features.run_echo_features(
        synthetic_adata, obs_col="combo_type", layers=["L"],
        sigma=0.1, ls=1.0, min_cells=10, verbose=False,
    )
    expected_layers = {
        "predicted_L_DM_EigenVectors_RNA_space",
        "predicted_L_DM_EigenVectors_ATAC_space",
        "predicted_L_DM_EigenVectors_RNA_space_residuals",
        "predicted_L_DM_EigenVectors_ATAC_space_residuals",
        "predicted_L_LFC_DM_EigenVectors_RNA_v_DM_EigenVectors_ATAC",
        "L_null",
    }
    missing = expected_layers - set(synthetic_adata.layers.keys())
    assert not missing, f"missing layers: {missing}"
    assert "reconstruction_results_L" in synthetic_adata.varm


def test_correctness_run_echo_features_matches_pipeline_components(synthetic_adata):
    """run_echo_features should produce the same numeric outputs as calling
    the three component functions individually with the same parameters.
    """
    a2 = synthetic_adata.copy()
    scEcho.echo_features.run_echo_features(
        synthetic_adata, obs_col="combo_type", layers=["L"],
        sigma=0.1, ls=1.0, min_cells=10, verbose=False,
    )

    scEcho.echo_features.embeddings_predict_layer(a2, ls=1.0, sigma=0.1, layer="L")
    scEcho.echo_features.get_desynch_stats(a2, obs_col="combo_type", layer="L")
    scEcho.echo_features.run_null_desynch_test(
        a2, obs_col="combo_type", layer="L",
        ls=1.0, sigma=0.1, min_cells=10,
    )

    res1 = synthetic_adata.varm["reconstruction_results_L"]
    res2 = a2.varm["reconstruction_results_L"]
    npt.assert_allclose(
        res1["MSE_combo_type_A_RNA"].to_numpy(),
        res2["MSE_combo_type_A_RNA"].to_numpy(),
        rtol=1e-6,
    )


# ── get_reconstruction_results ───────────────────────────────────────────────

def test_get_reconstruction_results_filters_to_group(synthetic_adata):
    scEcho.echo_features.run_echo_features(
        synthetic_adata, obs_col="combo_type", layers=["L"],
        sigma=0.1, ls=1.0, min_cells=10, verbose=False,
    )
    sub = scEcho.echo_features.get_reconstruction_results(
        synthetic_adata, layer="L", grouping="combo_type", group="A",
    )
    # all returned columns must reference group "A"
    for col in sub.columns:
        assert "_combo_type_A" in col, f"unexpected column for group A: {col}"
    # asking for a group that doesn't exist raises
    with pytest.raises(KeyError):
        scEcho.echo_features.get_reconstruction_results(
            synthetic_adata, layer="L", grouping="combo_type", group="Z",
        )


def test_get_reconstruction_results_min_cells_filter(synthetic_adata):
    scEcho.echo_features.run_echo_features(
        synthetic_adata, obs_col="combo_type", layers=["L"],
        sigma=0.1, ls=1.0, min_cells=10, verbose=False,
    )
    full = scEcho.echo_features.get_reconstruction_results(
        synthetic_adata, layer="L", grouping="combo_type", group="A",
    )
    filtered = scEcho.echo_features.get_reconstruction_results(
        synthetic_adata, layer="L", grouping="combo_type", group="A",
        min_cells=10**6,  # absurd cutoff
    )
    assert len(filtered) < len(full)


# ── compute_ncells (internal helper, called via get_desynch_stats) ───────────

def test_compute_ncells_correctness_on_seeded_input(synthetic_adata):
    """Hardcoded expected counts for the seed-0 fixture's Poisson(0.5) matrix."""
    res = {}
    compute_ncells(synthetic_adata, layer_key="L", col_name="ncells_L", res=res)
    expected_first_10 = np.array([48, 37, 35, 37, 44, 38, 36, 31, 42, 32])
    npt.assert_array_equal(res["ncells_L"][:10], expected_first_10)
    # total count of non-zero entries
    assert int(res["ncells_L"].sum()) == 1164


def test_compute_ncells_negative_layer_emits_warning(synthetic_adata):
    """Negative values → ncells is filled with NaN and a warning fires."""
    import warnings

    synthetic_adata.layers["signed"] = -synthetic_adata.layers["L"].astype(np.float32) + 1
    res = {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        compute_ncells(synthetic_adata, layer_key="signed", col_name="ncells_signed", res=res)
    assert any("Negative" in str(rec.message) for rec in w)
    assert np.isnan(res["ncells_signed"])
