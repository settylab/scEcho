"""Tests for `scEcho.echo_states` — currently `dn_comp_obsm` only.

Two-tier: smoke (function runs, expected columns added) + correctness
(numeric output matches hardcoded baseline captured on first green run).
Correctness uses `optimizer="L-BFGS-B"` so determinism is rock-solid
regardless of any future Mellon ADVI / PRNG-key changes.
"""
import numpy as np
import numpy.testing as npt
import pytest

import scEcho

# Mellon's L-BFGS-B + JAX compilation makes each dn_comp_obsm call ~25s on
# the 100-cell fixture (subsequent calls in the same session hit JAX's
# cached lowering and drop to ~3s). Both tests get the slow marker so the
# default `pytest -m "not slow"` dev-fast suite stays under ~45s; full
# `pytest` is the pre-release / CI-slow suite.
pytestmark = pytest.mark.slow


def test_smoke_dn_comp_obsm_writes_expected_obs_columns(synthetic_adata):
    scEcho.echo_states.dn_comp_obsm(
        synthetic_adata,
        ls_factor=2,
        log_fold_change_threshold=0.5,
        optimizer="L-BFGS-B",
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
    assert not missing, f"missing obs columns: {missing}"

    direction = synthetic_adata.obs["direction_RNA_v_ATAC"]
    assert set(direction.cat.categories) == {
        "RNA variability higher",
        "ATAC variability higher",
        "neutral",
    }


def test_correctness_dn_comp_obsm_values(synthetic_adata):
    """Hardcoded baseline captured on the first green run with seed-0 input
    and `optimizer="L-BFGS-B"`. Tolerance: rtol=1e-4 (Mellon L-BFGS has
    minor numerical jitter across jax/jaxopt patch versions).
    """
    scEcho.echo_states.dn_comp_obsm(
        synthetic_adata,
        ls_factor=2,
        log_fold_change_threshold=0.5,
        optimizer="L-BFGS-B",
    )

    expected_log_density_rna = np.array([
        -2.8607371423195183, -2.8555435446201844, -2.834912088552553,
        -2.8474089879574365, -2.868569307032633,
    ])
    expected_log_density_atac = np.array([
        -2.903745274029026, -2.887269317242774, -2.911338568944693,
        -2.908117695578099, -2.8787799672958307,
    ])
    expected_lfc = np.array([
        -0.04300813170950768, -0.03172577262258969, -0.07642648039213995,
        -0.060708707620662494, -0.010210660263197724,
    ])

    npt.assert_allclose(
        synthetic_adata.obs["log_density_RNA"].to_numpy()[:5],
        expected_log_density_rna, rtol=1e-4, atol=1e-6,
    )
    npt.assert_allclose(
        synthetic_adata.obs["log_density_ATAC"].to_numpy()[:5],
        expected_log_density_atac, rtol=1e-4, atol=1e-6,
    )
    npt.assert_allclose(
        synthetic_adata.obs["density_lfc_RNA_vs_ATAC"].to_numpy()[:5],
        expected_lfc, rtol=1e-4, atol=1e-6,
    )

    # below the 0.5 LFC threshold → all "neutral" on this synthetic.
    assert (synthetic_adata.obs["direction_RNA_v_ATAC"].astype(str) == "neutral").all()
