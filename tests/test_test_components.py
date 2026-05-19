"""Tests for `scEcho.test_components` — the diffusion-component sweep
utility (audit punch-list item 15 renames this file to
diffusion_component_sweep.py; that's the structure stream's scope).
"""
import numpy as np
import pandas as pd
import pytest

import scEcho


def test_smoke_sweep_diffusion_components(synthetic_adata):
    scEcho.test_components.sweep_diffusion_components(
        synthetic_adata,
        layer="L",
        obsm_key="DM_EigenVectors_RNA",
        ls=1.0,
        sigma=0.1,
        min_components=2,
    )
    # for the 6-dim DM, sweep goes 2..6 → 5 imputation layers
    n_components = synthetic_adata.obsm["DM_EigenVectors_RNA"].shape[1]
    for n in range(2, n_components + 1):
        key = f"predicted_L_DM_EigenVectors_RNA_{n}dims_space"
        assert key in synthetic_adata.layers
        assert f"{key}_residuals" in synthetic_adata.layers
        assert key in synthetic_adata.layers
    # temporary sweep key is cleaned up
    assert "DM_EigenVectors_RNA_sweep_tmp" not in synthetic_adata.obsm


def test_correctness_collect_sweep_residual_means(synthetic_adata):
    scEcho.test_components.sweep_diffusion_components(
        synthetic_adata,
        layer="L",
        obsm_key="DM_EigenVectors_RNA",
        ls=1.0,
        sigma=0.1,
        min_components=2,
    )
    res = scEcho.test_components.collect_sweep_residual_means(
        synthetic_adata,
        layer="L",
        obsm_key="DM_EigenVectors_RNA",
        min_components=2,
    )
    assert isinstance(res, pd.DataFrame)
    n_components = synthetic_adata.obsm["DM_EigenVectors_RNA"].shape[1]
    # one column per component count tested
    assert set(res.columns) == set(range(2, n_components + 1))
    assert len(res) == synthetic_adata.n_vars
    # residual means must be non-negative (they're squared residuals)
    assert (res.values >= -1e-8).all()


def test_sweep_diffusion_components_bad_min_components(synthetic_adata):
    with pytest.raises(AssertionError):
        scEcho.test_components.sweep_diffusion_components(
            synthetic_adata, layer="L", obsm_key="DM_EigenVectors_RNA",
            min_components=1,
        )
    with pytest.raises(AssertionError):
        scEcho.test_components.sweep_diffusion_components(
            synthetic_adata, layer="L", obsm_key="DM_EigenVectors_RNA",
            min_components=999,
        )
