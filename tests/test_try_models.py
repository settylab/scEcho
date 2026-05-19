"""Tests for `scEcho.try_models`. Covers all of try_models.__all__:
try_models, read_test_results, plot_model_heatmap.
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from matplotlib.axes import Axes

import scEcho


@pytest.fixture
def adata_after_try_models(synthetic_adata):
    """Run try_models once on a 2×2 grid and cache the result for the
    read_test_results / plot_model_heatmap tests.
    """
    scEcho.try_models.try_models(
        synthetic_adata,
        ls_vals=[1.0, 2.0],
        sigmas=[0.1, 0.5],
        layer="L",
        random_state=0,
    )
    return synthetic_adata


# ── try_models ───────────────────────────────────────────────────────────────

def test_smoke_try_models_writes_error_var_ratio_columns(adata_after_try_models):
    train_cols = [c for c in adata_after_try_models.var.columns
                  if c.endswith("_error_var_ratio_train")]
    test_cols  = [c for c in adata_after_try_models.var.columns
                  if c.endswith("_error_var_ratio_test")]
    # 2 ls × 2 sigma × 2 embeddings
    assert len(train_cols) == 8
    assert len(test_cols)  == 8
    assert "tr_tst" in adata_after_try_models.obs.columns
    assert set(adata_after_try_models.obs["tr_tst"].unique()) == {"train", "test"}


def test_correctness_try_models_error_var_ratio_values(adata_after_try_models):
    """Hardcoded baseline for the first ratio column (alphabetical).
    Audit punch-list item 9: the actual column name is `_error_var_ratio_*`,
    NOT the docstring's `MSE_*`.
    """
    col = "predicted_L_ATAC_ls1.0_sigma0.1_error_var_ratio_train"
    expected = np.array([
        0.00034634550471742336, 0.0005124712763097726, 0.00032549019808296664,
        0.0002668144306942424, 0.0002366698977873915,
    ])
    actual = adata_after_try_models.var[col].to_numpy()[:5]
    npt.assert_allclose(actual, expected, rtol=1e-4, atol=1e-7)


def test_try_models_invalid_layer_raises(synthetic_adata):
    with pytest.raises(AssertionError):
        scEcho.try_models.try_models(
            synthetic_adata, ls_vals=[1.0], sigmas=[0.1],
            layer="not_a_layer",
        )


# ── read_test_results ────────────────────────────────────────────────────────

def test_smoke_read_test_results_parses_columns(adata_after_try_models):
    res = scEcho.try_models.read_test_results(adata_after_try_models, layer="L")
    assert isinstance(res, pd.DataFrame)
    expected_cols = {"variable", "error_var_ratio", "tr_tst", "sigma", "ls", "embedding"}
    assert expected_cols.issubset(set(res.columns))


def test_correctness_read_test_results_shape_and_categories(adata_after_try_models):
    res = scEcho.try_models.read_test_results(adata_after_try_models, layer="L")
    # 2 ls × 2 sigma × 2 emb × 2 (train/test) × n_features rows
    n_features = adata_after_try_models.n_vars
    assert len(res) == 2 * 2 * 2 * 2 * n_features
    assert set(res["tr_tst"].unique()) == {"train", "test"}
    assert set(res["ls"].unique()) == {1.0, 2.0}
    assert set(res["sigma"].unique()) == {0.1, 0.5}
    assert set(res["embedding"].unique()) == {"RNA", "ATAC"}


# ── plot_model_heatmap ───────────────────────────────────────────────────────

def test_smoke_plot_model_heatmap(adata_after_try_models):
    plt.close("all")
    res = scEcho.try_models.read_test_results(adata_after_try_models, layer="L")
    ax = scEcho.try_models.plot_model_heatmap(res, embedding="RNA", tr_tst="test")
    assert isinstance(ax, Axes)


def test_correctness_plot_model_heatmap_axis_labels(adata_after_try_models):
    plt.close("all")
    res = scEcho.try_models.read_test_results(adata_after_try_models, layer="L")
    ax = scEcho.try_models.plot_model_heatmap(res, embedding="RNA", tr_tst="train")
    assert ax.get_xlabel() == "sigma"
    assert ax.get_ylabel() == "ls"
    assert "RNA" in ax.get_title()


def test_plot_model_heatmap_unknown_embedding_raises(adata_after_try_models):
    res = scEcho.try_models.read_test_results(adata_after_try_models, layer="L")
    with pytest.raises(AssertionError):
        scEcho.try_models.plot_model_heatmap(res, embedding="NOT_AN_EMBEDDING")
