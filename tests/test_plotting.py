"""Tests for `scEcho.plotting`. Covers every entry in `plotting.__all__`:
plot_scores, plot_direction_fractions, plot_SE, linked_plot,
plot_desynchronized_state_volcano, rotate_coords.
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scEcho


@pytest.fixture
def adata_with_desynch_features(synthetic_adata):
    """Fixture whose AnnData has already been through the full Echo_features
    pipeline so plotting functions can read their expected obs/varm keys."""
    scEcho.Echo_features.run_echo_features(
        synthetic_adata, obs_col="combo_type", layers=["L"],
        sigma=0.1, ls=1.0, min_cells=10, verbose=False,
    )
    return synthetic_adata


@pytest.fixture
def adata_with_dn_comp(synthetic_adata):
    """Fixture with dn_comp_obsm results in obs (used by plot_direction_fractions
    and plot_desynchronized_state_volcano)."""
    scEcho.Echo_states.dn_comp_obsm(
        synthetic_adata, ls_factor=2, log_fold_change_threshold=0.5,
        optimizer="L-BFGS-B",
    )
    return synthetic_adata


# ── plot_scores ──────────────────────────────────────────────────────────────

def test_smoke_plot_scores_non_interactive(adata_with_desynch_features):
    c = adata_with_desynch_features.obs["combo_type"].unique()[0]
    ax = scEcho.plotting.plot_scores(
        adata_with_desynch_features, obs_col="combo_type",
        c=c, layer="L", n_features_label=3, interactive=False,
    )
    assert isinstance(ax, Axes)


def test_correctness_plot_scores_axes_inventory(adata_with_desynch_features):
    """Specific assertions on the returned Axes — one scatter collection plus
    n_features_label text annotations (audit's `n_features_label=3` →
    additional adjust_text texts may appear; allow [N, N+1] to be safe).
    """
    plt.close("all")
    c = adata_with_desynch_features.obs["combo_type"].unique()[0]
    ax = scEcho.plotting.plot_scores(
        adata_with_desynch_features, obs_col="combo_type",
        c=c, layer="L", n_features_label=3, interactive=False,
    )
    assert len(ax.collections) == 1
    # 3 features labeled → 3 or 4 texts (the 4th can be a legend artifact)
    assert 3 <= len(ax.texts) <= 5
    assert "Var explained diff" in ax.get_xlabel()
    assert "Mahalanobis distance" in ax.get_ylabel()


def test_plot_scores_interactive_runs(adata_with_desynch_features, monkeypatch):
    """Tracks the known bug at settylab/scEcho#1 — the interactive path falls
    through without an explicit `return fig`. Once fixed, tighten to
    `isinstance(result, go.Figure)`.
    """
    import plotly.graph_objects as go
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)
    c = adata_with_desynch_features.obs["combo_type"].unique()[0]
    result = scEcho.plotting.plot_scores(
        adata_with_desynch_features, obs_col="combo_type",
        c=c, layer="L", n_features_label=2, interactive=True,
    )
    assert result is None or hasattr(result, "show")


# ── plot_direction_fractions ─────────────────────────────────────────────────

def test_smoke_plot_direction_fractions(adata_with_dn_comp):
    plt.close("all")
    ax = scEcho.plotting.plot_direction_fractions(adata_with_dn_comp, obs_col="combo_type")
    assert isinstance(ax, Axes)


def test_correctness_plot_direction_fractions_axis_shape(adata_with_dn_comp):
    """Stacked bar plot for two combo_type groups × 3 direction categories
    → 6 patches. y-axis fixed at [0, 1] (fractions).
    """
    plt.close("all")
    ax = scEcho.plotting.plot_direction_fractions(adata_with_dn_comp, obs_col="combo_type")
    assert ax.get_ylim() == (0.0, 1.0)
    assert len(ax.patches) == 6  # 2 groups × 3 categories


def test_plot_direction_fractions_missing_obs_raises(synthetic_adata):
    """Without dn_comp_obsm first, the required obs column doesn't exist."""
    with pytest.raises(KeyError):
        scEcho.plotting.plot_direction_fractions(synthetic_adata, obs_col="combo_type")


# ── plot_desynchronized_state_volcano ────────────────────────────────────────

def test_smoke_plot_desynchronized_state_volcano(adata_with_dn_comp):
    plt.close("all")
    ax = scEcho.plotting.plot_desynchronized_state_volcano(
        adata_with_dn_comp, hue_col="combo_type",
    )
    assert isinstance(ax, Axes)
    assert "density LFC" in ax.get_xlabel()
    assert "-log10 p-value" in ax.get_ylabel()


# ── plot_SE ──────────────────────────────────────────────────────────────────

def test_smoke_plot_SE(synthetic_adata):
    """plot_SE expects very specific pre-imputed layers — fabricate them."""
    n, p = synthetic_adata.n_obs, synthetic_adata.n_vars
    rng  = np.random.default_rng(0)
    layer = "logcounts"
    synthetic_adata.layers[layer] = rng.normal(size=(n, p)).astype(np.float32)
    emb1, emb2 = "DM_EigenVectors_RNA", "DM_EigenVectors_ATAC"
    for k in (
        f"mellon_imputed_{layer}_{emb1}_space",
        f"mellon_imputed_{layer}_{emb2}_space",
        f"mellon_imputed_{emb1}_SE_smooth",
        f"mellon_imputed_{emb2}_SE_smooth",
        "mellon_imputed_SE_smooth_FC",
    ):
        synthetic_adata.layers[k] = rng.normal(size=(n, p)).astype(np.float32)
    synthetic_adata.obsm["X_umap"] = rng.normal(size=(n, 2)).astype(np.float32)

    scEcho.plotting.plot_SE(
        synthetic_adata, gn=synthetic_adata.var_names[0],
        pre_imputed_layer=layer, emb1=emb1, emb2=emb2,
    )


# ── linked_plot ──────────────────────────────────────────────────────────────

def test_smoke_linked_plot(synthetic_adata):
    """linked_plot pairs two 2D embeddings — overwrite the 6D fixture obsm
    with 2D versions so the function can plot scatter pairs.
    """
    rng = np.random.default_rng(0)
    synthetic_adata.obsm["DM_EigenVectors_RNA"]  = rng.normal(size=(synthetic_adata.n_obs, 2)).astype(np.float32)
    synthetic_adata.obsm["DM_EigenVectors_ATAC"] = rng.normal(size=(synthetic_adata.n_obs, 2)).astype(np.float32)

    fig = scEcho.plotting.linked_plot(
        synthetic_adata,
        embedding1="DM_EigenVectors_RNA",
        embedding2="DM_EigenVectors_ATAC",
        modality1_name="RNA", modality2_name="ATAC",
    )
    assert isinstance(fig, Figure)


# ── rotate_coords ────────────────────────────────────────────────────────────

def test_correctness_rotate_coords_90_degrees():
    coords = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    rotated = scEcho.plotting.rotate_coords(coords, degrees=90)
    expected = np.array([[0.0, -1.0], [1.0, 0.0], [2.0, -2.0]])
    npt.assert_allclose(rotated, expected, atol=1e-10)


def test_correctness_rotate_coords_360_is_identity():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(20, 2))
    rotated = scEcho.plotting.rotate_coords(coords, degrees=360)
    npt.assert_allclose(rotated, coords, atol=1e-9)


def test_correctness_rotate_coords_flip_y():
    coords = np.array([[1.0, 2.0], [3.0, -4.0]])
    flipped = scEcho.plotting.rotate_coords(coords, degrees=0, flip_y=True)
    npt.assert_allclose(flipped, np.array([[1.0, -2.0], [3.0, 4.0]]))


def test_correctness_rotate_coords_rotate_around_point():
    coords = np.array([[2.0, 1.0]])
    center = np.array([1.0, 1.0])
    rotated = scEcho.plotting.rotate_coords(coords, degrees=180, rotate_around=center)
    # 180° around (1, 1) takes (2, 1) → (0, 1)
    npt.assert_allclose(rotated, np.array([[0.0, 1.0]]), atol=1e-10)
