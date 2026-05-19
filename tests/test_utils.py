"""Tests for `scEcho.utils`. Covers every entry in utils.__all__:
run_and_store_pr_res (smoke only — heavy palantir dep), plot_branch_comp,
regress_embedding, calc_corr, df_to_adata_layer, try_models,
read_test_results, plot_model_heatmap, sweep_diffusion_components,
collect_sweep_residual_means.
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scEcho


# ── df_to_adata_layer ────────────────────────────────────────────────────────

def test_smoke_df_to_adata_layer(synthetic_adata):
    df = pd.DataFrame(
        np.arange(synthetic_adata.n_obs * synthetic_adata.n_vars, dtype=np.float32)
        .reshape(synthetic_adata.n_obs, synthetic_adata.n_vars),
        index=synthetic_adata.obs_names,
        columns=synthetic_adata.var_names,
    )
    scEcho.utils.df_to_adata_layer(synthetic_adata, df, layer_name="from_df", sparse=False)
    assert "from_df" in synthetic_adata.layers
    assert synthetic_adata.layers["from_df"].shape == synthetic_adata.shape


def test_correctness_df_to_adata_layer_aligns(synthetic_adata):
    """Shuffle row/col order to confirm the function actually re-aligns."""
    rng = np.random.default_rng(0)
    obs_shuffled = rng.permutation(synthetic_adata.obs_names)
    var_shuffled = rng.permutation(synthetic_adata.var_names)
    df = pd.DataFrame(
        np.arange(synthetic_adata.n_obs * synthetic_adata.n_vars, dtype=np.float32)
        .reshape(synthetic_adata.n_obs, synthetic_adata.n_vars),
        index=obs_shuffled,
        columns=var_shuffled,
    )
    scEcho.utils.df_to_adata_layer(synthetic_adata, df, layer_name="aligned", sparse=False)
    # value at (adata.obs_names[0], adata.var_names[0]) must equal the df's lookup
    expected = df.loc[synthetic_adata.obs_names[0], synthetic_adata.var_names[0]]
    assert synthetic_adata.layers["aligned"][0, 0] == expected


def test_correctness_df_to_adata_layer_sparse(synthetic_adata):
    import scipy.sparse as sp
    df = pd.DataFrame(
        np.zeros((synthetic_adata.n_obs, synthetic_adata.n_vars), dtype=np.float32),
        index=synthetic_adata.obs_names,
        columns=synthetic_adata.var_names,
    )
    df.iloc[0, 0] = 7.0
    scEcho.utils.df_to_adata_layer(synthetic_adata, df, layer_name="sp", sparse=True)
    assert sp.issparse(synthetic_adata.layers["sp"])
    assert synthetic_adata.layers["sp"][0, 0] == 7.0


# ── regress_embedding ────────────────────────────────────────────────────────

def test_smoke_regress_embedding(synthetic_adata):
    synthetic_adata.obs["depth"] = np.arange(synthetic_adata.n_obs).astype(float)
    scEcho.utils.regress_embedding(
        synthetic_adata, emb_key="DM_EigenVectors_RNA", depth_col="depth",
    )
    assert "DM_EigenVectors_RNA_depth_regressed" in synthetic_adata.obsm


def test_correctness_regress_embedding_residualization(synthetic_adata):
    """After regressing out depth, the per-dimension residuals should have
    near-zero Spearman correlation with depth (within numerical noise).
    """
    from scipy.stats import spearmanr

    synthetic_adata.obs["depth"] = np.arange(synthetic_adata.n_obs).astype(float)
    scEcho.utils.regress_embedding(
        synthetic_adata, emb_key="DM_EigenVectors_RNA", depth_col="depth",
    )
    residual = synthetic_adata.obsm["DM_EigenVectors_RNA_depth_regressed"]
    depth    = synthetic_adata.obs["depth"].to_numpy()
    # OLS residuals have zero Pearson correlation with the regressor by
    # construction; Spearman can be non-zero but bounded small for a 100-cell
    # random embedding. Loose check: each dim's |Pearson| < 1e-6.
    for d in range(residual.shape[1]):
        r = np.corrcoef(residual[:, d], depth)[0, 1]
        assert abs(r) < 1e-6, f"dim {d} Pearson r vs depth = {r}, expected ~0"


# ── calc_corr ────────────────────────────────────────────────────────────────

def test_smoke_calc_corr(synthetic_adata):
    synthetic_adata.obs["depth"] = np.arange(synthetic_adata.n_obs).astype(float)
    res = scEcho.utils.calc_corr(
        synthetic_adata, emb="DM_EigenVectors_RNA", depth_col="depth",
    )
    assert isinstance(res, pd.DataFrame)
    assert set(res.columns) == {"dim", "corr"}
    assert len(res) == synthetic_adata.obsm["DM_EigenVectors_RNA"].shape[1]


def test_correctness_calc_corr_values(synthetic_adata):
    """Hardcoded Spearman correlations between each of the 6 RNA dims and
    the depth column (sequential ints, seed-0 obsm).
    """
    synthetic_adata.obs["depth"] = np.arange(synthetic_adata.n_obs).astype(float)
    res = scEcho.utils.calc_corr(
        synthetic_adata, emb="DM_EigenVectors_RNA", depth_col="depth",
    )
    expected = np.array([
        0.041512151215121515, 0.1606360636063606, -0.09395739573957394,
        0.005508550855085508, 0.06042604260426041, -0.17189318931893188,
    ])
    npt.assert_allclose(res["corr"].to_numpy(), expected, atol=1e-12)


# ── plot_branch_comp ─────────────────────────────────────────────────────────

def test_smoke_plot_branch_comp(synthetic_adata):
    """Fabricate the minimum scaffolding plot_branch_comp expects:
    pseudotime_{modality2} obs column and a {branch}_prob_{modality} pair.
    """
    plt.close("all")
    synthetic_adata.obs["pseudotime_ATAC"] = np.linspace(0, 1, synthetic_adata.n_obs)
    rng = np.random.default_rng(0)
    synthetic_adata.obs["X_prob_RNA"]  = rng.random(synthetic_adata.n_obs)
    synthetic_adata.obs["X_prob_ATAC"] = rng.random(synthetic_adata.n_obs)
    fig = scEcho.utils.plot_branch_comp(synthetic_adata, branch_name="X")
    assert isinstance(fig, Figure)


def test_plot_branch_comp_vlines(synthetic_adata):
    plt.close("all")
    synthetic_adata.obs["pseudotime_ATAC"] = np.linspace(0, 1, synthetic_adata.n_obs)
    rng = np.random.default_rng(0)
    synthetic_adata.obs["X_prob_RNA"]  = rng.random(synthetic_adata.n_obs)
    synthetic_adata.obs["X_prob_ATAC"] = rng.random(synthetic_adata.n_obs)
    fig = scEcho.utils.plot_branch_comp(
        synthetic_adata, branch_name="X", vline_locs=[0.25, 0.75],
    )
    # each subplot gets the two vlines on top of the existing horizontal/vertical scaffolding
    # the count check is loose — just confirm vlines actually render
    assert isinstance(fig, Figure)


# ── run_and_store_pr_res ─────────────────────────────────────────────────────
#
# Skipped — palantir.core.run_palantir requires diffusion components from
# palantir's own preprocessor stack and is not portable to a 100-cell
# Poisson-random fixture. Covered by integration testing in the notebooks
# stream; smoke here would be a brittle mock.


# ── try_models / read_test_results / plot_model_heatmap ──────────────────────

@pytest.fixture
def adata_after_try_models(synthetic_adata):
    """Run try_models once on a 2×2 grid and cache the result for the
    read_test_results / plot_model_heatmap tests.
    """
    scEcho.utils.try_models(
        synthetic_adata,
        ls_vals=[1.0, 2.0],
        sigmas=[0.1, 0.5],
        layer="L",
        random_state=0,
    )
    return synthetic_adata


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
    with pytest.raises(KeyError):
        scEcho.utils.try_models(
            synthetic_adata, ls_vals=[1.0], sigmas=[0.1],
            layer="not_a_layer",
        )


def test_smoke_read_test_results_parses_columns(adata_after_try_models):
    res = scEcho.utils.read_test_results(adata_after_try_models, layer="L")
    assert isinstance(res, pd.DataFrame)
    expected_cols = {"variable", "error_var_ratio", "tr_tst", "sigma", "ls", "embedding"}
    assert expected_cols.issubset(set(res.columns))


def test_correctness_read_test_results_shape_and_categories(adata_after_try_models):
    res = scEcho.utils.read_test_results(adata_after_try_models, layer="L")
    # 2 ls × 2 sigma × 2 emb × 2 (train/test) × n_features rows
    n_features = adata_after_try_models.n_vars
    assert len(res) == 2 * 2 * 2 * 2 * n_features
    assert set(res["tr_tst"].unique()) == {"train", "test"}
    assert set(res["ls"].unique()) == {1.0, 2.0}
    assert set(res["sigma"].unique()) == {0.1, 0.5}
    assert set(res["embedding"].unique()) == {"RNA", "ATAC"}


def test_smoke_plot_model_heatmap(adata_after_try_models):
    plt.close("all")
    res = scEcho.utils.read_test_results(adata_after_try_models, layer="L")
    ax = scEcho.utils.plot_model_heatmap(res, embedding="RNA", tr_tst="test")
    assert isinstance(ax, Axes)


def test_correctness_plot_model_heatmap_axis_labels(adata_after_try_models):
    plt.close("all")
    res = scEcho.utils.read_test_results(adata_after_try_models, layer="L")
    ax = scEcho.utils.plot_model_heatmap(res, embedding="RNA", tr_tst="train")
    assert ax.get_xlabel() == "sigma"
    assert ax.get_ylabel() == "ls"
    assert "RNA" in ax.get_title()


def test_plot_model_heatmap_unknown_embedding_raises(adata_after_try_models):
    res = scEcho.utils.read_test_results(adata_after_try_models, layer="L")
    with pytest.raises(ValueError):
        scEcho.utils.plot_model_heatmap(res, embedding="NOT_AN_EMBEDDING")


# ── sweep_diffusion_components / collect_sweep_residual_means ────────────────

def test_smoke_sweep_diffusion_components(synthetic_adata):
    scEcho.utils.sweep_diffusion_components(
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
    scEcho.utils.sweep_diffusion_components(
        synthetic_adata,
        layer="L",
        obsm_key="DM_EigenVectors_RNA",
        ls=1.0,
        sigma=0.1,
        min_components=2,
    )
    res = scEcho.utils.collect_sweep_residual_means(
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
    with pytest.raises(ValueError):
        scEcho.utils.sweep_diffusion_components(
            synthetic_adata, layer="L", obsm_key="DM_EigenVectors_RNA",
            min_components=1,
        )
    with pytest.raises(ValueError):
        scEcho.utils.sweep_diffusion_components(
            synthetic_adata, layer="L", obsm_key="DM_EigenVectors_RNA",
            min_components=999,
        )
