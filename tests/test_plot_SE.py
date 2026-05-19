"""Smoke test for plotting.plot_SE.

plot_SE expects very specific pre-imputed layers (e.g.
mellon_imputed_{layer}_{emb}_space, mellon_imputed_{emb}_SE_smooth,
mellon_imputed_SE_smooth_FC) plus an embedding in .obsm. We fabricate
the minimum scaffolding needed for the function to enter and exit
without raising.
"""
import numpy as np
import pytest

import scEcho


def test_plot_SE_runs(synthetic_adata):
    ad = synthetic_adata
    n, p = ad.n_obs, ad.n_vars
    rng  = np.random.default_rng(0)

    layer = "logcounts"
    ad.layers[layer] = rng.normal(size=(n, p)).astype(np.float32)

    emb1 = "DM_EigenVectors_RNA"
    emb2 = "DM_EigenVectors_ATAC"

    # pre-imputed layers
    ad.layers[f"mellon_imputed_{layer}_{emb1}_space"]    = rng.normal(size=(n, p)).astype(np.float32)
    ad.layers[f"mellon_imputed_{layer}_{emb2}_space"]    = rng.normal(size=(n, p)).astype(np.float32)
    ad.layers[f"mellon_imputed_{emb1}_SE_smooth"]         = rng.normal(size=(n, p)).astype(np.float32)
    ad.layers[f"mellon_imputed_{emb2}_SE_smooth"]         = rng.normal(size=(n, p)).astype(np.float32)
    ad.layers["mellon_imputed_SE_smooth_FC"]              = rng.normal(size=(n, p)).astype(np.float32)

    # 2D embedding under the basis name the function expects
    ad.obsm["X_umap"] = rng.normal(size=(n, 2)).astype(np.float32)

    gn = ad.var_names[0]

    # scanpy emits a few categorical-related warnings on synthetic data; just
    # confirm the function call doesn't raise.
    try:
        scEcho.plotting.plot_SE(ad, gn=gn, pre_imputed_layer=layer, emb1=emb1, emb2=emb2)
    except Exception as e:  # pragma: no cover
        pytest.fail(f"plot_SE raised unexpectedly: {e!r}")
