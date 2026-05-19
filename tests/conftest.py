import matplotlib

matplotlib.use("Agg")  # headless backend so plotting tests work in CI

import anndata as ad
import numpy as np
import pytest


@pytest.fixture
def synthetic_adata():
    """Small (100 cells x 30 features) synthetic AnnData with the layout
    scEcho's public functions expect:
      - .X / .layers["L"]
      - .obsm["DM_EigenVectors_RNA" | "DM_EigenVectors_ATAC"]
      - .obs["combo_type"] categorical
    """
    np.random.seed(0)
    n = 100
    n_features = 30
    X = np.random.poisson(0.5, size=(n, n_features)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.layers["L"] = X.copy()
    adata.obsm["DM_EigenVectors_RNA"]  = np.random.randn(n, 6).astype(np.float32)
    adata.obsm["DM_EigenVectors_ATAC"] = np.random.randn(n, 6).astype(np.float32)
    adata.obs["combo_type"] = np.random.choice(["A", "B"], size=n)
    return adata
