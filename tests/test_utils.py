import numpy as np
import pandas as pd

import scEcho
from scEcho.Echo_features import compute_ncells


def test_df_to_adata_layer_aligns_and_writes(synthetic_adata):
    ad = synthetic_adata
    # shuffle row + col order to confirm the function actually re-aligns
    obs_shuffled = np.random.RandomState(0).permutation(ad.obs_names)
    var_shuffled = np.random.RandomState(1).permutation(ad.var_names)

    df = pd.DataFrame(
        np.arange(ad.n_obs * ad.n_vars, dtype=np.float32).reshape(ad.n_obs, ad.n_vars),
        index=obs_shuffled,
        columns=var_shuffled,
    )
    scEcho.utils.df_to_adata_layer(ad, df, layer_name="aligned", sparse=False)

    assert "aligned" in ad.layers
    assert ad.layers["aligned"].shape == ad.shape
    # value at obs[0], var[0] should equal df's lookup for those names
    expected = df.loc[ad.obs_names[0], ad.var_names[0]]
    assert ad.layers["aligned"][0, 0] == expected


def test_compute_ncells_non_negative_layer(synthetic_adata):
    """compute_ncells is the small Echo_features utility flagged in the audit;
    smoke-test the non-negative-layer branch where it writes per-feature counts.
    """
    res = {}
    compute_ncells(synthetic_adata, layer_key="L", col_name="ncells_L", res=res)
    ncells = res["ncells_L"]
    assert ncells.shape == (synthetic_adata.n_vars,)
    # poisson(0.5) draws produce some positive entries per feature
    assert (ncells >= 0).all()
    assert (ncells <= synthetic_adata.n_obs).all()
