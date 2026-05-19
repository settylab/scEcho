import scEcho


def test_try_models_writes_error_var_ratio_columns(synthetic_adata):
    scEcho.try_models.try_models(
        synthetic_adata,
        ls_vals=[1.0, 2.0],
        sigmas=[0.1, 0.5],
        layer="L",
        random_state=0,
    )

    # try_models.py:193–194 writes _error_var_ratio_train / _test (the
    # docstring's MSE_train/test names are wrong — see audit punch list 9).
    train_cols = [c for c in synthetic_adata.var.columns if c.endswith("_error_var_ratio_train")]
    test_cols  = [c for c in synthetic_adata.var.columns if c.endswith("_error_var_ratio_test")]

    # 2 ls × 2 sigma × 2 embeddings = 8 combinations
    assert len(train_cols) == 8, f"expected 8 train columns, got {len(train_cols)}: {train_cols}"
    assert len(test_cols)  == 8, f"expected 8 test  columns, got {len(test_cols)}: {test_cols}"

    assert "tr_tst" in synthetic_adata.obs.columns
