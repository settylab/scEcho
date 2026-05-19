from matplotlib.axes import Axes

import scEcho


def test_plot_scores_non_interactive_returns_axes(synthetic_adata):
    scEcho.Echo_features.run_echo_features(
        synthetic_adata,
        obs_col="combo_type",
        layers=["L"],
        sigma=0.1,
        ls=1.0,
        min_cells=10,
        verbose=False,
    )

    c = synthetic_adata.obs["combo_type"].unique()[0]
    ax = scEcho.plotting.plot_scores(
        synthetic_adata,
        obs_col="combo_type",
        c=c,
        layer="L",
        n_features_label=2,
        interactive=False,
    )

    assert isinstance(ax, Axes), f"expected matplotlib Axes, got {type(ax)}"


def test_plot_scores_interactive_runs(synthetic_adata, monkeypatch):
    """Interactive path currently falls through without an explicit return
    (tracked in settylab/scEcho#1). This smoke test asserts it runs cleanly
    end-to-end and doesn't raise. When the bug is fixed and the function
    returns a plotly Figure, the assertion below can be tightened.
    """
    scEcho.Echo_features.run_echo_features(
        synthetic_adata,
        obs_col="combo_type",
        layers=["L"],
        sigma=0.1,
        ls=1.0,
        min_cells=10,
        verbose=False,
    )

    # Suppress fig.show() — plotly opens a browser window otherwise.
    import plotly.graph_objects as go

    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)

    c = synthetic_adata.obs["combo_type"].unique()[0]
    result = scEcho.plotting.plot_scores(
        synthetic_adata,
        obs_col="combo_type",
        c=c,
        layer="L",
        n_features_label=2,
        interactive=True,
    )

    # Currently returns None (see settylab/scEcho#1). Once fixed, tighten
    # to: assert isinstance(result, go.Figure).
    assert result is None or hasattr(result, "show")
