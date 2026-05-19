import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import scanpy as sc
import scipy.sparse as sparse
import warnings



def plot_scores(
    ad,
    obs_col,
    c,
    layer,
    modality1="RNA",
    modality2="ATAC",
    ncells_cutoff=20,
    ncells_layer=None,
    interactive=False,
    s=5,
    features_label=None,
    features_highlight=None,
    n_features_label=10,
    highlight_s_scaling=5,
    expand=(2.0, 2.0),
    force_text=(1.5, 1.5),
    force_points=(0.5, 0.5),
    max_move_frac=0.1,
    iter_lim=1000,
    **adjust_text_kwargs,
):
    # ── Resolve column names ───────────────────────────────────────────────────

    if ncells_layer is None:
        ncells_layer = layer

    ncells_col    = f"ncells_{ncells_layer}_{obs_col}_{c}"
    var_exp_col   = f"var_explained_diff_{layer}_{obs_col}_{c}"
    MHD_col       = f"MHD_{obs_col}_{c}_{modality1}_vs_{modality2}"
    direction_col = f"var_explained_diff_{layer}_{obs_col}_{c}_direction"
    colors_key    = f"desynch_direction_{layer}_{modality1}_v_{modality2}_colors"

    # ── Pull results from varm ────────────────────────────────────────────────

    res = ad.varm[f"reconstruction_results_{layer}"]

    # ── Apply ncells filter ───────────────────────────────────────────────────

    if ncells_col not in res.columns:
        warnings.warn(
            f"'{ncells_col}' not found in reconstruction_results_{layer} — skipping ncells filter. "
            f"Run get_desynch_stats with layer='{ncells_layer}' or include it in extra_ncells_layers."
        )
        plot_df = res
    elif res[ncells_col].isna().all():
        warnings.warn(
            f"'{ncells_col}' is all NaN — likely because layer '{ncells_layer}' contains negative values. "
            f"ncells filter will be skipped."
        )
        plot_df = res
    else:
        plot_df = res[res[ncells_col] > ncells_cutoff]

    # ── Clean up non-finite values ────────────────────────────────────────────

    n_before  = len(plot_df)
    plot_df   = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[var_exp_col, MHD_col])
    n_dropped = n_before - len(plot_df)

    if n_dropped > 0:
        warnings.warn(
            f"{n_dropped} features dropped due to NaN or infinite values in "
            f"'{var_exp_col}' or '{MHD_col}'."
        )

    # ── Resolve which features to label ──────────────────────────────────────

    if features_highlight is not None:
        if isinstance(features_highlight, str):
            features_highlight = [features_highlight]
        if features_label is not None:
            warnings.warn("Both features_highlight and features_label were provided — features_label will be ignored.")
        if n_features_label is not None:
            warnings.warn("Both features_highlight and n_features_label were provided — n_features_label will be ignored.")
        label_features = features_highlight

    elif n_features_label is not None:
        if features_label is not None:
            warnings.warn("Both n_features_label and features_label were provided — features_label will be ignored.")
        plot_df["_rank"] = (
            plot_df[var_exp_col].abs().rank() + plot_df[MHD_col].rank()
        )
        label_features = plot_df["_rank"].nlargest(n_features_label).index.tolist()
        plot_df.drop(columns=["_rank"], inplace=True)

    elif features_label is not None:
        if isinstance(features_label, str):
            features_label = [features_label]
        label_features = features_label

    else:
        label_features = None

    # ── Resolve direction coloring ────────────────────────────────────────────

    if direction_col in plot_df.columns:
        colors       = ad.uns[colors_key] if colors_key in ad.uns else ["#ff7f0e", "#1f77b4", "lightgrey"]
        categories   = plot_df[direction_col].cat.categories
        color_map    = dict(zip(categories, colors))
        point_colors = plot_df[direction_col].map(color_map)
    else:
        warnings.warn(
            f"'{direction_col}' not found — plotting without direction coloring. "
            f"Run run_null_desynch_test first to enable coloring."
        )
        color_map    = None
        point_colors = "lightgrey"

    # ── Plot ──────────────────────────────────────────────────────────────────

    if interactive:
        if direction_col in plot_df.columns:
            fig = px.scatter(
                plot_df,
                x=var_exp_col,
                y=MHD_col,
                hover_name="feature",
                color=direction_col,
                color_discrete_map=color_map,
            )
        else:
            fig = px.scatter(
                plot_df,
                x=var_exp_col,
                y=MHD_col,
                hover_name="feature",
            )
        fig.update_layout(autosize=False, width=1000, height=800, legend_title=None)
        fig.show()

    else:
        ax = sns.scatterplot(
            plot_df,
            x=var_exp_col,
            y=MHD_col,
            hue=direction_col if direction_col in plot_df.columns else None,
            palette=color_map,
            s=s,
            linewidth=0,
        )
        ax.set_xlabel(f"Var explained diff ({modality1} - {modality2})\n{obs_col}: {c}")
        ax.set_ylabel(f"Mahalanobis distance\n{obs_col}: {c}")
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title=None)

        # ── Highlight and label selected features with repulsion ──────────────

        if label_features is not None:
            label_df = plot_df.loc[plot_df.index.isin(label_features)]

            if features_highlight is not None:
                sns.scatterplot(
                    label_df,
                    x=var_exp_col,
                    y=MHD_col,
                    s=s * highlight_s_scaling,
                    linewidth=0,
                    ax=ax,
                    color="black",
                    zorder=5,
                )

            texts = [
                ax.text(
                    row[var_exp_col],
                    row[MHD_col],
                    feature,
                    fontsize=8,
                )
                for feature, row in label_df.iterrows()
            ]

            # scale max_move to data range so labels don't drift too far
            x_range  = plot_df[var_exp_col].abs().max()
            y_range  = plot_df[MHD_col].abs().max()
            max_move = float(np.mean([x_range, y_range]) * max_move_frac)

            adjust_text(
                texts,
                x=plot_df[var_exp_col].values,
                y=plot_df[MHD_col].values,
                ax=ax,
                arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
                expand=expand,
                force_text=force_text,
                force_points=force_points,
                max_move=max_move,
                iter_lim=iter_lim,
                **adjust_text_kwargs,
            )

        return ax
    
    
    
    

def plot_direction_fractions(
    ad,
    obs_col,
    modality1_name="RNA",
    modality2_name="ATAC",
    figsize=None,
    ax=None,
):
    
    direction_col   = f"direction_{modality1_name}_v_{modality2_name}"
    colors_key      = f"{direction_col}_colors"

    # ── Validate inputs ───────────────────────────────────────────────────────

    if direction_col not in ad.obs.columns:
        raise KeyError(
            f"'{direction_col}' not found in ad.obs. Run dn_comp_obsm first."
        )
    if colors_key not in ad.uns:
        raise KeyError(
            f"'{colors_key}' not found in ad.uns. Run dn_comp_obsm first."
        )

    # ── Compute fractions ─────────────────────────────────────────────────────

    # use the category order stored by dn_comp_obsm
    categories = ad.obs[direction_col].cat.categories
    colors     = ad.uns[colors_key]

    counts = (
        ad.obs.groupby(obs_col, observed=True)[direction_col]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=categories, fill_value=0)  # preserve category order
    )
    fractions = counts.div(counts.sum(axis=1), axis=0)

    # ── Plot ──────────────────────────────────────────────────────────────────

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (len(fractions) * 0.6 + 1, 4))

    fractions.plot(
        kind="bar",
        stacked=True,
        color=colors,
        ax=ax,
        width=0.8,
    )

    ax.set_xlabel(obs_col)
    ax.set_ylabel("Fraction of cells")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(
        title=direction_col,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    return ax

         
    
    
    

            
def plot_SE(ad,
            gn,
            prc_clip = 100,
            b = "umap",
            prc_clip_FC = 95,
            so = False,
            pre_imputed_layer = "logcounts",
            emb1 = "DM_EigenVectors_RNA",
            emb2 = "DM_EigenVectors_ATAC"):
    
    
    
    fig, axs = plt.subplots(3, 2, figsize=(12, 17))
    
    imp_rna = f"mellon_imputed_{pre_imputed_layer}_{emb1}_space"
    imp_atac = f"mellon_imputed_{pre_imputed_layer}_{emb2}_space"
    
    imp_rna_SE = f"mellon_imputed_{emb1}_SE_smooth" #f"mellon_imputed_{pre_imputed_layer}_{emb1}_SE_smooth"
    imp_atac_SE = f"mellon_imputed_{emb2}_SE_smooth" #f"mellon_imputed_{pre_imputed_layer}_{emb2}_SE_smooth"
    
    # set scales
    
    maxv_val = max(np.percentile(ad[:,gn].layers[imp_rna], prc_clip), 
             np.percentile(ad[:,gn].layers[imp_atac], prc_clip))
    minv_val = min(np.percentile(ad[:,gn].layers[imp_rna], 100-prc_clip), 
             np.percentile(ad[:,gn].layers[imp_atac], 100-prc_clip))
    
    
    maxv = max(np.percentile(ad[:,gn].layers[imp_rna_SE], prc_clip), 
             np.percentile(ad[:,gn].layers[imp_atac_SE], prc_clip))
    minv = min(np.percentile(ad[:,gn].layers[imp_rna_SE], 100-prc_clip), 
             np.percentile(ad[:,gn].layers[imp_atac_SE], 100-prc_clip))
    
    sc.pl.embedding(ad, 
                    color=gn, 
                    basis = b, 
                    use_raw=False,
                    title=f"{gn}\n{pre_imputed_layer}", 
                    vmin = f"p{100 - prc_clip}",
                    vmax = f"p{prc_clip}",
                    ax = axs[0,0], 
                    layer = pre_imputed_layer,
                    show=False, 
                    sort_order = so)
    
    sc.pl.embedding(ad, 
                    color=gn, 
                    basis = b, 
                    layer = imp_rna, 
                    title=f"{gn}\nRNA imputed {pre_imputed_layer}", 
                    vmin = maxv_val,
                    vmax = minv_val,
                    ax = axs[1,0], 
                    show=False, 
                    sort_order = so)
    
    
    
    
    
    sc.pl.embedding(ad, 
                    color=gn, 
                    basis = b, 
                    layer = imp_rna_SE, 
                    title=f"{gn}\nRNA SE smoothed", 
                    vmin = minv,
                    vmax = maxv,
                    ax = axs[1,1], 
                    show=False, 
                    sort_order = so)
    
    sc.pl.embedding(ad, 
                    color=gn, 
                    basis = b, 
                    layer = imp_atac, 
                    title=f"{gn}\nATAC imputed {pre_imputed_layer}", 
                    vmin = maxv_val,
                    vmax = minv_val,
                    ax = axs[2,0], 
                    show=False, 
                    sort_order = so)
    
    sc.pl.embedding(ad, 
                    color=gn, 
                    basis = b, 
                    layer = imp_atac_SE, 
                    title=f"{gn}\nATAC SE smoothed", 
                    vmin = minv,
                    vmax = maxv,
                    ax = axs[2,1], 
                    show=False, 
                    sort_order = so)
    
    
    
    
    clr_extent = np.percentile(np.abs(ad[:,gn].layers["mellon_imputed_SE_smooth_FC"]), prc_clip)
    sc.pl.embedding(ad, color=gn, 
                basis = b, 
                layer ="mellon_imputed_SE_smooth_FC", 
                # vcenter = 0,
                    vmin=clr_extent, #f"p{prc_clip_FC}",
                    vmax=-clr_extent, #f"p{100 - prc_clip_FC}",
                cmap="RdBu_r",
               title=f"{gn}\n RNA SE - ATAC SE",
                   ax = axs[0,1], 
                    sort_order = so)
    
    
    



def rotate_coords(
    coords,
    degrees,
    rotate_around=np.zeros(2),
    flip_x=False,
    flip_y=False,
):
    """Rotate (and optionally flip) 2D coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (n_cells, 2).
    degrees : float
        Degrees to rotate coordinates.
    rotate_around : np.ndarray
        Point to rotate around. Defaults to origin.
    flip_x : bool
        Whether to flip the x-axis after rotation.
    flip_y : bool
        Whether to flip the y-axis after rotation.

    Returns
    -------
    np.ndarray of rotated (and optionally flipped) coordinates.
    """

    radians      = degrees * np.pi / 180
    rotation_mtx = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians),  np.cos(radians)],
    ])

    new_coords = (coords - rotate_around) @ rotation_mtx + rotate_around

    if flip_x:
        new_coords[:, 0] = -1 * new_coords[:, 0]
    if flip_y:
        new_coords[:, 1] = -1 * new_coords[:, 1]

    return new_coords


def make_lines_df(
    obj1,
    coords1,
    coords2,
    modality1_name,
    modality2_name,
    line_mask,
    line_thickness,
    downsample_lines_frac,
):
    """Build a DataFrame of line segments connecting paired cells across two embeddings.

    Parameters
    ----------
    obj1 : anndata.AnnData
        AnnData object (used for cell names).
    coords1 : np.ndarray
        Coordinates for the first modality.
    coords2 : np.ndarray
        Coordinates for the second modality.
    modality1_name : str
        Name for the first modality.
    modality2_name : str
        Name for the second modality.
    line_mask : list or array-like
        Cell names to draw lines for. Empty list draws lines for all cells.
    line_thickness : float
        Thickness of connecting lines.
    downsample_lines_frac : float
        Fraction of lines to draw (<=1). Used to avoid overplotting.

    Returns
    -------
    line_df : pd.DataFrame
        DataFrame of line segment coordinates.
    cells_highlight : pd.Index
        Cells selected by line_mask (used for highlighting in scatter plot).
    """

    line_df = pd.DataFrame(
        np.hstack((coords1, coords2)),
        index=obj1.obs_names,
        columns=[
            f"{modality1_name}_UMAP1", f"{modality1_name}_UMAP2",
            f"{modality2_name}_UMAP1", f"{modality2_name}_UMAP2",
        ],
    )
    line_df["line_thickness"] = line_thickness

    if len(line_mask) > 0:
        line_df         = line_df.loc[line_mask, :]
        cells_highlight = line_df.index
    else:
        cells_highlight = []

    if downsample_lines_frac > 1:
        warnings.warn("downsample_lines_frac must be <= 1. No downsampling will occur.")
    elif downsample_lines_frac < 1:
        line_df = line_df.sample(frac=downsample_lines_frac)

    return line_df, cells_highlight


def make_points_df(
    obj1,
    coords1,
    coords2,
    modality1_name,
    modality2_name,
    color_by,
    color_values,
    cells_highlight,
    highlight_border,
    pt_size_highlight,
    border_thickness,
    pt_size,
):
    """Build a long-form DataFrame of cell coordinates for scatter plotting.

    Parameters
    ----------
    obj1 : anndata.AnnData
        AnnData object.
    coords1 : np.ndarray
        Coordinates for the first modality.
    coords2 : np.ndarray
        Coordinates for the second modality.
    modality1_name : str
        Name for the first modality.
    modality2_name : str
        Name for the second modality.
    color_by : str
        Column name or feature name to color points by.
    color_values : pd.Series or np.ndarray
        Pre-resolved values to color by (from obs or layer).
    cells_highlight : pd.Index
        Cells to highlight.
    highlight_border : bool
        Whether to draw a border around highlighted cells.
    pt_size_highlight : float
        Point size for highlighted cells.
    border_thickness : float
        Thickness of highlighted cell borders.
    pt_size : float
        Point size for non-highlighted cells.

    Returns
    -------
    pd.DataFrame in long format.
    """

    mtx1_df             = pd.DataFrame(coords1, columns=["UMAP1", "UMAP2"])
    mtx1_df["cell"]     = obj1.obs_names
    mtx1_df["modality"] = modality1_name

    mtx2_df             = pd.DataFrame(coords2, columns=["UMAP1", "UMAP2"])
    mtx2_df["cell"]     = obj1.obs_names
    mtx2_df["modality"] = modality2_name

    if color_by != "modality":
        mtx1_df[color_by] = color_values
        mtx2_df[color_by] = color_values

    points_df = pd.concat((mtx1_df, mtx2_df), axis=0)

    points_df["highlight"]        = points_df["cell"].isin(cells_highlight)
    points_df["size"]             = np.where(points_df["highlight"], pt_size_highlight, pt_size)
    points_df["size"]             = points_df["size"].astype(float)
    points_df["border_thickness"] = border_thickness if highlight_border else 0

    return points_df


def linked_plot(
    obj1,
    embedding1,
    embedding2,
    modality1_name=None,
    modality2_name=None,
    color_by=None,
    layer=None,
    vmin=None,
    vmax=None,
    offset=None,
    figsize=(8, 12),
    pt_size=10,
    border_thickness=0,
    pt_size_highlight=None,
    highlight_border=True,
    palette=None,
    title=None,
    line_thickness=1,
    line_alpha=0.5,
    line_mask=[],
    downsample_lines_frac=0.3,
):
    """Plot two embeddings side by side with connecting lines between paired cells.

    Parameters
    ----------
    obj1 : anndata.AnnData
        AnnData object containing both embeddings.
    embedding1 : str
        Key in obj1.obsm for the first embedding.
    embedding2 : str
        Key in obj1.obsm for the second embedding.
    modality1_name : str, optional
        Display name for the first modality. Defaults to embedding1.
    modality2_name : str, optional
        Display name for the second modality. Defaults to embedding2.
    color_by : str, optional
        Column in ad.obs or feature in ad.var_names to color points by.
        Defaults to 'modality'. If in ad.var_names, layer must be specified.
    layer : str, optional
        Key in ad.layers to pull feature values from when color_by is in ad.var_names.
    offset : tuple of float, optional
        (x, y) offset to apply to the second embedding. Auto-computed if None.
    figsize : tuple
        Figure size.
    pt_size : float
        Point size for non-highlighted cells.
    border_thickness : float
        Border thickness for highlighted cells.
    pt_size_highlight : float, optional
        Point size for highlighted cells. Defaults to pt_size.
    highlight_border : bool
        Whether to draw a border around highlighted cells.
    palette : str or dict, optional
        Color palette. Defaults to 'Spectral_r' for numeric data.
    title : str, optional
        Plot title. Auto-generated if None.
    line_thickness : float
        Thickness of connecting lines.
    line_alpha : float
        Opacity of connecting lines.
    line_mask : list
        Cell names to draw lines for. Empty list draws lines for all cells.
    downsample_lines_frac : float
        Fraction of lines to draw (<=1).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    # ── Resolve modality names and defaults ───────────────────────────────────

    if modality1_name is None:
        modality1_name = embedding1
    if modality2_name is None:
        modality2_name = embedding2
    if color_by is None:
        color_by = "modality"

    # ── Validate inputs ───────────────────────────────────────────────────────

    if embedding1 not in obj1.obsm:
        raise KeyError(
            f"'{embedding1}' not found in obj1.obsm. Available keys: {list(obj1.obsm.keys())}"
        )
    if embedding2 not in obj1.obsm:
        raise KeyError(
            f"'{embedding2}' not found in obj1.obsm. Available keys: {list(obj1.obsm.keys())}"
        )

    if pt_size_highlight is None:
        pt_size_highlight = pt_size

    # ── Resolve color values ──────────────────────────────────────────────────

    in_obs      = color_by in obj1.obs.columns
    in_var      = color_by in obj1.var_names
    in_modality = color_by == "modality"

    if in_obs and in_var:
        raise ValueError(
            f"'{color_by}' is present in both ad.obs.columns and ad.var_names — "
            f"cannot determine how to color. Rename one to disambiguate."
        )
    elif in_var:
        if layer is None:
            raise ValueError(
                f"'{color_by}' is in ad.var_names — a layer must be specified via the layer argument."
            )
        if layer not in obj1.layers:
            raise KeyError(
                f"Layer '{layer}' not found in ad.layers. "
                f"Available layers: {list(obj1.layers.keys())}"
            )
        vals = obj1[:, color_by].layers[layer]
        if sparse.issparse(vals):
            vals = vals.toarray()
        color_values = vals.flatten()
    elif in_obs:
        color_values = obj1.obs[color_by].values

        # use scanpy-style stored colors if available and palette not explicitly set
        if palette is None and f"{color_by}_colors" in obj1.uns:
            palette = dict(zip(
                obj1.obs[color_by].cat.categories,
                obj1.uns[f"{color_by}_colors"],
            ))
            
    elif in_modality:
        color_values = None  # handled inside make_points_df via modality column
    else:
        raise ValueError(
            f"'{color_by}' not found in ad.obs.columns or ad.var_names. "
            f"Check spelling or available columns."
        )

    # ── Get coordinates ───────────────────────────────────────────────────────

    coords1 = obj1.obsm[embedding1]
    coords2 = obj1.obsm[embedding2]

    # ── Compute offset ────────────────────────────────────────────────────────

    if offset is None:
        y_offset = (coords2[:, 1].max() - coords1[:, 1].min()) * 1.1
        x_offset = coords2[:, 0].mean() - coords1[:, 0].mean()
        offset   = (x_offset, y_offset)

    coords2 = coords2 - offset

    # ── Build line and point dataframes ───────────────────────────────────────

    line_df, cells_highlight = make_lines_df(
        obj1,
        coords1,
        coords2,
        modality1_name,
        modality2_name,
        line_mask,
        line_thickness,
        downsample_lines_frac,
    )

    points_df = make_points_df(
        obj1,
        coords1,
        coords2,
        modality1_name,
        modality2_name,
        color_by,
        color_values,
        cells_highlight,
        highlight_border,
        pt_size_highlight,
        border_thickness,
        pt_size,
    )

    if (palette is None) and is_numeric_dtype(points_df[color_by]):
        palette = "Spectral_r"
    
    # ── Resolve vmin / vmax ───────────────────────────────────────────────────

    color_vals_numeric = points_df[color_by] if is_numeric_dtype(points_df[color_by]) else None

    def resolve_bound(bound, color_vals):
        if bound is None or color_vals is None:
            return bound
        if isinstance(bound, str):
            if not bound.startswith("p"):
                raise ValueError(
                    f"String vmin/vmax must be of the form 'p{{number}}' (e.g. 'p5', 'p97.5'). Got: '{bound}'"
                )
            q = float(bound[1:])
            if not (0 <= q <= 100):
                raise ValueError(f"Percentile must be between 0 and 100. Got: {q}")
            return np.nanpercentile(color_vals, q)
        return bound

    vmin = resolve_bound(vmin, color_vals_numeric)
    vmax = resolve_bound(vmax, color_vals_numeric)

    # ── Build figure ──────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=figsize)

    line_tensor = np.stack(
        (line_df.values[:, 0:2], line_df.values[:, 2:4]),
        axis=1,
    )
    lc = LineCollection(
        line_tensor,
        color="grey",
        alpha=line_alpha,
        zorder=0,
        linewidths=line_df["line_thickness"],
    )
    ax.add_collection(lc)

    sns.scatterplot(
        points_df,
        x="UMAP1",
        y="UMAP2",
        hue=color_by,
        ax=ax,
        size="highlight",
        sizes=(pt_size_highlight, pt_size),
        linewidth=border_thickness,
        palette=palette,
        **{"hue_norm": (vmin, vmax)} if is_numeric_dtype(points_df[color_by]) else {},
    )

    # ── Legend / colorbar ─────────────────────────────────────────────────────

    if is_numeric_dtype(points_df[color_by]):
        ax.get_legend().remove()
        norm = plt.Normalize(
            vmin=vmin if vmin is not None else points_df[color_by].min(),
            vmax=vmax if vmax is not None else points_df[color_by].max(),
        )
        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        ax.figure.colorbar(sm, ax=ax)
    else:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 0.9))

    # ── Clean up axes ─────────────────────────────────────────────────────────

    ax.spines[["right", "top", "left", "bottom"]].set_visible(False)
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    ax.set_title(title if title is not None else f"colored by {color_by}")

    return fig





def plot_desynchronized_state_volcano(
    ad,
    hue_col,
    modality1_name="RNA",
    modality2_name="ATAC",
    lfc_threshold=0.7,
    pval_threshold=0.05,
    palette=None,
    sig_size=50,
    bg_color="lightgrey",
    figsize=None,
    ax=None,
):
    """Volcano-style plot of density-comparison results.

    Plots the density log-fold change between two modalities against its
    significance. Non-significant cells (direction == "neutral") are drawn in
    a flat background color; significant cells are colored by `hue_col`.
    Threshold guide lines are drawn for the LFC cutoff and the p-value cutoff.

    Parameters
    ----------
    ad : anndata.AnnData
        The annotated data matrix. Expects density-comparison results in
        ad.obs (e.g. from dn_comp_obsm).
    hue_col : str
        Column in ad.obs used to color significant points (e.g. "combo_type").
    modality1_name : str
        Name of the first modality; used to build column names.
    modality2_name : str
        Name of the second modality; used to build column names.
    lfc_threshold : float
        Absolute density LFC at which to draw the vertical guide lines.
    pval_threshold : float
        P-value at which to draw the horizontal guide line. The line is drawn
        at -log10(pval_threshold).
    palette : dict, str, or None
        Palette for the significant points. If None, falls back to
        ad.uns[f"{hue_col}_colors"] when present, otherwise lets seaborn pick.
    sig_size : float
        Marker size for significant points.
    bg_color : str
        Color for non-significant (background) points.
    figsize : tuple or None
        Figure size, used only when `ax` is None.
    ax : matplotlib.axes.Axes or None
        Axis to draw on. Created if None.

    Returns
    -------
    matplotlib.axes.Axes
        The axis the plot was drawn on.
    """
    # ── Build column names from the modality pair ─────────────────────────────

    pair          = f"{modality1_name}_vs_{modality2_name}"
    direction_col = f"direction_{modality1_name}_v_{modality2_name}"
    x_col         = f"density_lfc_{pair}"
    y_col         = f"density_lfc_ml10pval_{pair}"

    # ── Validate inputs ───────────────────────────────────────────────────────

    for col in (direction_col, x_col, y_col, hue_col):
        if col not in ad.obs.columns:
            raise KeyError(
                f"'{col}' not found in ad.obs. Run the density comparison first."
            )

    # ── Resolve palette ───────────────────────────────────────────────────────

    if palette is None:
        colors_key = f"{hue_col}_colors"
        palette = ad.uns.get(colors_key, None)

    # ── Split significant vs. background cells ────────────────────────────────

    msk = ad.obs[direction_col] != "neutral"

    # ── Plot ──────────────────────────────────────────────────────────────────

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (6, 5))

    # Background: non-significant cells, drawn first so they sit underneath.
    sns.scatterplot(
        ad.obs.loc[~msk],
        x=x_col,
        y=y_col,
        color=bg_color,
        linewidth=0,
        ax=ax,
    )

    # Foreground: significant cells, colored by hue_col.
    sns.scatterplot(
        ad.obs.loc[msk],
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=palette,
        linewidth=0,
        s=sig_size,
        ax=ax,
    )

    # ── Threshold guide lines ─────────────────────────────────────────────────

    ax.axvline(-lfc_threshold, color="black", linestyle="--")
    ax.axvline(lfc_threshold, color="black", linestyle="--")
    ax.axhline(-np.log10(pval_threshold), color="grey", linestyle="--")

    ax.set_xlabel(f"density LFC ({modality1_name} vs {modality2_name})")
    ax.set_ylabel("-log10 p-value")

    return ax