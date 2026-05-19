# scEcho

**scEcho** is a statistical framework for identifying *desynchronized* cell
states — states where gene expression and chromatin accessibility change out of
step with each other — from paired single-cell RNA and ATAC-seq data.

During differentiation and disease, expression and accessibility are coupled
through regulation but do not change in lockstep: enhancers can open before
their target genes are expressed, and transcriptional programs can shift without
matching chromatin changes. scEcho quantifies this desynchronization directly
and identifies the genes and regulatory elements that drive it.

The framework has two components:

- **Echo States** builds cell-state representations separately for RNA and ATAC,
  estimates cell-state density in each modality (via Mellon), and compares them
  to flag states that are better resolved in one modality than the other.
- **Echo Features** identifies the genes (for RNA-resolved states) and
  regulatory loci or TF motifs (for ATAC-resolved states) underlying each
  desynchronized state, using Gaussian Process regression to compare predictive
  accuracy across the two state spaces.

Together they turn desynchronization into an interpretable readout of how
expression and chromatin accessibility interact during cell-state transitions.

## Installation

Install directly from GitHub with `pip`:

```bash
pip install git+https://github.com/settylab/scEcho.git
```

To install a specific tagged version:

```bash
pip install git+https://github.com/settylab/scEcho.git@v0.0.5
```

If the install hangs or fails while building `h5py` from sdist (the host is
missing system `libhdf5`), force pip to pick a wheel:

```bash
pip install --only-binary=:all: git+https://github.com/settylab/scEcho.git
```

### Development install

Clone and install in editable mode with the `dev` extras (pytest, build, twine):

```bash
git clone https://github.com/settylab/scEcho.git
cd scEcho
pip install -e ".[dev]"
```

The same `--only-binary=:all:` workaround applies to the editable install if
`h5py` is built from source:

```bash
pip install --only-binary=:all: -e ".[dev]"
```

## Required AnnData inputs

scEcho operates on an `AnnData` object that has already been preprocessed into
per-modality low-dimensional representations. Building those representations is
upstream of scEcho. The pipeline expects, at minimum:

- `adata.obsm["DM_EigenVectors_RNA"]` — diffusion-map eigenvectors (or another
  low-dim embedding) computed from the RNA modality.
- `adata.obsm["DM_EigenVectors_ATAC"]` — the matching ATAC embedding, over the
  same cells in the same order.
- `adata.obs[<grouping>]` — a categorical cell grouping (cell type, lineage,
  cluster label, etc.); the column name is passed as `obs_col` to
  `run_echo_features`. The example notebook uses `"combo_type"`.
- At least one `adata.layers[<layer>]` of non-negative feature counts (e.g.
  log-normalized RNA counts or ATAC peak/motif scores) to analyze.

The `obsm` keys, `obs` column name, and modality labels are all configurable
via keyword arguments — see the docstrings of `Echo_states.dn_comp_obsm` and
`Echo_features.run_echo_features`.

## Usage

End-to-end pattern: compare density across modalities (Echo States), identify
the features driving each desynchronized state (Echo Features), then visualize
the per-feature results.

```python
import anndata
import scEcho

adata = anndata.read_h5ad("paired_rna_atac.h5ad")

# 1. Echo States — flag cell states that resolve better in one modality.
scEcho.Echo_states.dn_comp_obsm(
    adata,
    obsm_key1="DM_EigenVectors_RNA",
    obsm_key2="DM_EigenVectors_ATAC",
    pval_threshold=0.05,
    log_fold_change_threshold=0.7,
    ls_factor=2,
)

# 2. Echo Features — find the features driving desynchronization per group.
scEcho.Echo_features.run_echo_features(
    adata,
    obs_col="combo_type",
    layers=["RNA_lognorm_counts"],
    sigma=0.1,
    ls=10 ** -0.5,
)

# 3. Plot the per-feature volcano for one group.
ax = scEcho.plotting.plot_scores(
    adata,
    obs_col="combo_type",
    c="MPC",
    layer="RNA_lognorm_counts",
    n_features_label=25,
    ncells_cutoff=30,
    interactive=False,
)

# 4. Pull the desynchronized feature table out of .varm.
results = scEcho.Echo_features.get_reconstruction_results(
    adata,
    "RNA_lognorm_counts",
    grouping="combo_type",
    group="MPC",
    min_cells=30,
)
```

See `notebooks/example.ipynb` for the full walkthrough (linked side-by-side
embeddings, volcano of desynchronized states, per-cell-type direction
fractions, and downstream functional-enrichment via Kompot's StringDB report).

## Modules

- `Echo_states` — per-modality density estimation and cross-modality density
  comparison (`dn_comp_obsm`); writes per-cell direction labels into `.obs`.
- `Echo_features` — feature-level desynchronization pipeline
  (`run_echo_features`, plus `embeddings_predict_layer`, `get_desynch_stats`,
  `run_null_desynch_test`, `get_reconstruction_results`).
- `plotting` — visualization (`plot_scores`, `plot_direction_fractions`,
  `plot_desynchronized_state_volcano`, `linked_plot`, `plot_SE`).
- `try_models` — hyperparameter sweep over Mellon GP settings
  (`try_models`, `read_test_results`, `plot_model_heatmap`).
- `utils` — Palantir wrapper (`run_and_store_pr_res`), embedding-depth
  regression (`regress_embedding`, `calc_corr`), and AnnData layer helpers.
- `test_components` — diffusion-component sweep
  (`sweep_diffusion_components`, `collect_sweep_residual_means`). Despite the
  name, this is not a pytest suite.

## Data

The example notebook (`notebooks/example.ipynb`) loads a paired RNA + ATAC day-59
human fetal retina dataset (`D59_retina.h5ad`). That file is not currently
distributed with the package — ask the maintainer if you need it. To run the
notebook against your own data, substitute any paired RNA + ATAC `AnnData`
that meets the [Required AnnData inputs](#required-anndata-inputs) above; the
notebook documents the expected shape and conventions.

## License

GPL-3.0 — see [LICENSE](LICENSE).
