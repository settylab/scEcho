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

```bash
git clone https://github.com/settylab/scEcho.git
cd scEcho
pip install -e ".[dev]"
```

## Modules

Imported as `import scEcho` (capital E):

- `Echo_states` — per-modality density estimation and cross-modality density
  comparison; writes per-cell direction labels into `.obs`.
- `Echo_features` — feature-level desynchronization pipeline (imputation,
  per-feature statistics, null-model significance testing).
- `plotting` — visualization (volcano plots, linked side-by-side embeddings,
  per-group direction fractions).
- `try_models` — hyperparameter sweep over Mellon GP settings.
- `utils` — Palantir wrapper, embedding-depth regression, AnnData layer
  helpers.
- `test_components` — diffusion-component sweep (not a pytest suite, despite
  the name).

## Usage

See [`notebooks/example.ipynb`](notebooks/example.ipynb) for the canonical end-to-end pipeline. The basic
shape is `scEcho.Echo_states.dn_comp_obsm(adata, ...)` followed by
`scEcho.Echo_features.run_echo_features(adata, ...)`; see each function's
docstring for the required `.obsm` / `.obs` / `.layers` keys.

## License

GPL-3.0 — see [LICENSE](LICENSE).
