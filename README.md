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

### Development install

For an editable install with development tools:

```bash
git clone https://github.com/settylab/scEcho.git
cd scEcho
pip install -e ".[dev]"
```

## Usage

```python
import scecho

scecho.impute
scecho.density_comp
scecho.plotting
scecho.priming
scecho.try_models
scecho.utils
scecho.layers
scecho.test_components
```

## Modules

- `density_comp` — density comparison utilities
- `impute` — imputation routines
- `layers` — AnnData layer helpers
- `plotting` — plotting and visualization
- `priming` — core priming analysis
- `try_models` — model fitting / comparison
- `utils` — shared utilities
- `test_components` — component testing helpers

## License

GPL-3.0 — see [LICENSE](LICENSE).
