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

scEcho is installed directly from GitHub with `pip`.

> **Note:** This repository is currently **private**. Installing it requires
> GitHub credentials with access to the repo (an SSH key or a personal access
> token). The plain `https` commands below only work once the repository is
> public.

### Install via SSH (recommended while private)

If you have an SSH key set up with GitHub:

```bash
pip install git+ssh://git@github.com/ConnorFinkbeiner/scEcho.git
```

### Install via HTTPS with a personal access token

Generate a token at **GitHub → Settings → Developer settings → Personal access
tokens** with `repo` scope, then:

```bash
pip install git+https://<YOUR_TOKEN>@github.com/ConnorFinkbeiner/scEcho.git
```

### Install a specific version or branch

Append `@` followed by a tag, branch, or commit:

```bash
pip install git+ssh://git@github.com/ConnorFinkbeiner/scEcho.git@v0.0.5
```

### Install via HTTPS (once the repository is public)

```bash
pip install git+https://github.com/ConnorFinkbeiner/scEcho.git
```

### Development install (editable, with dev tools)

```bash
git clone git@github.com:ConnorFinkbeiner/scEcho.git
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
