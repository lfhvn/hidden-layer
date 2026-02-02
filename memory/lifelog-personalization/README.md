# Lifelog Personalization Gatekeeper

This project packages the lifelog retrieval + personalization evaluation harness as a notebook-friendly workflow. It lives inside the broader **Memory & Personalization** research area.

## Prerequisites

- Python 3.10+ with Hidden Layer repository set up
- At least one LLM provider configured (Ollama, MLX, or API)
- Lifelog datasets (instructions in notebook)

**New to Hidden Layer?** See [/QUICKSTART.md](../../QUICKSTART.md) for initial setup.

## Installation

This project uses the shared harness infrastructure. From the repository root:

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python check_setup.py
```

## Contents

- `gatekeeper/` – Python package implementing dataset loaders, runners, metrics, and promotion policies
- `notebooks/` – Ready-to-run notebooks for configuring datasets, caching features, and executing evaluation scorecards

## Quick Start

### Jupyter Notebooks

Launch Jupyter from the repository root:

```bash
jupyter lab memory/lifelog-personalization/notebooks/00_quickstart.ipynb
```

The notebook will guide you through:
1. Configuring dataset paths
2. Loading cached parquet files
3. Running evaluation runners
4. Analyzing personalization metrics

The notebooks automatically add `memory/lifelog_personalization` to `sys.path`, so imports such as `from memory.lifelog_personalization.gatekeeper.runners import run_lifelog` work without extra setup.

### Command-Line Usage

See `gatekeeper/README.md` for the underlying command-line usage and promotion gates.
