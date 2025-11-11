# Lifelog Personalization Gatekeeper

This project packages the lifelog retrieval + personalization evaluation harness as a notebook-friendly workflow. It lives inside the broader **Memory & Personalization** research area.

## Contents

- `gatekeeper/` – Python package implementing dataset loaders, runners, metrics, and promotion policies.
- `notebooks/` – Ready-to-run notebooks for configuring datasets, caching features, and executing evaluation scorecards.

## Getting Started

1. Install dependencies (use the repo root `requirements.txt`).
2. Launch Jupyter from the repository root:
   ```bash
   jupyter notebook memory/lifelog-personalization/notebooks/00_quickstart.ipynb
   ```
3. Follow the notebook cells to configure dataset paths, load cached parquet files, and run the evaluation runners.

The notebooks automatically add `memory/lifelog_personalization` to `sys.path`, so imports such as `from memory.lifelog_personalization.gatekeeper.runners import run_lifelog` work without extra setup.

See `gatekeeper/README.md` for the underlying command-line usage and promotion gates.
