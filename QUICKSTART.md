# Hidden Layer Quick Start

Zero-to-notebook instructions for every supported workflow. Choose the path that matches your hardware and goals, run a quick validation, and start experimenting.

---

## 0. Prerequisites

- macOS (M-series recommended) or Linux with Python ≥ 3.10.
- For local models: [Homebrew](https://brew.sh) + Ollama **or** Apple MLX.
- Optional (API path): Anthropic/OpenAI API keys.
- Recommended tooling:
  - `uv` (`pip install uv`) for fast virtualenv management, or the built-in `python -m venv`.
  - `make` (ships with macOS; install via `xcode-select --install` if missing).

---

## 1. Create a Python Environment

Pick one of the following:

### Option A – Repo-managed virtualenv (simple)
```bash
make setup        # creates venv/, upgrades pip, installs requirements.txt
source venv/bin/activate
```

### Option B – Custom environment (uv example)
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

> Need conda or poetry? Install the deps from `requirements.txt` and skip to the next section.

**Tip:** Hidden Layer targets Python 3.10–3.12 (matching current MLX wheels). If `python3 --version` reports something outside that range (e.g., 3.9 or 3.13), install 3.11 and run:
```bash
PYTHON=python3.11 make setup
```

---

## 2. Choose a Model Path

### Local via Ollama (recommended for fast iteration)
```bash
brew install ollama                 # one-time
ollama serve                        # keep running in a separate terminal
ollama pull llama3.2:latest         # ~3B; great starter model
# optional extras
ollama pull mistral-small:latest    # 7B reasoning
```

### Local via MLX (for on-device Apple Silicon research)
> Only available on Apple Silicon (arm64 macOS running ≥ macOS 13 and Python 3.10–3.12). If `pip install mlx` fails with “no matching distribution,” skip this step and stick with Ollama/API.
```bash
pip install mlx mlx-lm              # inside your env
python -m mlx_lm.download mlx-community/Llama-3.2-3B-Instruct-4bit
# use other IDs from MLX_MODELS_2025.md as desired
```

### API Providers (Anthropic / OpenAI)
```bash
cp .env.example .env
open .env                           # or nano .env
# set ANTHROPIC_API_KEY / OPENAI_API_KEY
```
Then export the keys for the current shell or rely on `python-dotenv` to load them.

> You can mix paths. For example, run Ollama locally and fall back to Anthropic for comparison.

---

## 3. Validate the Installation

Run the bundled check script from the repo root:
```bash
python check_setup.py
```
What you should see:
- Python ≥ 3.10
- Required packages installed
- Harness import succeeds
- Ollama status (if installed)
- MLX status (if installed)

Fix anything marked ✗ before continuing. The script lists common remedies (e.g., missing models).

---

## 4. Launch a Ready-to-Run Notebook

### Easiest path (make target)
```bash
make notebook          # activates venv, launches Jupyter Lab in notebooks/
```

### Manual alternative
```bash
source venv/bin/activate   # or your own env
python -m jupyter lab notebooks/
```

Open `00_quickstart.ipynb`. It:
- Detects available providers (Ollama, MLX, Anthropic, OpenAI).
- Runs a smoke-test `llm_call`.
- Demonstrates `run_strategy` and `ExperimentTracker`.

---

## 5. Project-Specific Jump Points

- **Harness Core** – `notebooks/01_baseline_experiments.ipynb`
  - Compare strategies, run benchmarks, log experiments.
- **CRIT (Design Critique)** – `notebooks/crit/01_multi_perspective_baseline.ipynb`
  - Multi-perspective critiques, synthesis workflows.
- **SELPHI (Theory of Mind)** – `notebooks/selphi/01_scenarios.ipynb`
  - Batch scenario evaluation, model comparisons, scoring.
- **Latent Lens** (SAE + interpretability GUI)
  - Backend uses the shared tracker; see `latent-lens/SETUP.md` after core setup.
- **Steerability Dashboard**
  - Activation steering experiments; integrate via shared API after following `steerability-dashboard/README.md`.

Each subsystem reuses the same configuration files in `config/` and the harness utilities you just installed.

---

## 6. Switching Providers Inside a Notebook

```python
from harness import llm_call, run_strategy

# Auto-detect default provider/model from config
resp = llm_call("Hello Hidden Layer!")

# Force Ollama
resp = llm_call("Local call", provider="ollama", model="llama3.2:latest")

# Force MLX
resp = llm_call("MLX call", provider="mlx", model="mlx-community/Llama-3.2-3B-Instruct-4bit")

# Anthropic (requires API key)
resp = llm_call("API call", provider="anthropic", model="claude-3-5-haiku-20241022")
```

---

## 7. Troubleshooting and Variations

- **No Ollama models listed** – run `ollama pull llama3.2:latest`.
- **Notebook needs remote access** – start Jupyter with `jupyter lab --ip=0.0.0.0 --port=8888` and SSH tunnel (`ssh -L 8888:localhost:8888 user@host`).
- **Want a clean notebook repo** – enable `nbstripout` (`pip install nbstripout && nbstripout --install`).
- **Reinstall dependencies** – `make clean && make setup`.
- **Switching providers mid-experiment** – use `config/models.yaml` presets or CLI: `python code/cli.py "task" --provider anthropic`.

---

## Next Moves

1. Explore `00_quickstart.ipynb`.
2. Run baseline experiments and log results (`ExperimentTracker`).
3. Branch into CRIT/SELPHI/Latent Lens/Steerability using the shared harness.
4. Customize `config/models.yaml` for your favorite models or system prompts.

Refer to `README.md` for a deeper system overview and `SETUP.md` for hardware-specific tuning.
