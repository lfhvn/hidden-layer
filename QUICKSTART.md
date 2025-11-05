# Hidden Layer Quick Start

Zero-to-notebook instructions for every supported workflow. Choose the path that matches your hardware and goals, run a quick validation, and start experimenting.

---

## 0. Prerequisites

- macOS (M-series recommended) or Linux with Python ≥ 3.10.
- For local models: [Homebrew](https://brew.sh) + Ollama **or** Apple MLX.
- Optional (API path): Anthropic/OpenAI API keys.
- Recommended tooling:
  - `uv` (`pip install uv`) for fast virtualenv management, or the built-in `python3 -m venv`.
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

**Tip:** Hidden Layer works with Python 3.10+. MLX requires Python 3.10–3.12 on Apple Silicon. If you have Python 3.13+ and want MLX support, use Python 3.11 instead:
```bash
PYTHON=python3.11 make setup
```
Otherwise, MLX will be automatically skipped and you can use Ollama or API providers instead.

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
> Only available on Apple Silicon (arm64 macOS running ≥ macOS 13 and Python 3.10–3.12). If `pip install mlx` fails with "no matching distribution," skip this step and stick with Ollama/API.
```bash
pip install mlx mlx-lm              # inside your env
python3 -m mlx_lm.download mlx-community/Llama-3.2-3B-Instruct-4bit
# use other IDs from docs/hardware/mlx-models.md as desired
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
python3 check_setup.py
```
What you should see:
- Python ≥ 3.10
- Required packages installed
- Harness import succeeds
- Ollama status (if installed)
- MLX status (if installed)

Fix anything marked ✗ before continuing. The script lists common remedies (e.g., missing models).

---

## 4. Launch Your First Notebook

### Multi-Agent Communication (Easiest Start)
```bash
# Navigate to multi-agent notebooks
cd communication/multi-agent
source ../../venv/bin/activate      # if using repo venv
jupyter lab notebooks/
```

Open `00_quickstart.ipynb`. It:
- Detects available providers (Ollama, MLX, Anthropic, OpenAI).
- Runs a smoke-test `llm_call`.
- Demonstrates `run_strategy` and `ExperimentTracker`.

### Alternative: Use Make Target
```bash
make notebook          # launches Jupyter Lab from root
```
Then navigate to `communication/multi-agent/notebooks/00_quickstart.ipynb`

---

## 5. Research Areas and Notebooks

Hidden Layer is organized into four research areas:

### 📡 Communication
**Location**: `communication/`

**multi-agent** - Multi-agent coordination strategies
- `notebooks/00_quickstart.ipynb` - Start here!
- `notebooks/01_baseline_experiments.ipynb` - Single model baselines
- `notebooks/02_debate_experiments.ipynb` - Debate strategy
- `notebooks/02_multi_agent_comparison.ipynb` - Compare strategies
- `notebooks/03_introspection_experiments.ipynb` - Introspection tasks
- `notebooks/crit/01_basic_critique_experiments.ipynb` - Design critique

**ai-to-ai-comm** - Direct LLM communication via cache-to-cache
- `notebooks/01_c2c_quickstart.ipynb` - Cache-to-Cache demo
- `notebooks/02_efficiency_evaluation.ipynb` - C2C vs text benchmarks

### 🧠 Theory of Mind
**Location**: `theory-of-mind/`

**selphi** - Theory of mind evaluation
- `notebooks/01_basic_tom_tests.ipynb` - Basic ToM scenarios
- `notebooks/02_benchmark_evaluation.ipynb` - Benchmark evaluation

**introspection** - Model introspection experiments
- `notebooks/01_concept_vectors.ipynb` - Build concept libraries
- `notebooks/02_activation_steering.ipynb` - Steer with concepts

### 🎨 Representations
**Location**: `representations/latent-space/`

**lens** - SAE interpretability web app
- `notebooks/01_sae_training.ipynb` - Train sparse autoencoders
- Backend/frontend: See `lens/SETUP.md` for web app setup

**topologies** - Mobile latent space exploration
- `notebooks/01_embedding_exploration.ipynb` - Visualize concept spaces
- Mobile app: See `topologies/SETUP_DEV.md` for React Native setup

### 🎯 Alignment
**Location**: `alignment/`

**steerability** - Steering vectors and adherence metrics
- `notebooks/01_dashboard_testing.ipynb` - Test dashboard API
- Dashboard: See `steerability/README.md` for web dashboard setup

---

## 6. Python Import Pattern

Hidden Layer uses a special import pattern to handle hyphens in directory names. Each research area has both:
- `theory-of-mind/` (actual code and notebooks)
- `theory_of_mind/` (Python import shim - small, just for imports)

This allows clean Python imports:
```python
# Import shims make this work (Python can't import hyphens)
from theory_of_mind.selphi import run_scenario, SALLY_ANNE
from theory_of_mind.introspection import ConceptLibrary, ActivationSteerer

# Same pattern for communication projects
from communication.multi_agent import run_strategy
from communication.ai_to_ai_comm import RosettaModel
```

**Note**: The underscore directories are tiny (just shim code). All actual code, notebooks, and documentation live in the hyphenated directories.

---

## 7. Switching Providers Inside a Notebook

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

## 8. Experiment Tracking

All notebooks integrate with the harness experiment tracker:

```python
from harness import ExperimentConfig, ExperimentResult, get_tracker

# Configure experiment
config = ExperimentConfig(
    experiment_name="my_experiment",
    task_type="benchmark",
    strategy="debate",
    provider="ollama",
    model="llama3.2:latest",
)

# Track results
tracker = get_tracker()
run_dir = tracker.start_experiment(config)
# ... run your experiment ...
tracker.log_result(result)
summary = tracker.finish_experiment()

print(f"Results logged to: {run_dir}")
```

Experiments are logged to `experiments/` with timestamps and full reproducibility metadata.

---

## 9. Troubleshooting and Variations

- **No Ollama models listed** – run `ollama pull llama3.2:latest`.
- **Notebook needs remote access** – start Jupyter with `jupyter lab --ip=0.0.0.0 --port=8888` and SSH tunnel (`ssh -L 8888:localhost:8888 user@host`).
- **Want a clean notebook repo** – enable `nbstripout` (`pip install nbstripout && nbstripout --install`).
- **Reinstall dependencies** – `make clean && make setup`.
- **Import errors** – Make sure you're in the repo root or the specific project directory.

---

## 10. Next Steps by Interest

### Want to explore multi-agent strategies?
→ Start with `communication/multi-agent/notebooks/00_quickstart.ipynb`

### Interested in theory of mind?
→ Try `theory-of-mind/selphi/notebooks/01_basic_tom_tests.ipynb`

### Want to understand latent representations?
→ Explore `theory-of-mind/introspection/notebooks/01_concept_vectors.ipynb`

### Curious about interpretability?
→ Check out `representations/latent-space/lens/notebooks/01_sae_training.ipynb`

### Building alignment tools?
→ See `alignment/steerability/notebooks/01_dashboard_testing.ipynb`

### Experimenting with AI-to-AI communication?
→ Start with `communication/ai-to-ai-comm/notebooks/01_c2c_quickstart.ipynb`

---

## Additional Resources

- **System Overview**: `README.md`
- **Development Guide**: `CLAUDE.md`
- **Architecture Details**: `docs/ARCHITECTURE.md`
- **Research Themes**: `RESEARCH.md`
- **Infrastructure**: `harness/README.md`
- **Hardware Setup**: `docs/hardware/local-setup.md`

All projects share the same `harness/` infrastructure, `shared/` resources, and `config/` files for maximum consistency and reusability.
