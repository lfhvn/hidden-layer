# Hidden Layer - Frequently Asked Questions

> Quick answers to common questions about setup, usage, and troubleshooting

---

## Table of Contents

- [Getting Started](#getting-started)
- [Environment & Setup](#environment--setup)
- [LLM Providers](#llm-providers)
- [Project Selection](#project-selection)
- [Import & Module Errors](#import--module-errors)
- [Performance & Optimization](#performance--optimization)
- [Notebooks & Jupyter](#notebooks--jupyter)
- [Experiment Tracking](#experiment-tracking)
- [Common Errors](#common-errors)
- [Advanced Topics](#advanced-topics)

---

## Getting Started

### Q: I'm new to Hidden Layer. Where should I start?

**A**: Follow this path:
1. Read [QUICKSTART.md](QUICKSTART.md) to set up your environment (5 minutes)
2. Run `python check_setup.py` to verify everything works
3. Browse [PROJECT_GUIDE.md](PROJECT_GUIDE.md) to find a project that interests you
4. Open that project's `00_quickstart.ipynb` or `01_*.ipynb` notebook
5. Start experimenting!

### Q: Do I need API keys to use Hidden Layer?

**A**: No! You can use local models (Ollama or MLX) for everything. API keys (Anthropic/OpenAI) are optional and only needed if you want to use frontier models.

### Q: What's the minimum hardware requirement?

**A**:
- **Basic use (API only)**: Any computer with Python 3.10+
- **Local models (Ollama)**: 8GB RAM minimum, 16GB recommended
- **MLX (Apple Silicon)**: M1/M2/M3 Mac with 8GB+ unified memory
- **GPU acceleration**: CUDA-compatible GPU with 8GB+ VRAM (optional)

### Q: Which projects can I run without GPUs?

**A**: All of them!
- **Notebook-based projects**: Use API providers (no GPU needed)
- **Local models**: Ollama works on CPU (slower but functional)
- **Web apps**: Backend can use API providers
- **MLX**: Runs on Apple Silicon unified memory (no separate GPU needed)

---

## Environment & Setup

### Q: I get a Python version error. What version do I need?

**A**: Python 3.10, 3.11, or 3.12 recommended.
- **MLX support**: Requires Python 3.10-3.12 (not 3.13+)
- **General use**: Python 3.10+ works

```bash
# Check your version
python --version

# Use a specific version with pyenv
pyenv install 3.11.0
pyenv local 3.11.0
```

### Q: Should I use venv, conda, or uv?

**A**: Any of them work! Pick what you're comfortable with:

```bash
# Option 1: Built-in venv (simplest)
make setup
source venv/bin/activate

# Option 2: uv (fastest)
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Option 3: conda
conda create -n hidden-layer python=3.11
conda activate hidden-layer
pip install -r requirements.txt
```

### Q: Do I need to install dependencies for every project?

**A**: No! All research projects share the same dependencies from the root `requirements.txt`. Install once, use everywhere:

```bash
cd /path/to/hidden-layer  # Repository root
pip install -r requirements.txt
```

### Q: How do I update to the latest version?

**A**:
```bash
git pull origin main
pip install -r requirements.txt --upgrade
python check_setup.py  # Verify everything still works
```

---

## LLM Providers

### Q: Which provider should I use?

**A**: Depends on your goal:

| Goal | Provider | Why |
|------|----------|-----|
| **Fast iteration** | Ollama | Local, no API costs, fast feedback |
| **Apple Silicon** | MLX | Native M-series optimization, fast |
| **Best quality** | Anthropic (Claude) | State-of-the-art reasoning |
| **OpenAI models** | OpenAI (GPT) | Access to GPT-4, GPT-4o |
| **No setup** | Any API | Works immediately, no local install |

### Q: How do I set up Ollama?

**A**:
```bash
# Install (macOS/Linux)
brew install ollama

# Start server (keep running in separate terminal)
ollama serve

# Pull a model
ollama pull llama3.2:latest  # 3B - fast, good quality
ollama pull mistral-small:latest  # 7B - better quality

# Test it
ollama run llama3.2:latest "Hello!"
```

### Q: How do I set up MLX?

**A**: MLX only works on Apple Silicon (M1/M2/M3/M4):

```bash
# Check if you have Apple Silicon
uname -m  # Should output: arm64

# Install MLX
pip install mlx mlx-lm

# Download a model
python -m mlx_lm.download mlx-community/Llama-3.2-3B-Instruct-4bit

# Or use MLX Lab CLI
mlx-lab models download qwen3-8b-4bit
```

### Q: How do I use API providers?

**A**:
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your keys
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...

# Keys are loaded automatically by python-dotenv
```

### Q: Can I use multiple providers in the same script?

**A**: Yes! Mix and match:

```python
from harness import llm_call

# Local for fast iteration
response1 = llm_call("Draft an outline", provider="ollama")

# API for final quality
response2 = llm_call("Refine this outline", provider="anthropic")

# Compare
print(f"Local: {response1.text}")
print(f"API: {response2.text}")
```

### Q: Why is my Ollama connection failing?

**A**: Common causes:
```bash
# 1. Ollama not running
ollama serve  # Keep this running

# 2. Wrong model name
ollama list  # See what's installed
ollama pull llama3.2:latest  # Pull if missing

# 3. Port conflict
# Ollama uses port 11434 by default
# Check: lsof -i :11434
```

---

## Project Selection

### Q: Which project should I try first?

**A**: Based on your interests:

- **New to AI research?** → [Multi-Agent](communication/multi-agent/) - Easy to understand, immediate results
- **Want to see inside models?** → [Latent Lens](representations/latent-space/lens/) - Visual, interactive
- **Interested in psychology?** → [SELPHI](theory-of-mind/selphi/) - Theory of mind tests
- **Want to control behavior?** → [Steerability](alignment/steerability/) - Direct model control
- **Like building products?** → [AgentMesh](agentmesh/) - Production orchestration

See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for full decision guide.

### Q: Can I run multiple projects at once?

**A**: Yes! All projects share the same infrastructure (harness), so you can:
- Run experiments in one project, visualize in another
- Use concept vectors from Introspection in Steerability
- Feed multi-agent outputs into SELPHI evaluation
- Train SAEs in Latent Lens, load in State Explorer

### Q: Which projects work offline?

**A**: With local models (Ollama/MLX), all notebook-based projects work fully offline:
- Multi-Agent
- SELPHI
- Introspection
- AI-to-AI Communication
- CALM
- Lifelog Personalization

Web apps need internet to load frontend dependencies initially.

---

## Import & Module Errors

### Q: `ModuleNotFoundError: No module named 'harness'`

**A**:
```bash
# Make sure you're in the repo root
cd /path/to/hidden-layer

# Verify harness exists
ls harness/

# Install in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH=/path/to/hidden-layer:$PYTHONPATH
```

### Q: `ModuleNotFoundError: No module named 'communication.multi_agent'`

**A**: Import paths should work from repository root:

```python
# ✓ Correct - from repo root
from communication.multi_agent import run_strategy

# ✗ Wrong - don't do this
from multi_agent import run_strategy  # Won't work
```

### Q: Jupyter notebook can't find modules

**A**:
```python
# Add to first cell of notebook
import sys
from pathlib import Path

# Add repository root to path
repo_root = Path.cwd().parent.parent  # Adjust depth as needed
sys.path.insert(0, str(repo_root))

# Now imports work
from harness import llm_call
```

Or start Jupyter from repository root:
```bash
cd /path/to/hidden-layer
jupyter lab
```

---

## Performance & Optimization

### Q: Experiments are running slow. How can I speed them up?

**A**:

**For local models:**
- Use smaller models: `llama3.2:latest` (3B) instead of `llama3.1:70b`
- Use quantized models: `-4bit` or `-8bit` variants
- Enable GPU: Make sure Ollama is using CUDA/Metal
- Use MLX on Apple Silicon: Faster than Ollama on M-series

**For API models:**
- Use cheaper/faster models: `claude-3-5-haiku` instead of `claude-3-5-sonnet`
- Reduce context length: Shorter prompts = faster responses
- Use parallel requests where possible
- Cache repeated calls (harness does this automatically)

**For multi-agent:**
- Reduce number of agents: `n_debaters=2` instead of 3+
- Use single-round debates instead of multi-round
- Switch to self-consistency (faster than debate)

### Q: I'm hitting API rate limits. What should I do?

**A**:
```python
# Option 1: Add delays
from time import sleep
for item in batch:
    result = llm_call(item, provider="anthropic")
    sleep(1)  # 1 second between requests

# Option 2: Use harness rate limiting (if available)
from harness import llm_call
result = llm_call(prompt, provider="anthropic", rate_limit=True)

# Option 3: Switch to local models (no limits!)
result = llm_call(prompt, provider="ollama")
```

### Q: How much does it cost to run experiments?

**A**:

**Local models (Ollama/MLX)**: $0 (free!)

**API models** (approximate):
- Claude 3.5 Haiku: $0.25 per million input tokens, $1.25 per million output
- Claude 3.5 Sonnet: $3 per million input tokens, $15 per million output
- GPT-4o: $2.50 per million input tokens, $10 per million output

**Example**:
- 100 debate rounds with Haiku: ~$0.05-0.10
- Full ToMBench (388 samples) with Sonnet: ~$1-2
- Typical research session: $5-20

**To minimize costs**: Use local models for development, API only for final evaluations.

---

## Notebooks & Jupyter

### Q: How do I start Jupyter?

**A**:
```bash
# From repository root
make notebook

# Or manually
jupyter lab

# Or for remote access
jupyter lab --ip=0.0.0.0 --port=8888
```

### Q: My notebook kernel keeps dying

**A**: Common causes:
- Out of memory: Use smaller models or reduce batch size
- Model loading failed: Check Ollama/MLX is running
- Dependencies missing: Run `pip install -r requirements.txt`

### Q: Can I run notebooks in VS Code?

**A**: Yes!
1. Install Jupyter extension in VS Code
2. Open a `.ipynb` file
3. Select Python interpreter (your venv)
4. Run cells normally

### Q: How do I save notebook outputs?

**A**:
```bash
# Strip outputs before committing (keep notebooks clean)
pip install nbstripout
nbstripout --install  # Auto-strip on git commit

# Or manually export
jupyter nbconvert --to html notebook.ipynb
```

---

## Experiment Tracking

### Q: How do I log experiments?

**A**:
```python
from harness import get_tracker

tracker = get_tracker()

# Run experiment
result = run_strategy("debate", task_input="question")

# Log it
tracker.log_result("debate_experiment_1", result)

# View logged experiments
tracker.summarize()

# Export to CSV
tracker.export("experiments.csv")
```

### Q: Where are experiments saved?

**A**: By default in `experiments/` directory in repository root. You can change this:

```python
from harness import ExperimentTracker

tracker = ExperimentTracker(save_dir="my_experiments")
```

### Q: Can I compare experiments across runs?

**A**: Yes! The tracker persists data:

```python
from harness import get_tracker

tracker = get_tracker()

# Load previous experiments
previous = tracker.load_experiment("debate_experiment_1")

# Compare with new run
new = run_strategy("debate", task_input="question")

# Side-by-side comparison
tracker.compare([previous, new])
```

---

## Common Errors

### Q: `RuntimeError: No default provider configured`

**A**:
```bash
# Option 1: Set environment variable
export LLM_PROVIDER=ollama

# Option 2: Pass explicitly
from harness import llm_call
response = llm_call("question", provider="ollama")

# Option 3: Update config/models.yaml
default_provider: ollama
```

### Q: `ConnectionRefusedError: [Errno 61] Connection refused`

**A**: Ollama server not running:
```bash
# Start Ollama
ollama serve  # Keep running in separate terminal
```

### Q: `torch.cuda.OutOfMemoryError`

**A**: GPU out of memory:
```python
# Use smaller model
model = "gpt2"  # Instead of larger models

# Reduce batch size
batch_size = 1  # Instead of larger batches

# Use CPU instead
device = "cpu"

# Or use quantized models
model = "llama3.2:latest-4bit"  # Ollama
model = "mlx-community/Llama-3.2-3B-Instruct-4bit"  # MLX
```

### Q: `JSONDecodeError` when parsing responses

**A**: Model returned invalid JSON:
```python
# Use harness robust parsing
from harness import llm_call

response = llm_call(
    prompt="Return JSON: {...}",
    provider="ollama",
    response_format="json",  # Enforce JSON format
    max_retries=3  # Retry on parse errors
)
```

### Q: Notebook says "Kernel appears to have died"

**A**:
- Check Jupyter logs: `jupyter lab --debug`
- Restart kernel: Kernel → Restart Kernel
- Check dependencies: `pip list | grep -E "(torch|transformers|mlx)"`
- Try smaller model to test if it's a memory issue

---

## Advanced Topics

### Q: Can I add my own LLM provider?

**A**: Yes! Extend the harness:

```python
from harness import LLMProvider, register_provider

class MyProvider(LLMProvider):
    def call(self, prompt, **kwargs):
        # Your implementation
        return response

register_provider("myprovider", MyProvider)

# Use it
from harness import llm_call
response = llm_call("question", provider="myprovider")
```

### Q: Can I use custom models not in the config?

**A**: Yes:
```python
from harness import llm_call

# Ollama: Any pulled model
response = llm_call("question", provider="ollama", model="llama3.1:70b")

# MLX: Any downloaded model
response = llm_call("question", provider="mlx", model="mlx-community/Qwen3-8B-4bit")

# API: Any available model
response = llm_call("question", provider="anthropic", model="claude-opus-4-20250514")
```

### Q: How do I contribute to Hidden Layer?

**A**:
1. Read [CLAUDE.md](CLAUDE.md) for development guidelines
2. Pick a project and read its `CLAUDE.md`
3. Follow the theoretical discipline: every feature should test a hypothesis
4. Maintain backward compatibility
5. Add tests and documentation
6. Submit a pull request

### Q: Can I use Hidden Layer for commercial projects?

**A**: Check the LICENSE file. Generally:
- Harness: Can likely be open-sourced and used commercially
- Research projects: Primarily for research, check individual licenses
- Web tools: Built for deployment, check specific licenses
- Always verify license terms before commercial use

### Q: How do I deploy projects to production?

**A**: See individual project READMEs. General pattern:

```bash
# Web apps (Latent Lens, Steerability, etc.)
cd project-directory
docker-compose -f docker-compose.prod.yml up -d

# AgentMesh
cd agentmesh
# See agentmesh/README.md for production deployment

# Research notebooks
# Convert to scripts, deploy as APIs or batch jobs
```

### Q: Can I run experiments on a remote server?

**A**: Yes:
```bash
# On remote server
git clone repo
cd hidden-layer
./setup.sh
python check_setup.py

# For notebooks
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# On local machine
ssh -L 8888:localhost:8888 user@remote-server
# Then open http://localhost:8888 in browser
```

---

## Still Have Questions?

1. **Check project-specific docs**: Each project has a `README.md` and `CLAUDE.md`
2. **Run check_setup.py**: Often reveals configuration issues
3. **Check the troubleshooting sections**: In individual project READMEs
4. **Search issues**: See if others have encountered the same problem
5. **Ask**: Open an issue with your question

---

**Last updated**: 2025-11-19

**Related Documentation**:
- [QUICKSTART.md](QUICKSTART.md) - Initial setup
- [PROJECT_GUIDE.md](PROJECT_GUIDE.md) - Project selection and navigation
- [CLAUDE.md](CLAUDE.md) - Development guide
- [RESEARCH.md](RESEARCH.md) - Research themes and questions
