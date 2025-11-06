# MLX Lab Integration with Hidden Layer Infrastructure

## Overview

`mlx-lab` is a **companion CLI tool** for setup and management. It **complements** (not replaces) your existing research infrastructure.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hidden Layer Research Lab                     │
└─────────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
         ┌──────▼──────┐ ┌────▼─────┐ ┌─────▼──────┐
         │   Harness   │ │ mlx-lab  │ │  Notebooks │
         │ (Research)  │ │ (Setup)  │ │ (Experiments)│
         └──────┬──────┘ └────┬─────┘ └─────┬──────┘
                │              │              │
                └──────────────┼──────────────┘
                               │
                      ┌────────▼────────┐
                      │  Shared Layer   │
                      │                 │
                      │ • HF Cache      │
                      │ • Concepts      │
                      │ • Config        │
                      └─────────────────┘
```

## Integration Points

### 1. Harness (Core Research Infrastructure)

**Location**: `/harness/`

**What the harness does:**
- LLM provider abstraction (MLX, Ollama, Anthropic, OpenAI)
- Experiment tracking and logging
- System prompt management
- Response formatting

**How mlx-lab integrates:**
- ✅ **Reads** harness config (`harness/defaults.py`, `config/models.yaml`)
- ✅ **Uses** harness `LLMProvider` for testing models
- ✅ **Does NOT modify** harness code
- ✅ **Does NOT replace** harness functionality

**Example:**
```python
# Your notebook code (UNCHANGED)
from harness import LLMProvider

llm = LLMProvider()
response = llm.call("Hello", provider="mlx", model="mlx-community/Qwen3-8B-4bit")
```

**What mlx-lab does:**
```bash
# Before using the harness, ensure model is downloaded
mlx-lab models download qwen3-8b-4bit
mlx-lab models test qwen3-8b-4bit  # Check performance

# Then use harness normally in notebooks
```

---

### 2. Model Storage (Shared HuggingFace Cache)

**Location**: `~/.cache/huggingface/hub/`

**What both systems do:**
```python
# Harness (in notebooks)
from mlx_lm import load
model, tokenizer = load("mlx-community/Qwen3-8B-4bit")  # Uses HF cache

# mlx-lab (CLI)
mlx-lab models download qwen3-8b-4bit  # Downloads to HF cache
```

**Key Point**: They use the **SAME cache**. Models downloaded via mlx-lab are immediately available to harness and vice versa.

**Integration:**
- ✅ **No duplication** - One copy of each model
- ✅ **No conflicts** - Both tools read/write same location
- ✅ **Seamless** - Download with CLI, use in notebooks

---

### 3. Introspection Research (Activation Steering)

**Location**: `/theory-of-mind/introspection/`

**What introspection code does:**
```python
# theory-of-mind/introspection/code/activation_steering.py
from mlx_lm import load
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

steerer = ActivationSteerer(model, tokenizer)
concept_vec = steerer.extract_activation(...)
```

**How mlx-lab helps:**
```bash
# Ensure model is ready
mlx-lab models test llama3.2-3b-4bit

# Browse available concepts
mlx-lab concepts list

# Verify concept compatibility
mlx-lab concepts info honesty
```

**Integration:**
- ✅ **Manages** concept vectors in `/shared/concepts/`
- ✅ **Browsers** concept metadata
- ✅ **Does NOT modify** introspection code
- ✅ **Validates** MLX is installed (required for steering)

---

### 4. Concepts (Shared Resource)

**Location**: `/shared/concepts/`

**What notebooks do:**
```python
# Load concept in notebook
import pickle
with open("shared/concepts/honesty.pkl", "rb") as f:
    honesty_vec = pickle.load(f)
```

**What mlx-lab does:**
```bash
mlx-lab concepts list        # Browse what's available
mlx-lab concepts info honesty # Show dimensions, source model
```

**Integration:**
- ✅ **Read-only** - mlx-lab doesn't create/modify concepts
- ✅ **Discovery** - Makes concepts discoverable
- ✅ **Metadata** - Shows which models they came from

---

### 5. Configuration (Harness Defaults)

**Location**: `config/models.yaml`, `harness/defaults.py`

**What's in harness/defaults.py:**
```python
# Auto-detect best available provider
if check_mlx_installed():
    DEFAULT_PROVIDER = "mlx"
    DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"
elif check_ollama_installed():
    DEFAULT_PROVIDER = "ollama"
    DEFAULT_MODEL = "gpt-oss:20b"
# ...
```

**What mlx-lab does:**
```bash
mlx-lab config show      # Display current defaults
mlx-lab config validate  # Check if defaults will work
```

**Integration:**
- ✅ **Reads** defaults to show configuration
- ✅ **Validates** that provider/model are available
- ✅ **Does NOT modify** defaults automatically
- ✅ **Informs** user about configuration state

---

## Workflow Integration

### Before mlx-lab (Manual Process)

```bash
# 1. Install dependencies manually
pip install mlx mlx-lm

# 2. Open Python and download model
python
>>> from mlx_lm import load
>>> model, tokenizer = load("mlx-community/Qwen3-8B-4bit")  # Wait... no feedback
>>> # Is it downloading? How long? How much space?
>>> # No idea!

# 3. Test manually in notebook
# Create test notebook
# Write code to test model
# Check memory usage manually with Activity Monitor

# 4. Find concepts manually
ls shared/concepts/
# What's in these files? No idea without opening them

# 5. Debug setup issues
python check_setup.py  # Only validates, doesn't help fix
```

### With mlx-lab (Streamlined)

```bash
# 1. One-step validation
mlx-lab config validate  # Shows what's missing + how to fix

# 2. Download with feedback
mlx-lab models download qwen3-8b-4bit  # Progress bar, size info

# 3. Performance test
mlx-lab models test qwen3-8b-4bit
# Speed: 47 tok/s, Memory: 5.2GB, Latency: 148ms
# ✅ Fast - Good for interactive experiments

# 4. Compare options
mlx-lab models compare qwen3-8b-4bit gpt-oss-20b-4bit
# See side-by-side performance on YOUR hardware

# 5. Browse concepts
mlx-lab concepts list  # See what's available for steering
```

**Then use harness/notebooks exactly as before!**

---

## Use Cases by Role

### Research Workflow (Notebooks)

**mlx-lab does NOT change this:**
```python
# theory-of-mind/introspection/notebooks/02_activation_steering.ipynb

# Same as before
from mlx_lm import load
from theory_of_mind.introspection import ActivationSteerer
from harness import ExperimentConfig, get_tracker

model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
steerer = ActivationSteerer(model, tokenizer)

# Run experiments...
```

**mlx-lab helps beforehand:**
```bash
# Before opening notebook
mlx-lab models test qwen3-8b-4bit  # Is it fast enough?
mlx-lab concepts list              # What concepts are available?
```

---

### Setup / Onboarding

**New lab member joins:**

**Before:**
1. Read docs/hardware/local-setup.md
2. Manually install MLX
3. Manually download models
4. Hope it works
5. Debug issues

**After:**
```bash
mlx-lab setup  # Interactive wizard handles everything
```

---

### Model Selection

**Need to pick a model for experiment:**

**Before:**
- Read docs/hardware/mlx-models.md
- Guess which model is best
- Download and test manually
- Repeat if too slow

**After:**
```bash
mlx-lab models compare qwen3-8b-4bit gpt-oss-20b-4bit llama3.2-3b-4bit
# See actual performance on your hardware
# Make informed decision
```

---

## Repository Structure

```
hidden-layer/
├── harness/                    # Core research infrastructure
│   ├── llm_provider.py         # Uses mlx_lm.load() directly
│   └── defaults.py             # Auto-detects MLX (READ by mlx-lab)
│
├── mlx_lab/                    # NEW: Setup/management CLI
│   ├── models.py               # Wraps mlx_lm.load() for download
│   ├── benchmark.py            # Tests models before research
│   ├── concepts.py             # Browses shared/concepts/
│   └── config.py               # Validates harness setup
│
├── shared/
│   └── concepts/               # Shared by introspection + mlx-lab
│
├── theory-of-mind/
│   └── introspection/
│       ├── code/
│       │   └── activation_steering.py  # Requires MLX (mlx-lab validates)
│       └── notebooks/          # Use harness + mlx_lm directly
│
├── config/
│   └── models.yaml             # READ by both harness and mlx-lab
│
├── docs/
│   └── hardware/
│       └── mlx-models.md       # Documentation (mlx-lab provides live data)
│
└── check_setup.py              # Enhanced by mlx-lab config validate
```

---

## Key Design Principles

### 1. Non-Invasive
- ✅ mlx-lab does NOT modify harness code
- ✅ mlx-lab does NOT change notebook imports
- ✅ mlx-lab does NOT replace existing tools

### 2. Complementary
- ✅ Adds setup/management capabilities
- ✅ Fills gaps (model discovery, performance testing)
- ✅ Works alongside harness, not instead of it

### 3. Same Resources
- ✅ Uses same HuggingFace cache
- ✅ Reads same config files
- ✅ Manages same concepts directory

### 4. Optional
- ✅ Existing workflows still work without mlx-lab
- ✅ You can ignore mlx-lab and use harness directly
- ✅ mlx-lab is a convenience, not a requirement

---

## What mlx-lab Does NOT Do

❌ **Does NOT** replace harness LLMProvider
❌ **Does NOT** run experiments
❌ **Does NOT** track experiment results
❌ **Does NOT** manage notebooks
❌ **Does NOT** modify concept vectors
❌ **Does NOT** implement activation steering
❌ **Does NOT** change your research code

---

## What mlx-lab DOES Do

✅ **Downloads** models with progress feedback
✅ **Lists** what models you have
✅ **Tests** model performance on your hardware
✅ **Browses** available concept vectors
✅ **Validates** that MLX/harness are set up correctly
✅ **Guides** new users through setup

---

## Migration Path

**You don't need to migrate anything!** mlx-lab is purely additive.

**Existing notebooks continue to work:**
```python
# This still works exactly as before
from harness import LLMProvider
from mlx_lm import load

# No changes needed
```

**Just add mlx-lab commands when convenient:**
```bash
# Use mlx-lab for setup tasks
mlx-lab models list
mlx-lab models test qwen3-8b-4bit

# Then use harness for research (as always)
jupyter notebook theory-of-mind/introspection/notebooks/
```

---

## Integration Checklist

When using mlx-lab with your existing setup:

✅ **Models**: Downloaded via mlx-lab are usable in harness
✅ **Models**: Downloaded via harness are visible to mlx-lab
✅ **Concepts**: Created in notebooks are browsable by mlx-lab
✅ **Config**: Changes to harness config are reflected in mlx-lab
✅ **Benchmarks**: Cached in `~/.mlx-lab/benchmarks.json` (doesn't affect harness)

---

## Summary

**mlx-lab is a tool FOR the harness, not a replacement**

- **Harness** = Research infrastructure (experiments, tracking, providers)
- **mlx-lab** = Setup/management tool (download, test, validate)

They work together:
```
mlx-lab (setup) → Harness (research) → Results
     ↓                  ↓
   Models           Experiments
   Concepts         Publications
   Validation       Insights
```

**Think of it like:**
- **Harness** = Your research lab equipment
- **mlx-lab** = The tool for setting up and maintaining the equipment

You still do research with the harness. mlx-lab just makes setup easier.
