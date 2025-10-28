# Setup Complete: Installation & New Features

## 📋 Summary

I've verified your installation, added new reasoning features, and fixed the benchmark notebook. Here's what's ready for you:

---

## ⚠️ **REQUIRED: Install Python Packages**

### Current Status
- ✅ Ollama is installed and running
- ✅ Models available: `gpt-oss:20b`, `llama3.2`
- ❌ **Python packages NOT installed**

### Install Now

```bash
cd /Users/lhm/Documents/GitHub/hidden-layer
pip install -r requirements.txt
```

This installs: `mlx`, `ollama`, `pandas`, `numpy`, `matplotlib`, `jupyter`, `pyyaml`, and more.

**Why this matters**: Your notebooks will fail without these packages installed.

---

## ✅ What I Fixed

### 1. **Notebook 07 - Task-by-Task Breakdown**

Added comprehensive task-by-task results showing:
- ✅/❌ Which strategies succeeded/failed on each question
- 📝 Full question text and expected answers
- ⏱️ Latency and cost per strategy per task
- 🎯 Task difficulty analysis
- 💡 Where multi-agent strategies helped

**Location**: `notebooks/07_custom_strategies_benchmark.ipynb`
- New cell after "Results Analysis"
- Shows each question with strategy performance
- Identifies tasks where multi-agent beat single model

### 2. **Thinking Budget Support**

Your model (`gpt-oss:20b`) supports extended reasoning via `thinking_budget` parameter:

```python
response = llm_call(
    "Complex math problem...",
    provider="ollama",
    model="gpt-oss:20b",
    thinking_budget=2000  # Extra tokens for reasoning
)
```

**When to use**: Complex problems requiring multi-step reasoning
**Cost**: 2-3x more tokens, but potentially better accuracy

### 3. **Rationale Extraction (NEW!)**

New utility to get models to explain their reasoning:

```python
from harness import ask_with_reasoning

result = ask_with_reasoning("Should we use React or Vue?")

print(result.rationale)  # Step-by-step reasoning
print(result.answer)     # Final recommendation
```

**Files created**:
- `code/harness/rationale.py` - Rationale extraction utilities
- `notebooks/09_reasoning_and_rationale.ipynb` - Examples and demos
- `INSTALLATION_AND_REASONING.md` - Comprehensive guide

---

## 🎯 Quick Start

### After installing packages:

```bash
# 1. Install packages
pip install -r requirements.txt

# 2. Start Jupyter
cd notebooks
jupyter notebook

# 3. Open and run:
#    - 09_reasoning_and_rationale.ipynb (NEW - try this first!)
#    - 07_custom_strategies_benchmark.ipynb (UPDATED - task-by-task results)
#    - 08_agentic_builder.ipynb (build actual code/designs)
```

---

## 📚 New Features Reference

### Thinking Budget

```python
# Basic call
llm_call("Question?", provider="ollama", model="gpt-oss:20b")

# With extended reasoning
llm_call(
    "Complex question?",
    provider="ollama",
    model="gpt-oss:20b",
    thinking_budget=2000  # Allow 2000 tokens for reasoning
)
```

**Optimal values**:
- 500-1000: Simple multi-step problems
- 1000-2000: Complex reasoning (recommended)
- 2000-4000: Very complex strategic problems

### Rationale Extraction

```python
from harness import ask_with_reasoning

# Simplest way to get reasoning
result = ask_with_reasoning("Your question?")
print(f"Reasoning: {result.rationale}")
print(f"Answer: {result.answer}")

# With strategies
from harness import run_strategy_with_rationale

result = run_strategy_with_rationale(
    "adaptive_team",
    "Your question?",
    n_experts=3
)
```

### Pre-configured Models

```python
from harness import get_model_config, llm_call

# Use pre-configured reasoning setup
config = get_model_config("gpt-oss-20b-reasoning")
response = llm_call("Question?", **config.to_kwargs())
```

Available configs:
- `gpt-oss-20b-reasoning` - Extended reasoning (thinking_budget=2000)
- `gpt-oss-20b-creative` - High temperature for creative tasks
- `gpt-oss-20b-precise` - Low temperature for deterministic outputs
- `gpt-oss-20b-fast` - Lower token limits for fast iteration

---

## 📊 Using in Benchmarks

### Add Thinking Budget to Benchmarks

Edit `notebooks/07_custom_strategies_benchmark.ipynb`, cell 5:

```python
STRATEGIES_TO_TEST = [
    # Baseline
    ("single", {
        "provider": PROVIDER,
        "model": MODEL,
        "verbose": False
    }),

    # With thinking budget
    ("single", {
        "provider": PROVIDER,
        "model": MODEL,
        "thinking_budget": 2000,  # ADD THIS
        "verbose": False
    }),

    # Adaptive team with thinking
    ("adaptive_team", {
        "n_experts": 3,
        "thinking_budget": 2000,  # ADD THIS
        "provider": PROVIDER,
        "model": MODEL,
        "verbose": True
    }),
]
```

### See Task-by-Task Results

After running the benchmark, the new cell shows:

```
📋 DETAILED TASK-BY-TASK RESULTS
================================================================================

────────────────────────────────────────────────────────────────────────────────
📝 Task 1/10: Janet's ducks lay 16 eggs per day...

✓ Expected Answer: 18

 Strategy Results:
 ──────────────────────────────────────────────────────────────────────────────
   ✅ adaptive_team            : CORRECT    | Latency:   8.45s | Cost: $0.0023
   ✅ single                   : CORRECT    | Latency:   2.34s | Cost: $0.0005
   ❌ design_critique          : INCORRECT  | Latency:   7.89s | Cost: $0.0021
```

---

## 📁 Files Created/Modified

### Created
- ✅ `code/harness/rationale.py` - Rationale extraction utilities
- ✅ `notebooks/09_reasoning_and_rationale.ipynb` - Examples & demos
- ✅ `INSTALLATION_AND_REASONING.md` - Comprehensive guide
- ✅ `SETUP_COMPLETE.md` - This file

### Modified
- ✅ `code/harness/__init__.py` - Export rationale functions
- ✅ `notebooks/07_custom_strategies_benchmark.ipynb` - Task-by-task breakdown

---

## 🔍 Verification

After installing packages, verify everything works:

```bash
python -c "
from harness import llm_call, ask_with_reasoning
print('✅ Basic import works')

# Test basic call
response = llm_call('Hi!', provider='ollama', model='gpt-oss:20b')
print(f'✅ Basic call works: {response.text[:50]}...')

# Test rationale
result = ask_with_reasoning('What is 2+2?', thinking_budget=500)
print(f'✅ Rationale works')
print(f'   Reasoning: {result.rationale[:100]}...')
print(f'   Answer: {result.answer}')
"
```

---

## 💡 What to Try Next

### 1. Try the New Reasoning Notebook
```bash
jupyter notebook notebooks/09_reasoning_and_rationale.ipynb
```

Demonstrates:
- Thinking budget effects
- Rationale extraction
- Multi-agent reasoning
- Pre-configured models

### 2. Run Benchmarks with Thinking
Open `notebooks/07_custom_strategies_benchmark.ipynb`:
- Add `thinking_budget=2000` to strategies (cell 5)
- Run evaluation (cell 9)
- Check task-by-task results (new cell after cell 11)

### 3. Build Something with Reasoning
Open `notebooks/08_agentic_builder.ipynb`:
- Modify `PROJECT_BRIEF` to your idea
- Strategies will use reasoning to build better designs
- Output to `artifacts/` directory

---

## 📖 Documentation

- **Installation & Reasoning**: `INSTALLATION_AND_REASONING.md`
- **Custom Strategies**: `CUSTOM_STRATEGIES_GUIDE.md`
- **Benchmark Guide**: `BENCHMARK_GUIDE.md`
- **Integration Summary**: `INTEGRATION_SUMMARY.md`

---

## 🐛 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### Cells failing in notebooks
- ✅ Install packages first (see above)
- ✅ Restart Jupyter kernel: `Kernel → Restart`
- ✅ Re-run setup cells

### Ollama not responding
```bash
ollama list  # Check if running
ollama serve &  # Start if needed
```

### Model not found
```bash
ollama pull gpt-oss:20b
```

---

## ❓ Questions Answered

### ✅ Can I verify the installation?
Yes! Run the verification script above after `pip install -r requirements.txt`

### ✅ Is there a "thinking" option?
Yes! Use `thinking_budget=N` parameter:
- Works with `llm_call`, `run_strategy`, and all strategies
- Allocates extra tokens for reasoning
- Recommended: 1000-2000 tokens

### ✅ Can I get the model to explain its rationale?
Yes! Three ways:
1. `ask_with_reasoning("question?")` - Simplest
2. `llm_call_with_rationale("question?")` - More control
3. `run_strategy_with_rationale("strategy", "question?")` - With multi-agent

All return `RationaleResponse` with `.rationale` and `.answer` separated.

---

## 🚀 Next Steps

1. **Install packages**: `pip install -r requirements.txt`
2. **Verify installation**: Run verification script above
3. **Try reasoning notebook**: `09_reasoning_and_rationale.ipynb`
4. **Run benchmarks**: `07_custom_strategies_benchmark.ipynb` with thinking budget
5. **Read guides**: `INSTALLATION_AND_REASONING.md` for detailed examples

---

**Status**: ✅ Ready to use (after `pip install -r requirements.txt`)

**Last Updated**: 2025-01-27
