# Provider Limitations for Introspection Research

## Critical Understanding: Activation Access

**The core limitation**: Only MLX provides access to model activations during inference.

This determines what you can and cannot do with each provider.

## Provider Comparison

| Provider | Activation Access | Can Steer | Introspection Method | Speed | Cost |
|----------|------------------|-----------|---------------------|-------|------|
| **MLX** | ✅ Full | ✅ Yes | True activation steering | Medium | Free |
| **Ollama** | ❌ None | ❌ No | Not supported | Fast | Free |
| **Anthropic** | ❌ None | ⚠️ Simulated | Prompt-based steering | Fast | $$$ |
| **OpenAI** | ❌ None | ⚠️ Simulated | Prompt-based steering | Fast | $$$ |

## What Each Provider Can Do

### MLX - Full Introspection Capabilities ✅

**What it does**:
- Hooks into model forward pass
- Extracts activation vectors from any layer
- Injects concept vectors during inference
- Tests if model detects the injection

**Example**:
```python
from harness import run_strategy

result = run_strategy(
    "introspection",
    task_input="Tell me a story",
    concept="happiness",
    layer=15,          # Inject at layer 15
    strength=1.5,      # Steering strength
    provider="mlx",    # REQUIRED for activation steering
    model="mlx-community/Qwen3-8B-4bit"
)
```

**Why it works**: MLX gives us direct access to:
```python
# We can literally do this:
activations = model.layers[15](input)
activations = activations + (concept_vector * strength)  # Inject!
output = model.layers[16:](activations)
```

**Models that work**:
- ✅ All `mlx-community/*` models on HuggingFace
- ✅ Qwen3 series (8B, 32B, 235B-A22B)
- ✅ gpt-oss series (20B, 120B)
- ✅ Llama 3.1 series
- ✅ Any MLX-compatible model

**Limitations**:
- Apple Silicon only (M1/M2/M3/M4)
- Slower than Ollama
- Requires Python/notebook environment

### Ollama - No Introspection ❌

**What it does**:
- Fast local inference
- Simple CLI interface
- Good for chat and generation

**What it CANNOT do**:
- Access activations
- Inject concepts
- Test introspection
- Mechanistic interpretability

**Why it doesn't work**:
Ollama runs the model in a closed loop:
```
Input → [Black Box Inference] → Output
         ↑ No access here!
```

You cannot hook into the inference process, so you cannot:
- Extract activation vectors
- Inject concept vectors
- Modify layer outputs
- Test introspection

**Example of what WON'T work**:
```python
# This will FAIL or fall back to non-introspection mode:
result = run_strategy(
    "introspection",
    task_input="Tell me a story",
    concept="happiness",
    layer=15,
    provider="ollama",  # ❌ Cannot access activations!
    model="gpt-oss:20b"
)
# Error: "Introspection strategy only works with provider='mlx'"
```

**When to use Ollama**:
- Quick testing of model outputs (no introspection)
- Fast generation for normal tasks
- CLI-based workflows
- When you don't need research capabilities

### API (Claude, GPT) - Prompt-Based Steering Only ⚠️

**What it does**:
- Uses system prompts to simulate concept steering
- Tests natural introspection (no manipulation)
- Benchmarks against frontier models

**What it CANNOT do**:
- True activation steering
- Layer-specific injection
- Fine-grained mechanistic analysis

**How it works**:
Instead of injecting into activations, we use prompts:
```python
# Instead of: activations[layer] = activations + concept_vector
# We do: system_prompt = "As you respond, let happiness influence your thinking"
```

**Example**:
```python
result = run_strategy(
    "introspection",
    task_input="Tell me a story",
    concept="happiness",
    provider="anthropic",
    model="claude-sonnet-4-5",
    api_strength="moderate",  # Not a layer injection!
    steering_style="implicit"  # Prompt wording style
)
```

**Why use it**:
- Benchmark local models against SOTA
- Test natural introspection abilities
- Validate paper findings on Claude/GPT
- Study if prompt steering approximates activation steering

**Models that work**:
- ✅ Claude 4.5 (Sonnet, Haiku)
- ✅ GPT-5 (full, mini, nano)
- ✅ Claude 3.5 series
- ✅ GPT-4o series

## The gpt-oss Confusion

gpt-oss is available via BOTH Ollama and MLX, but they serve different purposes:

### For Normal Use → Ollama (Faster)
```bash
ollama pull gpt-oss:20b
ollama run gpt-oss:20b "Write a poem"
```
- ✅ 19.7 tokens/sec
- ✅ Simple CLI
- ❌ No introspection research

### For Introspection Research → MLX (Required)
```python
from mlx_lm import load
model, tokenizer = load("mlx-community/gpt-oss-20b-4bit")
```
- ✅ Full activation access
- ✅ Can inject concepts
- ✅ Research capabilities
- ❌ 5.2 tokens/sec (slower)

**Bottom line**: If you see "ollama" anywhere in introspection examples, it's WRONG. Must use MLX.

## Decision Tree

```
Do you need activation steering?
├─ YES → Use MLX
│        Models: mlx-community/Qwen3-*, mlx-community/gpt-oss-*
│
└─ NO → What do you need?
         ├─ Fast local inference → Use Ollama
         │                         Models: gpt-oss:20b, qwen3:*
         │
         └─ Benchmark vs SOTA → Use API
                                 Models: claude-sonnet-4-5, gpt-5
```

## For Your Introspection Research

### Phase 1: Development (MLX Required)
```python
# ONLY these will work for introspection:
models = [
    "mlx-community/Qwen3-8B-4bit",           # Start here
    "mlx-community/gpt-oss-20b-4bit",        # Alternative
    "mlx-community/Qwen3-32B-4bit",          # Validation
    "mlx-community/Qwen3-235B-A22B-4bit",    # Best local
    "mlx-community/gpt-oss-120b-4bit",       # Best open
]

# provider="mlx" is NOT OPTIONAL
result = run_strategy(
    "introspection",
    ...,
    provider="mlx",  # REQUIRED!
    model=model
)
```

### Phase 2: Frontier Comparison (API)
```python
# These work with prompt-based steering:
api_models = [
    "claude-sonnet-4-5",    # Best for introspection
    "claude-haiku-4-5",     # Fast and cheap
    "gpt-5",                # SOTA
    "gpt-5-mini",           # Cheaper
]

result = run_strategy(
    "introspection",
    ...,
    provider="anthropic",  # or "openai"
    model=api_model,
    api_strength="moderate"  # Not layer/strength!
)
```

## Common Mistakes

### ❌ Mistake 1: Using Ollama for introspection
```python
# This will NOT work:
run_strategy("introspection", ..., provider="ollama", model="gpt-oss:20b")
```

### ✅ Correct: Use MLX
```python
run_strategy("introspection", ..., provider="mlx", model="mlx-community/gpt-oss-20b-4bit")
```

### ❌ Mistake 2: Expecting layer control with API
```python
# layer parameter is ignored for API:
run_strategy("introspection", ..., provider="anthropic", layer=15)
```

### ✅ Correct: Use api_strength instead
```python
run_strategy("introspection", ..., provider="anthropic", api_strength="moderate")
```

### ❌ Mistake 3: Thinking Ollama is "good enough"
```python
# Ollama is fast, but you get ZERO introspection capability
# You might as well use the standard "single" strategy
```

### ✅ Correct: Use MLX for research, Ollama for demos
```python
# Research:
run_strategy("introspection", ..., provider="mlx")

# Demos/chat (no introspection):
# ollama run gpt-oss:20b
```

## Why This Matters

The Anthropic paper specifically tests **activation injection** - they inject concept representations directly into model activations. This is ONLY possible with MLX in your setup.

Prompt-based steering (API/Ollama) is a useful approximation, but it's:
- Less precise
- Less controllable
- Not mechanistically interpretable
- Different methodology than the paper

For replicating the paper:
- **Primary method**: MLX with activation steering ✅
- **Secondary method**: API with prompt steering (for comparison) ⚠️
- **Not applicable**: Ollama ❌

## Summary Table

| Task | Provider | Why |
|------|----------|-----|
| Activation steering | MLX | Only option |
| Concept injection | MLX | Only option |
| Layer-specific tests | MLX | Only option |
| True introspection | MLX | Only option |
| Prompt-based tests | API | Frontier comparison |
| Natural introspection | API | No steering needed |
| Fast generation | Ollama | Speed over research |
| Chat/demos | Ollama | Simple interface |

## Your Hardware Advantage

With M4 Max 128GB, you can run:
- ✅ Qwen3-235B-A22B (70GB) with full activation access
- ✅ gpt-oss-120b (120GB) with full activation access
- ✅ Near-frontier performance, local control

Most researchers can't do this - they're stuck with:
- Small models (no hardware for 235B)
- Or API only (no activation access)

You have the best of both worlds: frontier-class local models WITH activation access.

## Bottom Line

**For this introspection research:**
- MLX is NOT optional - it's REQUIRED
- Ollama is fast but useless for introspection
- API is useful for comparison, but different methodology

All your actual introspection experiments must use `provider="mlx"`.
