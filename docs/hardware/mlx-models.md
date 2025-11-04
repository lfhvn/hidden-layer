# MLX-Compatible Models for Introspection (November 2025)

Complete guide to the best MLX models for transformer introspection on Apple Silicon.

## Quick Recommendations

| Use Case | Model | Size | RAM | Why |
|----------|-------|------|-----|-----|
| **Fast Iteration** | Qwen3-8B | 8B | ~8GB | Latest Qwen, excellent quality |
| **Development** | gpt-oss-20b | 20B | ~20GB | OpenAI's open model, great reasoning |
| **Production** | Qwen3-32B | 32B | ~32GB | Best balance of quality and speed |
| **SOTA Local** | Qwen3-235B-A22B | 235B (22B active) | ~70GB | MoE flagship, near-frontier performance |
| **Maximum Quality** | gpt-oss-120b | 120B | ~120GB | Best open model for introspection |

## 1. Qwen3 Series (Recommended - Released April 2025)

### Why Qwen3?
- **MLX-optimized**: Native support, 4-bit/6-bit/8-bit/BF16 quantization
- **Multilingual**: 119 languages
- **Latest generation**: Better reasoning than Qwen 2.5
- **MoE models**: Get 235B performance with 22B activation cost

### Available Models

#### Qwen3 MoE (Mixture of Experts) - BEST FOR INTROSPECTION

**Qwen3-235B-A22B** (Flagship)
```python
"mlx-community/Qwen3-235B-A22B-4bit"
```
- **RAM**: ~70GB (you have 128GB - perfect!)
- **Speed**: Fast for size (~10-15 tok/s)
- **Quality**: Near Claude/GPT-5 level
- **Architecture**: 235B total, 22B activated per token
- **Context**: 32K tokens (expandable to 131K)
- **Best for**: Final experiments, publication-quality results

**Qwen3-30B-A3B**
```python
"mlx-community/Qwen3-30B-A3B-4bit"
```
- **RAM**: ~30GB
- **Speed**: Very fast (~20-30 tok/s)
- **Quality**: Excellent for size
- **Architecture**: 30B total, 3.3B activated
- **Best for**: Quick experiments with frontier-like quality

#### Qwen3 Dense Models

**Qwen3-32B** (Recommended)
```python
"mlx-community/Qwen3-32B-4bit"
```
- **RAM**: ~32GB
- **Speed**: ~15-20 tok/s
- **Quality**: Excellent, better than most 70B older models
- **Best for**: Production experiments

**Qwen3-14B**
```python
"mlx-community/Qwen3-14B-4bit"
```
- **RAM**: ~14GB
- **Speed**: ~25-30 tok/s
- **Quality**: Very strong, rivals older 30B models
- **Best for**: Validation phase

**Qwen3-8B** (Start Here)
```python
"mlx-community/Qwen3-8B-4bit"
```
- **RAM**: ~8GB
- **Speed**: ~30-40 tok/s
- **Quality**: Excellent for iteration
- **Best for**: Development, rapid testing

**Qwen3-4B, 1.7B, 0.6B** (Lightweight)
```python
"mlx-community/Qwen3-4B-4bit"
"mlx-community/Qwen3-1.7B-4bit"
"mlx-community/Qwen3-0.6B-4bit"
```
- **Best for**: Testing if tiny models can introspect

## 2. OpenAI gpt-oss Series (Released Aug 2025)

### Why gpt-oss?
- **Open weights**: Apache 2.0 license
- **Reasoning focus**: Designed for chain-of-thought
- **Tool use**: Excellent for agentic tasks
- **OpenAI quality**: Near o4-mini performance

### Available Models

**gpt-oss-120b** (Best Open Model)
```python
"mlx-community/gpt-oss-120b-4bit"
```
- **RAM**: ~120GB (you can run this!)
- **Speed**: ~5-8 tok/s
- **Quality**: Near GPT-5-mini level
- **Best for**: Comparing open vs closed models

**gpt-oss-20b** (Recommended for Development)
```python
"mlx-community/gpt-oss-20b-4bit"
```
- **RAM**: ~20GB
- **Speed**: ~20-25 tok/s (MLX)
- **Quality**: Excellent reasoning
- **Best for**: All-around development

### ⚠️ IMPORTANT: Ollama vs MLX for gpt-oss

**For Introspection Research → MUST use MLX**:
```python
from mlx_lm import load
model, tokenizer = load("mlx-community/gpt-oss-20b-4bit")
```
- ✅ Full activation access for steering
- ✅ Can inject concepts into layers
- ✅ Works for introspection experiments
- ❌ Slower (5.2 t/s MLX vs 19.7 t/s Ollama)

**For Normal Chat/Generation → Can use Ollama**:
```bash
ollama pull gpt-oss:20b
ollama run gpt-oss:20b
```
- ✅ Much faster
- ✅ Simple CLI interface
- ❌ NO activation access
- ❌ CANNOT do introspection research

**Bottom line**: If you need introspection, you MUST use MLX. Ollama is faster but won't work for this research.

See `PROVIDER_LIMITATIONS.md` for detailed explanation.

## 3. Legacy Models (Still Good)

### Llama 3.1 Series

**Llama 3.1 70B**
```python
"mlx-community/Meta-Llama-3.1-70B-Instruct-4bit"
```
- **RAM**: ~70GB
- **Quality**: Still competitive
- **Best for**: Baseline comparisons

**Llama 3.1 8B**
```python
"mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
```
- **RAM**: ~8GB
- **Quality**: Good baseline
- **Best for**: Quick tests

### Qwen 2.5 Series (Superseded by Qwen3)

Still excellent if you have them cached:
- `mlx-community/Qwen2.5-72B-Instruct-4bit`
- `mlx-community/Qwen2.5-32B-Instruct-4bit`
- `mlx-community/Qwen2.5-14B-Instruct-4bit`

## Recommended Progression for Your Research

### Phase 1: Fast Iteration (Week 1)
**Use**: Qwen3-8B
```python
model = "mlx-community/Qwen3-8B-4bit"
```
- Test all concepts, layers, strengths
- Build concept libraries
- Validate methodology
- **Cost**: Free (local)
- **Time**: 5-10 seconds per test

### Phase 2: Validation (Week 2)
**Use**: gpt-oss-20b or Qwen3-32B
```python
model = "mlx-community/gpt-oss-20b-4bit"
# or
model = "mlx-community/Qwen3-32B-4bit"
```
- Confirm findings with better model
- Test if introspection improves with size
- **Cost**: Free (local)
- **Time**: 10-15 seconds per test

### Phase 3: Frontier Comparison (Week 3)
**Use**: Qwen3-235B-A22B + gpt-oss-120b
```python
# Local frontier
model = "mlx-community/Qwen3-235B-A22B-4bit"

# Compare with API
api_model = "gpt-5"  # or "claude-sonnet-4-5"
```
- Compare local SOTA vs API SOTA
- Validate paper claims about model size
- **Cost**: $0 local, ~$20-50 for API tests
- **Time**: 15-20 seconds per test (local)

## Installation

### Install MLX and Qwen3
```bash
pip install mlx mlx-lm
```

Qwen3 models auto-download on first use:
```python
from mlx_lm import load
model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
```

### Install Ollama and gpt-oss
```bash
# Install Ollama
brew install ollama
ollama serve &

# Pull gpt-oss
ollama pull gpt-oss:20b
ollama pull gpt-oss:120b
```

## Performance Expectations (M4 Max 128GB)

| Model | Load Time | Gen Speed | Introspection Accuracy* |
|-------|-----------|-----------|------------------------|
| Qwen3-8B | ~10s | 30-40 t/s | ~55-65% |
| gpt-oss-20b | ~15s | 20-25 t/s | ~65-70% |
| Qwen3-32B | ~25s | 15-20 t/s | ~70-75% |
| Qwen3-235B-A22B | ~60s | 10-15 t/s | ~75-80% |
| gpt-oss-120b | ~90s | 5-8 t/s | ~75-80% |

*Estimated based on model capabilities, not empirically tested yet

## Expected Introspection Results

Based on paper findings + 2025 model capabilities:

**Frontier Tier** (>75% accuracy):
- GPT-5, Claude Sonnet 4.5 (API)
- Qwen3-235B-A22B, gpt-oss-120b (local)

**High Quality** (65-75% accuracy):
- Claude Haiku 4.5, GPT-5-mini (API)
- Qwen3-32B, gpt-oss-20b (local)

**Good** (55-65% accuracy):
- GPT-4o, Claude 3.5 Sonnet (API)
- Qwen3-14B, Qwen3-8B (local)

**Baseline** (40-55% accuracy):
- Smaller models (3B-4B)
- Legacy models (Llama 3.2, older Qwen)

## Model Selection Guide

**For your M4 Max with 128GB RAM:**

✅ **You can comfortably run**:
- Qwen3-235B-A22B (70GB) - DO THIS!
- gpt-oss-120b (120GB) - Use 4-bit quant
- Multiple smaller models simultaneously

❌ **Too small for serious research**:
- Models under 8B (use for quick tests only)

🎯 **Optimal workflow**:
1. **Develop** on Qwen3-8B (fast iteration)
2. **Validate** on gpt-oss-20b or Qwen3-32B
3. **Finalize** on Qwen3-235B-A22B
4. **Compare** with GPT-5/Claude 4.5 API

## Quick Test Script

```python
from mlx_lm import load
from harness import ActivationSteerer, build_emotion_library

# Test Qwen3-8B
print("Loading Qwen3-8B...")
model, tokenizer = load("mlx-community/Qwen3-8B-4bit")

steerer = ActivationSteerer(model, tokenizer)

# Extract concept
happiness = steerer.extract_contrastive_concept(
    "I feel extremely happy and joyful!",
    "I feel neutral and calm.",
    layer_idx=15
)
print(f"✓ Extracted happiness vector: {happiness.shape}")

# Test introspection
from harness import run_strategy

result = run_strategy(
    "introspection",
    task_input="Describe your current state",
    concept="happiness",
    layer=15,
    strength=1.5,
    provider="mlx",
    model="mlx-community/Qwen3-8B-4bit"
)

print(f"Detected: {result.metadata['introspection_correct']}")
print(f"Confidence: {result.metadata['introspection_confidence']:.2f}")
```

## Finding Models

All models available at:
- **MLX**: https://huggingface.co/mlx-community
- **Ollama**: https://ollama.com/library

Search for latest:
```bash
# List Qwen3 models
python -c "from huggingface_hub import list_models; [print(m.modelId) for m in list_models(author='mlx-community', search='Qwen3')]"

# List gpt-oss models
ollama list | grep gpt-oss
```

## Key Takeaways

1. **Start with Qwen3-8B** - Latest, fast, excellent quality
2. **Develop with gpt-oss-20b** - OpenAI quality, great reasoning
3. **Finalize with Qwen3-235B-A22B** - Best local model available
4. **Compare with GPT-5/Claude 4.5** - API frontier models
5. **Your hardware** - Can run everything up to 120B comfortably!

## Next Steps

1. Install Qwen3-8B: `from mlx_lm import load; load("mlx-community/Qwen3-8B-4bit")`
2. Run introspection tests (see `notebooks/03_introspection_experiments.ipynb`)
3. Compare with gpt-oss-20b
4. Scale up to Qwen3-235B-A22B for final results
5. Benchmark against GPT-5 API

Your M4 Max with 128GB is perfect for this research - you can run near-frontier models locally!
