# Cache-to-Cache (C2C) Communication

**Direct Semantic Communication Between Large Language Models**

This is a reproduction of the C2C paper for the Hidden Layer research lab, implementing direct semantic communication between LLMs through KV-Cache projection.

📄 **Paper**: [Cache-to-Cache: Direct Semantic Communication Between Large Language Models](https://arxiv.org/abs/2510.03215)
🔗 **Original Repo**: [thu-nics/C2C](https://github.com/thu-nics/C2C)

## Overview

Cache-to-Cache (C2C) enables LLMs to communicate directly through their KV-Caches, bypassing text generation entirely. By projecting and fusing KV-Caches between models, C2C achieves:

- **8.5-10.5% higher accuracy** than individual models
- **3.0-5.0% better performance** than text-based communication
- **2× speedup** in latency compared to traditional methods

## Key Concept

Instead of having models communicate through generated text:

```
Model A → Text Generation → Text → Model B Encoding → Understanding
```

C2C enables direct semantic transfer:

```
Model A → KV-Cache → C2C Projector → Model B KV-Cache → Understanding
```

This preserves higher-level semantic information and avoids the information bottleneck of text tokenization.

## Architecture

### 1. C2CProjector

The core projection network that transforms KV-Caches between different model architectures:

```python
C2CProjector(
    source_dim=128,      # Dimension per head in source model
    target_dim=128,      # Dimension per head in target model
    source_num_heads=8,  # Number of attention heads in source
    target_num_heads=8,  # Number of attention heads in target
    hidden_dim=1024,     # Hidden dimension for MLPs
    num_layers=3,        # Number of MLP layers
)
```

**Architecture**:
1. Concatenate source and target KV features
2. Project to hidden dimension via MLP
3. Dual path processing:
   - **Projection path**: Transform semantic content
   - **Weight path**: Compute blending coefficients
4. Gated residual: `output = target + gate * weight * projected`

### 2. RosettaModel

Wrapper that orchestrates multiple LLMs for cache-to-cache communication:

```python
RosettaModel(
    model_list=[base_model, source_model],
    base_model_idx=0,
    projector_list=projectors
)
```

**Workflow**:
1. **Prefill Phase**: Run all models in parallel to build KV-Caches
2. **Projection Phase**: Transform source caches to target space
3. **Fusion Phase**: Integrate projected caches into base model
4. **Decode Phase**: Generate with fused semantic understanding

### 3. KV-Cache Utilities

Helper functions for cache manipulation:

- `generate_kv_cache_index()`: Create cache routing configurations
- `clone_kv_cache()`: Clone caches for independent manipulation
- `extract_layer_cache()`: Extract cache sections
- `print_cache_stats()`: Analyze cache memory usage

## Installation

The implementation uses standard dependencies:

```bash
# Core dependencies
pip install torch transformers

# Optional: for training
pip install wandb datasets accelerate
```

## Quick Start

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from code import RosettaModel, create_c2c_projector, generate_kv_cache_index

# Load models
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
source_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# Create projectors (one per layer)
projectors = [
    create_c2c_projector(source_model.config, base_model.config)
    for _ in range(base_model.config.num_hidden_layers)
]

# Initialize RosettaModel
rosetta = RosettaModel(
    model_list=[base_model, source_model],
    base_model_idx=0,
    projector_list=projectors
)

# Configure layer-wise projections
for layer_idx in range(base_model.config.num_hidden_layers):
    rosetta.set_projector_config(
        source_model_idx=1,
        source_model_layer_idx=layer_idx,
        target_model_idx=0,
        target_model_layer_idx=layer_idx,
        projector_idx=layer_idx
    )

# Tokenize input
prompt = "Explain quantum computing:"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate cache index
# Use source model for understanding, base model for generation
kv_cache_index = generate_kv_cache_index(
    instruction_length=inputs["input_ids"].shape[1] - 1,
    response_length=1,
    source_model_idx=1,  # Use source model
    target_model_idx=0,  # Output to base model
)

# Generate with C2C
output = rosetta.generate(
    kv_cache_index=kv_cache_index,
    input_ids=inputs["input_ids"],
    max_new_tokens=50
)

print(tokenizer.decode(output[0]))
```

### Running the Example

```bash
cd projects/ai-to-ai-comm/code
python example_c2c.py
```

This will:
1. Load two compatible models (e.g., Qwen2-0.5B and Qwen2-1.5B)
2. Set up C2C projectors
3. Run inference with cache-to-cache communication
4. Compare with baseline (base model only)

## Project Structure

```
projects/ai-to-ai-comm/
├── code/
│   ├── __init__.py              # Package exports
│   ├── c2c_projector.py         # C2CProjector implementation
│   ├── rosetta_model.py         # RosettaModel wrapper
│   ├── kv_cache_utils.py        # Cache utilities
│   └── example_c2c.py           # Usage example
├── c2c-reference/               # Original C2C repository (reference)
├── CLAUDE.md                    # Development guide
├── README.md                    # Project overview
└── C2C_README.md               # This file (C2C documentation)
```

## How It Works

### 1. Cache Index Configuration

The `kv_cache_index` controls which model processes each token:

```python
# Format: [source_model_idx, target_model_idx]

# Use projection (source model → target model)
[[1, 0]]  # Use model 1 for understanding, project to model 0

# No projection (base model only)
[[-1, 0]]  # Use base model 0 directly
```

**Example**:
```python
kv_cache_index = [
    torch.tensor([[1, 0]]).repeat(10, 1).unsqueeze(0),  # First 10 tokens: use projection
    torch.tensor([[-1, 0]]).repeat(1, 1).unsqueeze(0),  # Last token: no projection
]
```

### 2. Multi-Model Processing

During prefill:
1. Base model processes tokens → generates base KV-Cache
2. Source model processes same tokens → generates source KV-Cache
3. C2CProjector transforms source cache to target space
4. Transformed cache fused into base model's cache

During decode:
- Only base model active (efficient generation)
- Uses fused cache with richer semantic understanding

### 3. Projection Mechanism

```
Source KV (B, H_s, N, D_s)  ┐
Target KV (B, H_t, N, D_t)  ┴→ Concat → MLP → [Projected, Weights]
                                                    ↓
                    Output = Target + Gate * Weight * Projected
```

**Key components**:
- **Concatenation**: Combines source and target features
- **Dual MLPs**: One for projection, one for weights
- **Gated residual**: Learnable blending of original and projected
- **Temperature annealing**: Gradually sharpen gates during training

## Training (To Be Implemented)

Training C2C projectors involves:

1. **Data**: Instruction-following datasets (e.g., Alpaca, ShareGPT)
2. **Objective**: Minimize divergence between:
   - C2C model output
   - Source model (teacher) output
3. **Process**:
   - Freeze both base and source models
   - Train only C2C projectors
   - Use temperature annealing for gates

Training script will be added in future updates.

## Evaluation (To Be Implemented)

Evaluation benchmarks to be added:

- **Accuracy**: MMLU, GSM8K, HumanEval
- **Efficiency**: Latency, throughput, memory usage
- **Comparison**: vs. text-based, vs. individual models

## Research Questions

This implementation enables exploration of:

1. **Cross-Architecture Communication**: How well does C2C work across different model families?
2. **Information Preservation**: What semantic information is preserved vs. lost?
3. **Scaling**: How does performance scale with model size differences?
4. **Layer Alignment**: Which layer mappings are most effective?
5. **Multi-Model Ensembles**: Can we fuse 3+ models effectively?

## Limitations

Current implementation:

- ✅ Core C2C mechanism (projector + wrapper)
- ✅ Inference and generation
- ✅ Example usage
- ⏳ Training scripts (to be added)
- ⏳ Evaluation benchmarks (to be added)
- ⏳ Pre-trained projector weights (to be added)

## Differences from Original

This reproduction:

- **Simplified**: Focuses on core mechanism
- **Modular**: Designed for Hidden Layer stack
- **Educational**: Clear documentation and examples
- **Research-oriented**: Easy to extend and experiment

Original C2C:

- **Production-ready**: Full training pipeline
- **Optimized**: Performance optimizations
- **Pre-trained**: Includes trained projectors
- **Comprehensive**: Full evaluation suite

## Citation

If you use this implementation, please cite the original C2C paper:

```bibtex
@article{fu2025c2c,
    title={Cache-to-Cache: Direct Semantic Communication Between Large Language Models},
    author={Tianyu Fu and Zihan Min and Hanling Zhang and Jichao Yan and Guohao Dai and Wanli Ouyang and Yu Wang},
    journal={arXiv preprint arXiv:2510.03215},
    year={2025},
}
```

## Resources

- 📄 [Original Paper](https://arxiv.org/abs/2510.03215)
- 🔗 [Original Repository](https://github.com/thu-nics/C2C)
- 🌐 [Project Page](https://fuvty.github.io/C2C_Project_Page/)
- 🤗 [HuggingFace Collection](https://huggingface.co/collections/nics-efc/c2c-68e66ef54b977bd7e58d2d74)

## Contributing

This is part of the Hidden Layer research lab. See main project documentation for contribution guidelines.

## License

This reproduction follows the original C2C Apache-2.0 license.

---

**Hidden Layer Lab** | Research in AI Communication & Coordination
