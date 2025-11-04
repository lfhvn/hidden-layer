# AI-to-AI Communication

**Status**: C2C Implementation Complete | Training & Evaluation In Progress

Non-linguistic communication between LLMs via latent representations, starting with Cache-to-Cache (C2C) direct semantic communication.

## Research Question

Can LLMs communicate more efficiently through latent representations instead of natural language?

## Implemented: Cache-to-Cache (C2C)

We've reproduced the C2C paper which enables direct semantic communication between LLMs through KV-Cache projection:

### Key Results (from original paper)
- **8.5-10.5% higher accuracy** than individual models
- **3.0-5.0% better** than text-based communication
- **2× speedup** in latency

### Quick Start

```python
from code import RosettaModel, create_c2c_projector, generate_kv_cache_index
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load models
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
source_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

# Create projectors and RosettaModel
projectors = [create_c2c_projector(source_model.config, base_model.config)
              for _ in range(base_model.config.num_hidden_layers)]
rosetta = RosettaModel([base_model, source_model], projector_list=projectors)

# Configure and generate
for i in range(base_model.config.num_hidden_layers):
    rosetta.set_projector_config(1, i, 0, i, i)

cache_idx = generate_kv_cache_index(10, 1)
output = rosetta.generate(cache_idx, input_ids, max_new_tokens=50)
```

See **[C2C_README.md](C2C_README.md)** for complete documentation.

## Project Structure

```
ai-to-ai-comm/
├── code/
│   ├── __init__.py              # Package exports
│   ├── c2c_projector.py         # C2C projection networks
│   ├── rosetta_model.py         # Multi-model wrapper
│   ├── kv_cache_utils.py        # KV-Cache utilities
│   └── example_c2c.py           # Usage example
├── c2c-reference/               # Original C2C repo (reference)
├── CLAUDE.md                    # Development guide
├── README.md                    # This file
└── C2C_README.md               # Detailed C2C documentation
```

## Research Questions

- ✅ **Architecture**: How to transform KV-Caches between models? (C2C solved)
- 🔄 **Training**: What data and objectives work best? (In progress)
- 🔄 **Evaluation**: Performance vs. text-based communication? (In progress)
- ⏳ **Protocols**: What communication protocols emerge?
- ⏳ **Generalization**: Does it work across different model families?
- ⏳ **Multi-model**: Can we fuse 3+ models effectively?

## Next Steps

1. ✅ ~~Implement C2C core mechanism~~
2. 🔄 Add training scripts for projector networks
3. 🔄 Implement evaluation benchmarks (MMLU, GSM8K, etc.)
4. ⏳ Experiment with cross-architecture communication
5. ⏳ Integration with multi-agent systems
6. ⏳ Explore alternative approaches (concept vectors, etc.)

## Resources

- 📄 [C2C Paper](https://arxiv.org/abs/2510.03215)
- 🔗 [Original Repository](https://github.com/thu-nics/C2C)
- 📘 [Our C2C Documentation](C2C_README.md)
- 📋 [Development Guide](CLAUDE.md)

## Citation

```bibtex
@article{fu2025c2c,
    title={Cache-to-Cache: Direct Semantic Communication Between Large Language Models},
    author={Tianyu Fu and Zihan Min and Hanling Zhang and Jichao Yan and Guohao Dai and Wanli Ouyang and Yu Wang},
    journal={arXiv preprint arXiv:2510.03215},
    year={2025},
}
```

---

**Hidden Layer Lab** | Research in AI Communication & Coordination
