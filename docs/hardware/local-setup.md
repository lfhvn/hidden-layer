# Setup Guide - MLX + Local Models for M4 Max

## Overview
This guide helps you set up your M4 Max (128GB RAM) for local ML experimentation with MLX and Ollama, plus optional API access.

## Prerequisites
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10 or later
- At least 50GB free disk space for models

## 1. Install Core Dependencies

### Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Install MLX (Apple's Framework)
```bash
# MLX and MLX-LM for local model inference and fine-tuning
pip install mlx mlx-lm

# Test installation
python -c "import mlx.core as mx; print(mx.__version__)"
```

### Install Ollama (Easy Model Management)
```bash
# Download from ollama.ai or use homebrew
brew install ollama

# Start Ollama service
ollama serve &

# Pull a model (Llama 3.2 3B is a good start)
ollama pull llama3.2:latest

# Optional: Pull Mistral 7B
ollama pull mistral:latest

# Test
ollama run llama3.2 "What is 2+2?"
```

## 2. Install Project Dependencies

```bash
cd /path/to/project
pip install -r requirements.txt
```

If requirements.txt doesn't exist yet:
```bash
pip install pandas numpy matplotlib seaborn jupyter ipykernel
pip install python-dotenv tqdm
```

## 3. Optional: API Provider Setup

### Anthropic (Claude)
```bash
pip install anthropic

# Add to .env file
echo "ANTHROPIC_API_KEY=your_key_here" >> .env
```

### OpenAI
```bash
pip install openai

# Add to .env file
echo "OPENAI_API_KEY=your_key_here" >> .env
```

## 4. Download MLX Models

MLX models are available on HuggingFace. Download via MLX-LM:

```python
from mlx_lm import load

# This downloads model to ~/.cache/huggingface/
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
```

### Recommended Models for M4 Max 128GB:

**For baseline experiments (fast):**
- `mlx-community/Llama-3.2-3B-Instruct-4bit` (~2GB)
- `mlx-community/Mistral-7B-Instruct-v0.3-4bit` (~4GB)

**For high-quality baselines:**
- `mlx-community/Llama-3.1-8B-Instruct-4bit` (~5GB)
- `mlx-community/Meta-Llama-3.1-70B-Instruct-4bit` (~40GB - you can run this!)

**For fine-tuning:**
- Start with 7B models (Llama/Mistral)
- 128GB RAM allows comfortable fine-tuning of 13B models

## 5. Verify Setup

Test the harness:

```bash
cd code/harness
python -c "from llm_provider import llm_call; print(llm_call('test', provider='ollama'))"
```

## 6. Jupyter Notebook Setup

```bash
# Install jupyter kernel
python -m ipykernel install --user --name=research-lab --display-name "Research Lab"

# Start Jupyter
cd notebooks
jupyter notebook
```

## 7. Performance Tips

### MLX Optimization
- MLX is optimized for unified memory on Apple Silicon
- Batch size: Start with 1-4, increase based on available RAM
- Use 4-bit quantized models for speed/memory balance

### Ollama Tips
- Models cached in `~/.ollama/models`
- Use `OLLAMA_NUM_PARALLEL=2` to run multiple models
- Set `OLLAMA_MAX_LOADED_MODELS=2` to keep models in memory

### Memory Management
With 128GB, you can:
- Run 3-4 small models (7B) simultaneously for multi-agent debates
- Fine-tune 13B models with LoRA
- Load 70B models for high-quality baselines
- Keep multiple models cached for quick switching

## 8. Fine-Tuning Setup (LoRA with MLX)

MLX-LM has excellent fine-tuning support:

```bash
# Prepare your dataset (JSONL format)
# Each line: {"text": "prompt [INST] instruction [/INST] response"}

# Fine-tune with LoRA
mlx_lm.lora \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --train \
    --data data/train.jsonl \
    --iters 1000 \
    --learning-rate 1e-5 \
    --batch-size 4 \
    --lora-layers 16
```

Or use the Python API in notebooks (recommended for experimentation).

## 9. Common Issues

### Ollama not responding
```bash
# Kill and restart
killall ollama
ollama serve &
```

### MLX out of memory
- Use smaller models or 4-bit quantization
- Reduce batch size
- Close other memory-intensive apps

### Model download fails
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
rm -rf ~/.ollama/models/
```

## 10. Next Steps

1. Run baseline notebook: `notebooks/01_baseline_experiments.ipynb`
2. Test multi-agent: `notebooks/02_multi_agent_comparison.ipynb`
3. Start fine-tuning experiments

## Useful Resources

- MLX Examples: https://github.com/ml-explore/mlx-examples
- MLX-LM Docs: https://github.com/ml-explore/mlx-examples/tree/main/llms
- Ollama Library: https://ollama.ai/library
- HuggingFace MLX Models: https://huggingface.co/mlx-community

## Quick Reference Card

```bash
# Activate environment
source venv/bin/activate

# Start Ollama
ollama serve &

# Start Jupyter
cd notebooks && jupyter notebook

# Run experiment from CLI
cd code/harness
python strategies.py --strategy single --task "What is 2+2?"

# Monitor resources
htop  # CPU/memory
nvidia-smi  # If you somehow have NVIDIA on Mac (you don't)
```
