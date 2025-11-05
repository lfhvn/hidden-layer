# Model Management - Centralized Storage Strategy

**Problem**: Multiple projects downloading models to separate `./model_cache` directories, wasting disk space.

**Solution**: Centralize all model storage to a single location with proper environment variable configuration.

---

## Centralized Model Storage Locations

### 1. Hugging Face Models (MLX, Transformers)

**Centralized Location**: `~/.cache/huggingface/`

This is the default Hugging Face cache location. All projects should use this instead of local `./model_cache` directories.

**Environment Variables**:
```bash
# In your ~/.bashrc, ~/.zshrc, or shell profile:
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/hub
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets
```

**Benefits**:
- Models downloaded once, shared across all projects
- Standard location recognized by all HF libraries
- Easy to manage and clean up

### 2. Ollama Models

**Location**: `~/.ollama/models/` (managed by Ollama)

Ollama automatically manages this - no action needed.

**List models**: `ollama list`
**Remove model**: `ollama rm <model-name>`
**Check space**: `du -sh ~/.ollama/models`

### 3. Custom/Project-Specific Models

**Location**: `~/models/` (optional, for non-HF models)

For any custom models not from Hugging Face:
```bash
export CUSTOM_MODEL_DIR=~/models
```

---

## Current Issues Found

### Projects with Local Model Caches

These projects are configured to download to **local** `./model_cache` directories:

1. **representations/latent-space/lens/**
   - `.env.example`: `HF_CACHE_DIR=./model_cache`
   - `backend/app/config.py`: `hf_cache_dir: str = "./model_cache"`

2. **alignment/steerability/**
   - `.env.example`: `HF_CACHE_DIR=./model_cache`
   - `backend/app/config.py`: `hf_cache_dir: str = "./model_cache"`

3. **web-tools/steerability/**
   - `.env.example`: `HF_CACHE_DIR=./model_cache`
   - `backend/app/config.py`: `hf_cache_dir: str = "./model_cache"`

4. **web-tools/latent-lens/**
   - Likely has similar configuration

**Problem**: Each project downloads its own copy of models!

---

## Migration Plan

### Step 1: Set Global Environment Variables

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
# Centralized model storage
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/hub
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets

# Optional: Custom models
export CUSTOM_MODEL_DIR=~/models
```

Then reload: `source ~/.bashrc` (or restart terminal)

### Step 2: Update Root .env.example

Add centralized model configuration to `/home/user/hidden-layer/.env.example`:

```bash
# Model Storage Configuration
# These paths are shared across all projects to avoid duplicate downloads
HF_HOME=~/.cache/huggingface
TRANSFORMERS_CACHE=~/.cache/huggingface/hub
HF_DATASETS_CACHE=~/.cache/huggingface/datasets
HF_CACHE_DIR=~/.cache/huggingface/hub

# Optional: Custom models directory
CUSTOM_MODEL_DIR=~/models
```

### Step 3: Update Project .env.example Files

Change `HF_CACHE_DIR=./model_cache` to `HF_CACHE_DIR=~/.cache/huggingface/hub` in:
- `representations/latent-space/lens/.env.example`
- `alignment/steerability/.env.example`
- `web-tools/steerability/.env.example`
- `web-tools/latent-lens/.env.example` (if exists)

### Step 4: Update Config Files

Change default in Python config files from `"./model_cache"` to `os.path.expanduser("~/.cache/huggingface/hub")`:
- `representations/latent-space/lens/backend/app/config.py`
- `alignment/steerability/backend/app/config.py`
- `web-tools/steerability/backend/app/config.py`
- `web-tools/latent-lens/backend/app/config.py` (if exists)

### Step 5: Update .gitignore

Ensure local model caches are ignored (already done):
```
# Model caches
.cache/
*.gguf
*.bin
model_cache/

# MLX/HuggingFace caches
.mlx_cache/
huggingface/
```

### Step 6: Clean Up Local Caches

After migration, remove local `model_cache` directories:
```bash
# Find all local model caches
find /home/user/hidden-layer -type d -name "model_cache" 2>/dev/null

# Remove them (AFTER verifying centralized cache works!)
find /home/user/hidden-layer -type d -name "model_cache" -exec rm -rf {} +
```

---

## Verification

### Check Disk Usage

**Before migration**:
```bash
# Check for duplicate model caches
du -sh $(find . -type d -name "model_cache" 2>/dev/null)
```

**After migration**:
```bash
# Check centralized cache
du -sh ~/.cache/huggingface/

# Check Ollama
du -sh ~/.ollama/models/
```

### Test Model Loading

```python
# Test that models load from centralized cache
import os
from transformers import AutoModel

# Should use ~/.cache/huggingface/hub
model = AutoModel.from_pretrained("gpt2")
print(f"Cache dir: {os.getenv('TRANSFORMERS_CACHE')}")
```

For MLX:
```python
from mlx_lm import load

# MLX respects HF_HOME environment variable
model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
print(f"HF Home: {os.getenv('HF_HOME')}")
```

---

## Best Practices

### 1. Always Use Environment Variables

Never hardcode `./model_cache` - always use environment variables:

```python
import os
cache_dir = os.getenv("HF_CACHE_DIR", os.path.expanduser("~/.cache/huggingface/hub"))
```

### 2. Document Cache Location

In project README files, note:
```markdown
## Model Storage

This project uses centralized model storage at `~/.cache/huggingface/`.
See `MODEL_MANAGEMENT.md` in the repo root for details.
```

### 3. Check Before Downloading

Use `--cache-dir` parameter when explicitly needed:
```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "gpt2",
    cache_dir=os.getenv("TRANSFORMERS_CACHE")
)
```

### 4. Regular Cleanup

Periodically clean up unused models:
```bash
# List models in cache
ls -lh ~/.cache/huggingface/hub/

# Remove specific model
rm -rf ~/.cache/huggingface/hub/models--<model-name>

# Or use huggingface-cli (if installed)
huggingface-cli delete-cache
```

---

## Disk Space Management

### Estimate Storage Needs

**Typical Model Sizes** (4-bit quantized for MLX):
- 0.6B model: ~500 MB
- 8B model: ~5-8 GB
- 14B model: ~10-14 GB
- 32B model: ~25-32 GB
- 70B model: ~50-70 GB
- 235B MoE model: ~60-80 GB

**Recommended**: Keep 200-300 GB free for model experimentation.

### Monitor Usage

```bash
# Check cache size
du -sh ~/.cache/huggingface/

# Check available space
df -h ~

# List largest cached models
du -sh ~/.cache/huggingface/hub/models--* | sort -h | tail -10
```

### Clean Up Strategy

1. **Keep frequently used models**: Don't delete models used in active research
2. **Remove old versions**: Delete older model snapshots
3. **Archive large models**: Move rarely-used large models to external storage

```bash
# Archive large model
mv ~/.cache/huggingface/hub/models--<name> /external-drive/model-archive/

# Symlink if needed later
ln -s /external-drive/model-archive/models--<name> ~/.cache/huggingface/hub/
```

---

## MLX-Specific Notes

### MLX Model Storage

MLX uses Hugging Face cache automatically:
- **Location**: `~/.cache/huggingface/hub/models--mlx-community--<model-name>`
- **Format**: Safetensors or MLX-specific formats
- **No separate cache**: Shares HF cache with Transformers

### MLX Model Conversion

If converting models to MLX format:
```bash
# Convert and save to centralized location
python -m mlx_lm.convert \
    --hf-path <hf-model-name> \
    --mlx-path ~/.cache/huggingface/hub/models--mlx-community--<custom-name>
```

---

## Ollama-Specific Notes

### Ollama Model Management

Ollama handles its own model storage at `~/.ollama/models/`:

```bash
# List installed models
ollama list

# Pull a model (downloads to ~/.ollama/models/)
ollama pull llama3.2

# Remove a model to free space
ollama rm llama3.2

# Check Ollama storage usage
du -sh ~/.ollama/models/
```

### Ollama Model Storage Format

- **Location**: `~/.ollama/models/blobs/`
- **Format**: Ollama-specific format (not compatible with HF)
- **Deduplication**: Ollama deduplicates model layers automatically

**Note**: Ollama models cannot be shared with HF/MLX - they're separate ecosystems.

---

## Summary

### ✅ DO

- Set `HF_HOME=~/.cache/huggingface` globally
- Use environment variables in all code
- Document cache location in project READMEs
- Monitor disk usage regularly
- Clean up unused models

### ❌ DON'T

- Use local `./model_cache` directories
- Hardcode cache paths in code
- Download the same model multiple times
- Let cache grow unbounded

### 📊 Expected Savings

**Before**: Each project has its own cache
- Project 1: 20 GB
- Project 2: 20 GB (same models!)
- Project 3: 15 GB (some overlap)
- **Total**: ~55 GB

**After**: Centralized cache
- Shared cache: 25 GB (no duplicates)
- **Savings**: ~30 GB (55% reduction)

---

## Next Steps

1. ✅ Read this document
2. ⬜ Set environment variables in shell profile
3. ⬜ Update `.env.example` files
4. ⬜ Update Python config files
5. ⬜ Test model loading from centralized cache
6. ⬜ Remove local `model_cache` directories
7. ⬜ Document in project READMEs
8. ⬜ Monitor disk usage

---

**Questions?** See `docs/infrastructure/llm-providers.md` or ask in the lab discussion.
