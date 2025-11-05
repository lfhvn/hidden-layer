# Development Setup — Latent Topologies

Complete setup guide for developing Latent Topologies on Apple Silicon (M4 Max).

---

## 🖥️ Hardware Requirements

**Recommended**:
- Apple Silicon (M3/M4) with ≥ 32GB unified memory
- 50GB free storage (models, data, dependencies)

**This project is optimized for**:
- M4 Max with 128GB RAM
- Can run embedding models entirely on-device
- Fast UMAP projections with large datasets

---

## 🐍 Python Environment Setup

### 1. Install Python 3.10+

```bash
# Using Homebrew
brew install python@3.11

# Verify
python3 --version  # Should be 3.10+
```

### 2. Create Virtual Environment

```bash
cd latent-topologies
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Optional: Install development tools
pip install ipython jupyter black ruff
```

### 4. Verify Installation

```bash
python3 -c "import sentence_transformers; import umap; print('✓ All imports successful')"
```

---

## 📊 Data Pipeline Setup

### 1. Test Basic Pipeline

```bash
# Navigate to project root
cd latent-topologies

# Test embedding generation (uses sample corpus)
python scripts/embed_corpus.py \
  --input data/corpus.csv \
  --embeddings data/test_embeddings.npy \
  --coords data/test_coords.npy \
  --umap

# Test database generation
python scripts/build_seed_db.py \
  --corpus data/corpus.csv \
  --embeddings data/test_embeddings.npy \
  --coords data/test_coords.npy \
  --out data/test_seed.db \
  --knn 12
```

### 2. Download Pre-trained Models

Models are downloaded automatically on first use, but you can pre-cache:

```bash
python3 << EOF
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"Model cached at: {model._model_card_vars['model_name']}")
EOF
```

**Model sizes**:
- `all-MiniLM-L6-v2`: ~90MB (recommended for dev)
- `all-mpnet-base-v2`: ~420MB (higher quality)
- `sentence-t5-base`: ~220MB (alternative)

---

## 📱 React Native / Expo Setup

### 1. Install Node.js

```bash
# Using Homebrew
brew install node

# Verify
node --version  # Should be 18+
npm --version   # Should be 9+
```

### 2. Install Expo CLI

```bash
npm install -g expo-cli

# Verify
expo --version
```

### 3. Initialize Expo Project (when ready)

```bash
# This will be done in M1 phase
npx create-expo-app mobile-app --template blank-typescript
cd mobile-app
npm install
```

### 4. Install iOS Simulator (for testing)

- Download Xcode from App Store (required for iOS builds)
- Open Xcode → Preferences → Components → Install iOS Simulator
- Test: `expo start` then press `i` to open iOS simulator

---

## 🧠 Model Conversion Setup (for on-device inference)

### Core ML (iOS)

```bash
# Install coremltools
pip install coremltools torch transformers

# Conversion script will be in scripts/convert_to_coreml.py
```

### TensorFlow Lite (Android - optional)

```bash
# Install TFLite converter
pip install tensorflow tf-keras

# Conversion script will be in scripts/convert_to_tflite.py
```

---

## 🔧 Development Tools

### Jupyter Notebooks (for exploration)

```bash
# Already installed if you followed optional step
jupyter notebook

# Or use JupyterLab
pip install jupyterlab
jupyter lab
```

### Code Quality

```bash
# Format Python code
black scripts/ --line-length 100

# Lint
ruff check scripts/

# Type checking (optional)
pip install mypy
mypy scripts/
```

---

## 📂 Project Structure After Setup

```
latent-topologies/
├── venv/                    # Python virtual environment
├── data/
│   ├── corpus.csv          # Source data (tracked)
│   ├── embeddings.npy      # Generated (gitignored)
│   ├── coords_umap.npy     # Generated (gitignored)
│   └── seed.db             # Generated (gitignored)
├── models/                  # Model cache (gitignored, auto-created)
├── scripts/
│   ├── embed_corpus.py
│   ├── build_seed_db.py
│   ├── generate_corpus.py  # To be created
│   └── convert_to_coreml.py # To be created
├── mobile-app/              # Expo project (created in M1)
└── research/
    ├── notebooks/           # Jupyter notebooks
    └── outputs/             # Study results (gitignored)
```

---

## 🚀 Quick Start Workflow

### Daily Development

```bash
# 1. Activate environment
cd latent-topologies
source venv/bin/activate

# 2. Run data pipeline (if corpus changed)
python scripts/embed_corpus.py --input data/corpus.csv --embeddings data/embeddings.npy --coords data/coords_umap.npy --umap
python scripts/build_seed_db.py --corpus data/corpus.csv --embeddings data/embeddings.npy --coords data/coords_umap.npy --out data/seed.db

# 3. Start mobile app (when available)
cd mobile-app && npx expo start
```

### Testing Changes

```bash
# Test embedding generation
python scripts/embed_corpus.py --input data/corpus.csv --embeddings data/test.npy --model all-MiniLM-L6-v2

# Inspect database
sqlite3 data/seed.db
> .schema
> SELECT COUNT(*) FROM item;
> .quit
```

---

## 🐛 Troubleshooting

### Python Import Errors

```bash
# Ensure virtual environment is activated
which python  # Should point to venv/bin/python

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### UMAP Installation Issues on M1/M2/M4

```bash
# If UMAP fails to build, install dependencies first
brew install llvm libomp

# Then reinstall
pip install --no-binary umap-learn umap-learn
```

### Sentence Transformers Cache Location

Models are cached at:
- macOS: `~/.cache/torch/sentence_transformers/`

To change cache location:
```bash
export SENTENCE_TRANSFORMERS_HOME=/path/to/cache
```

### Expo Not Starting

```bash
# Clear cache
npx expo start --clear

# Reset
rm -rf node_modules package-lock.json
npm install
```

### Memory Issues (Large Corpus)

```bash
# Process in batches (modify embed_corpus.py to use smaller batch_size)
python scripts/embed_corpus.py --input data/corpus.csv --embeddings data/embeddings.npy --batch-size 32

# Or use swap on disk for UMAP
# Modify umap.UMAP(low_memory=True)
```

---

## 🧪 Validation Checklist

Before moving to mobile development:

- [ ] Python environment activates without errors
- [ ] Can generate embeddings from sample corpus
- [ ] UMAP projection completes successfully
- [ ] SQLite database builds with proper schema
- [ ] Node.js and Expo installed
- [ ] Can run basic Expo app in simulator

---

## 📚 Additional Resources

**Sentence Transformers**:
- Docs: https://www.sbert.net/
- Models: https://huggingface.co/sentence-transformers

**UMAP**:
- Docs: https://umap-learn.readthedocs.io/
- Params guide: https://umap-learn.readthedocs.io/en/latest/parameters.html

**Core ML**:
- Apple Docs: https://developer.apple.com/documentation/coreml
- coremltools: https://coremltools.readme.io/

**Expo**:
- Docs: https://docs.expo.dev/
- React Native: https://reactnative.dev/

---

## 🔄 Keeping Dependencies Updated

```bash
# Update Python packages
pip list --outdated
pip install --upgrade sentence-transformers umap-learn

# Update Node packages (when mobile app exists)
cd mobile-app
npm outdated
npm update
```

---

**Next**: After setup is complete, proceed to M0 (Foundation) in TASKS.md to build your first corpus and validate the full pipeline.
