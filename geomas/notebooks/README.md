# GeoMAS Notebooks

Interactive experiments for geometric memory analysis in multi-agent systems.

---

## Notebooks Overview

### 00_quick_start_synthetic.ipynb
**Purpose**: Test geometric analysis tools on synthetic data
**Time**: 10-15 minutes
**Prerequisites**: None

**What it does**:
- Validates geometric metrics on known structures
- Tests clustered vs random vs manifold data
- Verifies visualization tools work
- Establishes baseline metric ranges

**Run this first** to ensure tools work before testing on real models.

---

### 01_geometric_validation.ipynb
**Purpose**: Reproduce paper findings on local models (Phase 1)
**Time**: 30-45 minutes
**Prerequisites**: Ollama running with llama3.2

**What it does**:
- Generates path-star graph tasks
- Tests models on adversarial reasoning
- Extracts hidden states (when implemented)
- Validates geometric memory emergence
- Compares to Noroozizadeh et al. (2025) results

**Status**: ✓ Task generation validated | Hidden state extraction pending

---

### 02_multi_agent_geometric_comparison.ipynb
**Purpose**: Compare geometric structures across strategies (Phase 3)
**Time**: 45-60 minutes
**Prerequisites**: Harness available, Ollama running

**What it does**:
- Runs same task through single, debate, manager-worker
- Extracts and compares geometric structures
- Tests hypotheses about multi-agent refinement
- Cost-benefit analysis
- Visualizes geometric differences

**Status**: ✓ Framework ready | Needs real hidden states for conclusions

---

## Quick Start

```bash
# 1. Activate environment
cd /path/to/hidden-layer
source venv/bin/activate

# 2. Install GeoMAS dependencies (if not done)
pip install scipy scikit-learn umap-learn plotly

# 3. Start Jupyter
cd geomas/notebooks
jupyter notebook

# 4. Open 00_quick_start_synthetic.ipynb
# Run all cells to validate tools
```

---

## Notebook Workflow

### Phase 1: Validation (Weeks 1-3)
1. **00_quick_start_synthetic** → Validate tools ✓
2. **01_geometric_validation** → Test on real models
3. Iterate until geometric memory confirmed

### Phase 2: Baseline (Weeks 4-6)
- Create `03_single_model_baseline.ipynb` (future)
- Profile geometric quality across tasks
- Build task-model-geometry database

### Phase 3: Multi-Agent (Weeks 7-10)
1. **02_multi_agent_geometric_comparison** → Compare strategies
2. Run on 20-30 diverse tasks
3. Collect geometric evolution data

### Phase 4: Predictive (Weeks 11-14)
- Create `04_predictive_framework.ipynb` (future)
- Train regression: geometry → multi-agent benefit
- Validate on held-out tasks

### Phase 5: Interpretability (Weeks 15-16)
- Create `05_deep_interpretability.ipynb` (future)
- Layer-by-layer analysis
- Attention pattern correlations

---

## Dependencies

**Core**:
```
numpy
scipy
scikit-learn
matplotlib
```

**Optional** (for best visualizations):
```
umap-learn
plotly
seaborn
```

**Harness** (for real model testing):
```
# From hidden-layer project
from harness import llm_call, run_strategy
```

---

## Common Issues

### "Harness not available"
- Ensure you're in hidden-layer project directory
- Check `sys.path.append('../..')` points correctly
- Notebooks work in simulation mode without harness

### "Hidden state extraction not implemented"
- Expected for now - validation uses simulated data
- See `geomas/code/geometric_probes.py` for TODO
- MLX extraction is priority (easier than Ollama)

### "UMAP not installed"
- Run: `pip install umap-learn`
- Falls back to PCA if unavailable
- Not critical for validation

---

## Output

Each notebook saves results to:
```
geomas/experiments/
├── validation/
│   └── validation_summary.json
├── multi_agent/
│   └── comparison_results.json
└── ... (more as you run experiments)
```

---

## Next Steps

After completing these notebooks:

1. **Implement hidden state extraction**
   - Start with MLX (use model hooks)
   - Add Ollama support (embedding endpoint)

2. **Run systematic experiments**
   - 20+ tasks across difficulty levels
   - Multiple models (3B, 7B, 70B)
   - All strategies (single, debate, manager-worker, self-consistency)

3. **Build predictive model**
   - Collect geometry → performance data
   - Train regression/classifier
   - Integrate into harness

4. **Write paper**
   - Novel findings on multi-agent geometry
   - Practical guidelines for strategy selection
   - Submit to ICLR/NeurIPS/ICML

---

**Happy experimenting! 🧪🔬**

For questions, see main project README or research proposal.
