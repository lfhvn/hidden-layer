# Integration with Hidden Layer Harness

How **Latent Topologies** connects to the main **Hidden Layer** multi-agent experimentation harness.

---

## Overview

**Hidden Layer** (`/code/harness/`) is a research harness for comparing single-model vs multi-agent LLM strategies on Apple Silicon.

**Latent Topologies** (`/latent-topologies/`) is a mobile app for exploring embedding latent space phenomenologically.

### Shared Research Themes

1. **Interpretability**: Understanding how models represent knowledge
2. **Embodied AI**: Making abstract AI structures tangible
3. **Local-first**: Running experiments on M4 Max hardware
4. **Reproducibility**: Experiment tracking and versioning

---

## Connection Points

### 1. **Corpus Generation via Multi-Agent Strategies**

Use Hidden Layer's **multi-agent strategies** to generate high-quality, diverse corpus entries for Latent Topologies.

#### Example: Debate Strategy for Concept Expansion

```python
# In /code/harness/
from harness import run_strategy, get_tracker

# Generate diverse definitions for "consciousness"
task = """Generate 5 distinct but related philosophical definitions of 'consciousness',
each from a different theoretical perspective (phenomenology, functionalism,
neuroscience, etc.). Format each as: term: definition"""

result = run_strategy(
    "debate",
    task,
    provider="ollama",
    model="llama3.2:latest",
    n_debaters=3,
    temperature=0.8
)

# Parse result.output into corpus entries
# Save to /latent-topologies/data/corpus.csv
```

**Why this helps**:
- Debates produce **diverse perspectives** → better coverage of semantic space
- Multi-agent approaches avoid single-model bias
- Systematic generation of 5-10k entries

---

### 2. **Embedding Model Comparison via Harness**

Use the harness to **systematically compare embedding models** on downstream tasks.

#### Script: `/latent-topologies/scripts/compare_embedding_models.py`

```python
#!/usr/bin/env python3
"""
Use Hidden Layer harness to compare embedding models.
Metrics: neighbor prediction accuracy, cluster coherence.
"""
import sys
sys.path.insert(0, '../code')

from harness import get_tracker, ExperimentConfig
from sentence_transformers import SentenceTransformer
import numpy as np

MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "sentence-t5-base",
]

def evaluate_embedding_model(model_name: str, corpus_path: str):
    """Evaluate embedding quality for Latent Topologies use case."""

    # Load corpus
    df = pd.read_csv(corpus_path)
    texts = df["text"].tolist()

    # Generate embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Metrics
    metrics = {}

    # 1. Neighbor coherence: Do kNN share topics?
    neighbors = compute_knn(embeddings, k=12)
    coherence = compute_topic_coherence(neighbors, df["topic"])
    metrics["neighbor_coherence"] = coherence

    # 2. Cluster quality: Silhouette score
    from sklearn.metrics import silhouette_score
    clusters = detect_clusters(embeddings)
    metrics["silhouette_score"] = silhouette_score(embeddings, clusters)

    # 3. Semantic similarity correlation
    # Compare model similarities to human judgments (if available)
    # metrics["human_correlation"] = ...

    return metrics


# Run experiment with tracking
tracker = get_tracker()

for model_name in MODELS:
    exp_dir = tracker.start_experiment(ExperimentConfig(
        experiment_name=f"embedding_comparison_{model_name}",
        strategy="single",
        model=model_name,
        provider="local",
    ))

    metrics = evaluate_embedding_model(model_name, "data/corpus.csv")

    tracker.log_result(ExperimentResult(
        task_id=model_name,
        strategy_name="embedding_eval",
        output=str(metrics),
        metrics=metrics,
    ))

    tracker.finish_experiment()

# Compare results
# python code/cli.py --compare-experiments embedding_comparison_*
```

**Benefit**: Systematic comparison tracked in `/experiments/`, reusable for paper.

---

### 3. **Annotation Quality via LLM-as-Judge**

Use harness **LLM-as-judge** eval to assess quality of user annotations in Latent Topologies.

#### Scenario: User Annotates a Cluster

User groups "justice", "mercy", "compassion" and labels cluster as "moral virtues".

**Validation via harness**:

```python
from harness import llm_call

prompt = f"""
Evaluate if the following concepts belong together under the label '{cluster_label}':
{concept_list}

Rate coherence 1-5 and explain reasoning.
"""

response = llm_call(prompt, provider="anthropic", model="claude-3-5-sonnet-20241022")

# Parse rating and log
coherence_score = extract_rating(response.text)
```

**Use case**:
- Validate user-generated cluster labels
- Suggest alternative labels if coherence < 3
- Track annotation quality over time

---

### 4. **Self-Consistency for Corpus Diversity**

Use **self-consistency strategy** to generate diverse phrasings of same concept.

```python
# Generate 10 variations of "emergence"
task = "Define 'emergence' in systems theory in one sentence."

result = run_strategy(
    "self_consistency",
    task,
    provider="ollama",
    model="llama3.2:latest",
    n_samples=10,
    temperature=0.9
)

# Result contains 10 diverse phrasings
# Add all to corpus to increase semantic coverage around "emergence"
```

---

### 5. **Experiment Tracking for Corpus Pipeline**

Use harness **experiment tracker** to version corpus generation runs.

```python
# In generate_corpus.py
import sys
sys.path.insert(0, '../code')

from harness import get_tracker, ExperimentConfig, ExperimentResult

tracker = get_tracker()

exp_dir = tracker.start_experiment(ExperimentConfig(
    experiment_name="corpus_generation_v2",
    strategy="template",
    notes="Added 200 AI/ML concepts, used debate strategy for expansion"
))

# Generate corpus
entries = generate_base_corpus(size=500)
expanded = expand_with_llm(entries, provider="ollama")

# Log metadata
tracker.log_result(ExperimentResult(
    task_id="corpus_generation",
    strategy_name="template + debate",
    output=f"Generated {len(expanded)} entries",
    metrics={
        "num_entries": len(expanded),
        "num_domains": len(set(e["topic"] for e in expanded)),
        "avg_length": np.mean([len(e["text"]) for e in expanded]),
    }
))

tracker.finish_experiment()
```

**Benefit**: Full reproducibility—can regenerate corpus from tracked config.

---

## Integration Architecture

```
hidden-layer/
├── code/harness/                   # Multi-agent experimentation
│   ├── llm_provider.py            # Unified LLM interface
│   ├── strategies.py              # Debate, self-consistency, etc.
│   ├── experiment_tracker.py      # Versioning & logging
│   └── evals.py                   # LLM-as-judge
│
├── latent-topologies/             # Latent space exploration
│   ├── scripts/
│   │   ├── generate_corpus.py    # → Uses harness.llm_call
│   │   ├── compare_models.py     # → Uses harness.experiment_tracker
│   │   └── validate_annotations.py # → Uses harness.evals
│   ├── data/
│   │   └── corpus.csv            # ← Generated via harness strategies
│   └── mobile-app/                # React Native app
│
└── experiments/                   # Shared experiment logs
    ├── corpus_generation_v2_*/
    ├── embedding_comparison_*/
    └── annotation_quality_*/
```

---

## Concrete Integration Workflows

### Workflow 1: Generate Corpus with Debate Strategy

```bash
# 1. Use harness to generate diverse definitions
cd /code
python cli.py "Generate 10 definitions of 'embodiment' from different perspectives" \
  --strategy debate \
  --n-debaters 3 \
  --output ../latent-topologies/data/raw_concepts.txt

# 2. Parse and add to corpus
cd ../latent-topologies
python scripts/parse_and_add_concepts.py \
  --input data/raw_concepts.txt \
  --corpus data/corpus.csv

# 3. Regenerate embeddings
python scripts/embed_corpus.py \
  --input data/corpus.csv \
  --embeddings data/embeddings.npy \
  --coords data/coords_umap.npy \
  --umap
```

---

### Workflow 2: Compare Embedding Models

```bash
# Run comparison experiment
cd latent-topologies
python scripts/compare_embedding_models.py \
  --corpus data/corpus.csv \
  --models all-MiniLM-L6-v2 all-mpnet-base-v2

# View results
cd ../code
python cli.py --compare-experiments embedding_comparison_*
```

---

### Workflow 3: Validate User Annotations

```bash
# User annotated cluster in mobile app → exports JSON
# data/user_annotations.json

# Validate with LLM-as-judge
python scripts/validate_annotations.py \
  --annotations data/user_annotations.json \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022

# Logs to /experiments/annotation_quality_*/
```

---

## Shared Configuration

Both projects use `config/models.yaml` for model presets:

```yaml
# Add Latent Topologies specific configs
latent-small:
  model: all-MiniLM-L6-v2
  provider: local
  temperature: 0.1
  use_case: embedding

latent-quality:
  model: all-mpnet-base-v2
  provider: local
  temperature: 0.1
  use_case: embedding

corpus-generation:
  model: llama3.2:latest
  provider: ollama
  temperature: 0.9
  use_case: diverse_generation
```

---

## Research Synergies

### Joint Research Questions

1. **When do multi-agent approaches improve corpus quality?**
   - Hypothesis: Debate → more diverse semantic coverage
   - Measure: UMAP cluster separation, neighbor coherence

2. **Can users predict embedding neighbors better than chance?**
   - Latent Topologies user study → collect predictions
   - Harness evals → compute precision/recall

3. **What's the ROI of local models vs API models?**
   - Cost tracking in harness
   - Quality comparison (embedding correlation, user preference)

---

## Data Sharing

### From Harness → Latent Topologies

- **Generated corpus**: `/experiments/corpus_generation_*/corpus.csv`
- **Model configs**: `/config/models.yaml`
- **Evaluation results**: `/experiments/*/summary.json`

### From Latent Topologies → Harness

- **User annotations**: Can be eval dataset for LLM-as-judge
- **Embedding benchmarks**: Test bed for strategy selection
- **Phenomenological insights**: Inform interpretability research

---

## Development Workflow

### Daily Development

```bash
# Morning: Check harness experiments
cd /code
python cli.py --list-experiments

# Generate new concepts via debate
python cli.py "Expand concept of 'affordance'" \
  --strategy debate \
  --save-to ../latent-topologies/data/new_concepts.txt

# Afternoon: Update Latent Topologies corpus
cd ../latent-topologies
python scripts/add_concepts.py --input data/new_concepts.txt
python scripts/embed_corpus.py --input data/corpus.csv --umap

# Evening: Test mobile app with updated embeddings
cd mobile-app
npx expo start
```

---

## Future Integration Opportunities

1. **Live Embedding Updates**: Mobile app calls harness API to embed new user text
2. **Collaborative Filtering**: User annotations → fine-tune embedding model
3. **Multi-Agent Annotation**: Use debate strategy to suggest cluster labels
4. **Temporal Tracking**: Track how user's latent space navigation changes over time

---

## Dependency Management

Both projects share Python environment:

```bash
# Install both sets of dependencies
cd /
pip install -r code/requirements.txt
pip install -r latent-topologies/requirements.txt

# Or use unified requirements
cat code/requirements.txt latent-topologies/requirements.txt > requirements_combined.txt
pip install -r requirements_combined.txt
```

---

## References

- **Hidden Layer Harness**: See `/CLAUDE.md`, `/README.md`
- **Latent Topologies**: See `/latent-topologies/PRD.md`
- **Experiment Tracking**: See `/code/harness/experiment_tracker.py`

---

**Bottom Line**: Latent Topologies is **both a standalone app AND a research instrument** that leverages the Hidden Layer harness for systematic corpus generation, model comparison, and evaluation—all while maintaining the local-first, reproducible philosophy.
