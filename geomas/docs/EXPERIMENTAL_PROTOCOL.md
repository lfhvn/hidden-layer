# Experimental Protocol - GeoMAS

**Detailed procedures for validating and analyzing geometric memory in multi-agent systems**

---

## Overview

This document provides step-by-step protocols for all GeoMAS experiments, from validation through predictive framework development.

---

## Phase 1: Validation Experiments (Weeks 1-3)

### Goal
Reproduce Noroozizadeh et al. (2025) findings on local models to validate our geometric analysis tools.

### 1.1 Path-Star Task Implementation

**Purpose**: Create the adversarial reasoning task that requires geometric vs associative memory.

**Procedure**:

1. **Graph Generation**
   ```python
   def generate_path_star_graph(d: int, ℓ: int) -> Graph:
       """
       Generate path-star graph with:
       - d paths branching from central node
       - Each path has length ℓ
       - Random node labels (e.g., names, concepts)
       """
       # Implementation details in geomas/code/tasks.py
   ```

2. **Task Format**
   - **In-weights memorization**: Model sees all edges during training/context
   - **Path-finding query**: "What is at the end of the path starting with X?"
   - **Success criterion**: Correct answer requires ℓ-hop composition

3. **Difficulty Levels**
   - **Easy**: d=5, ℓ=3 (15 nodes, 3-hop paths)
   - **Medium**: d=10, ℓ=5 (50 nodes, 5-hop paths)
   - **Hard**: d=20, ℓ=10 (200 nodes, 10-hop paths)

**Expected Results**:
- Models should achieve >90% accuracy despite theoretical difficulty
- Geometric quality score should correlate with accuracy
- Spectral gap should be larger than random baseline

---

### 1.2 Hidden State Extraction Validation

**Purpose**: Ensure we can reliably extract hidden states from local models.

**Procedure**:

1. **MLX Model Extraction**
   ```python
   # Test with small MLX model first
   model = "mlx-community/Llama-3.2-3B-Instruct-4bit"

   # Extract from multiple layers
   layers_to_test = [-1, -2, -3, -5, -10]

   # Verify extraction shape and properties
   assert hidden_states.shape == (n_samples, hidden_dim)
   assert not np.isnan(hidden_states).any()
   ```

2. **Ollama Model Extraction**
   ```python
   # Test with Ollama model
   model = "llama3.2:latest"

   # May need to use embedding endpoint or model inspection
   # Document limitations if full hidden state access unavailable
   ```

3. **Validation Checks**
   - Shape consistency across examples
   - Non-degenerate values (no all-zeros, no NaNs)
   - Layer-wise differences visible
   - Computational cost acceptable (<10s per extraction)

**Fallback**: If full hidden state extraction proves difficult, use model embeddings as first approximation.

---

### 1.3 Geometric Metrics Validation

**Purpose**: Verify that our geometric metrics behave as expected.

**Procedure**:

1. **Synthetic Data Tests**
   ```python
   # Test 1: Clustered data → high geometric quality
   clustered_data = generate_clustered_data(n_clusters=5)
   quality = geometric_quality_score(clustered_data)
   assert quality > 0.7

   # Test 2: Random data → low geometric quality
   random_data = np.random.randn(100, 128)
   quality = geometric_quality_score(random_data)
   assert quality < 0.5

   # Test 3: Linear manifold → intermediate quality
   linear_data = generate_linear_manifold()
   quality = geometric_quality_score(linear_data)
   assert 0.4 < quality < 0.7
   ```

2. **Graph Laplacian Validation**
   ```python
   # Verify eigenvalue properties
   spectral = compute_spectral_structure(data)

   # Should have 0 as smallest eigenvalue
   assert np.isclose(spectral['eigenvalues'][0], 0, atol=1e-6)

   # Eigenvalues should be non-negative and sorted
   assert np.all(spectral['eigenvalues'] >= -1e-6)
   assert np.all(np.diff(spectral['eigenvalues']) >= 0)
   ```

3. **Fiedler Vector Validation**
   ```python
   # For known graph structures, verify Fiedler vector
   # creates sensible partition

   # Example: Two clearly separated clusters
   # Fiedler vector should have different signs for each cluster
   ```

**Success Criteria**:
- All synthetic tests pass
- Metrics discriminate between different data structures
- Results match mathematical expectations

---

### 1.4 Baseline vs. Model Comparison

**Purpose**: Compare our local models to paper's findings.

**Procedure**:

1. **Run path-star task on local models**
   - Llama 3.2 3B (fast iteration)
   - Mistral 7B (quality baseline)
   - Llama 3.1 70B (high-end, if time permits)

2. **Measure geometric properties**
   - Spectral gap
   - Fiedler vector structure
   - Cluster coherence
   - Quality score

3. **Compare to paper benchmarks**
   - Do we see similar geometric emergence?
   - Similar spectral gaps?
   - Similar clustering patterns?

4. **Document differences**
   - Model architecture effects
   - Quantization effects (4-bit vs full precision)
   - Local vs. API models

**Deliverable**: Validation report confirming geometric memory in local models.

---

## Phase 2: Single-Model Baseline (Weeks 4-6)

### Goal
Establish geometric quality profiles for different tasks and models.

### 2.1 Task Suite Definition

**Purpose**: Create diverse tasks with varying geometric complexity.

**Task Categories**:

| Category | Example Tasks | Expected Geometric Demand |
|----------|---------------|---------------------------|
| **Multi-hop reasoning** | Path-finding, logical chains, transitive inference | High |
| **Analogical reasoning** | A:B :: C:?, pattern completion | High |
| **Planning** | Multi-step procedures, constraint satisfaction | Medium-High |
| **Math** | Word problems, equation solving | Medium |
| **Causal reasoning** | Cause-effect chains, counterfactuals | Medium |
| **Factual QA** | Single-hop retrieval, classification | Low |
| **Simple tasks** | Sentiment analysis, keyword extraction | Low |

**Task Selection**:
- 5-10 tasks per category
- Varying difficulty within category
- Ground truth available for evaluation
- Reuse existing harness tasks where possible

---

### 2.2 Geometric Profiling Procedure

**For each (task, model) pair**:

1. **Run task with single-model strategy**
   ```python
   result = run_strategy("single", task_input, provider="ollama", model=model)
   ```

2. **Extract hidden states**
   ```python
   hidden_states = extract_hidden_states(result, layer=-1)
   ```

3. **Compute geometric metrics**
   ```python
   analysis = probe.analyze(hidden_states)

   metrics = {
       'spectral_gap': analysis.spectral_gap,
       'cluster_coherence': analysis.cluster_coherence,
       'quality_score': analysis.quality_score,
       'global_structure_score': analysis.global_structure_score
   }
   ```

4. **Evaluate task performance**
   ```python
   from harness import evaluate_task
   accuracy = evaluate_task(result, task.ground_truth)
   ```

5. **Record correlation**
   ```python
   record = {
       'task': task.name,
       'task_category': task.category,
       'model': model,
       'geometric_metrics': metrics,
       'accuracy': accuracy,
       'latency_s': result.latency_s,
       'tokens': result.tokens_in + result.tokens_out
   }
   ```

**Analysis**:
- Compute correlation: `geometric_quality ↔ task_accuracy`
- Identify tasks with poor geometry but good accuracy (anomalies)
- Identify tasks with good geometry but poor accuracy (evaluation issues?)
- Create "geometric difficulty" ratings

**Deliverable**: Geometric profile database for all (task, model) combinations.

---

### 2.3 Model Comparison

**Purpose**: Understand architectural effects on geometric memory.

**Models to Compare**:
- **Local**: Llama 3.2 3B, Mistral 7B, Qwen 7B, Llama 3.1 70B
- **API** (optional): GPT-4o-mini, Claude 3.5 Haiku (for comparison)

**Metrics to Compare**:
- Average geometric quality across tasks
- Spectral gap distribution
- Correlation with performance
- Computational cost (for local models)

**Research Questions**:
- Do larger models have stronger geometric structure?
- Do instruction-tuned models differ from base models?
- Do different architectures show different geometric biases?

**Deliverable**: Model geometric capability comparison report.

---

## Phase 3: Multi-Agent Analysis (Weeks 7-10)

### Goal
Compare geometric structures across reasoning strategies.

### 3.1 Strategy-Specific Protocols

#### 3.1.1 Debate Strategy Analysis

**Procedure**:

1. **Run debate with hidden state tracking**
   ```python
   result = run_strategy(
       "debate",
       task_input=task,
       n_debaters=3,
       n_rounds=2,
       provider="ollama"
   )
   ```

2. **Extract states per round**
   ```python
   round_1_states = extract_from_debate_round(result, round=1)
   round_2_states = extract_from_debate_round(result, round=2)
   final_states = extract_from_debate_round(result, round='final')
   ```

3. **Analyze geometric evolution**
   ```python
   evolution = []
   for round_states in [round_1_states, round_2_states, final_states]:
       analysis = probe.analyze(round_states)
       evolution.append(analysis)

   # Measure improvement
   quality_improvement = evolution[-1].quality_score - evolution[0].quality_score
   spectral_gap_improvement = evolution[-1].spectral_gap - evolution[0].spectral_gap
   ```

**Research Questions**:
- Does geometric quality improve across rounds?
- Does spectral gap increase (stronger structure)?
- Do clusters become more separated?
- Can we visualize geometric "triangulation"?

---

#### 3.1.2 Manager-Worker Analysis

**Procedure**:

1. **Run manager-worker with subtask tracking**
   ```python
   result = run_strategy(
       "manager_worker",
       task_input=complex_task,
       n_workers=3,
       provider="ollama"
   )
   ```

2. **Extract states per subtask**
   ```python
   full_task_states = extract_single_model_states(complex_task)
   subtask_states = [extract_states(subtask) for subtask in result.subtasks]
   synthesis_states = extract_synthesis_states(result)
   ```

3. **Compare geometric cleanliness**
   ```python
   full_task_quality = probe.geometric_quality_score(full_task_states)
   subtask_qualities = [probe.geometric_quality_score(s) for s in subtask_states]

   # Hypothesis: subtasks should have higher average quality
   avg_subtask_quality = np.mean(subtask_qualities)

   decomposition_benefit = avg_subtask_quality - full_task_quality
   ```

**Research Questions**:
- Do subtasks have cleaner geometric structure?
- Does decomposition create separable geometric subspaces?
- How does synthesis combine geometric structures?

---

#### 3.1.3 Self-Consistency Analysis

**Procedure**:

1. **Run self-consistency with multiple samples**
   ```python
   result = run_strategy(
       "self_consistency",
       task_input=task,
       n_samples=5,
       temperature=0.8,
       provider="ollama"
   )
   ```

2. **Extract states per sample**
   ```python
   sample_states = [extract_states(sample) for sample in result.samples]
   ```

3. **Measure geometric consistency**
   ```python
   # Compute pairwise geometric distance between samples
   from scipy.spatial.distance import pdist

   # For each sample, get Fiedler vector as geometric signature
   fiedler_vectors = [probe.compute_spectral_structure(s)['fiedler_vector']
                      for s in sample_states]

   # Measure consistency: low variance = high consistency
   consistency_score = 1.0 - np.mean(pdist(fiedler_vectors, metric='cosine'))
   ```

**Research Questions**:
- Do correct samples share geometric structure?
- Do incorrect samples have different geometry?
- Can we identify "geometric consensus"?

---

### 3.2 Cross-Strategy Comparison

**For each task**:

1. Run all strategies: single, debate, manager-worker, self-consistency
2. Extract and analyze geometric structures
3. Record performance and cost
4. Compute geometric advantage metrics

**Comparison Metrics**:

```python
comparison = {
    'task': task.name,
    'single': {
        'geometric_quality': ...,
        'accuracy': ...,
        'latency_s': ...,
        'cost_usd': ...
    },
    'debate': {
        'geometric_quality': ...,
        'accuracy': ...,
        'latency_s': ...,
        'cost_usd': ...,
        'quality_improvement_vs_single': ...,
        'accuracy_gain_vs_single': ...
    },
    # ... similar for other strategies
}
```

**Key Analyses**:
- Correlation: `geometric_improvement ↔ accuracy_gain`
- Cost-benefit: `accuracy_gain / (latency_ratio × cost_ratio)`
- Identification of when multi-agent helps most

**Deliverable**: Strategy comparison database with geometric metrics.

---

## Phase 4: Predictive Framework (Weeks 11-14)

### Goal
Build model to predict multi-agent benefit from geometric analysis.

### 4.1 Feature Engineering

**Input Features** (from single-model baseline):
- `spectral_gap`: Strength of geometric structure
- `cluster_coherence`: Separation of concepts
- `quality_score`: Overall geometric quality
- `n_samples`: Number of samples in task
- `task_category`: One-hot encoded category
- `model_size`: Model parameters (3B, 7B, 70B)

**Target Variable**:
- `multi_agent_benefit`: Accuracy gain (multi_agent_acc - single_acc)

Or multiple targets for different strategies:
- `debate_benefit`
- `manager_worker_benefit`
- `self_consistency_benefit`

---

### 4.2 Model Training

**Approach 1: Simple Regression**

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Prepare data
X = features_df[['spectral_gap', 'cluster_coherence', 'quality_score', ...]]
y = targets_df['debate_benefit']

# Train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
scores = cross_val_score(model, X, y, cv=5)
print(f"CV R² score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

**Approach 2: Classification (High/Low Benefit)**

```python
from sklearn.ensemble import GradientBoostingClassifier

# Binarize target: benefit > threshold
y_binary = (y > 0.1).astype(int)

# Train classifier
clf = GradientBoostingClassifier(n_estimators=100)
clf.fit(X_train, y_binary)

# Evaluate
accuracy = clf.score(X_test, y_binary)
print(f"Prediction accuracy: {accuracy:.3f}")
```

**Success Criteria**:
- Classification accuracy > 70%
- Regression R² > 0.5
- Feature importance aligns with intuition (geometric quality should be top feature)

---

### 4.3 Validation on Held-Out Tasks

**Procedure**:

1. **Hold out 20% of tasks** for final validation (not used in training)

2. **For each held-out task**:
   ```python
   # 1. Run single model, analyze geometry
   single_result = run_strategy("single", task)
   geometry = probe.analyze(single_result)

   # 2. Predict benefit
   features = extract_features(geometry, task, model)
   predicted_benefit = predictor.predict([features])[0]

   # 3. Actually run multi-agent
   debate_result = run_strategy("debate", task)

   # 4. Measure actual benefit
   actual_benefit = debate_result.accuracy - single_result.accuracy

   # 5. Compute prediction error
   error = abs(predicted_benefit - actual_benefit)
   ```

3. **Aggregate validation metrics**:
   - Mean absolute error (MAE)
   - Prediction accuracy for high/low benefit
   - Correlation between predicted and actual

**Deliverable**: Validated predictive model with accuracy report.

---

### 4.4 Integration into Harness

**Create recommendation function**:

```python
# In geomas/code/recommendations.py

def recommend_strategy(
    task_input: str,
    model: str = "llama3.2:latest",
    provider: str = "ollama",
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Recommend which strategy to use based on geometric analysis.

    Returns:
        {
            'recommended_strategy': str,
            'confidence': float,
            'reasoning': str,
            'alternatives': List[Dict],
            'geometric_metrics': Dict
        }
    """
    # Run single model baseline
    single_result = run_strategy("single", task_input, provider, model)

    # Analyze geometry
    probe = GeometricProbe(model, provider)
    geometry = probe.analyze(single_result)

    # Predict benefit for each strategy
    predictor = load_trained_predictor()
    predictions = {
        'debate': predictor.predict_debate_benefit(geometry),
        'manager_worker': predictor.predict_manager_worker_benefit(geometry),
        'self_consistency': predictor.predict_self_consistency_benefit(geometry)
    }

    # Select best strategy
    best_strategy = max(predictions, key=predictions.get)
    confidence = predictions[best_strategy]

    # Generate reasoning
    if geometry.quality_score < 0.5:
        reasoning = f"Low geometric quality ({geometry.quality_score:.2f}) suggests multi-agent will help"
    else:
        reasoning = f"High geometric quality ({geometry.quality_score:.2f}) suggests single model sufficient"

    return {
        'recommended_strategy': best_strategy if confidence > confidence_threshold else 'single',
        'confidence': confidence,
        'reasoning': reasoning,
        'alternatives': [{'strategy': s, 'predicted_benefit': b} for s, b in predictions.items()],
        'geometric_metrics': {
            'quality_score': geometry.quality_score,
            'spectral_gap': geometry.spectral_gap,
            'cluster_coherence': geometry.cluster_coherence
        }
    }
```

---

## Phase 5: Deep Interpretability (Weeks 15-16)

### Goal
Understand *how* and *where* geometric memory emerges.

### 5.1 Layer-by-Layer Analysis

**Procedure**:

1. **Extract from all layers**
   ```python
   all_layers = list(range(model.n_layers))
   layer_analyses = {}

   for layer in all_layers:
       hidden_states = extract_hidden_states(task, layer=layer)
       analysis = probe.analyze(hidden_states)
       layer_analyses[layer] = analysis
   ```

2. **Plot geometric quality vs. layer depth**
   ```python
   qualities = [layer_analyses[l].quality_score for l in all_layers]
   plt.plot(all_layers, qualities)
   plt.xlabel("Layer")
   plt.ylabel("Geometric Quality")
   plt.title("Geometric Structure Emergence Across Layers")
   ```

3. **Identify critical layers**
   - Where does geometry first appear?
   - Which layer has strongest geometry?
   - Does it vary by task type?

---

### 5.2 Attention Pattern Analysis

**Hypothesis**: Attention structure correlates with geometric structure.

**Procedure**:

1. **Extract attention weights**
   ```python
   attention_patterns = extract_attention_weights(model, task)
   ```

2. **Compute attention-based graph**
   ```python
   # Treat attention as graph edges
   attention_graph = aggregate_attention_across_heads(attention_patterns)
   ```

3. **Compare to geometric structure**
   ```python
   # Compute Laplacian from attention graph
   attention_laplacian = compute_laplacian(attention_graph)

   # Compare eigenvectors to hidden state geometry
   correlation = compute_eigenvector_correlation(
       attention_laplacian,
       hidden_state_laplacian
   )
   ```

**Research Question**: Does attention structure predict geometric structure?

---

### 5.3 Visualization Tools

**Interactive Geometric Explorer**:

```python
# Create interactive tool to explore geometric structures

from geomas.visualizations import GeometricExplorer

explorer = GeometricExplorer()

explorer.add_layer_comparison(task, model, layers=[-1, -5, -10])
explorer.add_strategy_comparison(task, strategies=["single", "debate"])
explorer.add_evolution_plot(debate_result)

explorer.launch_browser()  # Opens interactive dashboard
```

**Features**:
- 2D/3D projections of geometric structure
- Layer-by-layer slider
- Strategy comparison side-by-side
- Debate round evolution animation
- Eigenvalue spectrum plots
- Cluster coherence heatmaps

---

## Logging and Reproducibility

### Experiment Logging

Every experiment should log:

```python
experiment_log = {
    'timestamp': datetime.now().isoformat(),
    'phase': 'validation',  # or 'baseline', 'multi_agent', etc.
    'task': task.name,
    'model': model_name,
    'provider': provider,
    'strategy': strategy_name,

    # Geometric metrics
    'geometric_analysis': {
        'spectral_gap': ...,
        'quality_score': ...,
        'cluster_coherence': ...,
        'fiedler_vector': ...,  # Serialize as list
        'eigenvalues': ...
    },

    # Task metrics
    'performance': {
        'accuracy': ...,
        'latency_s': ...,
        'tokens_in': ...,
        'tokens_out': ...,
        'cost_usd': ...
    },

    # Reproducibility
    'random_seed': seed,
    'git_hash': get_git_hash(),
    'config': {
        'temperature': ...,
        'n_debaters': ...,
        # ... all hyperparameters
    }
}
```

**Storage**:
- Save to `geomas/experiments/{experiment_name}_{timestamp}/`
- JSON for metadata, NPZ for arrays
- Link to harness experiment tracker

---

## Success Criteria Summary

| Phase | Success Criteria |
|-------|------------------|
| **Validation** | Reproduce geometric memory in local models; tools validated |
| **Baseline** | Geometric quality correlates with performance (R > 0.5) |
| **Multi-Agent** | Identify measurable geometric differences across strategies |
| **Predictive** | Predict multi-agent benefit with >70% accuracy |
| **Interpretability** | Identify layer of geometric emergence; create visualizations |

---

## Next Steps After Protocol Completion

1. **Paper Writing**: Draft "Geometric Memory in Multi-Agent LLM Systems"
2. **Tool Release**: Package GeoMAS as standalone library
3. **Integration**: Add geometric analysis to CRIT, SELPHI
4. **Extension**: Test on fine-tuned models, other architectures (Mamba, etc.)

---

**END OF EXPERIMENTAL PROTOCOL**
