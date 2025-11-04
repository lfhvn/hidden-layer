# Custom Strategies Integration - Summary

## What Was Done

Successfully integrated **design critique** and **interdisciplinary team** strategies into the benchmark evaluation framework, allowing these custom mental models to be tested on established benchmarks alongside standard multi-agent strategies.

## Changes Made

### 1. Added Two New Strategies to `code/harness/strategies.py`

#### `design_critique_strategy()`
- Generates initial draft
- Multiple critics provide feedback from different perspectives
- Iteratively refines based on critique
- Returns final refined output

**Parameters:**
- `n_iterations`: Number of critique/revision cycles (default: 2)
- `critique_panel`: Custom critics (optional, has sensible defaults)
- Standard params: `provider`, `model`, `temperature`, `verbose`

**Default Critics:**
- Clarity Critic (comprehension & structure)
- Accuracy Critic (technical correctness)
- Completeness Critic (coverage)

#### `interdisciplinary_team_strategy()`
- Multiple domain experts analyze problem
- Synthesizer integrates all perspectives
- Iteratively refines solution with expert feedback
- Returns final integrated solution

**Parameters:**
- `refinement_rounds`: Number of refinement iterations (default: 1)
- `expert_team`: Custom experts (optional, has sensible defaults)
- Standard params: `provider`, `model`, `temperature`, `verbose`

**Default Experts:**
- Technical Lead (engineering perspective)
- User Advocate (UX/design perspective)
- Business Strategist (business perspective)

### 2. Updated Strategy Registry

Both strategies added to `STRATEGIES` dict in `strategies.py`:
```python
STRATEGIES = {
    "single": single_model_strategy,
    "debate": debate_strategy,
    "self_consistency": self_consistency_strategy,
    "manager_worker": manager_worker_strategy,
    "consensus": consensus_strategy,
    "design_critique": design_critique_strategy,          # NEW!
    "interdisciplinary_team": interdisciplinary_team_strategy,  # NEW!
}
```

### 3. Updated Benchmark Notebook

**File:** `notebooks/04_benchmark_evaluation.ipynb`

**Changes:**
- Added new markdown cell explaining the custom strategies
- Added commented-out examples in `STRATEGIES_TO_TEST` configuration
- Shows how to use default configurations
- Shows how to customize critics/experts for specific benchmarks

**Example Usage:**
```python
STRATEGIES_TO_TEST = [
    # ... existing strategies ...

    # Design Critique with defaults
    ("design_critique", {
        "n_iterations": 2,
        "provider": PROVIDER,
        "model": MODEL,
        "verbose": False
    }),

    # XFN Team with defaults
    ("interdisciplinary_team", {
        "refinement_rounds": 1,
        "provider": PROVIDER,
        "model": MODEL,
        "verbose": False
    }),
]
```

### 4. Created Documentation

#### `CUSTOM_STRATEGIES_GUIDE.md`
Comprehensive guide covering:
- Overview of both strategies
- When to use each strategy
- Basic and advanced configuration examples
- Benchmark-specific configurations (GSM8K, MMLU, TruthfulQA)
- Performance considerations (token usage, cost)
- Expected use cases and results interpretation
- Troubleshooting

#### Updated `README.md`
- Added design critique & XFN team to features list
- Added "Benchmark Evaluation" feature section
- Added documentation links section
- Updated project structure with all 6 notebooks

### 5. Cache Management

Cleared Python cache to ensure new strategies are immediately available:
```bash
find code/harness -type d -name "__pycache__" -exec rm -rf {} +
```

## How to Use

### Option 1: Use Defaults (Easiest)

Open `notebooks/04_benchmark_evaluation.ipynb`, uncomment the strategies:

```python
STRATEGIES_TO_TEST = [
    ("single", {...}),

    # Uncomment these:
    ("design_critique", {
        "n_iterations": 2,
        "provider": PROVIDER,
        "model": MODEL,
        "verbose": False
    }),

    ("interdisciplinary_team", {
        "refinement_rounds": 1,
        "provider": PROVIDER,
        "model": MODEL,
        "verbose": False
    }),
]
```

Run the notebook cells to evaluate!

### Option 2: Customize for Your Benchmark

For **GSM8K** (math problems), customize critics:
```python
("design_critique", {
    "n_iterations": 2,
    "critique_panel": [
        {
            "name": "Calculation Checker",
            "focus": "Arithmetic accuracy",
            "criteria": "Verify every calculation is correct"
        },
        {
            "name": "Logic Checker",
            "focus": "Reasoning validity",
            "criteria": "Ensure problem-solving approach is sound"
        }
    ],
    "provider": PROVIDER,
    "model": MODEL
})
```

For **MMLU** (knowledge questions), customize experts:
```python
("interdisciplinary_team", {
    "refinement_rounds": 1,
    "expert_team": [
        {
            "name": "Subject Expert",
            "role": "Domain Specialist",
            "perspective": "Deep subject knowledge",
            "system_prompt": "You are an expert. Focus on correctness."
        },
        {
            "name": "Skeptic",
            "role": "Critical Thinker",
            "perspective": "Question assumptions",
            "system_prompt": "You challenge answers and look for errors."
        }
    ],
    "provider": PROVIDER,
    "model": MODEL
})
```

### Option 3: Use in Custom Code

Import and use directly:
```python
from harness import run_strategy

# Design critique
result = run_strategy(
    "design_critique",
    task_input="Write a blog post about AI safety",
    n_iterations=3,
    provider="ollama",
    model="gpt-oss:20b"
)

print(result.output)  # Final refined version
print(f"Cost: ${result.cost_usd:.4f}")

# Access all versions
for version in result.metadata['all_versions']:
    print(f"Version {version['version']}: {len(version['content'])} chars")

# XFN team
result = run_strategy(
    "interdisciplinary_team",
    task_input="Should we build a mobile app or web app first?",
    refinement_rounds=2,
    provider="ollama",
    model="gpt-oss:20b"
)

print(result.output)  # Integrated solution

# Access expert analyses
for analysis in result.metadata['expert_analyses']:
    print(f"{analysis['expert']}: {analysis['analysis'][:100]}...")
```

## What to Expect

### Performance Characteristics

**Design Critique:**
- Token usage: ~13 LLM calls per task (2 iterations, 3 critics)
- Latency: 5-10x baseline (depends on iterations)
- Best for: Writing, creative tasks, quality-sensitive work

**Interdisciplinary Team:**
- Token usage: ~7 LLM calls per task (3 experts, 1 refinement)
- Latency: 3-7x baseline (depends on experts)
- Best for: Complex problems, strategic decisions, multi-faceted tasks

### When They Help

**Design Critique Works Well:**
- ✅ Tasks where initial drafts have fixable issues
- ✅ Writing and generation tasks
- ✅ Quality-sensitive applications
- ❌ Simple factoid questions
- ❌ Tasks where model is already at ceiling

**XFN Team Works Well:**
- ✅ Complex multi-dimensional problems
- ✅ Tasks requiring different expertise
- ✅ Strategic planning and analysis
- ❌ Simple single-domain questions
- ❌ Tasks where all experts would agree

### Results Interpretation

**Good signs:**
- Accuracy improves 5-10%+ over baseline
- Consistent improvement across tasks
- Cost/latency tradeoff is reasonable

**Warning signs:**
- No accuracy improvement
- High variance (works sometimes, fails others)
- Cost 10x higher with minimal benefit

## Testing Recommendations

1. **Start small**: Test with `NUM_TASKS = 5-10` first
2. **Use local model**: Iterate with `gpt-oss:20b` to keep costs down
3. **Compare to baseline**: Always include `("single", {...})` in strategies
4. **Monitor costs**: Check `cost_usd` in results
5. **Analyze metadata**: Look at intermediate versions/analyses

## Next Steps

1. **Restart Jupyter kernel** to pick up new strategies
2. **Open benchmark notebook**: `notebooks/04_benchmark_evaluation.ipynb`
3. **Choose a benchmark**: Start with GSM8K or MMLU
4. **Uncomment custom strategies** in configuration cell
5. **Run evaluation** and analyze results!

## Troubleshooting

**"Unknown strategy" error:**
```bash
# Clear cache
find code/harness -type d -name "__pycache__" -exec rm -rf {} +

# Restart Jupyter kernel
# Re-import in notebook
```

**Not seeing improvement:**
- This is valuable data! Document it
- Try customizing expert/critic prompts
- Check if baseline is already near ceiling
- Test on different benchmark

**High costs:**
- Reduce `n_iterations` or `refinement_rounds`
- Use fewer critics/experts
- Test on smaller `NUM_TASKS`

## Files Modified

- ✅ `code/harness/strategies.py` - Added 2 new strategies (~360 lines)
- ✅ `notebooks/04_benchmark_evaluation.ipynb` - Updated config cell + new docs cell
- ✅ `CUSTOM_STRATEGIES_GUIDE.md` - Created comprehensive guide
- ✅ `README.md` - Updated features, structure, docs section
- ✅ Cache cleared

## What This Enables

You can now:

1. **Benchmark custom strategies** on standard datasets (GSM8K, MMLU, etc.)
2. **Compare to SOTA** - See how iterative refinement stacks up
3. **Validate mental models** - Prove these approaches work (or don't!)
4. **Publish findings** - You have reproducible metrics
5. **Tune configurations** - Find optimal critics/experts for each task type

## Resources

- **Guide**: `CUSTOM_STRATEGIES_GUIDE.md`
- **Benchmark Guide**: `BENCHMARK_GUIDE.md`
- **Design Critique Notebook**: `notebooks/06_design_critique.ipynb`
- **XFN Team Notebook**: `notebooks/05_interdisciplinary_team.ipynb`
- **Implementation**: `code/harness/strategies.py` (lines 727-1091)

---

**Ready to test?** Open the benchmark notebook and uncomment the custom strategies!
