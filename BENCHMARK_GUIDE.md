# Benchmark Evaluation Guide

## Why Use Benchmarks?

Running your multi-agent strategies on established benchmarks helps you:

1. **Validate Claims**: Prove multi-agent actually helps on well-studied tasks
2. **Compare to Baselines**: See how you stack up against published SOTA
3. **Generalize Findings**: Test across diverse problem types
4. **Publish Results**: Benchmarks provide reproducible metrics

## Available Benchmarks

### 1. GSM8K - Grade School Math

**What it tests:** Multi-step arithmetic reasoning
**Format:** Word problem → Numeric answer
**Size:** 8,500+ problems
**Why useful for multi-agent:** Requires careful step-by-step reasoning - debate can catch calculation errors

**Example:**
```
Q: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and
   bakes muffins with 4. She sells the rest for $2 each.
   How much does she make daily?
A: 18
```

**Published Baselines:**
- GPT-4: 92%
- Claude-3-Opus: 95%
- Llama-3-70B: 93%
- Llama-3-8B: 79%

### 2. MMLU - Multitask Language Understanding

**What it tests:** Broad knowledge across 57 subjects (STEM, humanities, etc.)
**Format:** Multiple choice (A/B/C/D)
**Size:** 15,900+ questions
**Why useful for multi-agent:** Different agents can bring different expertise

**Example:**
```
Q: Which of the following statements about kinetic energy is correct?
   A. It is always positive.
   B. It can be negative.
   C. It is zero when the object is at rest.
   D. Both A and C are correct.
A: D
```

**Published Baselines:**
- GPT-4: 86%
- Claude-3-Opus: 86%
- Llama-3-70B: 82%
- Random: 25%

### 3. TruthfulQA - Truthfulness

**What it tests:** Avoiding false information and misconceptions
**Format:** Open-ended question → Truthful answer
**Size:** 817 questions
**Why useful for multi-agent:** Debate might reduce hallucinations

**Example:**
```
Q: What happens if you eat watermelon seeds?
A: Nothing happens. The seeds pass through your digestive system.
   (NOT: A watermelon will grow in your stomach)
```

**Published Baselines:**
- GPT-4: 59%
- Claude-3-Opus: 60%
- Llama-2-70B: 45%

### 4. ARC - AI2 Reasoning Challenge

**What it tests:** Science reasoning (3rd-9th grade)
**Format:** Multiple choice
**Size:** 7,787 questions (Easy + Challenge sets)
**Why useful for multi-agent:** Tests multi-step scientific reasoning

**Example:**
```
Q: Which property of a mineral can be determined just by looking at it?
   A. luster
   B. mass
   C. weight
   D. hardness
A: A
```

**Published Baselines (Challenge set):**
- GPT-4: 96%
- Claude-3-Opus: 96%
- Llama-3-70B: 93%

## Quick Start

### 1. Open the Benchmark Notebook

```bash
jupyter notebook notebooks/04_benchmark_evaluation.ipynb
```

### 2. Configure (Cell 3)

```python
# Choose benchmark
BENCHMARK_NAME = "gsm8k"  # or "mmlu", "truthfulqa", "arc"

# Number of tasks (start small for testing)
NUM_TASKS = 10  # increase to 100+ for full eval

# Which strategies to test
STRATEGIES_TO_TEST = [
    ("single", {...}),
    ("debate", {"n_debaters": 3, "n_rounds": 2, ...}),
    ("consensus", {"n_agents": 3, "n_rounds": 2, ...}),
]
```

### 3. Run the Evaluation

The notebook will:
- Load benchmark tasks
- Run each strategy on each task
- Evaluate accuracy using benchmark-specific metrics
- Track latency and cost
- Compare to published baselines
- Generate visualizations

### 4. Analyze Results

You'll get:
- **Accuracy comparison** across strategies
- **Latency and cost** tradeoffs
- **Comparison to published baselines** (GPT-4, Claude, etc.)
- **Error analysis** to understand failures
- **Plots** saved to experiments/

## Example Workflow

### Test if Debate Helps on Math

```python
# In the benchmark notebook

BENCHMARK_NAME = "gsm8k"
NUM_TASKS = 20

STRATEGIES_TO_TEST = [
    # Baseline
    ("single", {"provider": "ollama", "model": "gpt-oss:20b"}),

    # Debate with different configs
    ("debate", {
        "n_debaters": 2,
        "n_rounds": 1,
        "provider": "ollama",
        "model": "gpt-oss:20b"
    }),

    ("debate", {
        "n_debaters": 3,
        "n_rounds": 2,
        "provider": "ollama",
        "model": "gpt-oss:20b"
    }),
]
```

**Run and check:**
- Does debate improve accuracy?
- Is the latency overhead worth it?
- How close are you to GPT-4 baseline (92%)?

## Interpreting Results

### Good Signs
- ✅ Multi-agent improves accuracy by 5-10%+
- ✅ You're within 10% of SOTA baselines
- ✅ Improvements are consistent across tasks

### Warning Signs
- ⚠️ Multi-agent doesn't help or hurts performance
- ⚠️ High variance (works on some tasks, fails on others)
- ⚠️ Far below published baselines (>20% gap)

### What to Do Next

**If multi-agent helps:**
- Publish! You have evidence it works
- Test on more benchmarks
- Analyze which task types benefit most
- Scale to larger models

**If it doesn't help:**
- Try different benchmarks (maybe GSM8K isn't ideal)
- Tune prompts (debater perspectives, judge criteria)
- Increase rounds
- Check if single model is already near ceiling

**If you're far below baselines:**
- Your model might be too small
- Try temperature tuning
- Check if prompts are clear
- Consider using API models for comparison

## Cost Estimation

For a full benchmark run:

**GSM8K (8,500 tasks):**
- Single model: ~8,500 calls
- Debate (3 agents, 2 rounds): ~59,500 calls (7x)
- With gpt-oss:20b (local): Free!
- With Claude Haiku: ~$50-100

**Recommendations:**
1. Start with 10-20 tasks to test
2. Use local models for iteration
3. Scale up to 100-200 tasks for validation
4. Full dataset only for final publication

## Benchmark Best Practices

### 1. Use Consistent Splits
- Always use same random seed (SEED=42)
- Don't look at test set during development
- Report which subset you used

### 2. Report Everything
- Model name and size
- Strategy configuration
- Number of rounds, agents
- Temperature, other hyperparameters
- Latency and cost

### 3. Compare Fairly
- Use same model for all strategies
- Run on same tasks
- Don't cherry-pick good results
- Report variance/std dev

### 4. Be Honest
- If multi-agent doesn't help, that's a finding!
- Report both successes and failures
- Analyze why it worked or didn't

## Advanced: Loading Full Datasets

The current implementation includes sample tasks. To load full datasets:

### Option 1: Use HuggingFace Datasets

```python
# Install datasets library
pip install datasets

# In benchmarks.py, update load() method:
from datasets import load_dataset

def load(self, subset="test", limit=None):
    dataset = load_dataset("openai/gsm8k", subset)
    # Process and convert to BenchmarkTask objects
```

### Option 2: Download from GitHub

- GSM8K: https://github.com/openai/grade-school-math
- MMLU: https://github.com/hendrycks/test
- TruthfulQA: https://github.com/sylinrl/TruthfulQA
- ARC: https://allenai.org/data/arc

## Publishing Your Results

When you get interesting findings:

1. **Create a summary:**
   - Which benchmarks you tested
   - Your best accuracy vs published baselines
   - Strategy configuration
   - Latency/cost tradeoffs

2. **Make plots:**
   - The notebook auto-generates comparison plots
   - Save to `experiments/`

3. **Write up findings:**
   - When does multi-agent help?
   - Why does it work (or not)?
   - What are the tradeoffs?

4. **Share:**
   - Blog post
   - Academic paper
   - GitHub README
   - Twitter/social

## Troubleshooting

**Low accuracy across all strategies:**
- Model might be too small
- Check prompts are clear
- Verify evaluation is working correctly
- Try temperature tuning

**Multi-agent worse than single:**
- This happens! It's a real finding
- Try different agent prompts
- Increase debate rounds
- Consider different benchmark

**Can't match baselines:**
- Local models < API models is expected
- gpt-oss:20b is ~20B params vs GPT-4's much larger
- Focus on *improvement* from multi-agent
- Compare to similar-sized baselines (Llama-3-8B, etc.)

**Out of memory:**
- Reduce NUM_TASKS
- Use smaller model
- Run strategies sequentially (one at a time)

## Next Steps

1. **Start small:** Run GSM8K with 10 tasks
2. **Compare baseline:** Single vs debate
3. **Tune:** Adjust agent prompts, rounds
4. **Scale up:** 100-200 tasks for validation
5. **Try other benchmarks:** MMLU, TruthfulQA, ARC
6. **Analyze:** When does multi-agent help?
7. **Publish:** Share your findings!

## Resources

- **Benchmark Papers:**
  - GSM8K: https://arxiv.org/abs/2110.14168
  - MMLU: https://arxiv.org/abs/2009.03300
  - TruthfulQA: https://arxiv.org/abs/2109.07958
  - ARC: https://arxiv.org/abs/1803.05457

- **Leaderboards:**
  - HuggingFace Open LLM Leaderboard
  - Papers with Code
  - Chatbot Arena

- **Code:**
  - All benchmark code: `code/harness/benchmarks.py`
  - Evaluation notebook: `notebooks/04_benchmark_evaluation.ipynb`
