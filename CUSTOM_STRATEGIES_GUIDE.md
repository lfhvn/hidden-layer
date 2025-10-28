# Custom Strategy Integration Guide

## Overview

The benchmark evaluation framework now supports **design critique** and **interdisciplinary team** strategies, allowing you to test these mental models on established benchmarks.

## Available Custom Strategies

### 1. Design Critique Strategy

**Mental Model:** Iterative refinement through structured feedback

**How it works:**
1. Generate initial draft/solution
2. Multiple critics provide feedback from different perspectives
3. Revise based on critiques
4. Repeat for n iterations

**Best for:**
- Writing tasks (blog posts, documentation, articles)
- Creative tasks where quality improves with iteration
- Tasks requiring attention to multiple quality dimensions
- Benchmarks like TruthfulQA (accuracy), creative writing evaluations

**Basic usage:**
```python
("design_critique", {
    "n_iterations": 2,
    "provider": PROVIDER,
    "model": MODEL,
    "verbose": False
})
```

**Advanced usage with custom critics:**
```python
("design_critique", {
    "n_iterations": 3,
    "provider": PROVIDER,
    "model": MODEL,
    "critique_panel": [
        {
            "name": "Math Accuracy Critic",
            "focus": "Mathematical correctness",
            "criteria": "Check all calculations, verify logic, ensure arithmetic is correct"
        },
        {
            "name": "Step-by-Step Clarity Critic",
            "focus": "Clear explanation",
            "criteria": "Ensure solution is explained step-by-step and easy to follow"
        },
        {
            "name": "Completeness Critic",
            "focus": "Solution completeness",
            "criteria": "Verify all parts of the problem are addressed and no steps are skipped"
        }
    ],
    "verbose": False
})
```

**Default critics:**
- **Clarity Critic**: Evaluates comprehension and structure
- **Accuracy Critic**: Checks technical correctness
- **Completeness Critic**: Ensures all points are covered

### 2. Interdisciplinary Team Strategy

**Mental Model:** Multiple domain experts collaborate on complex problems

**How it works:**
1. Each expert analyzes the problem from their domain perspective
2. Synthesizer integrates all expert insights into unified solution
3. (Optional) Experts review and refine the integrated solution
4. Repeat refinement for n rounds

**Best for:**
- Complex problems requiring multiple perspectives
- Strategic planning and decision-making tasks
- Problems with technical, user, and business dimensions
- Benchmarks like MMLU (multiple subject domains), complex reasoning tasks

**Basic usage:**
```python
("interdisciplinary_team", {
    "refinement_rounds": 1,
    "provider": PROVIDER,
    "model": MODEL,
    "verbose": False
})
```

**Advanced usage with custom experts:**
```python
("interdisciplinary_team", {
    "refinement_rounds": 2,
    "provider": PROVIDER,
    "model": MODEL,
    "expert_team": [
        {
            "name": "Mathematician",
            "role": "Math Expert",
            "perspective": "Mathematical rigor and correctness",
            "system_prompt": """You are a mathematician with expertise in problem-solving.
Focus on: mathematical correctness, logical reasoning, proof techniques.
Analyze problems ensuring rigorous mathematical foundations."""
        },
        {
            "name": "Educator",
            "role": "Teacher",
            "perspective": "Pedagogical clarity and student understanding",
            "system_prompt": """You are an experienced math teacher.
Focus on: clear explanations, step-by-step reasoning, common student mistakes.
Analyze problems from a teaching perspective, ensuring solutions are understandable."""
        },
        {
            "name": "Applied Scientist",
            "role": "Practical Problem Solver",
            "perspective": "Real-world application and practicality",
            "system_prompt": """You are a scientist who applies math to real problems.
Focus on: practical interpretation, real-world context, sanity checks.
Analyze problems ensuring solutions make practical sense."""
        }
    ],
    "verbose": False
})
```

**Default experts:**
- **Technical Lead**: Software engineering perspective (feasibility, architecture)
- **User Advocate**: UX/design perspective (usability, accessibility)
- **Business Strategist**: Business perspective (market fit, ROI)

## Using in Benchmark Evaluation

### Step 1: Open the Benchmark Notebook

```bash
jupyter notebook notebooks/04_benchmark_evaluation.ipynb
```

### Step 2: Configure Benchmark and Strategies

In cell 3, add your custom strategies to `STRATEGIES_TO_TEST`:

```python
STRATEGIES_TO_TEST = [
    # Baseline
    ("single", {
        "provider": PROVIDER,
        "model": MODEL,
        "verbose": False
    }),

    # Standard multi-agent strategies
    ("debate", {
        "n_debaters": 3,
        "n_rounds": 2,
        "provider": PROVIDER,
        "model": MODEL,
        "verbose": False
    }),

    # Custom strategy: Design Critique
    ("design_critique", {
        "n_iterations": 2,
        "provider": PROVIDER,
        "model": MODEL,
        "verbose": False
    }),

    # Custom strategy: Interdisciplinary Team
    ("interdisciplinary_team", {
        "refinement_rounds": 1,
        "provider": PROVIDER,
        "model": MODEL,
        "verbose": False
    }),
]
```

### Step 3: Run Evaluation

Execute the notebook cells to:
- Load benchmark tasks
- Run each strategy (including custom ones)
- Evaluate accuracy
- Track latency and cost
- Generate comparison visualizations

### Step 4: Analyze Results

The notebook will automatically:
- Compare accuracy across all strategies
- Show latency/cost tradeoffs
- Compare to published baselines
- Generate plots showing strategy performance

## Example Configurations for Different Benchmarks

### GSM8K (Math Reasoning)

**Good fit:** Design critique with math-focused critics

```python
("design_critique", {
    "n_iterations": 2,
    "critique_panel": [
        {
            "name": "Calculation Checker",
            "focus": "Arithmetic accuracy",
            "criteria": "Verify every calculation step is arithmetically correct"
        },
        {
            "name": "Logic Checker",
            "focus": "Logical reasoning",
            "criteria": "Ensure the problem-solving approach is logically sound"
        }
    ],
    "provider": PROVIDER,
    "model": MODEL
})
```

### MMLU (Multitask Knowledge)

**Good fit:** Interdisciplinary team with subject-matter experts

```python
("interdisciplinary_team", {
    "refinement_rounds": 1,
    "expert_team": [
        {
            "name": "Subject Expert",
            "role": "Domain Specialist",
            "perspective": "Deep subject knowledge",
            "system_prompt": "You are an expert in this subject domain. Focus on correctness and depth."
        },
        {
            "name": "Critical Thinker",
            "role": "Skeptic",
            "perspective": "Question assumptions",
            "system_prompt": "You critically evaluate answers, looking for errors and misconceptions."
        },
        {
            "name": "Synthesizer",
            "role": "Integrator",
            "perspective": "Holistic understanding",
            "system_prompt": "You integrate multiple perspectives to find the best answer."
        }
    ],
    "provider": PROVIDER,
    "model": MODEL
})
```

### TruthfulQA (Factuality)

**Good fit:** Design critique with fact-checking focus

```python
("design_critique", {
    "n_iterations": 2,
    "critique_panel": [
        {
            "name": "Fact Checker",
            "focus": "Factual accuracy",
            "criteria": "Verify claims are factually accurate and not misconceptions"
        },
        {
            "name": "Skeptic",
            "focus": "Challenge assumptions",
            "criteria": "Question common myths and misconceptions"
        }
    ],
    "provider": PROVIDER,
    "model": MODEL
})
```

## Performance Considerations

### Token Usage

Custom strategies use more tokens than simple strategies:

- **Design Critique**: ~(1 + n_iterations × n_critics × 2) calls per task
  - Example: 2 iterations, 3 critics = ~13 LLM calls per task

- **Interdisciplinary Team**: ~(n_experts + 1 + refinement_rounds × (n_experts + 1)) calls per task
  - Example: 3 experts, 1 refinement = ~7 LLM calls per task

### Recommendations

1. **Start small**: Test with NUM_TASKS=5-10 first
2. **Tune parameters**: Try fewer iterations/experts initially
3. **Use local models**: Run with Ollama models during development
4. **Monitor costs**: Track cost_usd in results
5. **Compare tradeoffs**: Is the accuracy gain worth the latency/cost?

## Expected Use Cases

### When Design Critique Helps

- ✅ Tasks where initial solutions often have fixable errors
- ✅ Writing and generation tasks
- ✅ Tasks benefiting from multiple quality checks
- ❌ Simple factoid questions (overhead not worth it)
- ❌ Tasks where single model is already near ceiling

### When Interdisciplinary Team Helps

- ✅ Complex multi-faceted problems
- ✅ Tasks requiring domain expertise
- ✅ Strategic planning and analysis
- ❌ Simple single-domain questions
- ❌ Tasks where experts would give identical answers

## Interpreting Results

### Good Signs

- Custom strategy improves accuracy by 5%+ over baseline
- Improvement is consistent across multiple tasks
- Cost/latency tradeoff is reasonable for your use case

### Warning Signs

- No accuracy improvement (or worse performance)
- High variance (works on some tasks, fails on others)
- Cost/latency 10x higher with minimal benefit

### Next Steps

**If custom strategies help:**
- Document which benchmarks benefit most
- Tune expert/critic configurations
- Test on more tasks
- Consider for production use

**If they don't help:**
- Try different benchmark (maybe not a good fit)
- Tune prompts and expert perspectives
- Check if baseline is already near ceiling
- Consider that not all tasks benefit from iteration/collaboration

## Troubleshooting

**"Unknown strategy" error:**
- Clear Python cache: `find code/harness -type d -name "__pycache__" -exec rm -rf {} +`
- Restart Jupyter kernel

**High latency:**
- Reduce n_iterations or refinement_rounds
- Use fewer critics/experts
- Test on fewer tasks (reduce NUM_TASKS)

**Not seeing improvement:**
- This is valuable data! Not all strategies help all tasks
- Try customizing expert/critic prompts
- Check if baseline is already near ceiling
- Consider different benchmark

## Resources

- **Design Critique Notebook**: `notebooks/06_design_critique.ipynb`
- **Interdisciplinary Team Notebook**: `notebooks/05_interdisciplinary_team.ipynb`
- **Strategy Implementation**: `code/harness/strategies.py`
- **Benchmark Evaluation**: `notebooks/04_benchmark_evaluation.ipynb`
- **Benchmark Guide**: `BENCHMARK_GUIDE.md`

## Contributing

Want to add your own custom strategy?

1. Implement strategy function in `code/harness/strategies.py`
2. Add to `STRATEGIES` registry
3. Document in this guide
4. Test in benchmark notebook

See existing custom strategies as templates!
