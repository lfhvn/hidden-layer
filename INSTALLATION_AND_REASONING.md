# Installation & Reasoning Features Guide

## 🚀 Installation

### Current Status
✅ Ollama is installed and running
✅ Models available: `gpt-oss:20b`, `llama3.2`
❌ Python packages NOT installed

### Install Python Packages

```bash
# Navigate to project directory
cd /Users/lhm/Documents/GitHub/hidden-layer

# Activate virtual environment (if using one)
source venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

This will install:
- **mlx** & **mlx-lm** - Apple Silicon optimized ML
- **ollama** - Local model interface
- **anthropic** & **openai** - API providers
- **pandas**, **numpy**, **matplotlib**, **seaborn** - Data analysis & visualization
- **jupyter**, **ipykernel**, **ipywidgets** - Notebook support
- **python-dotenv**, **tqdm**, **pyyaml** - Utilities

### Verify Installation

```bash
python -c "
from harness import llm_call
response = llm_call('Hi!', provider='ollama', model='gpt-oss:20b')
print(response.text)
"
```

---

## 🧠 Thinking/Reasoning Features

### 1. **Thinking Budget** (Extended Reasoning)

The `thinking_budget` parameter allocates extra tokens for internal reasoning before generating the final answer.

#### How It Works
```python
from harness import llm_call

# Standard call
response = llm_call(
    "What is 234 * 567?",
    provider="ollama",
    model="gpt-oss:20b",
    temperature=0.7
)

# With extended thinking
response = llm_call(
    "What is 234 * 567?",
    provider="ollama",
    model="gpt-oss:20b",
    temperature=0.7,
    thinking_budget=2000  # Allow up to 2000 tokens for reasoning
)
```

#### When to Use
- **Complex math problems** - More reasoning = better accuracy
- **Multi-step reasoning** - Breaking down complex problems
- **Strategic decisions** - Weighing trade-offs
- **Code generation** - Planning before writing

#### Pre-configured Reasoning Model
```python
from harness import get_model_config

# Use pre-configured reasoning setup
config = get_model_config("gpt-oss-20b-reasoning")
response = llm_call(
    "Complex problem...",
    **config.to_kwargs()
)
```

---

### 2. **Rationale Extraction** (NEW!)

Get the model to explain its reasoning before giving the final answer.

#### Basic Rationale Wrapper

I've created a new utility in `code/harness/rationale.py`:

```python
from harness import llm_call_with_rationale

# Ask for reasoning + answer
result = llm_call_with_rationale(
    "Should we build a mobile app or web app first?",
    provider="ollama",
    model="gpt-oss:20b"
)

print("REASONING:")
print(result.rationale)
print("\nFINAL ANSWER:")
print(result.answer)
```

#### How It Works

The wrapper uses a two-step process:
1. **First call**: Ask the model to think through the problem and provide reasoning
2. **Parse**: Extract the rationale and final answer

The prompt template:
```
Think through this problem step-by-step, showing your reasoning.

Problem: {user_question}

Format your response as:

REASONING:
[Your step-by-step thinking, analysis, considerations, trade-offs, etc.]

ANSWER:
[Your final, concise answer]
```

#### Integration with Strategies

Use rationale with any multi-agent strategy:

```python
from harness import run_strategy_with_rationale

result = run_strategy_with_rationale(
    strategy="adaptive_team",
    task_input="Should we use microservices or monolith?",
    n_experts=3,
    provider="ollama",
    model="gpt-oss:20b"
)

# Access each expert's reasoning
for expert in result.expert_analyses:
    print(f"\n{expert['expert']}:")
    print(f"  Reasoning: {expert['rationale'][:200]}...")
    print(f"  Conclusion: {expert['answer']}")

# Access synthesized reasoning
print(f"\nFINAL REASONING: {result.synthesis_rationale}")
print(f"\nFINAL ANSWER: {result.answer}")
```

---

## 📊 Using in Benchmarks

### Add Thinking Budget to Benchmark Evaluation

Edit `notebooks/07_custom_strategies_benchmark.ipynb`:

```python
STRATEGIES_TO_TEST = [
    # Baseline (no thinking budget)
    ("single", {
        "provider": PROVIDER,
        "model": MODEL,
        "verbose": False
    }),

    # With thinking budget
    ("single", {
        "provider": PROVIDER,
        "model": MODEL,
        "thinking_budget": 2000,  # Add this!
        "verbose": False
    }),

    # Adaptive team with thinking
    ("adaptive_team", {
        "n_experts": 3,
        "refinement_rounds": 1,
        "thinking_budget": 2000,  # Add this!
        "provider": PROVIDER,
        "model": MODEL,
        "verbose": True
    }),
]
```

### Track Reasoning in Results

The benchmark system automatically tracks:
- **Tokens used for reasoning** (in `thinking_budget` metadata)
- **Total latency** (including reasoning time)
- **Cost impact** (reasoning tokens count toward total cost)

Access in results:
```python
for result in results:
    if 'thinking_budget' in result.metadata:
        print(f"Used thinking budget: {result.metadata['thinking_budget']}")
```

---

## 🔧 Advanced: Custom Reasoning Strategies

### Chain-of-Thought Strategy

Create a dedicated reasoning strategy:

```python
def chain_of_thought_strategy(task_input, provider, model, **kwargs):
    """
    Strategy that enforces step-by-step reasoning
    """

    cot_prompt = f"""Solve this problem using step-by-step reasoning.

Problem: {task_input}

Show your work:
Step 1: [First step]
Step 2: [Second step]
...
Final Answer: [Your answer]
"""

    result = llm_call(
        cot_prompt,
        provider=provider,
        model=model,
        thinking_budget=2000,  # Give extra reasoning space
        **kwargs
    )

    return result
```

### Self-Critique with Reasoning

```python
def self_critique_with_rationale(task_input, provider, model, **kwargs):
    """
    Generate answer, explain reasoning, then critique it
    """

    # Step 1: Generate initial answer with reasoning
    initial = llm_call_with_rationale(task_input, provider, model, **kwargs)

    # Step 2: Critique the reasoning
    critique_prompt = f"""Review this reasoning and answer:

REASONING: {initial.rationale}
ANSWER: {initial.answer}

Critique:
1. Is the reasoning sound?
2. Are there any flaws or gaps?
3. Is the answer correct?
4. What could be improved?

Then provide your final answer.
"""

    final = llm_call_with_rationale(critique_prompt, provider, model, **kwargs)

    return {
        'initial_reasoning': initial.rationale,
        'initial_answer': initial.answer,
        'critique': final.rationale,
        'final_answer': final.answer
    }
```

---

## 💡 Best Practices

### When to Use Thinking Budget

✅ **Use thinking budget when:**
- Problem requires multi-step reasoning
- Math/logic problems
- Strategic decisions
- Code generation
- Complex analysis

❌ **Don't use thinking budget when:**
- Simple factoid questions
- Single-step lookups
- Speed is critical
- Cost is constrained

### Optimal Thinking Budget Values

- **500-1000 tokens**: Simple multi-step problems
- **1000-2000 tokens**: Complex reasoning (default for gpt-oss-20b-reasoning)
- **2000-4000 tokens**: Very complex strategic problems
- **> 4000 tokens**: Rarely needed, diminishing returns

### Cost Implications

Thinking budget increases token usage:
- **Without thinking**: ~100-200 tokens/response
- **With 2000 thinking budget**: ~300-500 tokens/response
- **Cost increase**: 2-3x (but accuracy may improve significantly)

**Recommendation**: Start with `thinking_budget=1000` and tune based on accuracy gains.

---

## 📝 Examples

### Example 1: Math Problem with Thinking

```python
from harness import llm_call

problem = """
Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning
and bakes muffins for her friends every day with four. She sells the remainder
at the farmers' market daily for $2 per fresh duck egg. How much in dollars
does she make every day at the farmers' market?
"""

# Without thinking
basic = llm_call(problem, provider="ollama", model="gpt-oss:20b")
print(f"Basic: {basic.text}")

# With thinking budget
reasoning = llm_call(
    problem,
    provider="ollama",
    model="gpt-oss:20b",
    thinking_budget=2000
)
print(f"\nWith reasoning: {reasoning.text}")
```

### Example 2: Strategic Decision with Rationale

```python
from harness import llm_call_with_rationale

decision = """
Our startup has $500k runway for 6 months. Should we:
A) Hire 3 engineers and build faster
B) Hire 1 engineer + 1 sales person to get revenue
C) Keep current team and extend runway to 12 months
"""

result = llm_call_with_rationale(
    decision,
    provider="ollama",
    model="gpt-oss:20b",
    thinking_budget=2000
)

print("REASONING:")
print(result.rationale)
print("\nRECOMMENDATION:")
print(result.answer)
```

### Example 3: Multi-Agent with Rationale

```python
from harness import run_strategy

# Use adaptive team - each expert explains their reasoning
result = run_strategy(
    "adaptive_team",
    "Should we use React or Vue for our new dashboard?",
    n_experts=3,
    thinking_budget=1500,  # Give each expert reasoning budget
    provider="ollama",
    model="gpt-oss:20b",
    verbose=True  # See each expert's reasoning in real-time
)

print(result.output)
```

---

## 🐛 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### Ollama not responding
```bash
# Check if running
ollama list

# Start if not running
ollama serve &
```

### Model not found
```bash
# Pull the model
ollama pull gpt-oss:20b
```

### Thinking budget not working
- ✅ Check model supports extended thinking (gpt-oss, llama3.2)
- ✅ Verify `thinking_budget` is passed to `llm_call` or strategy
- ✅ Some models ignore this parameter - check model docs

---

## 📚 Further Reading

- **Model configs**: `code/harness/model_config.py`
- **LLM provider**: `code/harness/llm_provider.py`
- **Rationale extraction**: `code/harness/rationale.py` (NEW!)
- **Strategies**: `code/harness/strategies.py`

---

**Last Updated**: 2025-01-27
