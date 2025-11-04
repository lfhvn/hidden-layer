# Multi-Agent Debate Guide

## Overview

The debate system now supports:
1. **Multiple debate rounds** - Agents refine their arguments over n rounds
2. **Individual debater prompts** - Give each agent a unique perspective/role
3. **Custom judge prompts** - Control how the judge evaluates arguments
4. **Consensus strategy** - Agents build consensus without a separate judge

## Quick Start

### 1. Debate Strategy (with Judge)

Location: `notebooks/02_debate_experiments.ipynb`

**Key Configuration Section (Cell 7):**

```python
# ========================================
# DEBATE CONFIGURATION - EDIT THESE VALUES
# ========================================

NUM_DEBATERS = 2  # Number of agents
NUM_ROUNDS = 2    # 🔧 Number of debate rounds (1 = no rebuttals)

# ========================================
# DEBATER PERSPECTIVES - EDIT THESE PROMPTS
# ========================================

DEBATER_PROMPTS = [
    """You are a critical thinker who questions assumptions...""",
    """You are a constructive thinker who explores possibilities...""",
]

# ========================================
# JUDGE PROMPT - EDIT THIS
# ========================================

JUDGE_PROMPT = """You are an expert judge evaluating perspectives.
Question: {task_input}
Evaluate the answers below..."""
```

### 2. Consensus Strategy (no Judge)

Location: `notebooks/03_consensus_experiments.ipynb`

**Key Configuration Section (Cell 7):**

```python
# ========================================
# CONSENSUS CONFIGURATION
# ========================================

NUM_AGENTS = 3   # Number of agents
NUM_ROUNDS = 3   # 🔧 Rounds of consensus building

# ========================================
# AGENT PERSPECTIVES
# ========================================

AGENT_PROMPTS = [
    """You are an analytical thinker...""",
    """You are a creative thinker...""",
    """You are a practical thinker...""",
]
```

## How It Works

### Debate Strategy (with Judge)

**Round 1:** Each debater provides initial argument
**Round 2+:** Each debater sees others' arguments and refines their position
**Final:** Judge evaluates all final arguments and selects best

```
Debater 1 → Argument 1
Debater 2 → Argument 2
   ↓ Round 2
Debater 1 (sees Arg 2) → Refined Argument 1
Debater 2 (sees Arg 1) → Refined Argument 2
   ↓ Judge
Judge (sees both) → Final Verdict
```

### Consensus Strategy (no Judge)

**Round 1:** Each agent provides initial position
**Round 2+:** Each agent sees others' positions and refines toward consensus
**Final:** One agent synthesizes all final positions into consensus answer

```
Agent 1 → Position 1
Agent 2 → Position 2
Agent 3 → Position 3
   ↓ Round 2
Agent 1 (sees 2, 3) → Refined Position 1
Agent 2 (sees 1, 3) → Refined Position 2
Agent 3 (sees 1, 2) → Refined Position 3
   ↓ Synthesis
Synthesize all final positions → Consensus Answer
```

## Parameters to Tune

### Number of Rounds

- **1 round**: Fast, initial arguments only
- **2 rounds**: Agents see each other's work, can respond
- **3+ rounds**: Deeper convergence, but diminishing returns

**Recommendation:** Start with 2 rounds for debate, 3 for consensus

### Debater/Agent Prompts

Give each agent a distinct perspective:

**Good Perspectives:**
- Analytical vs Creative vs Practical
- Optimistic vs Skeptical vs Neutral
- Domain experts (e.g., "economist", "engineer", "ethicist")
- Stakeholder roles (e.g., "user", "developer", "business")

**Example:**
```python
DEBATER_PROMPTS = [
    "You are an economist focused on cost-benefit analysis...",
    "You are a software engineer focused on technical feasibility...",
    "You are a UX designer focused on user experience...",
]
```

### Judge Prompt

Control how the judge weighs different factors:

**Examples:**
```python
# Emphasize evidence
JUDGE_PROMPT = """Evaluate based on factual evidence and logical reasoning.
Question: {task_input}"""

# Emphasize creativity
JUDGE_PROMPT = """Evaluate based on creativity and novel approaches.
Question: {task_input}"""

# Balanced
JUDGE_PROMPT = """Consider both logical reasoning and practical implications.
Question: {task_input}"""
```

## When to Use Each Strategy

### Debate (with Judge)

**Best for:**
- Subjective questions with multiple valid perspectives
- Creative/open-ended tasks
- When you want a clear "winner" selected

**Example tasks:**
- "What's the best approach to solve X?"
- "Should we prioritize A or B?"
- Design decisions, strategy questions

### Consensus (no Judge)

**Best for:**
- Objective questions with clear answers
- When you want agents to collaborate, not compete
- Testing how well agents converge on truth

**Example tasks:**
- Math/logic problems
- Factual questions
- Tasks where "correct" answer should emerge from discussion

## Performance Considerations

### Latency

- Debate with 2 agents, 2 rounds: ~4 LLM calls (2 debaters × 2 rounds + judge)
- Consensus with 3 agents, 3 rounds: ~10 LLM calls (3 agents × 3 rounds + synthesis)

**With streaming (`verbose=True`):** See output in real-time
**Without streaming (`verbose=False`):** Faster, less output

### Cost

Token usage scales with:
- Number of agents/debaters
- Number of rounds
- Context size (arguments get longer in later rounds)

**Tip:** Start with fewer rounds and agents, scale up if needed

## Examples

### Debate: 3 Debaters, 2 Rounds

```python
result = run_strategy(
    "debate",
    "Should we invest in renewable energy or nuclear power?",
    n_debaters=3,
    n_rounds=2,
    debater_prompts=[
        "You focus on environmental impact...",
        "You focus on economic feasibility...",
        "You focus on energy security...",
    ],
    judge_prompt="Evaluate based on long-term sustainability. Question: {task_input}",
    verbose=True
)
```

### Consensus: 4 Agents, 3 Rounds

```python
result = run_strategy(
    "consensus",
    "What is the average speed of the train?",
    n_agents=4,
    n_rounds=3,
    agent_prompts=[
        "You approach problems analytically with step-by-step reasoning...",
        "You verify calculations carefully...",
        "You look for common mistakes and edge cases...",
        "You synthesize different approaches...",
    ],
    verbose=True
)
```

## Tips

1. **Start Simple:** Begin with 2 agents/debaters, 1-2 rounds
2. **Test First:** Use `verbose=True` on a single task to see the full process
3. **Iterate:** Adjust prompts based on what you observe
4. **Compare:** Run same task with different configurations to find optimal setup
5. **Track Everything:** The experiment tracker logs all arguments/positions for analysis

## Troubleshooting

**Agents always agree immediately:**
- Increase diversity of agent_prompts
- Make perspectives more distinct/opposing

**No convergence in consensus:**
- Increase NUM_ROUNDS
- Adjust agent_prompts to be more collaborative
- Task may be too subjective for consensus

**Judge always picks first answer:**
- Refine judge_prompt to be more analytical
- Lower judge temperature (currently 0.3)

**Too slow:**
- Reduce NUM_ROUNDS
- Use `verbose=False`
- Use smaller/faster model

## Next Steps

1. **Run baseline:** Start with debate notebook, default settings
2. **Customize:** Edit prompts and rounds in configuration cell
3. **Compare:** Run consensus notebook on same tasks
4. **Analyze:** Compare when each strategy works better
5. **Experiment:** Try different agent perspectives and round counts

## Code Reference

All strategies are in: `code/harness/strategies.py`
- `debate_strategy()` - lines 156-367
- `consensus_strategy()` - lines 529-724

See function signatures for all available parameters.
