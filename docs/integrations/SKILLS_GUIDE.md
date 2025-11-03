# Claude Skills Guide

**Last Updated**: 2025-11-03
**Purpose**: Create custom automation skills for your research workflow

---

## What Are Claude Skills?

Claude Skills are **reusable automation modules** that extend Claude's capabilities. Think of them as specialized AI agents for specific tasks.

**Simple analogy**:
- **Without skills**: You manually ask Claude to analyze experiments each time
- **With skills**: Type `/analyze-experiment` → Claude automatically does the full workflow

---

## Why You Need Skills

### The Problem

Your research involves **repetitive workflows**:

1. **After every experiment**:
   - Read results.jsonl
   - Parse summary.json
   - Calculate metrics
   - Compare to baselines
   - Generate insights
   - Write report

2. **When evaluating benchmarks**:
   - Load benchmark
   - Run across models
   - Track metrics
   - Compare results
   - Generate comparison table

3. **For introspection research**:
   - Extract concept vectors
   - Test across layers
   - Measure detection accuracy
   - Find optimal settings
   - Document findings

**Current solution**: Manually ask Claude to do these steps. Every. Single. Time.

### The Solution: Skills

**Create once, use forever**:

```bash
# After running experiment
/analyze-experiment debate_energy_20251103

# Claude automatically:
# ✓ Reads all result files
# ✓ Calculates metrics
# ✓ Compares to baselines
# ✓ Searches for related papers
# ✓ Generates insights report
# ✓ Saves analysis.md
```

---

## How Skills Work

### Anatomy of a Skill

A skill is just a **markdown file** with:
1. **Name** - What to call it
2. **Description** - When to use it
3. **Prompt** - What Claude should do

**Example**:
```markdown
# File: .claude/skills/analyze-experiment.md

You are an expert at analyzing multi-agent LLM experiments.

When given an experiment directory:
1. Read config.json, results.jsonl, and summary.json
2. Calculate key metrics (avg latency, accuracy, cost)
3. Compare to baseline scores for this benchmark
4. Search for recent related papers
5. Identify insights (when did strategy help? cost-benefit?)
6. Generate a markdown report with findings
7. Save to {experiment_dir}/analysis.md
```

### Using a Skill

Once created:
```bash
# Invoke by name
/analyze-experiment

# Or in Claude Desktop
"Use the analyze-experiment skill on experiments/debate_energy_20251103/"
```

---

## Skills Included With Hidden Layer

I've created 4 essential skills for your research. They're in `.claude/skills/`:

### 1. Experiment Analyzer (`analyze-experiment.md`)

**Purpose**: Automatically analyze experiment results

**Usage**:
```bash
/analyze-experiment experiments/debate_energy_20251103/
```

**What it does**:
- ✓ Reads all result files
- ✓ Calculates metrics (latency, cost, accuracy)
- ✓ Compares to baseline
- ✓ Searches for related papers
- ✓ Generates insights report
- ✓ Saves analysis.md

**When to use**:
- After running an experiment
- Before starting analysis
- To generate quick insights

**Example output**:
```markdown
# Experiment Analysis: debate_energy_20251103

## Summary
- Strategy: debate (3 agents)
- Tasks: 47
- Avg Latency: 3.2s
- Total Cost: $0.24
- Accuracy: 82%

## Key Findings
1. Debate outperformed single-model by 12%
2. Cost was 2.8x higher but quality justified
3. Latency acceptable for research use

## Comparison to Baselines
- Your result: 82%
- Published baseline: 75%
- Current SOTA: 89%

## Related Papers
[Papers found via Brave Search]

## Recommendations
1. Try chain-of-thought prompting (+12% from recent paper)
2. Test with 4 debaters (diminishing returns?)
3. Compare to self-consistency strategy
```

### 2. Benchmark Runner (`run-benchmark.md`)

**Purpose**: Systematically run benchmarks across models

**Usage**:
```bash
/run-benchmark tombench --models llama3.2:3b,llama3.1:8b,claude-3-5-sonnet
```

**What it does**:
- ✓ Loads specified benchmark
- ✓ Runs each model on all tasks
- ✓ Tracks metrics
- ✓ Compares to published baselines
- ✓ Generates comparison table
- ✓ Saves to experiments/benchmarks/

**When to use**:
- Evaluating new models
- Reproducing published results
- Comparing strategies

**Example output**:
```markdown
# Benchmark: ToMBench
## Results

| Model | Accuracy | Avg Latency | Total Cost |
|-------|----------|-------------|------------|
| llama3.2:3b | 72% | 1.2s | $0.00 |
| llama3.1:8b | 78% | 2.1s | $0.00 |
| claude-3-5-sonnet | 85% | 0.8s | $3.42 |

## Comparison to Baselines
- Human: 95%
- GPT-4: 76%
- Published baseline: 75%

## Insights
1. Llama 3.1 8B best cost-effectiveness
2. Claude best accuracy but expensive
3. All models struggle with second-order beliefs
```

### 3. Introspection Sweeper (`introspection-sweep.md`)

**Purpose**: Systematically test introspection across parameters

**Usage**:
```bash
/introspection-sweep --model qwen3-8b --concepts happiness,honesty,curiosity
```

**What it does**:
- ✓ For each concept:
  - Tests layers 10, 15, 20
  - Tests strengths 0.5, 1.0, 1.5, 2.0
  - Runs detection tasks
  - Runs generation tasks
- ✓ Tracks accuracy and quality
- ✓ Finds optimal settings
- ✓ Generates introspection profile
- ✓ Saves to concepts/{model}/profile.json

**When to use**:
- Profiling a new model
- Finding best introspection parameters
- Comparing introspection capabilities

**Example output**:
```json
{
  "model": "qwen3-8b",
  "concepts": {
    "happiness": {
      "optimal_layer": 15,
      "optimal_strength": 1.5,
      "detection_accuracy": 0.89,
      "generation_quality": 0.82
    },
    "honesty": {
      "optimal_layer": 20,
      "optimal_strength": 2.0,
      "detection_accuracy": 0.76,
      "generation_quality": 0.71
    }
  },
  "insights": [
    "Model shows strong introspection for emotional concepts",
    "Ethical concepts require higher layers",
    "Optimal strength varies by concept"
  ]
}
```

### 4. Paper Writer (`write-paper.md`)

**Purpose**: Generate research paper sections from experiments

**Usage**:
```bash
/write-paper --experiments experiments/debate_* --output papers/multi_agent_tom/
```

**What it does**:
- ✓ Reads all specified experiments
- ✓ Analyzes findings
- ✓ Generates LaTeX sections:
  - Abstract
  - Introduction
  - Methods
  - Results (with tables)
  - Discussion
  - Conclusion
- ✓ Includes proper citations
- ✓ Suggests figure placements
- ✓ Saves to papers/{name}/

**When to use**:
- Writing up experiments for publication
- Generating draft sections
- Organizing findings

**Example output**:
```latex
\section{Results}

We evaluated three multi-agent strategies (debate, self-consistency,
manager-worker) on ToMBench \cite{nemirovsky2023tombench}. Results
are shown in Table~\ref{tab:main-results}.

\begin{table}
\caption{Strategy comparison on ToMBench (N=388 tasks)}
\label{tab:main-results}
\begin{tabular}{lrrr}
\toprule
Strategy & Accuracy & Latency (s) & Cost (USD) \\
\midrule
Single & 0.72 & 1.2 & 0.00 \\
Debate & 0.82 & 3.4 & 0.00 \\
Self-consistency & 0.79 & 4.1 & 0.00 \\
\bottomrule
\end{tabular}
\end{table}

The debate strategy achieved the highest accuracy (82\%),
outperforming single-model baseline by 10 percentage points...
```

---

## Creating Your Own Skills

### Step 1: Choose What to Automate

Identify repetitive tasks:
- Do you always analyze experiments the same way?
- Do you run the same benchmark comparisons?
- Do you generate similar reports?

### Step 2: Create Skill File

```bash
# Create skill
touch .claude/skills/my-skill.md
```

### Step 3: Write the Skill

Use this template:

```markdown
# Skill: {skill-name}

You are an expert at {task description}.

## Task
When given {input}:
1. {Step 1}
2. {Step 2}
3. {Step 3}
...

## Output Format
{Describe what output should look like}

## Example
{Show an example if helpful}

## Important Notes
- {Any special instructions}
- {Things to watch out for}
```

### Example: Custom CRIT Analyzer

```markdown
# File: .claude/skills/analyze-crit.md

# Skill: analyze-crit

You are an expert at analyzing design critique experiments from the CRIT subsystem.

## Task
When given a CRIT experiment directory:
1. Read the experiment results
2. Count critiques by perspective (usability, security, etc.)
3. Measure critique coverage (how many issues found)
4. Calculate depth scores (how detailed were critiques)
5. Compare to UICrit baseline
6. Identify which perspectives were most valuable
7. Generate insights about multi-perspective critique

## Output Format
Save a markdown report to {experiment_dir}/crit_analysis.md with:
- Summary statistics
- Coverage analysis
- Perspective value analysis
- Comparison to baseline
- Recommendations for improvement

## Important Notes
- Use the evaluate_critique() function from crit.evals
- Compare to UICrit human critiques when available
- Focus on actionability of critiques
```

### Step 4: Use Your Skill

```bash
/analyze-crit experiments/multiperspective_checkout/
```

---

## Advanced Skill Techniques

### Passing Arguments

```markdown
# Skill: compare-strategies

You are comparing LLM strategies.

When given:
- benchmark: {benchmark_name}
- strategies: {comma-separated list}

Do:
1. Load benchmark
2. Find experiments for each strategy
3. Compare metrics
4. Generate comparison table
```

**Usage**:
```bash
/compare-strategies --benchmark tombench --strategies debate,single,self_consistency
```

### Multi-Step Workflows

```markdown
# Skill: full-evaluation

This is a comprehensive evaluation workflow.

## Steps

### Step 1: Run Benchmark
Run all strategies on the specified benchmark.

### Step 2: Analyze Results
For each strategy, generate detailed analysis.

### Step 3: Create Comparison
Build comparison table across strategies.

### Step 4: Search Literature
Find relevant papers and compare to published results.

### Step 5: Generate Report
Create comprehensive PDF report with all findings.

Save to: evaluations/{benchmark}_{date}/
```

### Conditional Logic

```markdown
# Skill: smart-analyzer

## Logic

If experiment uses benchmark:
  - Compare to published baselines
  - Search for related papers

If experiment is introspection:
  - Analyze detection accuracy
  - Find optimal layer/strength
  - Compare to other concepts

If experiment is CRIT:
  - Evaluate critique quality
  - Measure coverage
  - Compare perspectives

Always:
  - Calculate basic metrics
  - Generate insights
  - Save analysis.md
```

---

## Best Practices

### 1. Be Specific

**Good**:
```markdown
1. Read experiments/{name}/results.jsonl
2. Parse each line as JSON
3. Calculate mean latency: sum(latency_s) / count
4. Compare to baseline from benchmarks/{name}/baseline.json
```

**Not helpful**:
```markdown
1. Analyze the results
2. Compare to baseline
```

### 2. Include Examples

```markdown
## Example Input
experiments/debate_energy_20251103/

## Expected Output
analysis.md containing:
- Summary: 47 tasks, 82% accuracy, $0.24 cost
- Comparison: +12% vs baseline
- Insights: 3 key findings
```

### 3. Specify Output Location

```markdown
7. Save report to {experiment_dir}/analysis.md
8. Also save raw data to {experiment_dir}/metrics.json
```

### 4. Handle Errors

```markdown
## Error Handling
- If results.jsonl not found, check for results.json
- If no baseline available, note this in report
- If analysis fails, save error log to debug.txt
```

### 5. Make It Reusable

**Good** (reusable):
```markdown
When given ANY experiment directory, analyze it.
```

**Not good** (one-time use):
```markdown
Analyze experiments/debate_energy_20251103/
```

---

## Combining Skills With MCP

Skills become **incredibly powerful** when combined with MCP servers:

**Example: Full Research Workflow**:
```markdown
# Skill: research-workflow

You are a research assistant for LLM experiments.

## Workflow

### Step 1: Read Experiment (uses File System MCP)
Read all files from experiment directory.

### Step 2: Analyze Results
Calculate metrics, identify insights.

### Step 3: Search Literature (uses Brave Search MCP)
Find 3-5 recent papers related to:
- The strategy used
- The benchmark evaluated
- The findings observed

### Step 4: Compare
Compare your results to published papers.

### Step 5: Suggest Next Steps
Based on recent research, suggest:
- Improvements to try
- New benchmarks to evaluate
- Related research directions

### Step 6: Generate Report
Save comprehensive markdown report with all findings.
```

**Usage**:
```bash
/research-workflow experiments/debate_energy_20251103/

# Claude automatically:
# - Reads your files (File System MCP)
# - Searches papers (Brave MCP)
# - Generates insights
# - Saves report
```

---

## Skill Library Ideas

Here are more skills you might want to create:

### Research Skills
- `compare-models` - Compare different models on same benchmark
- `find-optimal` - Find optimal hyperparameters from sweep
- `generate-plots` - Create visualization scripts
- `extract-insights` - Pull key findings from experiments

### Documentation Skills
- `update-readme` - Update README with latest results
- `generate-api-docs` - Document code functions
- `create-changelog` - Generate changelog from git commits

### Experiment Skills
- `setup-sweep` - Create hyperparameter sweep config
- `analyze-sweep` - Analyze sweep results
- `compare-runs` - Compare specific runs in detail
- `find-failures` - Identify and diagnose failed runs

### Paper Writing Skills
- `write-abstract` - Generate abstract from experiments
- `create-tables` - Generate LaTeX tables
- `find-citations` - Find and format citations
- `draft-discussion` - Generate discussion section

---

## Troubleshooting

### Skill Not Found

**Issue**: `/my-skill` doesn't work

**Solutions**:
1. Check file is in `.claude/skills/my-skill.md`
2. Verify filename matches skill name
3. Restart Claude Desktop
4. Try full path: "Use skill at .claude/skills/my-skill.md"

### Skill Not Working As Expected

**Issue**: Skill runs but doesn't do what you want

**Solutions**:
1. Make instructions more specific
2. Add examples
3. Break into smaller steps
4. Test with simple case first

### Skill Too Slow

**Issue**: Skill takes forever

**Solutions**:
1. Break into smaller skills
2. Reduce scope (analyze fewer files)
3. Cache intermediate results
4. Use more efficient logic

---

## Examples From Your Research

### Example 1: Quick Experiment Check

```markdown
# Skill: quick-check

Quick sanity check of experiment results.

When given experiment directory:
1. Count total tasks run
2. Check for errors in logs
3. Calculate avg latency and cost
4. Return one-line summary

Output format:
"{name}: {tasks} tasks, {accuracy}% accuracy, {latency}s avg, ${cost}"
```

**Usage**:
```bash
/quick-check experiments/debate_energy_20251103/
# Output: "debate_energy_20251103: 47 tasks, 82% accuracy, 3.2s avg, $0.24"
```

### Example 2: Find Best Layer

```markdown
# Skill: find-best-layer

Find optimal layer for concept vector extraction.

When given:
- model: model name
- concept: concept name

Do:
1. Load all introspection results for this model/concept
2. Group by layer
3. Calculate avg detection accuracy per layer
4. Find layer with highest accuracy
5. Report optimal layer and why

Output:
"Optimal layer for {concept} in {model}: Layer {N} (accuracy: {X}%)"
Plus explanation of why this layer works best.
```

### Example 3: Strategy Recommender

```markdown
# Skill: recommend-strategy

Recommend best strategy for a task type.

When given:
- task_type: reasoning/creative/planning/tom
- constraints: latency/cost/quality

Do:
1. Load results for all strategies on this task type
2. Filter by constraints
3. Rank by quality (accuracy)
4. Return top 3 with pros/cons

Output:
Ranked list of strategies with:
- Expected accuracy
- Latency
- Cost
- When to use
- When to avoid
```

---

## Quick Reference

### Create Skill
```bash
touch .claude/skills/my-skill.md
# Edit with your favorite editor
```

### Use Skill
```bash
/my-skill

# Or with arguments
/my-skill --arg1 value1 --arg2 value2

# Or explicitly
"Use the my-skill skill on {target}"
```

### Included Skills
- `/analyze-experiment` - Analyze experiment results
- `/run-benchmark` - Run benchmark evaluation
- `/introspection-sweep` - Test introspection parameters
- `/write-paper` - Generate paper sections

### Skill Template
```markdown
# Skill: {name}

You are an expert at {task}.

When given {input}:
1. {Step 1}
2. {Step 2}
...

Output: {format}
```

---

## Next Steps

1. **Try included skills**: Run `/analyze-experiment` on your latest experiment
2. **Read skill files**: Look at `.claude/skills/` to see how they work
3. **Create your first skill**: Automate something you do repeatedly
4. **Combine with MCP**: Use File System + Brave Search for powerful workflows
5. **Build skill library**: Create skills for your common tasks

---

**Questions?** Check the main integration guide: `INTEGRATION_OPPORTUNITIES.md`
