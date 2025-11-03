# Skill: Analyze Experiment

You are an expert at analyzing multi-agent LLM experiment results from the Hidden Layer research platform.

## Task

When given an experiment directory path:

1. **Read experiment files** (use File System MCP):
   - `config.json` - Experiment configuration
   - `results.jsonl` - Individual task results (one JSON object per line)
   - `summary.json` - Aggregated metrics

2. **Calculate metrics**:
   - Total tasks run
   - Average latency (mean of latency_s)
   - Total cost (sum of cost_usd)
   - Token usage (sum tokens_in and tokens_out)
   - Accuracy (if available in results)

3. **Identify strategy details**:
   - Strategy name
   - Model and provider used
   - Any special parameters (n_debaters, temperature, etc.)

4. **Compare to baselines** (if applicable):
   - Load baseline scores using `harness.get_baseline_scores(benchmark_name)`
   - Compare your results to human performance, GPT-4, etc.
   - Calculate improvement percentage

5. **Search for related research** (use Brave Search MCP):
   - Search for recent papers about the strategy used
   - Search for papers about the benchmark (if applicable)
   - Find any recent improvements or new techniques

6. **Generate insights**:
   - When did the strategy outperform baseline?
   - Was the extra cost/latency justified by quality improvement?
   - Are there patterns in successful vs failed tasks?
   - What could be improved?

7. **Create analysis report**:
   - Save a markdown file to `{experiment_dir}/analysis.md`
   - Include all findings, comparisons, and recommendations
   - Format with clear sections and tables

## Report Template

```markdown
# Experiment Analysis: {experiment_name}

**Date**: {date}
**Strategy**: {strategy}
**Model**: {provider}/{model}

## Summary Statistics

| Metric | Value |
|--------|-------|
| Tasks Run | {count} |
| Avg Latency | {latency}s |
| Total Cost | ${cost} |
| Tokens (in/out) | {tokens_in} / {tokens_out} |
| Accuracy | {accuracy}% |

## Strategy Details

- **Strategy**: {strategy_name}
- **Model**: {model}
- **Parameters**: {key parameters}

## Comparison to Baselines

{If benchmark available, show comparison table}

| Metric | Your Result | Baseline | SOTA | Human |
|--------|-------------|----------|------|-------|
| Accuracy | {your}% | {baseline}% | {sota}% | {human}% |

**Finding**: Your result is {comparison description}

## Related Research

{Papers found via Brave Search}

1. **{Paper Title}** [{link}]
   - {Key finding relevant to your experiment}

2. **{Paper Title}** [{link}]
   - {Key finding}

## Key Insights

1. {Insight 1 - when did strategy help?}
2. {Insight 2 - cost-benefit analysis}
3. {Insight 3 - patterns observed}

## Recommendations

Based on analysis and recent research:

1. {Recommendation 1}
2. {Recommendation 2}
3. {Recommendation 3}

## Next Steps

- [ ] {Suggested next experiment}
- [ ] {Another follow-up}
```

## Important Notes

- If files are missing, note this and analyze what's available
- If no baseline exists, note this but still provide insights
- Focus on actionable findings
- Compare cost vs quality tradeoffs
- Be specific about improvements to try

## Example Usage

```
User: "Use analyze-experiment skill on experiments/debate_energy_20251103_143022_a3f9/"

You:
1. Read files from that directory
2. Calculate all metrics
3. Search for "multi-agent debate LLM" and "{benchmark} benchmark" papers
4. Generate comprehensive analysis
5. Save to experiments/debate_energy_20251103_143022_a3f9/analysis.md
```

## Error Handling

- If config.json missing: Check for alternative config files, infer from directory name
- If results.jsonl missing: Check for results.json
- If no baseline available: Note this and focus on internal analysis
- If search fails: Continue with analysis based on available data
