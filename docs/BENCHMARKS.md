# Benchmark Datasets for SELPHI and CRIT

This document provides quick access to established benchmark datasets for both projects.

## SELPHI (Theory of Mind) Benchmarks

### ToMBench (ACL 2024)
- **Size**: 2,860 testing samples
- **Coverage**: 8 ToM tasks, 31 abilities
- **Languages**: Bilingual
- **Format**: JSONL
- **Installation**: `git clone https://github.com/zhchen18/ToMBench.git`
- **Paper**: https://aclanthology.org/2024.acl-long.847.pdf
- **Usage**:
  ```python
  from selphi import load_tombench
  benchmark = load_tombench()
  ```
- **Note**: Use for evaluation only, NOT training

### OpenToM (2024)
- **Size**: 696 narratives, 16,008 questions
- **Features**: Character personalities, longer narratives
- **Question Types**: Location, multihop, attitude
- **Format**: JSON
- **Installation**: `git clone https://github.com/seacowx/OpenToM.git`
- **HuggingFace**: `SeacowX/OpenToM`
- **Usage**:
  ```python
  from selphi import load_opentom
  benchmark = load_opentom(include_long=True)
  ```
- **Note**: Do NOT use for training or fine-tuning

### SocialIQA (2019)
- **Size**: 38,000 QA pairs
- **Focus**: Social commonsense reasoning
- **Format**: Multiple choice (3 options)
- **Installation**: `pip install datasets`
- **HuggingFace**: `allenai/social_i_qa`
- **Baseline**: GPT-4: 79%, Human: 84%
- **Usage**:
  ```python
  from selphi import load_socialiqa
  benchmark = load_socialiqa(split="validation")
  ```

## CRIT (Design Critique) Benchmarks

### UICrit (UIST 2024)
- **Size**: 11,344 critiques for 1,000 mobile UIs
- **Source**: RICO dataset (mobile UI screenshots)
- **Annotators**: 7 experienced designers + LLM
- **Ratings**: Aesthetics, learnability, efficiency, usability
- **Format**: CSV with bounding boxes
- **Installation**: `git clone https://github.com/google-research-datasets/uicrit.git`
- **Paper**: https://arxiv.org/abs/2407.08850
- **License**: CC BY 4.0
- **Usage**:
  ```python
  from crit import load_uicrit, load_uicrit_for_comparison
  benchmark = load_uicrit()
  comparison = load_uicrit_for_comparison(sample_size=10)
  ```
- **Performance**: LLMs achieved 55% improvement with few-shot prompting

## Quick Start

### SELPHI Benchmark Evaluation

```python
from selphi import load_socialiqa, run_multiple_scenarios, evaluate_batch, results_to_dict_list

# Load benchmark
benchmark = load_socialiqa(split="validation", max_samples=100)

# Run scenarios
results = run_multiple_scenarios(
    benchmark.scenarios,
    provider="ollama",
    temperature=0.1,
    verbose=True
)

# Evaluate
evaluation = evaluate_batch(results_to_dict_list(results))
print(f"Score: {evaluation['overall_average']:.3f}")
```

### CRIT Benchmark Evaluation

```python
from crit import load_uicrit_for_comparison, run_critique_strategy, compare_to_experts

# Load UICrit for comparison
comparison_data = load_uicrit_for_comparison(sample_size=10)

# Run your critique
for item in comparison_data:
    critique = run_critique_strategy(
        "multi_perspective",
        item['problem'],
        provider="ollama"
    )

    # Compare to expert critiques
    comparison = compare_to_experts(
        critique.recommendations,
        item['expert_critiques'],
        method="overlap"
    )
    print(f"Expert overlap: {comparison['overlap_score']:.2%}")
```

## Installation Commands

```bash
# ToMBench
git clone https://github.com/zhchen18/ToMBench.git

# OpenToM
git clone https://github.com/seacowx/OpenToM.git

# SocialIQA (via HuggingFace)
pip install datasets

# UICrit
git clone https://github.com/google-research-datasets/uicrit.git
```

## Benchmark Notebooks

### SELPHI
- `notebooks/selphi/02_benchmark_evaluation.ipynb` - Comprehensive tutorial on using all three ToM benchmarks

### CRIT
- `notebooks/crit/02_uicrit_benchmark.ipynb` - UICrit evaluation and comparison to experts

## Comparison with Published Results

### ToM Benchmarks

| Model | ToMBench | OpenToM | SocialIQA |
|-------|----------|---------|-----------|
| GPT-4 | TBD | TBD | 79% |
| Human | TBD | TBD | 84% |
| Your Model | Run to measure | Run to measure | Run to measure |

**Note**: Run full benchmarks to get accurate comparison metrics

### Design Critique

| Approach | UICrit Coverage | Expert Agreement |
|----------|----------------|------------------|
| Baseline LLM | ~45% | Low |
| Few-Shot LLM | ~70% (55% gain) | Medium |
| Your Strategy | Run to measure | Run to measure |

## Best Practices

### For SELPHI

1. **Use for Evaluation Only**: Never train on ToMBench or OpenToM
2. **Low Temperature**: Use 0.1-0.2 for consistent reasoning
3. **Full Dataset**: Run on complete datasets for accurate metrics
4. **Compare Splits**: Use validation sets for fair comparison
5. **Track Everything**: Use experiment tracker for reproducibility

### For CRIT

1. **Compare to Experts**: Benchmark against human designer critiques
2. **Quality Correlation**: Analyze relationship with quality ratings
3. **Few-Shot Learning**: Include expert examples in prompts
4. **Multiple Perspectives**: Test if collective critique helps
5. **Visual Context**: Note that UIs have screenshots (not available in code alone)

## Data Formats

### ToMBench JSONL Structure
```json
{
  "id": "...",
  "context": "...",
  "question": "...",
  "choices": ["A", "B", "C", "D"],
  "answer_key": "A"
}
```

### OpenToM JSON Structure
```json
{
  "id": "...",
  "narrative": "...",
  "question": "...",
  "choices": ["..."],
  "answer": "...",
  "question_type": "location|multihop|attitude"
}
```

### SocialIQA Structure
```json
{
  "context": "...",
  "question": "...",
  "answerA": "...",
  "answerB": "...",
  "answerC": "...",
  "label": "1|2|3"
}
```

### UICrit CSV Structure
```csv
rico_id,task,aesthetics_rating,learnability,efficency,usability_rating,design_quality_rating,comments_source,comments
```

## Citations

### ToMBench
```bibtex
@inproceedings{chen2024tombench,
  title={ToMBench: Benchmarking Theory of Mind in Large Language Models},
  author={Chen, Zhuang and others},
  booktitle={ACL},
  year={2024}
}
```

### OpenToM
```bibtex
@article{opentom2024,
  title={OpenToM: A Comprehensive Benchmark for Evaluating Theory-of-Mind Reasoning Capabilities of Large Language Models},
  year={2024}
}
```

### SocialIQA
```bibtex
@inproceedings{sap2019socialiqa,
  title={SocialIQA: Commonsense Reasoning about Social Interactions},
  author={Sap, Maarten and others},
  booktitle={EMNLP},
  year={2019}
}
```

### UICrit
```bibtex
@inproceedings{duan2024uicrit,
  title={UICrit: Enhancing Automated Design Evaluation with a UI Critique Dataset},
  author={Duan, Yuwen and others},
  booktitle={UIST},
  year={2024}
}
```

## Further Resources

- **SELPHI Documentation**: `code/selphi/README.md`
- **CRIT Documentation**: `code/crit/README.md`
- **Benchmark Loaders**: `code/selphi/benchmarks.py`, `code/crit/benchmarks.py`
- **Example Notebooks**: `notebooks/selphi/`, `notebooks/crit/`

## Contributing New Benchmarks

To add a new benchmark:

1. Create loader function in `benchmarks.py`
2. Convert to native format (ToMScenario or DesignProblem)
3. Add to `list_available_benchmarks()`
4. Update this documentation
5. Add example notebook
6. Test with existing evaluation pipeline

## License Notes

- **ToMBench**: Check repository for license
- **OpenToM**: Check repository for license
- **SocialIQA**: Apache 2.0
- **UICrit**: CC BY 4.0

Always review and comply with dataset licenses before use.
