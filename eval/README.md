# Hidden Layer Evaluation Harness

This directory implements the evaluation gatekeeper described in the lifelog + personalization spec. It provides:

- Structured configuration files under `configs/` for the mandatory test matrix (lifelog retrieval, personalization, long-context, TTL/TTT, and model editing).
- Dataset adapters in `loaders/` with caching support for the referenced benchmarks (LSC, NTCIR-18, ImageCLEF Lifelog, LoCoMo, LaMP, PrefEval, PersonaMem, etc.).
- Metric implementations in `scorers/` that cover retrieval quality, temporal correctness, preference adherence/drift, RAG faithfulness, and editing locality.
- Runner scripts in `runners/` that consume prediction logs and compute gate metrics based on a YAML configuration.
- Policy helpers in `policies/` encoding the promotion gates (GraphRAG/RAPTOR ablations, TTL budgets).
- Markdown/HTML scorecard rendering in `reports/` for CI artifacts.

## Usage

1. Download benchmarks to `eval/datasets/<dataset_name>/` (see papers for licensing terms). Alternatively set `EVAL_DATA_ROOT` to a custom directory.
2. Generate model predictions/logs for each dataset/retriever combination referenced by the configs. Use keys of the form `"{dataset}:{split}:{retriever}"` for lifelog retrieval or `"{dataset}:{split}"` elsewhere.
3. Call the runner entry point, e.g.:

```python
from pathlib import Path
from eval.runners.run_lifelog import evaluate_lifelog
from eval.reports.render_scorecard import render_scorecard

results = evaluate_lifelog(Path("eval/configs/lifelog.yaml"), predictions)
render_scorecard(results, title="Lifelog Retrieval", output=Path("lifelog.md"))
```

4. Enforce the promotion gates from the spec before shipping adapters/updates.

The repository intentionally keeps runners stateless: all behaviour is described in the YAML configs. Cached parquet copies of the datasets live under `eval/.cache` by default.
