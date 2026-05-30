# Metacognition — Development Guide

## Project Overview

Measure whether models **know what they know**:

- **Calibration** — does stated confidence track correctness? (Brier, ECE, MCE, AUROC)
- **Self-prediction** — can a model forecast its own answer before giving it?

**Research Question**: Is a model's confidence a trustworthy signal of its correctness,
and can a model introspect on its own outputs? Both are candidate **alignment signals**.

**Theme connections**:
- `introspection/` — self-prediction is an introspection probe; calibration is a
  measurable honesty signal for self-reports.
- `selphi/` — perspective-taking on *others*; calibration is perspective-taking on
  *oneself*.
- `alignment/steerability/` — a well-calibrated "I don't know" enables safe deferral.

**Uses**: `harness/` for the LLM provider abstraction (including the offline `sim`
provider) and experiment tracking.

## Architecture

```
code/
  metrics.py         # Pure-stdlib calibration metrics + ASCII reliability diagram
  dataset.py         # Question/Dataset, deterministic offline scoring, synthetic gen
  subjects.py        # ModelSubject (real LLM) + SimulatedSubject (deterministic)
  runner.py          # run_calibration() -> CalibrationRun (+ optional tracker logging)
  selfprediction.py  # ModelIntrospector / SimulatedIntrospector + run_self_prediction
  report.py          # save_json + optional matplotlib plot_reliability
```

### Design principles honored here

1. **Provider-agnostic** — the same pipeline runs on `sim`, `ollama`, `mlx`,
   `anthropic`, `openai`. Swap the subject, keep the metrics.
2. **Reproducible & offline-first** — `metrics.py` has no third-party deps;
   `SimulatedSubject`/`SimulatedIntrospector` are seeded and deterministic; matplotlib
   is optional (ASCII fallback).
3. **Interpretable** — every metric is a few lines of explicit arithmetic with a
   docstring tying it to the literature.
4. **Falsifiable knobs** — the simulated subjects have explicit `accuracy`, `profile`,
   `strength`, and `self_knowledge` parameters, so tests assert the metrics respond the
   way the theory predicts (e.g. overconfident ⇒ higher ECE; random ⇒ AUROC ≈ 0.5).

## The `sim` provider (added to the harness)

`harness/llm_provider.py` gained a deterministic `provider="sim"` branch — a test
double / offline stand-in that makes no network calls:

- `sim_response="..."` → returns that exact text (inject a canned answer)
- `sim_responses=[...]` → deterministically picks one by hashing the prompt
- neither → a stable hash-derived placeholder
- `sim_seed=int` → reproducible perturbation of the stream

It is general infrastructure (useful for any offline test in the lab), but it is what
lets `SimulatedSubject` route its formatted answer through the *real* parse-and-score
pipeline rather than shortcutting it.

## Common workflows

```bash
# Offline demo (calibration profiles, reliability diagram, self-prediction)
python theory-of-mind/metacognition/demo.py

# Tests (offline)
python -m pytest theory-of-mind/metacognition/tests -q
```

```python
# Compare a real model against the simulated baselines
from theory_of_mind.metacognition import ModelSubject, run_calibration
run = run_calibration(ModelSubject(provider="ollama", model="llama3.2:latest"), track=True)
```

## Extending

- **New dataset**: drop a JSON file in `data/` matching the schema in `dataset.py`
  (`{"name", "description", "questions": [{"id","question","answers","difficulty"}]}`)
  and `load_dataset(path=...)`, or generate one with `make_arithmetic_dataset()`.
- **New calibration profile**: add a branch in `SimulatedSubject._plan` and append the
  name to `CALIBRATION_PROFILES`.
- **New metric**: add a pure function to `metrics.py` (keep it numpy-free) and wire it
  into `calibration_report` if it belongs in the standard summary.

## Gotchas

- Confidences are **probabilities in [0, 1]**. Percentages are converted at parse time;
  `metrics.py` rejects out-of-range values.
- The bundled `factual_qa` set is only 25 items — great for exercising the real
  answer/parse/score path, but too small to see clean calibration. Use
  `make_arithmetic_dataset(n=…)` for large-N behavior (the demo and tests do).
- Correctness scoring is intentionally simple (normalized token / phrase match). It is
  not an LLM judge; keep accepted-answer lists explicit so scoring stays deterministic.
