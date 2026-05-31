# Metacognition — Do models know what they know?

Tools for measuring **metacognitive calibration** (does a model's stated confidence
track whether it is actually right?) and **introspective self-prediction** (can a model
forecast its own answer before producing it?).

Part of the Hidden Layer **Theory of Mind** research area, alongside
[`introspection`](../introspection/) (understanding self) and
[`selphi`](../selphi/) (understanding others). Confidence calibration and
self-prediction are two concrete, measurable handles on self-knowledge — and a
candidate **alignment signal**: a model that reliably knows when it doesn't know is
safer to deploy and easier to defer to a human.

## Why this exists

- **A real gap.** Nothing in the lab measured calibration before this.
- **Runs anywhere.** The core (`metrics.py`) is **pure stdlib** — no numpy, no
  network. Experiments use the harness `sim` provider, a new deterministic offline
  LLM stand-in, so the whole thing runs in CI or an air-gapped box with **zero API
  keys**. Plug in a real provider (`anthropic`, `ollama`, `mlx`, …) to measure a real
  model with the same code.

## Quick start

```bash
# Fully offline demo: calibration profiles + reliability diagram + self-prediction
python theory-of-mind/metacognition/demo.py
```

```python
from theory_of_mind.metacognition import (
    load_dataset, SimulatedSubject, run_calibration, ascii_reliability_diagram,
)

dataset = load_dataset()                                  # bundled factual Q&A
subject = SimulatedSubject(accuracy=0.7, profile="overconfident")
run = run_calibration(subject, dataset)

print(run.report.ece, run.report.brier, run.report.overconfidence)
print(ascii_reliability_diagram(run.report))
```

### Measuring a real model

One command (needs an API key, or use a local provider):

```bash
python theory-of-mind/metacognition/run_real.py --provider anthropic --model claude-3-5-sonnet-20241022
python theory-of-mind/metacognition/run_real.py --provider ollama --model llama3.2:latest
python theory-of-mind/metacognition/run_real.py --provider sim   # offline plumbing check
```

It prints calibration (ECE/MCE/Brier), discrimination (AUROC), and a reliability diagram,
logs a tracked run, and saves JSON/PNG to `output/`. Or via the API:

```python
from theory_of_mind.metacognition import ModelSubject, run_calibration

subject = ModelSubject(provider="anthropic", model="claude-3-5-sonnet-20241022")
run = run_calibration(subject, track=True)   # logs via the harness experiment tracker
print(run.report.to_dict())
```

The subject is asked each question in a fixed `Answer: … / Confidence: …%` format;
confidence is parsed robustly (handles `90%`, `0.9`, `90`, "85 percent sure"), and
correctness is scored offline by normalized token matching against accepted answers.

### Self-prediction (introspection)

```python
from theory_of_mind.metacognition import SimulatedIntrospector, run_self_prediction

run = run_self_prediction(SimulatedIntrospector(accuracy=0.7, self_knowledge=0.9))
print(run.self_prediction_accuracy, run.task_accuracy)
```

A subject first **predicts** its answer, then **answers**. Self-prediction accuracy
(does the forecast match the later answer?) is reported separately from task accuracy,
because genuine self-knowledge means predicting your own answer well *whether or not it
is correct*.

## Metrics (`metrics.py`)

| Metric | Meaning | Good value |
|---|---|---|
| **Brier score** | mean squared error of confidence vs. outcome (a *proper* scoring rule) | low (0 = perfect) |
| **ECE** | Expected Calibration Error: bin-weighted \|confidence − accuracy\| | low (0 = calibrated) |
| **MCE** | worst calibration gap across bins | low |
| **AUROC** | can confidence rank correct vs. incorrect answers? (discrimination) | high (0.5 = chance) |
| **Overconfidence** | mean confidence − accuracy (signed) | near 0 |
| **Reliability bins** | per-bin confidence vs. accuracy (for diagrams) | on the diagonal |

Calibration (ECE) and discrimination (AUROC) are **independent** axes — the demo's
`random` profile makes this concrete: confidences carry no signal, so AUROC ≈ 0.5 while
the calibrated/over/under profiles share the same discrimination.

## Layout

```
theory-of-mind/metacognition/
├── README.md              # this file
├── CLAUDE.md              # development guide
├── demo.py                # offline end-to-end demo
├── data/factual_qa.json   # bundled short-answer questions
├── code/
│   ├── metrics.py         # Brier, ECE, MCE, AUROC, reliability (pure stdlib)
│   ├── dataset.py         # Question/Dataset + deterministic scoring + synthetic gen
│   ├── subjects.py        # ModelSubject (real) + SimulatedSubject (deterministic)
│   ├── runner.py          # run_calibration -> CalibrationRun
│   ├── selfprediction.py  # self-prediction / introspection task
│   └── report.py          # JSON + optional matplotlib reliability plots
└── tests/                 # 32 tests, run fully offline
```

Importable as `theory_of_mind.metacognition` (the underscore package bridges to this
dash-named directory, per the lab's dual-directory convention).

## Tests

```bash
python -m pytest theory-of-mind/metacognition/tests -q
```

All tests run offline (no API keys, no network) via the `sim` provider.
