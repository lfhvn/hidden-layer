# Hidden Layer Runbook

The operational guide for executing research here: **hypothesis → test → publish**, plus
the everyday commands. Everything in this runbook runs **offline by default** (the
deterministic `sim` provider — no API keys, no network); swap in a real provider when you
want real model results.

- New to the repo / hardware setup? → [`QUICKSTART.md`](QUICKSTART.md)
- Picking a project? → [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)
- Stuck? → [`FAQ.md`](FAQ.md) or `python3 check_setup.py`

---

## TL;DR — the five commands

```bash
make setup                                            # 1. one-time environment setup
python scripts/new_project.py <area> <name>           # 2. scaffold a project (offline test included)
python -m harness.experiment run <spec>.yaml          # 3. run an experiment -> tracked runs + table
make -C papers results                                # 4. bind results into a paper (numbers from runs)
make ci                                               # 5. lint + 160+ offline tests + papers + docs
```

---

## 0. Setup (once)

```bash
./setup.sh            # or: make setup   (creates venv/, installs requirements.txt)
source venv/bin/activate
python3 check_setup.py   # verifies Python >=3.10, deps, providers
```

You do **not** need API keys or local models to run tests, experiments, demos, or the
papers pipeline — they default to the offline `sim` provider. Keys/models are only needed
to evaluate a *real* model (see [§5](#5-using-a-real-model)).

---

## 1. The research loop

```
   ┌─ scaffold ──► define experiment ──► run ──► analyze ──► bind to paper ─┐
   │  new_project    spec.yaml         run_experiment  analysis   make results│
   └──────────────────────────── make ci guards it all ◄─────────────────────┘
```

### 1a. Scaffold a project (optional, for new work)

```bash
python scripts/new_project.py theory-of-mind working-memory \
    --question "Do models maintain working memory across turns?"
python -m pytest theory-of-mind/working-memory/tests -q     # green immediately, offline
```

Creates the dash project dir (`README`, `CLAUDE`, `hypotheses.md`, `code/`, `tests/`),
the importable wrapper (`theory_of_mind/working_memory/`), an offline smoke test, and
wires the test dir into `make ci`. See [`CLAUDE.md`](CLAUDE.md) → "For new projects".

### 1b. Define an experiment

An experiment is one YAML spec: a hypothesis, a dataset of tasks, an eval, and one or
more **arms** to compare. Copy a template:

- `config/experiments/example_qa.yaml` — strategies/providers vs. accuracy
- `config/experiments/multi_agent_cost.yaml` — coordination cost (tokens) across strategies

```yaml
name: my_experiment
hypothesis: "Lower temperature improves exact-match accuracy on factual QA."
eval_type: keyword                     # exact_match | keyword | numeric | llm_judge
dataset:
  tasks:                               # inline, or use:  path: tasks.jsonl
    - { input: "Capital of France?", expected: paris }
    - { input: "2 + 2?", expected: "4", eval_type: numeric }   # per-task override
arms:                                  # each arm = one tracked run
  - { name: cold,  provider: sim, params: { sim_response: "paris 4", temperature: 0.0 } }
  - { name: debate, provider: sim, strategy: debate, params: { sim_response: "paris 4" } }
  - { name: sonnet, config: claude-sonnet }     # a config/models.yaml preset (needs API key)
```

Arm options: `provider`+`model`, a named `config` preset, a `strategy`
(`single`/`debate`/`consensus`/`self_consistency`/`manager_worker`/`crit`), and `params`
(any extra kwargs — `sim_response`, `n_debaters`, `temperature`, …).

### 1c. Run it

```bash
# CLI — writes tracked runs to ./experiments/ and prints a table
python -m harness.experiment run config/experiments/my_experiment.yaml --format markdown
```

```python
# Python — same thing, with the report object in hand
from harness import load_spec, run_experiment, analysis
report = run_experiment(load_spec("config/experiments/my_experiment.yaml"))
print(analysis.to_markdown(report.to_rows()))
```

Each arm becomes a reproducible run under `experiments/<name>.<arm>_<timestamp>_<hash>/`
(`config.json` + `results.jsonl` + `summary.json`), with the git hash recorded.

### 1d. Analyze

```bash
# Re-tabulate previously logged runs (markdown | latex | csv)
python -m harness.experiment report "experiments/my_experiment.*" --format latex
```

```python
from harness import analysis
rows = analysis.runs_to_rows(analysis.load_runs("experiments/my_experiment.*"))
print(analysis.to_markdown(rows))                       # comparison table
print(analysis.latex_macros(rows, metrics=["accuracy"]))  # \newcommand per (arm, metric)
analysis.bar_plot(rows, "accuracy", "fig.png")          # optional png (ASCII fallback)
```

Rows include eval metrics, `tokens_in`/`tokens_out`, `latency_s`, `cost_usd`.

### 1e. Bind results into a paper

So a paper cites numbers **from runs**, not hand-typed (which go stale):

1. Add a generator to `papers/generate_results.py` (`GENERATORS["my_paper"] = ...`) that
   runs the experiment and emits LaTeX via `analysis.latex_macros` / `analysis.to_latex`.
2. `\input{generated/my_paper_macros.tex}` in your `papers/my-paper.tex` preamble and the
   table in the body. See `papers/metacognition-calibration.tex` for a worked example.
3. Regenerate + build:

```bash
make -C papers results                # rewrites papers/generated/*.tex from experiments
make -C papers metacognition-calibration.pdf   # needs latexmk + a TeX install
```

Details: [`papers/README.md`](papers/README.md).

---

## 2. See it work end-to-end (copy-paste, offline)

```bash
# Calibration / introspection demo (prints metrics + reliability diagram)
python theory-of-mind/metacognition/demo.py

# Coordination-cost benchmark (token overhead per strategy, accuracy held fixed)
python -m harness.experiment run config/experiments/multi_agent_cost.yaml --format markdown

# Regenerate the numbers both bound papers cite, then validate them
make -C papers results && make -C papers check
```

---

## 3. Everyday commands

| Command | What it does |
|---|---|
| `make ci` | Full local gate: format-check, lint, **offline tests**, papers check, docs. Mirrors GitHub CI. |
| `make test` | Run all offline test suites (no keys/network). |
| `make format` / `make lint` | Auto-format / lint the maintained paths. |
| `make papers` | Regenerate + validate bound-paper results. |
| `make help` | List all targets. |
| `python -m harness.experiment run <spec>` | Run an experiment spec. |
| `python -m harness.experiment report <glob>` | Tabulate logged runs. |
| `python scripts/new_project.py <area> <name>` | Scaffold a new project. |

---

## 4. What runs on every push (CI)

`.github/workflows/ci.yml` mirrors `make ci` and is **fully offline** (Python
3.10/3.11/3.12): it runs the offline test suites, regenerates+checks bound papers, and
verifies docs/format/lint. New projects scaffolded with `new_project.py` are wired in
automatically. Heavy backends (torch/fastapi: lens, steerability, ai-to-ai-comm) are
intentionally excluded from the offline gate.

---

## 5. Using a real model

Everything above defaults to `provider="sim"`. To run against a real model, set keys and
point the arm/call at a provider:

```bash
cp .env.example .env        # add ANTHROPIC_API_KEY / OPENAI_API_KEY
# or for local models: `ollama serve` (+ `ollama pull llama3.2:latest`), or MLX on Apple Silicon
```

```python
from harness import llm_call, run_experiment, ExperimentSpec
llm_call("Hello", provider="anthropic", model="claude-3-5-sonnet-20241022")

# In a spec, an arm is just: { name: sonnet, provider: anthropic, model: claude-3-5-sonnet-20241022 }
# or reference a preset from config/models.yaml: { name: sonnet, config: claude-sonnet }
```

The runner, analysis, and paper-binding are identical — only the arm's provider changes.
Provider capabilities/limits: [`docs/infrastructure/provider-limitations.md`](docs/infrastructure/provider-limitations.md).

**A good first real run** — does a real model know what it knows?

```bash
python theory-of-mind/metacognition/run_real.py --provider anthropic --model claude-3-5-sonnet-20241022
# -> calibration (ECE/MCE/Brier), AUROC, reliability diagram; logs a tracked run + JSON/PNG.
```

---

## 6. Conventions that keep this working

- **Offline-first**: tests/demos/CI use `sim`; real providers are opt-in.
- **Reproducible**: experiments are seeded and logged; paper numbers come from runs.
- **One gate**: `make ci` is the single source of truth (GitHub CI just calls it).
- **Scaffold new work**: `new_project.py` bakes in the layout, offline test, and CI wiring.

For deeper architecture and research framing: [`CLAUDE.md`](CLAUDE.md), [`RESEARCH.md`](RESEARCH.md).
