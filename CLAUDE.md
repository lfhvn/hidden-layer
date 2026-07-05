# Claude Development Guide - Hidden Layer Lab

**Hidden Layer** is an independent research lab studying agent communication,
theory of mind, internal representations, and alignment. This file is loaded
into every agent session. Read the rules first; they exist because each one
was violated by an earlier agent session and had to be cleaned up by hand.

---

## Non-Negotiable Rules

1. **Never fabricate results.** No numbers, tables, plots, or "findings" that
   were not produced by an actual run. No invented citations, no fake arXiv
   IDs, no sample data attributed to real people. If you need illustrative
   data, label it `SYNTHETIC` in the filename and the content.
   *(This repo once contained 7 papers full of fabricated results. All were
   deleted. Do not recreate that problem.)*

2. **No claim without a results file.** Any number cited in any document must
   trace to a committed run directory under `results/` (config.json +
   results.jsonl + summary.json + git SHA). See `results/README.md`.
   `experiments/` is gitignored scratch space — citable runs go in `results/`.

3. **Verify before you report done.** Minimum bar for any change:
   ```bash
   python -m pytest tests/ -q        # must pass (runs in <1s, no excuses)
   python -m compileall -q . -x '\.git|node_modules'
   make docs                         # if you touched documentation paths
   ```
   If you claim something works, you ran it. If you couldn't run it (missing
   hardware, no API key), say so explicitly instead of implying success.

4. **One source of truth.** Never create a second copy of tests, docs, or
   configs "for the project" — stale duplicates have caused 27-test failure
   piles here. Root `tests/` is the test suite. If a doc exists, update it;
   don't write a parallel one.

5. **Don't leave the tree broken.** No committed merge-conflict markers, no
   imports of modules that don't exist, no docs describing apps that were
   deleted. If you move or delete a file, grep for references and fix them:
   ```bash
   grep -rn "old-name" --include="*.md" --include="*.py" .
   ```

6. **Status honesty in docs.** A README may only say "working" about code
   that runs. Aspirational features are listed under "Planned", not
   described in the present tense.

---

## Repo Map (with ground-truth status)

| Path | What it is | Status |
|---|---|---|
| `harness/` | LLM provider abstraction, experiment tracking, evals | Working; the core library |
| `shared/` | Concept vectors, datasets, utils | Working utils; datasets mostly empty |
| `communication/multi-agent/` | Debate/consensus/CRIT strategies | Working code; needs live LLM backend |
| `communication/ai-to-ai-comm/` | Cache-to-Cache KV transfer (torch) | Real code; needs GPU + HF models, unvalidated |
| `theory-of-mind/selphi/` | ToM scenario evaluation | Working; needs LLM backend |
| `theory-of-mind/introspection/` | Concept vectors, activation steering | Working; steering needs MLX |
| `representations/latent-space/lens/` | SAE training backend (FastAPI+torch) | Working backend; API-only (frontend removed) |
| `representations/latent-space/calm/` | Continuous latent modeling | Prototype (autoencoder + energy transformer) |
| `alignment/steerability/` | Activation steering dashboard | Real backend; needs torch + HF model |
| `memory/lifelog-personalization/` | Memory/retrieval eval harness | Real metrics; zero data downloaded |
| `agentmesh/` | Workflow orchestration platform | Skeleton works; frozen pending Phase-1 evidence |
| `mlx_lab/` | MLX model CLI | Working (Apple Silicon for model ops) |
| `ai_research_aggregator/` | arXiv digest + Substack publishing | Working fetchers |
| `web-tools/steerability/` | Public steering demo | Real backend, thin frontend |
| `results/` | Committed experiment results | See `results/README.md` |

Planning docs: `ROADMAP.md` (18-month plan) · `RESEARCH.md` (themes) ·
`docs/` (FAQ, PROJECT_GUIDE, SETUP, infrastructure, workflows).

---

## The Dual-Directory Import Pattern (read before touching imports)

Project files live in dash-named directories; Python imports go through
underscore-named loader packages:

| | Example |
|---|---|
| Project files (README, notebooks, code) | `communication/multi-agent/` |
| Importable package (loader shim) | `communication/multi_agent/` |

```python
# Correct — import through the underscore package:
from communication.multi_agent import run_strategy, STRATEGIES
from theory_of_mind.selphi import scenarios
from theory_of_mind.introspection import ConceptLibrary
from harness import llm_call, ExperimentTracker
```

Gotchas that have actually bitten:
- The underscore `__init__.py` files are ~40-line `importlib` shims that load
  the dash directory by **relative path**. If you move directories, update the
  shim's `Path(...)` arithmetic and **test the import** — a wrong `.parent`
  count fails silently or at import time.
- `harness` does NOT export the strategies. `run_strategy` lives in
  `communication.multi_agent`.
- Never `sys.path`-hack around the shims or create a `code/` copy; fix the
  shim instead.

---

## Harness Quick Reference

```python
from harness import llm_call, ExperimentConfig, ExperimentResult, ExperimentTracker, evaluate_task

# Provider-agnostic call (local first, API for frontier runs)
r = llm_call("prompt", provider="ollama", model="llama3.2:latest")
r = llm_call("prompt", provider="anthropic", model="claude-sonnet-5")
r.text, r.latency_s, r.tokens_in, r.tokens_out, r.cost_usd

# Tracked experiment (base_dir="results/..." for citable runs)
tracker = ExperimentTracker(base_dir="results/<project>/<experiment>")
tracker.start_experiment(ExperimentConfig(experiment_name=..., task_type=...,
                                          strategy=..., provider=..., model=...,
                                          git_hash=<short SHA>))
tracker.log_result(ExperimentResult(config=..., task_input=..., output=...,
                                    latency_s=..., eval_scores=evaluate_task(task, out)))
tracker.finish_experiment()   # writes summary.json
```

- Eval score keys prefixed `_` are metadata and excluded from aggregation.
- Named system prompts live in `config/system_prompts/`
  (`llm_call(..., system_prompt="researcher")`).
- Reference runner: `scripts/run_phase0.py` shows the full
  load-tasks → run-strategy → track → results/ loop.

## Environment Constraints

| Provider | Needs | Notes |
|---|---|---|
| `ollama` | local `ollama serve` | default for iteration |
| `mlx` | Apple Silicon only | temperature currently ignored |
| `anthropic` / `openai` | API key in env | spend on final runs, not iteration |

torch/transformers projects (lens, steerability, ai-to-ai-comm) need those
deps installed; they are not in the base requirements. CI (3.10–3.12) runs
the root suite only. Python ≥3.10 required.

---

## Commands That Work

```bash
make setup                # venv + requirements
python check_setup.py     # environment sanity check
python -m pytest tests/ -q            # root suite (~56 tests, <1s)
make docs                 # verify documentation files exist
flake8 harness/ shared/ tests/ --select=E9,F63,F7,F82   # CI hard-error gate
python scripts/run_phase0.py --provider ollama --model llama3.2:latest
```

---

## Development Workflows

**Working on a project**: `cd <area>/<project>/`, read its `CLAUDE.md`, make
changes, then run the root verification (Rule 3). Project docs list their own
run commands.

**Adding cross-project features**: core infrastructure → `harness/`;
utilities → `shared/utils/`; then update `docs/infrastructure/` and the
affected project guides.

**Adding a project**: dash directory for files, underscore shim for imports
(copy an existing shim and fix the paths), README + CLAUDE.md, entry in
`docs/PROJECT_GUIDE.md` and the map above.

**Research methodology** (full version in ROADMAP.md):
1. Frame the question and pre-register the design (hypotheses, cells, seeds).
2. Iterate on local models; log everything through the tracker.
3. Promote final runs to `results/` and cite only from there.
4. Write up with links to run directories; a null result is a result.

---

## Questions to Keep in Mind

- Does this maintain the local + API provider flexibility?
- Is this reproducible — config, seed, git SHA all recorded?
- Does this generalize across projects, or belong in one?
- What would falsify the claim I'm about to write down?

**For project-specific guidance**: `<area>/<project>/CLAUDE.md`
**For research areas**: `<area>/README.md` · **For the plan**: `ROADMAP.md`
