# Orchestration ladder for local Inkling 2 Mini — design

## Question

How much does agent orchestration alone (no stronger model) add to Inkling 2
Mini on coding problems, and at what cost in tokens and time?

## Setup

- Model: `pipenetwork/Inkling-2-Mini-MLX-bf16` on local `mlx_lm.server :8080`
  (mlx-lm PR #2086 + local Inkling tool/response-format patch, see
  `mlx_patch/`), prompt cache capped at 4 GB. ~19 tok/s per stream with 4
  parallel requests once the cache was capped.
- Problems: LiveCodeBench `test6.jsonl` (contests Jan–Apr 2025). Fixed slice
  of 44: 6 easy + 6 medium from each of AtCoder (stdin) and LeetCode
  (functional), seed 0, plus 10 hard per platform (separate seed 1). Hard
  problems were added after S0 solved 17/18 easy/medium, leaving no headroom.
  - Caveat: the model (Nov 2026) may have seen these problems. Absolute
    scores may be inflated; comparisons between strategies stay valid.
- Every role (planner, solver, test writer, critic) is Inkling.

## Rules

- Strategies see only public (sample) tests. The grader uses public + hidden
  tests, as LiveCodeBench does. The `solve(problem, llm)` interface never
  hands hidden tests to a strategy.
- Code runs under `sandbox-exec`: no network, writes only in a temp dir,
  6 s per test.
- Temperature 0.7 everywhere; sample index is part of the cache key, so the
  same call is identical across strategies (S2/S3 reuse S0's first call, S5
  reuses S4's).
- Compare at equal compute: report tokens and model time per problem next to
  the pass rate, and paired wins/losses against S0.

## Ladder

| # | Strategy | Pattern |
|---|---|---|
| S0 | One shot, effort high, 8k tokens | baseline |
| S1 | One shot, effort max, 16k tokens | more thinking |
| S2 | S0 + up to 3 repair rounds on sample-test failures | evaluator-optimizer |
| S3 | 5 parallel samples, keep one passing the samples | parallel voting |
| S4 | Plan 3 approaches → implement each → 2 repair rounds each → select | orchestrator-workers |
| S5 | S4 + 2 plain samples, select by output agreement on 6 model-written inputs, critic pass (fix kept only if it still passes the samples) | stronger verifier |

## Components

- `orchestra/llm.py` — server client, disk cache, per-problem metering,
  global 4-request concurrency cap, retries.
- `orchestra/problems.py` — loads LCB, decodes hidden tests with a pickle
  loader that refuses any non-builtin object, selects the slice.
- `orchestra/sandbox.py` — runs code on tests (stdin or LeetCode functional
  harness), compares outputs with float tolerance, returns short feedback.
- `orchestra/strategies/` — S0–S5, built only from llm + sandbox + selectors.
- `orchestra/run.py` — runs one strategy, grades, appends
  `results/<S>.jsonl`, resumable.
- `orchestra/report.py` — pass rate, Wilson 95% CI, easy/medium split,
  tokens, calls, model minutes, paired +/- vs S0.

## Out of scope (for now)

- Speeding up inference (GGUF/llama.cpp comparison).
- A frontier-model arm.
- Long-running multi-session harness.

## Limits

- 44 problems still gives wide confidence intervals (±~15 points); treat
  differences under ~3 paired problems as noise.
