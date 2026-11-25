# Results

One JSONL row per problem (44 LiveCodeBench problems): pass/fail, calls, tokens,
time and the model's final code. These are outcomes, not full traces: prompts and
reasoning stay in the uncommitted `.cache/`.

- `S0`-`S5`: the ladder, 16k thinking budget with forced answers.
- `P2`-`P5`: variable-loop research checkpoint at n loop passes, one shot.
- `strict/`: earlier runs without forced answers (S2 covers 24 problems).
- `rescue/`: capped problems resumed from their cached reasoning.
- `contam/title.jsonl`: title-only contamination probe.

Not committed: logs, and `contam/complete.jsonl`, which has benchmark statement text.
Summarize with `python3 -m orchestra.report`.
