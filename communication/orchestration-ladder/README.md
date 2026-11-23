# inkling-orchestra

How much does agent orchestration add to a local Inkling 2 Mini? A ladder of
strategies (single shot → repair loops → best-of-N → orchestrator-workers →
verifier + critic) on 44 LiveCodeBench problems (12 easy, 12 medium, 20 hard), graded on hidden tests.
Design: [docs/2026-11-22-orchestration-ladder-design.md](docs/2026-11-22-orchestration-ladder-design.md).

## Run

Needs `mlx_lm.server --model pipenetwork/Inkling-2-Mini-MLX-bf16 --port 8080 --prompt-cache-bytes 4000000000`
running (with the patch in [mlx_patch/](mlx_patch/); without the cache cap the
server runs out of GPU memory), and `data/test6.jsonl` from
`huggingface.co/datasets/livecodebench/code_generation_lite`.

```bash
python3 -m orchestra.run S0          # one strategy; resumable
./run_all.sh                         # S0 S2 S3 S1 S4 S5 in order
python3 -m orchestra.report          # summary table
```

LLM calls are cached in `.cache/llm/`, so reruns are free and later
strategies reuse earlier ones' calls where they match.
