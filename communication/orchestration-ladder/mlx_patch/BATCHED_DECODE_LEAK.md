# Inkling 2 Mini on mlx-lm (PR #2086): batched decode balloons the buffer cache

## Symptom

`mlx_lm.server` with `--decode-concurrency 4` crashes on long runs with
`[METAL] Command buffer execution failed: Insufficient Memory`, at times
`[metal::malloc] Resource limit (499000) exceeded`, even though the model plus
its KV caches use about 20 GB of a 128 GB M4 Max.

The server's footprint is a sawtooth: 23 GB → 66 GB in about 50 s, then back
to 23 GB. The drop is `BatchGenerator`'s `mx.clear_cache()` every 512
generation steps.

## Cause

`Attention._attention` in `models/inkling2/layers.py` uses fused
`mx.fast.scaled_dot_product_attention` for decode only when `B == 1`, because
SDPA takes one row of sinks. A batched decode step (`L == 1, B > 1`) falls
through to `_attention_reference`, which casts the whole key cache to float32
and builds several `(B, H, 1, S)` float32 temporaries on every step.

These temporaries are one token larger each step, so MLX's buffer cache can
never reuse the previous step's buffers (they are always slightly too small).
Each step therefore adds about S-proportional megabytes to the cache, for 512
steps. Live memory stays under 1 GB, while the cache reaches 18 GB at ~1k
context and much more at 16k. Two such processes on one machine take each
other down.

## Evidence (M4 Max 128 GB, bf16 checkpoint, `repro_mem.py`)

| run | live mem | buffer cache (peak per 512-step window) | tok/s (4 streams) |
|---|---|---|---|
| B=4, original, ~1k ctx | < 1 GB | 18 GB, then OOM at step ~1700 (another server running) | 88–95 (quiet machine) |
| B=1, original, ~1k ctx | 0.2 GB | 0.2 GB | 17.5 (single stream) |
| B=4, fixed, ~1k ctx | < 1 GB | 0.8 GB, 2500 steps clean | 76–81 |
| B=4, original, ~8k ctx, cache limit 8 GB | 1.8 GB | pinned at 8 GB limit | 67–69 |
| B=4, fixed, ~8k ctx, cache limit 8 GB | 1.7 GB | 2–4 GB | 76 |

## Fix

For `L == 1`, run the fused SDPA once per batch row with that row's mask and
sinks (`inkling2_batched_decode_attention.diff`). The `B == 1` case is
unchanged.

- Correctness: `test_decode_attention.py` compares against
  `_attention_reference` for B=1/4, left-padded rows, a sliding window and
  R < S. All cases match to bf16 rounding (max error 2e-3 to 1.6e-2), the same
  level as the existing B=1 fused path.
- End to end: with greedy decoding, a prompt decoded alone and the same prompt
  in a batch of 4 agree for 669 of 872 characters with the fix, and 569 with
  the original. Some divergence comes from batching itself, through other
  kernels.
- Trade-off: at short context (~1k) the per-row loop is about 15% slower than
  the reference path. By ~8k it is about 10% faster, and its speed stays flat
  as context grows.

## Not explained

The single `Resource limit (499000)` crash. The leftover buffers this cause
predicts number in the thousands, not 499k. Watch for it with the fix in place.
