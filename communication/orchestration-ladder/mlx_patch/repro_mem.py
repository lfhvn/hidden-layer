"""Repro for the mlx_lm.server memory blowup with Inkling 2 Mini.

Drives the same BatchGenerator the server uses, with no clear_cache, and logs
MLX active / cache / peak memory every N decode steps.

    python repro_mem.py --batch 4 --steps 3000
    python repro_mem.py --batch 1 --steps 3000
    python repro_mem.py --batch 4 --steps 3000 --cache-limit-gb 2
"""

import argparse
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

GB = 1024**3
PROMPT = (
    "Solve this competitive programming problem in Python. Think it through "
    "carefully, then give the code.\n\nGiven an array of N integers, count the "
    "pairs (i, j), i < j, whose sum is divisible by K. N up to 2e5."
)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="pipenetwork/Inkling-2-Mini-MLX-bf16")
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--every", type=int, default=100)
ap.add_argument("--cache-limit-gb", type=float, default=None)
ap.add_argument("--context", type=int, default=0, help="pad the prompt to about this many tokens")
args = ap.parse_args()

model, tok = load(args.model)
if args.cache_limit_gb is not None:
    mx.set_cache_limit(int(args.cache_limit_gb * GB))
weights = mx.get_active_memory()
print(f"weights loaded: active={weights / GB:.1f}GB cache_limit={args.cache_limit_gb}")

content = PROMPT
if args.context:
    filler = "Notes from an earlier attempt: " + " ".join(f"step {i} checked the edge case and found it fine." for i in range(4000))
    ftoks = tok.encode(filler)[: args.context]
    content = tok.decode(ftoks) + "\n\n" + PROMPT
msgs = [{"role": "user", "content": content}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, reasoning_effort="max")
# No stop tokens: every sequence decodes the full --steps.
gen = BatchGenerator(model, max_tokens=args.steps, stop_tokens=None,
                     sampler=make_sampler(temp=0.7))
gen.insert([list(ids)] * args.batch, [args.steps] * args.batch)
print(f"prompt tokens: {len(ids)}")

step, t0 = 0, time.time()
mx.reset_peak_memory()
print(f"{'step':>6s} {'active':>8s} {'cache':>8s} {'peak':>8s} {'tok/s':>7s}")
while responses := gen.next_generated():
    step += 1
    if step % args.every == 0:
        dt = time.time() - t0
        print(
            f"{step:6d} {(mx.get_active_memory() - weights) / GB:7.2f}G "
            f"{mx.get_cache_memory() / GB:7.2f}G {(mx.get_peak_memory() - weights) / GB:7.2f}G "
            f"{args.batch * args.every / dt:7.1f}",
            flush=True,
        )
        mx.reset_peak_memory()
        t0 = time.time()
gen.close()
