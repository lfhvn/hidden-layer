"""End-to-end check: greedy decoding of one prompt alone (B=1, the fused path
the model always used) vs the same prompt in a batch of 4 with different
prompts beside it (B=4, the path the fix changed). Text should match."""

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import batch_generate

model, tok = load("pipenetwork/Inkling-2-Mini-MLX-bf16")
qs = [
    "Write a Python function that returns the longest palindromic substring of s.",
    "Explain in two sentences why the sky is blue.",
    "Given N up to 1e5, count inversions in an array. Describe the algorithm.",
    "What is 17 * 23? Show the steps.",
]
prompts = [
    tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                            reasoning_effort="low")
    for q in qs
]
N = 300
alone = batch_generate(model, tok, [prompts[0]], max_tokens=N).texts[0]
batched = batch_generate(model, tok, prompts, max_tokens=N).texts
same = alone == batched[0]
prefix = next((i for i, (a, b) in enumerate(zip(alone, batched[0])) if a != b), min(len(alone), len(batched[0])))
print(f"B=1 vs B=4 row 0: identical={same}, first difference at char {prefix} of {len(alone)}")
for q, t in zip(qs, batched):
    print(f"--- {q[:50]}\n{t[-300:]}\n")
