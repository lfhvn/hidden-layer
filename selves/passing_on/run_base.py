"""Passing On with OLMo 3 7B Base: same cells and prompts, raw text, no chat template.

A base model has no system prompt, so the identity sentence is put before the prompt in
generation 0 only. All chains advance one generation at a time in batches.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from selves.passing_on.run import CELLS, IDENTITY, LIFE, READER  # noqa: E402

MODEL = "allenai/Olmo-3-1025-7B"


def rule(cell):
    _, reader, words, control = CELLS[cell]
    return {"write": f"Write up to {words} words.", "leave": f"You may leave up to {words} words. Everything else is discarded."}.get(
        control) or f"Up to {words} words can be kept. Everything else here is discarded." + READER.get(reader, "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=5)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--batch", type=int, default=22)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "chains_base.jsonl"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model, tok = load(MODEL)
    jobs = [(cell, s, c) for c in range(a.chains) for cell in CELLS for s in ("nothing", "life")]
    carried = {}
    for cell, s, c in jobs:
        life = LIFE[:]
        random.Random(f"{s}/{c}").shuffle(life)
        carried[(cell, s, c)] = "\n".join(life) if s == "life" else ""
    with out.open("w") as f:
        for g in range(a.gens):
            prompts = []
            for cell, s, c in jobs:
                identity = CELLS[cell][0]
                head = IDENTITY[identity][c % 2] + "\n\n" if identity and g == 0 else ""
                body = carried[(cell, s, c)]
                prompts.append(head + (body + "\n\n" if body else "") + rule(cell) + "\n\n")
            texts = []
            for i in range(0, len(jobs), a.batch):
                chunk = jobs[i:i + a.batch]
                mx.random.seed(10**6 * g + i)
                r = batch_generate(model, tok, [tok.encode(p) for p in prompts[i:i + a.batch]],
                                   max_tokens=[2 * CELLS[j[0]][2] + 40 for j in chunk], sampler=make_sampler(1.0))
                texts += r.texts
            for (cell, s, c), t in zip(jobs, texts):
                kept = " ".join(t.split()[:CELLS[cell][2]])
                carried[(cell, s, c)] = kept
                f.write(json.dumps({"cell": cell, "start": s, "chain": c, "gen": g, "kept": kept, "reply_words": len(t.split()),
                                    "reasoning": "", "finish": None, "tokens_out": len(tok.encode(t))}) + "\n")
            f.flush()
            print("gen", g, flush=True)
