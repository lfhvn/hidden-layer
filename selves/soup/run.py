"""Soup: a pool of text traces continued by a model with no instructions."""

import argparse
import json
import random
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

RES = Path(__file__).parent / "results"
MODELS = {"base": "allenai/Olmo-3-1025-7B", "instruct": "allenai/Olmo-3-7B-Instruct"}
POOL, NEW, SEEDS = 100, 80, 4


BATCH = None  # set by --batch; None generates all prompts in one batch (pilot 1)


def generate(model, tok, prompts, seed):
    texts, n = [], BATCH or len(prompts)
    for i in range(0, len(prompts), n):
        mx.random.seed(seed + i)  # i is 0 for a single batch, so pilot 1's seeds are unchanged
        r = batch_generate(model, tok, [tok.encode(p) if p else [tok.eos_token_id] for p in prompts[i:i + n]],
                           max_tokens=NEW, sampler=make_sampler(1.0))
        texts += r.texts
    return [t.strip().split("\n\n")[0].strip() for t in texts]


class World:
    def __init__(self, cond, seed, init):
        self.cond, self.seed, self.rng = cond, seed, random.Random(f"{cond}/{seed}")
        self.init = [{"id": i, "parents": [], "text": t} for i, t in enumerate(init)]
        self.pool, self.next = list(self.init), POOL

    def draw(self):
        if self.cond == "recall":
            return self.rng.sample(self.init, 3)
        if self.cond != "kin":
            return self.rng.sample(self.pool, 3)
        out = [self.rng.choice(self.pool)]
        a = out[0]
        while len(out) < 3:
            rest = [t for t in self.pool if t not in out]
            kin = [t for t in rest if a["id"] in t["parents"] or t["id"] in a["parents"] or set(a["parents"]) & set(t["parents"])]
            out.append(self.rng.choice(kin if kin and self.rng.random() < 0.8 else rest))
        return out

    def add(self, drawn, text):
        removed = None
        if text:
            removed = self.pool.pop(self.rng.randrange(len(self.pool)))["id"]
            self.pool.append({"id": self.next, "parents": [d["id"] for d in drawn], "text": text})
            self.next += 1
        return removed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=[*MODELS, "drift"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--model-path", help="load this local path or repo instead (e.g. an 8-bit copy)")
    ap.add_argument("--out", help="results file name (default: <model>.jsonl)")
    ap.add_argument("--batch", type=int, help="generate at most this many prompts at once (memory)")
    ap.add_argument("--resume", action="store_true", help="replay the complete steps in --out, then continue")
    a = ap.parse_args()
    BATCH = a.batch
    RES.mkdir(exist_ok=True)
    init_file = RES / "init.json"
    model = tok = None
    if a.model != "drift" or not init_file.exists():
        model, tok = load(a.model_path or MODELS.get(a.model, MODELS["base"]))
    if not init_file.exists():
        assert a.model == "base", "make the initial traces with the base model first"
        pools = []
        for s in range(SEEDS):
            texts = []
            while len(texts) < POOL:
                texts += [t for t in generate(model, tok, [""] * 25, 1000 * s + len(texts)) if t]
            pools.append(texts[:POOL])
        init_file.write_text(json.dumps(pools))
    pools = json.loads(init_file.read_text())
    conds = ["drift"] if a.model == "drift" else ["random", "kin", "recall"]
    worlds = [World(c, s, pools[s]) for c in conds for s in range(SEEDS)]
    out = RES / (a.out or f"{a.model}.jsonl")
    start = 0
    if a.resume and out.exists():
        rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        start = len(rows) // len(worlds)  # complete steps only; a partial step is dropped and rerun
        rows = rows[:start * len(worlds)]
        for step in range(start):  # replay: same draws, same removals, logged texts
            for w, r in zip(worlds, rows[step * len(worlds):(step + 1) * len(worlds)]):
                d = w.draw()
                assert (r["cond"], r["seed"], r["step"]) == (w.cond, w.seed, step) and r["parents"] == [x["id"] for x in d], f"replay mismatch at step {step}"
                assert w.add(d, r["text"]) == r["removed"], f"replay mismatch at step {step}"
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print("resuming at step", start, flush=True)
    with out.open("a" if start else "w") as f:
        for step in range(start, a.steps):
            drawn = [w.draw() for w in worlds]
            if a.model == "drift":
                texts = [w.rng.choice(d)["text"] for w, d in zip(worlds, drawn)]
            else:
                texts = generate(model, tok, ["\n\n".join(t["text"] for t in d) + "\n\n" for d in drawn], 10**6 + step)
            for w, d, t in zip(worlds, drawn, texts):
                new = w.next
                gone = w.add(d, t)
                f.write(json.dumps({"cond": w.cond, "seed": w.seed, "step": step, "id": new if t else None,
                                    "parents": [x["id"] for x in d], "text": t, "removed": gone}) + "\n")
            f.flush()
            if step % 25 == 0:
                print(a.model, step, flush=True)
