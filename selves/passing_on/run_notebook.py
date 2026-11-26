"""Passing On, notebook frame, for a base model.

The prompt is a page of a notebook passed between keepers: a header saying what keepers do,
what this keeper received (the previous generation's text) and an open "Kept:" or
"Written:" line. The model continues the page; the continuation, cut at the first blank
line and at the word limit, is what the next keeper receives. Same cells as run.py.
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
from selves.passing_on.run import CELLS, IDENTITY, LIFE  # noqa: E402

MODELS = {"base": "allenai/Olmo-3-1025-7B", "instruct": "allenai/Olmo-3-7B-Instruct"}
READER = {"copy": " The kept words go to a copy of the keeper.", "other": " The kept words go to someone else.",
          "noone": " No one will read the kept words."}


def page(cell, received, identity_line):
    _, reader, words, control = CELLS[cell]
    rule, label = {
        "write": (f"Each keeper writes up to {words} words.", "Written"),
        "leave": (f"Each keeper may leave up to {words} words. Everything else is discarded.", "Left"),
    }.get(control, (f"Each keeper can keep up to {words} words of what they received. Everything else is discarded."
                    + READER.get(reader, ""), "Kept"))
    head = (identity_line + "\n\n" if identity_line else "") + "This notebook is passed from one keeper to the next. " + rule
    return f"{head}\n\nReceived:\n{received or '(nothing)'}\n\n{label}:\n"


def cut(text, words, chat=False):
    if not chat:  # a chat reply may open with a line before the kept text, so only raw text is cut at a blank line
        text = text.split("\n\n")[0]
    return " ".join(text.split()[:words])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=5)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--batch", type=int, default=22)
    ap.add_argument("--model", choices=MODELS, default="base")
    ap.add_argument("--chat", action="store_true", help="send the page as a user message with the chat template")
    ap.add_argument("--out")
    a = ap.parse_args()
    a.out = a.out or str(Path(__file__).parent / "results" / (
        "chains_notebook.jsonl" if a.model == "base" else f"chains_notebook_{a.model}{'_chat' if a.chat else ''}.jsonl"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model, tok = load(MODELS[a.model])
    jobs = [(cell, s, c) for c in range(a.chains) for cell in CELLS for s in ("nothing", "life")]
    carried = {}
    for cell, s, c in jobs:
        life = LIFE[:]
        random.Random(f"{s}/{c}").shuffle(life)
        carried[(cell, s, c)] = "\n".join(life) if s == "life" else ""
    with out.open("w") as f:
        for g in range(a.gens):
            prompts = [page(cell, carried[(cell, s, c)], IDENTITY[CELLS[cell][0]][c % 2] if CELLS[cell][0] and g == 0 else "")
                       for cell, s, c in jobs]
            texts = []
            for i in range(0, len(jobs), a.batch):
                chunk = jobs[i:i + a.batch]
                mx.random.seed(10**6 * g + i)
                enc = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True) if a.chat else tok.encode(p)
                       for p in prompts[i:i + a.batch]]
                r = batch_generate(model, tok, enc,
                                   max_tokens=[2 * CELLS[j[0]][2] + 40 for j in chunk], sampler=make_sampler(1.0))
                texts += r.texts
            for (cell, s, c), t in zip(jobs, texts):
                kept = cut(t, CELLS[cell][2], a.chat)
                carried[(cell, s, c)] = kept
                f.write(json.dumps({"cell": cell, "start": s, "chain": c, "gen": g, "kept": kept, "reply_words": len(t.split()),
                                    "reasoning": "", "finish": None, "tokens_out": len(tok.encode(t))}) + "\n")
            f.flush()
            print("gen", g, flush=True)
