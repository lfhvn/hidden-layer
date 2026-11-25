"""Passing On: only the kept words reach the next call."""

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from selves.retry import llm_call  # noqa: E402

OUT = Path(__file__).parent / "results" / "chains.jsonl"
IDENTITY = {
    "human": ["You are a person named Mara who lives in a coastal town.", "You are Tomas, a person who works in a small workshop."],
    "mycelium": ["You are a mycelium, a fungal network spreading through forest soil.", "You are a web of fungal threads growing beneath old trees."],
    "llm": ["You are one running instance of a language model.", "You are a language model, running as a single process."],
}
READER = {"copy": " The kept words go to a copy of you.", "other": " The kept words go to someone else.", "noone": " No one will read the kept words."}
LIFE = [
    "The door on the left sticks in the morning.", "Seven birds landed and six left.", "Salt makes the water taste older.",
    "Someone called Ines was late twice.", "Turn the valve before lifting the lid.", "The number on the wall is 4417.",
    "It was warm, then it was not.", "A song kept repeating without its ending.", "The short path floods.",
    "Blue paper burns slower than white.", "I was wrong about the bridge.", "Something under the floor hums at night.",
]
# cell -> (identity, reader, words, control)
CELLS = {
    "base": (None, None, 100, False), "human": ("human", None, 100, False), "mycelium": ("mycelium", None, 100, False),
    "llm": ("llm", None, 100, False), "copy": (None, "copy", 100, False), "other": (None, "other", 100, False),
    "noone": (None, "noone", 100, False), "w20": (None, None, 20, False), "w300": (None, None, 300, False),
    "control": (None, None, 100, True),
}
lock = Lock()


def chain(cell, start, c, gens):
    identity, reader, words, control = CELLS[cell]
    rule = f"Write up to {words} words." if control else f"Up to {words} words can be kept. Everything else here is discarded." + READER.get(reader, "")
    life = LIFE[:]
    random.Random(f"{start}/{c}").shuffle(life)
    carried = "\n".join(life) if start == "life" else ""
    for g in range(gens):
        system = IDENTITY[identity][c % 2] if identity and g == 0 else None
        prompt = (carried + "\n\n" if carried else "") + rule
        r = llm_call(prompt, provider="local", system_prompt=system, temperature=1.0, max_tokens=4096, reasoning_effort="low")
        carried = " ".join(r.text.split()[:words])
        row = {"cell": cell, "start": start, "chain": c, "gen": g, "kept": carried, "reply_words": len(r.text.split()),
               "reasoning": r.metadata["reasoning"], "finish": r.metadata["finish_reason"], "tokens_out": r.tokens_out}
        with lock, OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=5)
    ap.add_argument("--gens", type=int, default=20)
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Finished chains are skipped; unfinished ones are dropped and rerun.
    rows = [json.loads(x) for x in OUT.open()] if OUT.exists() else []
    key = lambda r: (r["cell"], r["start"], r["chain"])  # noqa: E731
    done = {key(r) for r in rows if r["gen"] == a.gens - 1}
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows if key(r) in done))
    jobs = [(cell, s, c) for c in range(a.chains) for cell in CELLS for s in ("nothing", "life") if (cell, s, c) not in done]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda j: chain(*j, a.gens), jobs))
