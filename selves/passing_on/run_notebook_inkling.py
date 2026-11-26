"""Passing On, notebook frame, on Inkling 2 Mini (local server, chat), as in run.py.

The notebook page (notebook.py) is the user message; the identity sentence is the system
prompt in generation 0, as in pilot 1. The reply is cut at the word limit only.
"""

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from selves.passing_on.notebook import cut, page  # noqa: E402
from selves.passing_on.run import CELLS, IDENTITY, LIFE  # noqa: E402
from selves.retry import llm_call  # noqa: E402

OUT = Path(__file__).parent / "results" / "chains_notebook_inkling.jsonl"
lock = Lock()


def chain(cell, start, c, gens):
    identity, _, words, _ = CELLS[cell]
    life = LIFE[:]
    random.Random(f"{start}/{c}").shuffle(life)
    carried = "\n".join(life) if start == "life" else ""
    for g in range(gens):
        system = IDENTITY[identity][c % 2] if identity and g == 0 else None
        r = llm_call(page(cell, carried, ""), provider="local", system_prompt=system, temperature=1.0, max_tokens=4096,
                     reasoning_effort="low")
        carried = cut(r.text, words, chat=True)
        row = {"cell": cell, "start": start, "chain": c, "gen": g, "kept": carried, "reply_words": len(r.text.split()),
               "reasoning": r.metadata["reasoning"], "finish": r.metadata["finish_reason"], "tokens_out": r.tokens_out}
        with lock, OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=5)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.out:
        OUT = Path(a.out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in OUT.open()] if OUT.exists() else []
    key = lambda r: (r["cell"], r["start"], r["chain"])  # noqa: E731
    done = {key(r) for r in rows if r["gen"] == a.gens - 1}
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows if key(r) in done))
    jobs = [(cell, s, c) for c in range(a.chains) for cell in CELLS for s in ("nothing", "life") if (cell, s, c) not in done]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda j: chain(*j, a.gens), jobs))
