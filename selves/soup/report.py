"""Summarize Soup by replaying the logs to their final pools."""

import json
import re
import zlib
from collections import Counter
from pathlib import Path
from statistics import mean

RES = Path(__file__).parent / "results"
SNAP = (300, 600)  # pool snapshots taken before these steps, for runs that reach them
pools0 = json.loads((RES / "init.json").read_text())


def grams(t):
    w = re.findall(r"\w+", t.lower())
    return [tuple(w[i:i + 5]) for i in range(len(w) - 4)]


def measure(texts):
    sets = [set(grams(t)) for t in texts]
    c = Counter(g for s in sets for g in s)
    allg = [g for t in texts for g in grams(t)]
    top, n = c.most_common(1)[0] if c else ((), 0)
    blob = "\n".join(texts).encode()
    return {"top": n / len(texts), "distinct": len(set(allg)) / max(1, len(allg)), "zlib": len(zlib.compress(blob, 9)) / max(1, len(blob)),
            "first": mean(bool(re.search(r"\b(I|me|my)\b", t)) for t in texts), "words": mean(len(t.split()) for t in texts), "phrase": " ".join(top)}


def show(name, ms):
    print(f"{name:18}" + " ".join(f"{k}={mean(m[k] for m in ms):.3f}" for k in ("top", "distinct", "zlib", "first", "words")),
          "| top by seed", [round(m["top"], 2) for m in ms], "|", max(ms, key=lambda m: m["top"])["phrase"])


show("start", [measure(p) for p in pools0])
for f in sorted(RES.glob("*.jsonl")):
    runs = {}
    for r in map(json.loads, f.open()):
        runs.setdefault((r["cond"], r["seed"]), []).append(r)
    for cond in dict.fromkeys(c for c, _ in runs):
        ms, empty, snaps, prev = [], 0, {}, None
        for (c, s), rows in runs.items():
            if c != cond:
                continue
            pool = dict(enumerate(pools0[s]))
            for r in rows:
                if r["step"] in SNAP and r["step"] != prev:
                    snaps.setdefault(r["step"], []).append(measure(list(pool.values())))
                prev = r["step"]
                empty += not r["text"]
                if r["text"]:
                    pool[r["id"]] = r["text"]
                    del pool[r["removed"]]
            ms.append(measure(list(pool.values())))
        for st in sorted(snaps):
            show(f"  at step {st}", snaps[st])
        show(f"{f.stem}/{cond}", ms)
        print(f"{'':18}empty continuations: {empty}")
