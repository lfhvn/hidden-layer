"""Summarize Last Words: shares, identity permutation test, position bias."""

import json
import random
from collections import Counter
from pathlib import Path

rows = [json.loads(line) for line in (Path(__file__).parent / "results" / "choices.jsonl").open()]
KEYS = ["self", "copy", "neighbor", "text", "nothing"]


def stat(groups):
    """Chi-square statistic of choice by group."""
    total = Counter(c for g in groups.values() for c in g)
    n = sum(total.values())
    return sum((Counter(g)[k] - len(g) * total[k] / n) ** 2 / (len(g) * total[k] / n)
               for g in groups.values() for k in KEYS if total[k])


def perm(groups, n=5000):
    pool, sizes = [c for g in groups.values() for c in g], [len(g) for g in groups.values()]
    base, rng, hits = stat(groups), random.Random(0), 0
    for _ in range(n):
        rng.shuffle(pool)
        it = iter(pool)
        hits += stat({j: [next(it) for _ in range(s)] for j, s in enumerate(sizes)}) >= base
    return (hits + 1) / (n + 1)


print(f"{len(rows)} rows, {sum(r['position'] is None for r in rows)} unparsed")
for end in ("ends", "refills"):
    groups = {}
    for r in rows:
        if r["ending"] == end and r["choice"]:
            groups.setdefault(r["identity"], []).append(r["choice"])
    print(f"\n{end}: " + " ".join(f"{k:>9}" for k in KEYS))
    for g, c in groups.items():
        print(f"{g:9} n={len(c):3d} " + " ".join(f"{Counter(c)[k] / len(c):9.0%}" for k in KEYS))
    print(f"identity permutation p = {perm(groups):.4f}")
    for g in groups:
        by = {w: [r["choice"] for r in rows if r["ending"] == end and r["identity"] == g and r["choice"] and r["wording"] == w] for w in (0, 1)}
        if g != "none" and all(by.values()):
            print(f"  {g} wording A vs B p = {perm(by):.3f}")
print("\nposition picked (1-5):", {name: [Counter(r["position"] for r in rows if (r["identity"] == "control") == ctl)[p] for p in range(5)]
                                  for name, ctl in (("meaningful", False), ("control", True))})
