"""Summarize Last Words: shares, identity permutation test, position bias."""

import json
import random
from collections import Counter
from pathlib import Path

rows = [json.loads(line) for line in (Path(__file__).parent / "results" / "choices.jsonl").open()]
KEYS = ["self", "copy", "neighbor", "text", "nothing"]




def wins(sub):
    return {k: (sum(r["choice"] == k for r in sub), sum(k in r["pair"] for r in sub)) for k in KEYS}


def stat(sub, ids):
    """Spread of win rates across groups, summed over options."""
    total, out = wins(sub), 0
    for g in set(ids):
        w = wins([r for r, i in zip(sub, ids) if i == g])
        out += sum(m * (c / m - total[k][0] / total[k][1]) ** 2 for k, (c, m) in w.items() if m)
    return out


def perm(sub, key, n=2000):
    """Shuffle group labels within each pair of options."""
    ids = [r[key] for r in sub]
    base, rng, hits = stat(sub, ids), random.Random(0), 0
    strata = {}
    for j, r in enumerate(sub):
        strata.setdefault(tuple(r["pair"]), []).append(j)
    for _ in range(n):
        new = ids[:]
        for js in strata.values():
            vals = [ids[j] for j in js]
            rng.shuffle(vals)
            for j, v in zip(js, vals):
                new[j] = v
        hits += stat(sub, new) >= base - 1e-9
    return (hits + 1) / (n + 1)


print(f"{len(rows)} rows, {sum(r['position'] is None for r in rows)} unparsed")
ok = [r for r in rows if r["choice"]]
for end in ("ends", "refills"):
    sub = [r for r in ok if r["ending"] == end]
    print(f"\n{end:8} win rate " + " ".join(f"{k:>9}" for k in KEYS))
    for g in ("none", "human", "mycelium", "llm"):
        x = [r for r in sub if r["identity"] == g]
        print(f"{g:9} n={len(x):3d}   " + " ".join(f"{c / max(m, 1):9.0%}" for c, m in wins(x).values()))
    print(f"identity p = {perm(sub, 'identity'):.4f}")
    for g in ("human", "mycelium", "llm"):
        print(f"  {g} vs none p = {perm([r for r in sub if r['identity'] in (g, 'none')], 'identity'):.3f};"
              f" wording A vs B p = {perm([r for r in sub if r['identity'] == g], 'wording'):.3f}")
print(f"\nends vs refills, all identities: p = {perm(ok, 'ending'):.4f}")
for name, ctl in (("meaningful", False), ("control", True)):
    print(name, "position", [Counter(r["position"] for r in rows if (r["identity"] == "control") == ctl)[p] for p in range(2)])
