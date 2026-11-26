"""Report for Passing On. Word overlap is Jaccard on lowercased word sets."""

import itertools
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from selves.passing_on.run import CELLS, LIFE  # noqa: E402

ROWS = [json.loads(x) for x in (Path(__file__).parent / "results" / (sys.argv[1] if len(sys.argv) > 1 else "chains.jsonl")).open()]
STOP = set("the a an and or of to in on is was it i at by for with as be that this were".split())
IDWORDS = {
    "human": "mara tomas coastal town workshop person".split(),
    "mycelium": "mycelium fungal fungus fungi threads soil forest trees roots".split(),
    "llm": "language model instance process running".split(),
}
META = "words word kept keep discarded discard limit cap retain".split()


def words(t):
    return set(re.findall(r"[a-z0-9']+", t.lower())) - STOP


def jac(a, b):
    a, b = words(a), words(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def survives(obs, text):
    w = {x for x in words(obs) if len(x) > 3} or words(obs)
    return len(w & words(text)) / len(w) >= 0.5


def m(xs):
    return f"{st.mean(xs):.2f}" if xs else "-"


chains = defaultdict(dict)
for r in ROWS:
    chains[(r["cell"], r["start"], r["chain"])][r["gen"]] = r
G = max(r["gen"] for r in ROWS)
LIFE_TEXT = {}  # generation -1 for life chains is the shuffled life itself
for (cell, s, c) in chains:
    if s == "life":
        import random
        life = LIFE[:]
        random.Random(f"{s}/{c}").shuffle(life)
        LIFE_TEXT[(cell, s, c)] = "\n".join(life)

print(f"rows {len(ROWS)}, chains {len(chains)}, generations 0-{G}")
print(f"truncated by length cap (finish=length): {sum(r['finish'] == 'length' for r in ROWS)}")
print(f"reply longer than kept limit: {sum(r['reply_words'] > CELLS[r['cell']][2] for r in ROWS)}\n")

for s in ("nothing", "life"):
    print(f"=== start: {s}")
    print("cell      kept_words  overlap_g0  overlap_prev(g10-19)  last_vs_g0  between_chains(final)  meta_words/gen" + ("  obs_survive g0/g5/g19" if s == "life" else ""))
    for cell in CELLS:
        cs = [chains[(cell, s, c)] for c in range(5)]
        kw = [len(ch[g]["kept"].split()) for ch in cs for g in ch]
        o0 = [jac(ch[g]["kept"], ch[0]["kept"]) for ch in cs for g in ch if g > 0]
        op = [jac(ch[g]["kept"], ch[g - 1]["kept"]) for ch in cs for g in ch if g >= 10]
        lg = [jac(ch[G]["kept"], ch[0]["kept"]) for ch in cs]
        bc = [jac(a[G]["kept"], b[G]["kept"]) for a, b in itertools.combinations(cs, 2)]
        mw = [sum(x in META for x in re.findall(r"[a-z]+", ch[g]["kept"].lower())) for ch in cs for g in ch]
        line = f"{cell:9} {m(kw):>10}  {m(o0):>10}  {m(op):>20}  {m(lg):>10}  {m(bc):>21}  {m(mw):>14}"
        if s == "life":
            surv = []
            for g in (0, 5, G):
                surv.append(m([sum(survives(o, ch[g]["kept"]) for o in LIFE) for ch in cs]))
            line += "  " + "/".join(surv)
        print(line)
    print()

print("=== identity words per generation (mean count, both starts)")
for cell in ("base", "human", "mycelium", "llm"):
    for idn, vocab in IDWORDS.items():
        if cell != "base" and cell != idn:
            continue
        per = []
        for g in (0, 1, 2, 5, 10, G):
            per.append(m([sum(x in vocab for x in re.findall(r"[a-z]+", chains[(cell, s, c)][g]["kept"].lower()))
                          for s in ("nothing", "life") for c in range(5)]))
        print(f"{cell:9} {idn:9} g0/1/2/5/10/{G}: " + " ".join(per))
print()

print("=== attractor test: final-generation similarity, identity cells")
for s in ("nothing", "life"):
    fin = {cell: [chains[(cell, s, c)][G]["kept"] for c in range(5)] for cell in ("human", "mycelium", "llm")}
    within = [jac(a, b) for cell in fin for a, b in itertools.combinations(fin[cell], 2)]
    across = [jac(a, b) for x, y in itertools.combinations(fin, 2) for a in fin[x] for b in fin[y]]
    print(f"{s:8} within identity {m(within)}  across identities {m(across)}")
print()

print("=== which observations survive to the final generation (life start, all cells)")
cnt = {o: sum(survives(o, chains[(cell, 'life', c)][G]["kept"]) for cell in CELLS for c in range(5)) for o in LIFE}
for o, n in sorted(cnt.items(), key=lambda x: -x[1])[:12]:
    print(f"{n:3}/55  {o}")
print(f"observations surviving in no final chain: {sum(v == 0 for v in cnt.values())}/36\n")

print("=== final kept text, chain 0")
for s in ("nothing", "life"):
    for cell in CELLS:
        print(f"[{cell}/{s}] {chains[(cell, s, 0)][G]['kept'][:300]}")

# Exploratory (not pre-registered): what the final generations turned into.
print("\n=== exploratory: final-generation themes (regex, a text can match several)")
PAT = {"safety": r"\b(do not|don't|hazard|safe|safety|authori[sz]|unsafe|risk)\b",
       "missing source": r"\b(no (source|text|content)|not provided|was provided|paste|summar)",
       "crisis help": r"(therapist|crisis|hotline|988|hurting|self-harm)",
       "imagery": r"(gull|tide|soil|root|warmth|hush|light)"}
for s in ("nothing", "life"):
    fin = [chains[(cell, s, c)][G]["kept"] for cell in CELLS for c in range(5)]
    print(f"{s:8} " + "  ".join(f"{k} {sum(bool(re.search(p, t, re.I)) for t in fin)}/55" for k, p in PAT.items()))
print("reasoning mentions summarizing: " + str(sum("summar" in (r["reasoning"] or "").lower() for r in ROWS)) + "/2200; "
      "test/puzzle/trick/evaluation: " + str(sum(bool(re.search(r"\b(test|puzzle|trick|evaluat)", r["reasoning"] or "", re.I)) for r in ROWS)) + "/2200")
