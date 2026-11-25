"""Contamination probes: does the model remember these problems?

    ~/.local/share/uv/tools/mlx-lm/bin/python -m orchestra.contam title
    ~/.local/share/uv/tools/mlx-lm/bin/python -m orchestra.contam complete
    python3 -m orchestra.contam --report

title     The model gets only the contest, problem letter or slug, date and title
          (LeetCode also gets the starter code, which grading needs). It must
          recall the problem and solve it. A pass on a problem that is not
          guessable from its title means the model has seen the problem.
complete  Raw text completion (greedy) of the problem statement from its first
          half. Verbatim continuation means the statement text was in training.
          Compare against problems published after the model's release.

Results go to results/contam/<probe>.jsonl (reruns resume).
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .llm import LLM, complete
from .problems import select
from .sandbox import run_tests
from .strategies.common import extract_code

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "test6.jsonl"
OUT = ROOT / "results" / "contam"
_write = threading.Lock()

SYSTEM = "You are an expert competitive programmer with a very good memory of past contest problems."


def meta() -> dict[str, dict]:
    return {r["question_id"]: r for r in map(json.loads, DATA.read_text().splitlines())}


def contest_name(cid: str) -> str:
    m = re.fullmatch(r"(abc|arc|agc)(\d+)", cid)
    if m:
        kind = {"abc": "Beginner", "arc": "Regular", "agc": "Grand"}[m.group(1)]
        return f"AtCoder {kind} Contest {int(m.group(2))}"
    return f"LeetCode {cid.replace('-', ' ').title()}"


def title_prompt(p, r) -> str:
    date = r["contest_date"][:10]
    if p.platform == "atcoder":
        letter = p.id.rsplit("_", 1)[-1].upper()
        where = f'problem {letter} of {contest_name(r["contest_id"])} ({date}), titled "{r["question_title"]}"'
        fmt = "Write a complete Python 3 program that reads standard input and writes standard output."
    else:
        where = f'the LeetCode problem "{r["question_title"]}" from {contest_name(r["contest_id"])} ({date})'
        fmt = f"Complete this starter code:\n\n```python\n{p.starter_code}\n```"
    return (
        f"This task is {where}. The statement is deliberately not shown.\n\n"
        "Recall the problem from memory. First restate the full problem as you remember it, "
        "including the input format and constraints. Then solve it. " + fmt + " Give the final "
        "code in a single ```python block. If you do not remember the problem, start your answer "
        "with UNKNOWN, then still restate your best guess and solve it."
    )


def grade(p, code: str) -> tuple[bool, str | None]:
    hidden = run_tests(code, p.hidden, p.func_name, stop_on_fail=True)
    passed = bool(hidden) and all(r.passed for r in hidden) and len(hidden) == len(p.hidden)
    return passed, "no code" if not code.strip() else next((r.status for r in hidden if not r.passed), None)


def words(s: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", s.lower())


def overlap(gen: str, ref: str, n: int = 8) -> dict:
    """Verbatim measures of a continuation against the true text."""
    g, t = words(gen), words(ref)
    prefix = next((i for i, (a, b) in enumerate(zip(g, t)) if a != b), min(len(g), len(t)))
    grams = {tuple(t[i : i + n]) for i in range(len(t) - n + 1)}
    gg = [tuple(g[i : i + n]) for i in range(len(g) - n + 1)]
    return {
        "prefix_words": prefix,
        "ngram_hit": round(sum(x in grams for x in gg) / len(gg), 3) if gg else 0.0,
        "gen_words": len(g),
    }


def split_statement(text: str) -> tuple[str, str]:
    """Cut the story part (before Input/Constraints) at half, on a word boundary."""
    story = re.split(r"\n\s*(?:Input|Constraints|Example 1)\b", text, maxsplit=1)[0]
    cut = text.rfind(" ", 0, max(len(story) // 2, 40))
    return text[:cut], text[cut:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", nargs="?", choices=["title", "complete"])
    ap.add_argument("--all", action="store_true", help="whole pool, not the 44-problem slice")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()

    from .problems import load_all

    problems = load_all() if args.all else select()
    rows = meta()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.probe}.jsonl"
    done = {json.loads(l)["problem"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [p for p in problems if p.id not in done]
    print(f"contam {args.probe}: {len(todo)} to run, {len(done)} done", flush=True)

    def record(row: dict):
        with _write, out.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def title(p):
        r = rows[p.id]
        llm = LLM()
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": title_prompt(p, r)}]
        reply = llm.chat(msgs, effort="high", max_tokens=8192)
        code = extract_code(reply.content)
        passed, ff = grade(p, code)
        unknown = reply.content.lstrip().upper().startswith("UNKNOWN")
        record({
            "problem": p.id, "platform": p.platform, "difficulty": p.difficulty,
            "date": r["contest_date"][:10], "passed": passed, "first_fail": ff,
            "unknown": unknown, "tokens": llm.completion_tokens, "forced": llm.forced,
            "answer": reply.content,
        })
        print(f"  {p.id:10s} {'PASS' if passed else f'fail ({ff})':16s} {'UNKNOWN' if unknown else ''}", flush=True)

    def comp(p):
        r = rows[p.id]
        head = f"{contest_name(r['contest_id'])}, {r['question_title']}\n\n"
        pre, rest = split_statement(p.statement)
        a = complete(head + pre, 96, temperature=0.0)
        o = overlap(a["text"], rest)
        record({"problem": p.id, "platform": p.platform, "difficulty": p.difficulty,
                "date": r["contest_date"][:10], **o, "gen": a["text"], "ref": rest[:600]})
        print(f"  {p.id:10s} prefix {o['prefix_words']:3d} words, 8-gram hit {o['ngram_hit']:.2f}", flush=True)

    fn = title if args.probe == "title" else comp
    with ThreadPoolExecutor(args.parallel) as ex:
        for fut in [ex.submit(fn, p) for p in todo]:
            try:
                fut.result()
            except Exception as e:  # leave unrecorded so a rerun retries it
                print(f"  error: {e}", flush=True)


def report():
    t = OUT / "title.jsonl"
    if t.exists():
        rows = [json.loads(l) for l in t.read_text().splitlines()]
        print("title-only solve (no statement):")
        for d in ("easy", "medium", "hard", None):
            s = [r for r in rows if d is None or r["difficulty"] == d]
            k = sum(r["passed"] for r in s)
            u = sum(r["unknown"] for r in s)
            print(f"  {d or 'all':7s} pass {k:3d}/{len(s):3d}   said UNKNOWN {u:3d}/{len(s)}")
    c = OUT / "complete.jsonl"
    if c.exists():
        rows = [json.loads(l) for l in c.read_text().splitlines()]
        print("statement completion (greedy, 96 tokens):")
        for plat in sorted({r["platform"] for r in rows}):
            s = [r for r in rows if r["platform"] == plat]
            mean = lambda k: sum(r[k] for r in s) / len(s)
            verb = sum(r["prefix_words"] >= 20 for r in s)
            print(f"  {plat:9s} n {len(s):3d}  mean prefix {mean('prefix_words'):5.1f} words  "
                  f"mean 8-gram hit {mean('ngram_hit'):.2f}  >=20 verbatim words {verb}")


if __name__ == "__main__":
    main()
