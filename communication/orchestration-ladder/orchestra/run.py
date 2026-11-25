"""Run one strategy over the problem slice and grade on hidden tests.

    python -m orchestra.run S0 [--limit N] [--parallel 4]

Results append to results/<strategy>.jsonl; finished problems are skipped,
so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import llm as llm_mod
from .llm import LLM
from .problems import Problem, select
from .sandbox import run_tests
from .strategies import STRATEGIES

RESULTS = Path(__file__).resolve().parent.parent / "results"
_write = threading.Lock()


def solve_and_grade(name: str, p: Problem) -> dict:
    llm = LLM()
    start = time.monotonic()
    error = None
    try:
        code = STRATEGIES[name](p, llm)
    except Exception:
        code, error = "", traceback.format_exc(limit=3)
    wall = time.monotonic() - start
    hidden = run_tests(code, p.hidden, p.func_name, stop_on_fail=True)
    return {
        "problem": p.id,
        "platform": p.platform,
        "difficulty": p.difficulty,
        "passed": bool(hidden) and all(r.passed for r in hidden) and len(hidden) == len(p.hidden),
        "first_fail": "no code" if not code.strip() else next((r.status for r in hidden if not r.passed), None),
        "calls": llm.calls,
        "forced": llm.forced,
        "completion_tokens": llm.completion_tokens,
        "prompt_tokens": llm.prompt_tokens,
        "model_seconds": round(llm.model_seconds, 1),
        "wall_seconds": round(wall, 1),
        "error": error,
        "code": code,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy", choices=sorted(STRATEGIES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--parallel", type=int, default=4, help="problems in flight")
    ap.add_argument("--model", help="model name/path to request (default: llm.MODEL)")
    ap.add_argument("--url", help="server base URL (default: llm.BASE_URL)")
    ap.add_argument("--tag", help="results file name (default: the strategy name)")
    args = ap.parse_args()
    if args.model:
        llm_mod.MODEL = args.model
    if args.url:
        llm_mod.BASE_URL = args.url

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.tag or args.strategy}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["problem"] for l in out.read_text().splitlines() if l.strip()}
    problems = [p for p in select() if p.id not in done][: args.limit]
    print(f"{args.tag or args.strategy} [{llm_mod.MODEL}]: {len(problems)} to run, {len(done)} already done", flush=True)

    def one(p: Problem):
        row = solve_and_grade(args.strategy, p)
        if row["error"]:
            # Infrastructure failure (e.g. server down), not a wrong answer:
            # leave it unrecorded so the next run retries it.
            print(f"  {p.id:10s} SKIPPED: {row['error'].strip().splitlines()[-1]}", flush=True)
            return
        with _write, out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        mark = "PASS" if row["passed"] else f"fail ({row['first_fail'] or row['error'] and 'error'})"
        print(
            f"  {p.id:10s} {mark:16s} {row['calls']:2d} calls "
            f"{row['completion_tokens']:6d} tok {row['wall_seconds']:7.1f}s"
            + (f" ({row['forced']} forced)" if row["forced"] else ""),
            flush=True,
        )

    t0 = time.monotonic()
    with ThreadPoolExecutor(args.parallel) as ex:
        list(ex.map(one, problems))
    print(f"{args.strategy}: done in {(time.monotonic() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
