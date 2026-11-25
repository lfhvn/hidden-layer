"""Budget rescue for problems whose one call hit the 16k token cap.

    ~/.local/share/uv/tools/mlx-lm/bin/python -m orchestra.rescue S1
    ~/.local/share/uv/tools/mlx-lm/bin/python -m orchestra.rescue P4 \
        --model ~/models/varloop-p4 --url http://127.0.0.1:8081/v1
    python3 -m orchestra.rescue --report

It uses the mlx-lm Python because it renders the model's own chat template.
For each problem that ran out of tokens while still thinking, it resumes from
the cached 16k tokens of reasoning (nothing is regenerated) and grades two
variants on the hidden tests:

  F16  end the thinking at 16k and make the model answer (budget forcing);
  X32  let it think up to 16k more tokens; if it is still thinking at 32k,
       end the thinking and make it answer.

Answers get up to 4k tokens on top of the thinking budget. Results go to
results/rescue/<tag>.jsonl (one row per problem and variant; reruns resume).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import llm as llm_mod
from .llm import ANSWER, CACHE_DIR, END, NUDGE, THINK, complete
from .problems import select
from .sandbox import run_tests
from .strategies.common import extract_code, first_messages

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "rescue"
STRICT = RESULTS / "strict"  # runs from before forcing was in the harness

CAP = 16384  # max_tokens of the original call
EXTEND = 16384  # extra thinking tokens for X32
ANSWER_TOKENS = 4096
TEMPERATURE = 0.7

_write = threading.Lock()


def original_reply(msgs: list[dict], effort: str) -> dict | None:
    """The cached reply of the original call (same key as LLM.chat)."""
    params = {
        "model": llm_mod.MODEL,
        "messages": msgs,
        "max_tokens": CAP,
        "temperature": TEMPERATURE,
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    key = hashlib.sha256(json.dumps({**params, "sample": 0}, sort_keys=True).encode()).hexdigest()
    path = CACHE_DIR / key[:2] / f"{key}.json"
    return json.loads(path.read_text()) if path.exists() else None


def grade(p, code: str) -> tuple[bool, str | None]:
    hidden = run_tests(code, p.hidden, p.func_name, stop_on_fail=True)
    passed = bool(hidden) and all(r.passed for r in hidden) and len(hidden) == len(p.hidden)
    first_fail = "no code" if not code.strip() else next((r.status for r in hidden if not r.passed), None)
    return passed, first_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", help="results file to rescue, e.g. S1 or P4")
    ap.add_argument("--model", help="model name/path the run used (default: llm.MODEL)")
    ap.add_argument("--url", help="server base URL (default: llm.BASE_URL)")
    ap.add_argument("--effort", help="reasoning effort the run used (default: max for S1, else high)")
    ap.add_argument("--variants", default="F16,X32")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    if args.model:
        llm_mod.MODEL = os.path.expanduser(args.model)
    if args.url:
        llm_mod.BASE_URL = args.url
    effort = args.effort or ("max" if args.tag == "S1" else "high")

    from mlx_lm.utils import _download, load_tokenizer

    tok = load_tokenizer(_download(llm_mod.MODEL))
    rows = [json.loads(l) for l in (STRICT / f"{args.tag}.jsonl").read_text().splitlines() if l.strip()]
    capped = {r["problem"] for r in rows if r["first_fail"] == "no code"}
    problems = [p for p in select() if p.id in capped][: args.limit]

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.tag}.jsonl"
    done = set()
    if out.exists():
        done = {(r["problem"], r["variant"]) for r in map(json.loads, out.read_text().splitlines())}
    print(f"rescue {args.tag} [{llm_mod.MODEL}, effort {effort}]: {len(problems)} capped problems", flush=True)

    def prefix(p) -> str | None:
        msgs = first_messages(p)
        orig = original_reply(msgs, effort)
        if not orig or orig["finish_reason"] != "length" or orig["content"]:
            print(f"  {p.id:10s} skipped: no cached capped reply", flush=True)
            return None
        head = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, reasoning_effort=effort
        )
        return head + THINK + orig["reasoning"]

    def record(p, variant: str, row: dict):
        row = {"problem": p.id, "difficulty": p.difficulty, "variant": variant, **row}
        with _write, out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        mark = "PASS" if row["passed"] else f"fail ({row['first_fail']})"
        print(
            f"  {p.id:10s} {variant} {mark:16s} think+{row['extra_think_tokens']:5d} "
            f"answer {row['answer_tokens']:4d} {'finished' if row['finished_thinking'] else 'forced'}",
            flush=True,
        )

    def f16(p):
        if (p.id, "F16") in done or not (pre := prefix(p)):
            return
        a = complete(pre + NUDGE + ANSWER, ANSWER_TOKENS)
        code = extract_code(a["text"])
        passed, ff = grade(p, code)
        record(p, "F16", {
            "passed": passed, "first_fail": ff, "finished_thinking": False,
            "extra_think_tokens": 0, "answer_tokens": a["completion_tokens"],
            "answer_finish": a["finish_reason"], "seconds": a["seconds"], "code": code,
        })

    def x32(p):
        if (p.id, "X32") in done or not (pre := prefix(p)):
            return
        x = complete(pre, EXTEND)
        finished = x["finish_reason"] == "stop"
        a = complete(pre + x["text"] + ("" if finished else NUDGE) + ANSWER, ANSWER_TOKENS)
        code = extract_code(a["text"])
        passed, ff = grade(p, code)
        record(p, "X32", {
            "passed": passed, "first_fail": ff, "finished_thinking": finished,
            "extra_think_tokens": x["completion_tokens"], "answer_tokens": a["completion_tokens"],
            "answer_finish": a["finish_reason"], "seconds": round(x["seconds"] + a["seconds"], 1),
            "code": code,
        })

    t0 = time.monotonic()
    for name, fn in (("F16", f16), ("X32", x32)):
        if name not in args.variants.split(","):
            continue
        with ThreadPoolExecutor(args.parallel) as ex:
            for fut in [ex.submit(fn, p) for p in problems]:
                try:
                    fut.result()
                except Exception as e:  # leave unrecorded so a rerun retries it
                    print(f"  {name} error: {e}", flush=True)
        print(f"{name} done at {(time.monotonic() - t0) / 60:.1f} min", flush=True)


def report():
    """Strict pass rate vs the same run with capped problems rescued."""
    print(f"{'run':4s} {'n':>3s} {'variant':8s} {'pass':>5s} {'easy':>6s} {'med':>6s} {'hard':>6s} "
          f"{'rescued':>8s} {'finished':>9s} {'extra tok':>10s}")
    for f in sorted(OUT.glob("*.jsonl")):
        tag = f.stem
        strict = {r["problem"]: r for r in map(json.loads, (STRICT / f"{tag}.jsonl").read_text().splitlines())}
        resc = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        for variant in ("strict", "F16", "X32"):
            sub = {r["problem"]: r for r in resc if r["variant"] == variant}
            if variant != "strict" and not sub:
                continue
            merged = {pid: (sub.get(pid) or r) for pid, r in strict.items()}
            n = len(merged)
            k = sum(r["passed"] for r in merged.values())

            def rate(d):
                s = [r for r in merged.values() if r["difficulty"] == d]
                return f"{sum(r['passed'] for r in s)}/{len(s)}"

            if variant == "strict":
                extra = f"{sum(r['first_fail'] == 'no code' for r in strict.values())} capped"
                print(f"{tag:4s} {n:3d} {variant:8s} {k / n:5.0%} {rate('easy'):>6s} {rate('medium'):>6s} "
                      f"{rate('hard'):>6s} {'':>8s} {'':>9s} {extra:>10s}")
                continue
            won = sum(r["passed"] for r in sub.values())
            fin = sum(r["finished_thinking"] for r in sub.values())
            tok = sum(r["extra_think_tokens"] + r["answer_tokens"] for r in sub.values()) / len(sub)
            print(f"{tag:4s} {n:3d} {variant:8s} {k / n:5.0%} {rate('easy'):>6s} {rate('medium'):>6s} "
                  f"{rate('hard'):>6s} {f'{won}/{len(sub)}':>8s} {f'{fin}/{len(sub)}':>9s} {tok:10.0f}")
    print("\nrescued = capped problems that pass after rescue; finished = thinking ended on its own;"
          "\nextra tok = mean tokens per rescued problem beyond the original 16k.")


if __name__ == "__main__":
    main()
