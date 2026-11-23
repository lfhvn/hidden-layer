"""The strategy ladder. Each strategy: solve(problem, llm) -> final code.

Strategies see only public (sample) tests. Calls are shared where possible
(same messages, params and sample index), so the cache makes later strategies
reuse earlier strategies' work: S2 and S3 start from S0's call, S5 from S4's.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from ..llm import LLM
from ..problems import Problem
from .common import (
    Candidate,
    agreement_pick,
    best_by_public,
    check,
    extract_code,
    first_messages,
    generate,
    repair,
    task_prompt,
)

EFFORT = "high"
MAX_TOKENS = 8192
N_SAMPLES = 5
N_APPROACHES = 3
REPAIR_ROUNDS = 3
WORKER_REPAIR_ROUNDS = 2
N_GEN_INPUTS = 6


def s0_single(p: Problem, llm: LLM) -> str:
    """Baseline: one shot at the default (high) reasoning effort."""
    return generate(llm, p, first_messages(p), effort=EFFORT, max_tokens=MAX_TOKENS).code


def s1_max_effort(p: Problem, llm: LLM) -> str:
    """One shot, maximum reasoning effort and double the token budget."""
    return generate(llm, p, first_messages(p), effort="max", max_tokens=2 * MAX_TOKENS).code


def s2_repair(p: Problem, llm: LLM) -> str:
    """Evaluator-optimizer: S0, then up to 3 rounds of fixing sample-test failures."""
    cand = generate(llm, p, first_messages(p), effort=EFFORT, max_tokens=MAX_TOKENS)
    return repair(llm, p, cand, REPAIR_ROUNDS, effort=EFFORT, max_tokens=MAX_TOKENS).code


def _samples(p: Problem, llm: LLM, n: int) -> list[Candidate]:
    msgs = first_messages(p)
    with ThreadPoolExecutor(n) as ex:
        return list(
            ex.map(
                lambda i: generate(llm, p, msgs, sample=i, effort=EFFORT, max_tokens=MAX_TOKENS),
                range(n),
            )
        )


def s3_best_of_n(p: Problem, llm: LLM) -> str:
    """Parallel voting: 5 samples, keep one that passes the sample tests."""
    return best_by_public(_samples(p, llm, N_SAMPLES)).code


_JSON = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _json_block(text: str):
    blocks = _JSON.findall(text or "")
    for b in reversed(blocks):
        try:
            return json.loads(b)
        except ValueError:
            continue
    return None


def _plan(p: Problem, llm: LLM) -> list[str]:
    msgs = [
        {"role": "system", "content": "You are an expert competitive programmer and coach."},
        {
            "role": "user",
            "content": task_prompt(p)
            + f"\n\n### Your task\n\nDo NOT write code yet. Propose {N_APPROACHES} genuinely "
            "different algorithmic approaches that fit the constraints. For each, give the key "
            "idea, the edge cases it must handle and its time complexity. Answer with a "
            '```json block holding a list of objects {"name", "idea", "edge_cases", "complexity"}.',
        },
    ]
    reply = llm.chat(msgs, effort=EFFORT, max_tokens=MAX_TOKENS)
    plan = _json_block(reply.content)
    if not isinstance(plan, list) or not plan:
        return [""] * N_APPROACHES
    out = []
    for a in plan[:N_APPROACHES]:
        out.append(json.dumps(a, indent=1) if isinstance(a, dict) else str(a))
    return out + [""] * (N_APPROACHES - len(out))


def _workers(p: Problem, llm: LLM) -> list[Candidate]:
    """Orchestrator-workers: plan approaches, implement each, repair each."""
    approaches = _plan(p, llm)

    def work(i: int) -> Candidate:
        extra = (
            f"### Approach to use\n\nImplement this approach:\n{approaches[i]}"
            if approaches[i]
            else ""
        )
        cand = generate(llm, p, first_messages(p, extra), sample=i, effort=EFFORT, max_tokens=MAX_TOKENS)
        return repair(llm, p, cand, WORKER_REPAIR_ROUNDS, sample=i, effort=EFFORT, max_tokens=MAX_TOKENS)

    with ThreadPoolExecutor(N_APPROACHES) as ex:
        return list(ex.map(work, range(N_APPROACHES)))


def s4_orchestrator(p: Problem, llm: LLM) -> str:
    """Plan 3 approaches, implement and repair each in parallel, select."""
    return best_by_public(_workers(p, llm)).code


def _gen_inputs(p: Problem, llm: LLM) -> list[str]:
    if p.kind == "functional":
        fmt = (
            "Each input is a string with one JSON value per line, one line per argument of "
            f"`{p.func_name}`, in order (like the examples)."
        )
    else:
        fmt = "Each input is the exact text given on standard input."
    examples = "\n".join(f"Example input:\n{t.input}" for t in p.public[:2])
    msgs = [
        {"role": "system", "content": "You write tricky but valid test inputs."},
        {
            "role": "user",
            "content": f"### Problem\n\n{p.statement}\n\n{examples}\n\n### Your task\n\n"
            f"Write {N_GEN_INPUTS} valid test inputs that respect every constraint and probe "
            "edge cases (minimum sizes, equal values, boundaries, worst cases for greedy "
            f"mistakes). Keep each input small enough to write by hand. {fmt} Do not give the "
            "expected outputs. Answer with a ```json block holding a list of strings.",
        },
    ]
    data = _json_block(llm.chat(msgs, effort=EFFORT, max_tokens=MAX_TOKENS).content)
    if not isinstance(data, list):
        return []
    return [d if d.endswith("\n") else d + "\n" for d in data if isinstance(d, str) and d.strip()]


def _critic(p: Problem, llm: LLM, code: str) -> str | None:
    msgs = first_messages(p) + [
        {"role": "assistant", "content": f"```python\n{code}\n```"},
        {
            "role": "user",
            "content": "Review this solution as a strict judge. Check correctness on edge cases "
            "and whether it runs within the time limit at the maximum constraints. If it is "
            "correct and fast enough, reply with exactly APPROVED. Otherwise return the complete "
            "fixed code in a single ```python code block.",
        },
    ]
    reply = llm.chat(msgs, effort=EFFORT, max_tokens=MAX_TOKENS)
    if "APPROVED" in reply.content and not extract_code(reply.content):
        return None
    return extract_code(reply.content) or None


def s5_full(p: Problem, llm: LLM) -> str:
    """S4 + 2 extra plain samples, selection by agreement on model-written inputs,
    then a critic pass whose fix is kept only if it still passes the samples."""
    with ThreadPoolExecutor(2) as ex:
        f_workers = ex.submit(_workers, p, llm)
        f_inputs = ex.submit(_gen_inputs, p, llm)
        cands = f_workers.result() + _samples(p, llm, 2)
        inputs = f_inputs.result()
    passing = [c for c in cands if c.ok]
    pick = agreement_pick(p, passing, inputs) if passing else best_by_public(cands)
    fixed = _critic(p, llm, pick.code)
    if fixed:
        if all(r.passed for r in check(p, fixed)):
            return fixed
    return pick.code


STRATEGIES = {
    "S0": s0_single,
    "S1": s1_max_effort,
    "S2": s2_repair,
    "S3": s3_best_of_n,
    "S4": s4_orchestrator,
    "S5": s5_full,
}
