"""Prompts and helpers shared by the strategies."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm import LLM
from ..problems import Problem
from ..sandbox import Result, Test, run_tests

SYSTEM = (
    "You are an expert competitive programmer. Solve the problem correctly and "
    "efficiently in Python 3. Think about edge cases and the constraints' time "
    "limits before writing code."
)


def task_prompt(p: Problem) -> str:
    if p.kind == "functional":
        io = (
            "Complete the following starter code. Return the full class in a single "
            f"```python code block.\n\n```python\n{p.starter_code}\n```"
        )
    else:
        io = (
            "Read the input from standard input and write the answer to standard "
            "output. Return the full program in a single ```python code block."
        )
    return f"### Problem\n\n{p.statement}\n\n### Format\n\n{io}"


def first_messages(p: Problem, extra: str = "") -> list[dict]:
    user = task_prompt(p) + (f"\n\n{extra}" if extra else "")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


_CODE = re.compile(r"```(?:python|py|python3)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    blocks = _CODE.findall(text or "")
    return blocks[-1].strip() if blocks else ""


@dataclass
class Candidate:
    code: str
    public: list[Result] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)  # conversation that produced it

    @property
    def public_passed(self) -> int:
        return sum(r.passed for r in self.public)

    @property
    def ok(self) -> bool:
        return bool(self.public) and all(r.passed for r in self.public)


def check(p: Problem, code: str) -> list[Result]:
    return run_tests(code, p.public, p.func_name)


def feedback(cand: Candidate) -> str:
    fails = [r for r in cand.public if not r.passed][:2]
    lines = [f"Your code failed {len(cand.public) - cand.public_passed} of "
             f"{len(cand.public)} sample tests."]
    lines += [f"\n{r.detail}" for r in fails]
    lines.append(
        "\nFind the bug, then return the complete corrected code in a single "
        "```python code block."
    )
    return "\n".join(lines)


def repair(
    llm: LLM, p: Problem, cand: Candidate, rounds: int, sample: int = 0, **kw
) -> Candidate:
    """Evaluator-optimizer: feed sample-test failures back until they pass."""
    for _ in range(rounds):
        if cand.ok:
            break
        msgs = cand.messages + [
            {"role": "assistant", "content": f"```python\n{cand.code}\n```"},
            {"role": "user", "content": feedback(cand)},
        ]
        reply = llm.chat(msgs, sample=sample, **kw)
        code = extract_code(reply.content) or cand.code
        cand = Candidate(code, check(p, code), msgs)
    return cand


def generate(llm: LLM, p: Problem, msgs: list[dict], sample: int = 0, **kw) -> Candidate:
    reply = llm.chat(msgs, sample=sample, **kw)
    code = extract_code(reply.content)
    return Candidate(code, check(p, code), msgs)


def best_by_public(cands: list[Candidate]) -> Candidate:
    """First candidate passing the most sample tests (stable in sample order)."""
    return max(cands, key=lambda c: c.public_passed)


def agreement_pick(p: Problem, cands: list[Candidate], inputs: list[str]) -> Candidate:
    """Among candidates, pick one from the largest group that agrees on the
    outputs of extra inputs (AlphaCode/CodeT-style clustering)."""
    if len(cands) <= 1 or not inputs:
        return cands[0]
    tests = [Test(i, None, p.kind) for i in inputs]
    sigs = []
    for c in cands:
        outs = run_tests(c.code, tests, p.func_name)
        sigs.append(tuple(o.stdout.strip() if o.passed else f"<{o.status}>" for o in outs))
    counts: dict[tuple, int] = {}
    for s in sigs:
        counts[s] = counts.get(s, 0) + 1
    # Prefer clusters without crashes, then bigger clusters, then earlier samples.
    def score(i: int):
        s = sigs[i]
        crashes = sum(x.startswith("<") for x in s)
        return (-crashes, counts[s], -i)

    return cands[max(range(len(cands)), key=score)]
