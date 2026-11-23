"""Run candidate programs against tests in a sandbox.

Each test runs in its own `python3 -I` subprocess under macOS sandbox-exec:
no network, and no file writes outside a fresh temp directory. CPU time is
capped per test.
"""

from __future__ import annotations

import json
import math
import resource
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

TIMEOUT_S = 6.0
PYTHON = sys.executable

_PRELUDE = (
    "from typing import *\n"
    "import collections, heapq, bisect, math, itertools, functools, string, re\n"
    "from collections import *\nfrom functools import *\nfrom itertools import *\n"
    "from heapq import *\nfrom bisect import *\nfrom math import *\n"
)

# Calls Solution().<func>(*args) with one JSON argument per input line.
_FUNCTIONAL_HARNESS = """
import json, sys
sys.setrecursionlimit(1 << 20)
ns = {}
exec(compile(open(sys.argv[1]).read(), "solution.py", "exec"), ns)
args = [json.loads(l) for l in sys.stdin.read().split("\\n") if l.strip()]
res = getattr(ns["Solution"](), sys.argv[2])(*args)
print(json.dumps(res))
"""


@dataclass
class Test:
    input: str
    output: str | None  # None: input only (for comparing candidates' outputs)
    kind: str  # "stdin" or "functional"


@dataclass
class Result:
    passed: bool
    status: str  # ok | wrong | error | timeout
    stdout: str
    detail: str  # short, model-facing explanation


def _profile(tmp: str) -> str:
    real = str(Path(tmp).resolve())
    return (
        "(version 1)(allow default)(deny network*)(deny file-write*)"
        f'(allow file-write* (subpath "{tmp}") (subpath "{real}")'
        ' (literal "/dev/null") (literal "/dev/dtracehelper"))'
    )


def _limit_cpu():
    cpu = int(TIMEOUT_S) + 1
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))


def _tokens_equal(got: str, want: str) -> bool:
    a, b = got.split(), want.split()
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if not math.isclose(fx, fy, rel_tol=1e-6, abs_tol=1e-6):
            return False
    return True


def _json_equal(got: str, want: str) -> bool:
    try:
        g, w = json.loads(got), json.loads(want)
    except ValueError:
        return got.strip() == want.strip()

    def eq(x, y):
        if isinstance(x, float) or isinstance(y, float):
            try:
                return math.isclose(float(x), float(y), rel_tol=1e-6, abs_tol=1e-6)
            except (TypeError, ValueError):
                return False
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(eq(a, b) for a, b in zip(x, y))
        return x == y

    return eq(g, w)


def _clip(text: str, n: int = 300) -> str:
    text = text.strip()
    return text if len(text) <= n else text[: n // 2] + " ... " + text[-n // 2 :]


def run_one(code: str, test: Test, func_name: str | None) -> Result:
    tmp = tempfile.mkdtemp(prefix="orch-")
    try:
        sol = Path(tmp) / "solution.py"
        sol.write_text(_PRELUDE + code)
        if test.kind == "functional":
            harness = Path(tmp) / "harness.py"
            harness.write_text(_FUNCTIONAL_HARNESS)
            argv = [PYTHON, "-I", str(harness), str(sol), func_name or ""]
        else:
            argv = [PYTHON, "-I", str(sol)]
        try:
            proc = subprocess.run(
                ["sandbox-exec", "-p", _profile(tmp), *argv],
                input=test.input,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
                cwd=tmp,
                preexec_fn=_limit_cpu,
            )
        except subprocess.TimeoutExpired:
            return Result(False, "timeout", "", f"Time limit exceeded ({TIMEOUT_S:.0f}s).")
        if proc.returncode != 0:
            err = proc.stderr.strip().splitlines()
            tail = "\n".join(err[-4:]) if err else f"exit code {proc.returncode}"
            return Result(False, "error", proc.stdout, "Runtime error:\n" + _clip(tail, 600))
        if test.output is None:
            return Result(True, "ok", proc.stdout, "")
        same = (
            _json_equal(proc.stdout, test.output)
            if test.kind == "functional"
            else _tokens_equal(proc.stdout, test.output)
        )
        if same:
            return Result(True, "ok", proc.stdout, "")
        return Result(
            False,
            "wrong",
            proc.stdout,
            f"Wrong answer.\nInput:\n{_clip(test.input)}\n"
            f"Expected:\n{_clip(test.output)}\nGot:\n{_clip(proc.stdout)}",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_tests(
    code: str,
    tests: list[Test],
    func_name: str | None,
    *,
    stop_on_fail: bool = False,
    workers: int = 4,
) -> list[Result]:
    """Run tests in parallel. With stop_on_fail, tests after a failure may be skipped."""
    if not code.strip():
        return [Result(False, "error", "", "No code was produced.") for _ in tests]
    if not stop_on_fail:
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(lambda t: run_one(code, t, func_name), tests))
    results: list[Result] = []
    for i in range(0, len(tests), workers):
        with ThreadPoolExecutor(workers) as ex:
            batch = list(ex.map(lambda t: run_one(code, t, func_name), tests[i : i + workers]))
        results.extend(batch)
        if not all(r.passed for r in batch):
            break
    return results
