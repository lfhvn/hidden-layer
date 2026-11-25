"""LiveCodeBench problems, split into public tests (strategies see them) and
hidden tests (only the grader sees them)."""

from __future__ import annotations

import base64
import io
import json
import pickle
import random
import zlib
from dataclasses import dataclass
from pathlib import Path

from .sandbox import Test

DATA = Path(__file__).resolve().parent.parent / "data" / "test6.jsonl"


class _NoGlobals(pickle.Unpickler):
    """The hidden tests are a pickled JSON string; refuse anything that is not
    a plain builtin value, so a pickle cannot run code."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def _decode_private(blob: str) -> list[dict]:
    try:
        return json.loads(blob)
    except ValueError:
        raw = zlib.decompress(base64.b64decode(blob.encode()))
        return json.loads(_NoGlobals(io.BytesIO(raw)).load())


@dataclass
class Problem:
    id: str
    title: str
    platform: str
    difficulty: str
    statement: str
    starter_code: str
    func_name: str | None
    public: list[Test]
    hidden: list[Test]

    @property
    def kind(self) -> str:
        return self.public[0].kind if self.public else "stdin"


def _tests(rows: list[dict]) -> list[Test]:
    return [Test(r["input"], r["output"], r["testtype"]) for r in rows]


def load_all() -> list[Problem]:
    problems = []
    for line in DATA.read_text().splitlines():
        r = json.loads(line)
        meta = json.loads(r.get("metadata") or "{}")
        public = _tests(json.loads(r["public_test_cases"]))
        hidden = _tests(_decode_private(r["private_test_cases"]))
        problems.append(
            Problem(
                id=r["question_id"],
                title=r["question_title"],
                platform=r["platform"],
                difficulty=r["difficulty"],
                statement=r["question_content"],
                starter_code=r.get("starter_code") or "",
                func_name=meta.get("func_name"),
                public=public,
                # Grade on public + hidden, as LiveCodeBench does.
                hidden=public + hidden,
            )
        )
    return problems


def select(n_per_cell: int = 6, n_hard: int = 10, seed: int = 0) -> list[Problem]:
    """A fixed slice: n easy + n medium from each platform, plus n_hard hard
    problems per platform. Hard problems use their own RNG so adding them
    left the original easy/medium picks unchanged."""
    rng = random.Random(seed)
    out = []
    by_cell: dict[tuple[str, str], list[Problem]] = {}
    for p in load_all():
        if p.public:
            by_cell.setdefault((p.platform, p.difficulty), []).append(p)
    for cell in sorted(c for c in by_cell if c[1] in ("easy", "medium")):
        items = sorted(by_cell[cell], key=lambda p: p.id)
        out.extend(rng.sample(items, min(n_per_cell, len(items))))
    hard_rng = random.Random(seed + 1)
    for cell in sorted(c for c in by_cell if c[1] == "hard"):
        items = sorted(by_cell[cell], key=lambda p: p.id)
        out.extend(hard_rng.sample(items, min(n_hard, len(items))))
    return out
