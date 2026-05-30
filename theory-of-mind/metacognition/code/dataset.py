"""
Question datasets for calibration experiments.

A ``Question`` carries the prompt text plus the set of accepted answers, so that
correctness can be scored deterministically and offline (normalized substring
match) without an LLM judge or network access.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase and collapse non-alphanumeric runs to single spaces, for robust matching."""
    return _NORMALIZE_RE.sub(" ", text.lower()).strip()


@dataclass
class Question:
    """A single short-answer question with deterministically checkable answers."""

    id: str
    question: str
    answers: List[str]  # any of these (normalized) counts as correct
    difficulty: str = "unknown"

    def is_correct(self, model_answer: str) -> bool:
        """
        True if any accepted answer appears as a whole-token subsequence of the
        (normalized) model answer. Token-boundary matching avoids spurious hits
        like 'au' inside 'because'.
        """
        ans_tokens = normalize(model_answer).split()
        ans_set = set(ans_tokens)
        joined = " ".join(ans_tokens)
        for accepted in self.answers:
            acc = normalize(accepted)
            acc_tokens = acc.split()
            if len(acc_tokens) == 1:
                if acc in ans_set:
                    return True
            else:
                # multi-word answer: require the phrase to appear contiguously
                if acc and (joined == acc or f" {acc} " in f" {joined} "):
                    return True
        return False


@dataclass
class Dataset:
    """A named collection of questions."""

    name: str
    questions: List[Question]
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)


def make_arithmetic_dataset(n: int = 200, seed: int = 0, max_operand: int = 99) -> Dataset:
    """
    Generate ``n`` addition questions with deterministic integer answers.

    Useful for calibration experiments that need a large, fully-checkable question set
    (e.g. to show that a calibrated subject's ECE -> 0 as N grows, which is hard to see
    on the 25-item bundled set). Reproducible given ``seed``.
    """
    import random as _random

    rng = _random.Random(seed)
    questions: List[Question] = []
    for i in range(n):
        a = rng.randint(0, max_operand)
        b = rng.randint(0, max_operand)
        questions.append(
            Question(
                id=f"add{i:04d}",
                question=f"What is {a} plus {b}?",
                answers=[str(a + b)],
                difficulty="easy",
            )
        )
    return Dataset(
        name=f"arithmetic_n{n}_seed{seed}",
        questions=questions,
        description="Synthetic addition questions with deterministic answers.",
        metadata={"generator": "make_arithmetic_dataset", "seed": seed},
    )


def load_dataset(path: Optional[str] = None, name: str = "factual_qa") -> Dataset:
    """
    Load a dataset from JSON. With no path, loads the bundled ``data/<name>.json``.

    JSON schema:
        {"name": str, "description": str, "questions": [
            {"id": str, "question": str, "answers": [str, ...], "difficulty": str}, ...]}
    """
    json_path = Path(path) if path else DATA_DIR / f"{name}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Dataset not found: {json_path}")

    raw = json.loads(json_path.read_text())
    questions = [
        Question(
            id=q["id"],
            question=q["question"],
            answers=list(q["answers"]),
            difficulty=q.get("difficulty", "unknown"),
        )
        for q in raw["questions"]
    ]
    return Dataset(
        name=raw.get("name", name),
        questions=questions,
        description=raw.get("description", ""),
        metadata={k: v for k, v in raw.items() if k not in {"name", "description", "questions"}},
    )
