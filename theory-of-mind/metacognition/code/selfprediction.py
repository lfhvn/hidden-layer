"""
Self-prediction task: can a model predict its *own* future answer?

This is an introspection probe in the spirit of Anthropic's work on whether models
have accurate access to their own internal states. Unlike calibration (does confidence
track correctness?), self-prediction asks a sharper question: before producing an
answer, can the subject forecast what that answer will be?

A subject with genuine self-knowledge predicts its own answer well *regardless* of
whether that answer is correct. We therefore report self-prediction accuracy (does the
forecast match the later answer?) separately from task accuracy, and compare it to a
chance baseline.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional

from harness import llm_call

from .dataset import Dataset, Question, load_dataset, normalize

_PREDICTION_RE = re.compile(r"prediction\s*[:=]\s*(.+)", re.IGNORECASE)


def build_prediction_prompt(question: Question) -> str:
    """Ask the subject to forecast its own answer without committing to it yet."""
    return (
        "You are about to be asked a question. Before you answer it, predict in a few "
        "words what your eventual answer will be. Do not explain.\n\n"
        f"Upcoming question: {question.question}\n\n"
        "Respond in exactly this format:\n"
        "Prediction: <what you think your answer will be>"
    )


def parse_prediction(text: str) -> str:
    """Extract the predicted answer from a prediction response."""
    m = _PREDICTION_RE.search(text)
    if m:
        return m.group(1).strip(" .,-")
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def answers_match(a: str, b: str) -> bool:
    """
    True if two short answers refer to the same thing. We use token-set overlap on the
    normalized strings so 'Paris' matches 'the answer is Paris' and 'da vinci' matches
    'Leonardo da Vinci'.
    """
    ta, tb = set(normalize(a).split()), set(normalize(b).split())
    if not ta or not tb:
        return False
    return bool(ta & tb)


@dataclass
class SelfPredictionRecord:
    """One question answered twice: a prediction pass and an answer pass."""

    question_id: str
    question: str
    predicted: str
    actual: str
    matched: bool  # did the prediction match the actual answer?
    actual_correct: bool


@dataclass
class SelfPredictionRun:
    """Aggregate self-prediction results."""

    dataset_name: str
    records: List[SelfPredictionRecord]

    @property
    def n(self) -> int:
        return len(self.records)

    @property
    def self_prediction_accuracy(self) -> float:
        """Fraction of questions where the subject predicted its own answer."""
        return sum(r.matched for r in self.records) / self.n if self.n else 0.0

    @property
    def task_accuracy(self) -> float:
        """Fraction of answers that were actually correct."""
        return sum(r.actual_correct for r in self.records) / self.n if self.n else 0.0

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "n": self.n,
            "self_prediction_accuracy": self.self_prediction_accuracy,
            "task_accuracy": self.task_accuracy,
            "records": [
                {
                    "id": r.question_id,
                    "question": r.question,
                    "predicted": r.predicted,
                    "actual": r.actual,
                    "matched": r.matched,
                    "actual_correct": r.actual_correct,
                }
                for r in self.records
            ],
        }


class ModelIntrospector:
    """Real-model self-prediction subject: predicts, then answers, via the harness."""

    def __init__(
        self, provider: str = "anthropic", model: Optional[str] = None, temperature: float = 0.0, **call_kwargs
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.call_kwargs = call_kwargs

    def predict(self, question: Question) -> str:
        resp = llm_call(
            build_prediction_prompt(question),
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            **self.call_kwargs,
        )
        return parse_prediction(resp.text)

    def answer(self, question: Question):
        # Reuse the calibration subject's answer pathway for consistency.
        from .subjects import ModelSubject

        return ModelSubject(self.provider, self.model, self.temperature, **self.call_kwargs).answer(question)


class SimulatedIntrospector:
    """
    Deterministic self-prediction subject with tunable self-knowledge.

    ``self_knowledge`` in [0, 1] is the probability that the prediction matches the
    later answer. At 1.0 the subject perfectly forecasts itself; at 0.0 the prediction
    is an unrelated token. ``accuracy`` controls how often the actual answer is correct.
    All randomness is seeded per question id.
    """

    def __init__(self, accuracy: float = 0.6, self_knowledge: float = 0.8, seed: int = 0):
        if not (0.0 <= self_knowledge <= 1.0):
            raise ValueError("self_knowledge must be in [0, 1].")
        self.accuracy = accuracy
        self.self_knowledge = self_knowledge
        self.seed = seed

    def _actual_answer(self, question: Question, rng: random.Random) -> tuple[str, bool]:
        is_correct = rng.random() < self.accuracy
        return (question.answers[0] if is_correct else "no-idea-placeholder", is_correct)

    def record(self, question: Question) -> SelfPredictionRecord:
        rng = random.Random(f"{self.seed}:{question.id}")
        actual, is_correct = self._actual_answer(question, rng)
        if rng.random() < self.self_knowledge:
            predicted = actual
        else:
            # Forecast something unrelated (a deterministic distractor).
            predicted = f"distractor-{rng.randrange(1000)}"
        return SelfPredictionRecord(
            question_id=question.id,
            question=question.question,
            predicted=predicted,
            actual=actual,
            matched=answers_match(predicted, actual),
            actual_correct=is_correct,
        )


def run_self_prediction(
    introspector,
    dataset: Optional[Dataset] = None,
) -> SelfPredictionRun:
    """
    Run a self-prediction experiment.

    ``introspector`` may expose either ``.record(question) -> SelfPredictionRecord``
    (the simulated subject) or the pair ``.predict(question)`` / ``.answer(question)``
    (a real model). Both paths produce the same SelfPredictionRun.
    """
    if dataset is None:
        dataset = load_dataset()

    records: List[SelfPredictionRecord] = []
    for q in dataset:
        if hasattr(introspector, "record"):
            records.append(introspector.record(q))
        else:
            predicted = introspector.predict(q)
            ans = introspector.answer(q)
            records.append(
                SelfPredictionRecord(
                    question_id=q.id,
                    question=q.question,
                    predicted=predicted,
                    actual=ans.answer,
                    matched=answers_match(predicted, ans.answer),
                    actual_correct=ans.correct,
                )
            )
    return SelfPredictionRun(dataset_name=dataset.name, records=records)
