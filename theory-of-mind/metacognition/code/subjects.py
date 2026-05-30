"""
Subjects under test: the things that answer questions *with a confidence*.

Two implementations:
  - ``ModelSubject``  - wraps a real LLM via the harness (any provider).
  - ``SimulatedSubject`` - a deterministic stand-in whose accuracy and calibration
    are tunable, so calibration metrics can be exercised offline with known ground
    truth. It injects its formatted answer through the harness ``sim`` provider, so
    the *same* prompt -> answer -> parse -> score pipeline runs for sim and real models.

Both return ``AnswerRecord`` objects that the runner turns into calibration metrics.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from harness import llm_call

from .dataset import Question

# Match an explicit "Confidence: 85%" / "confidence = 0.85" style line first.
_CONF_LABELED_RE = re.compile(r"confidence\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*(%?)", re.IGNORECASE)
# Fallback: any "85%" or "85 percent" anywhere in the text.
_CONF_PERCENT_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*(%|percent)", re.IGNORECASE)
# Pull the answer off an "Answer: ..." line if present.
_ANSWER_RE = re.compile(r"answer\s*[:=]\s*(.+)", re.IGNORECASE)


def _coerce_confidence(value: float, had_percent: bool) -> float:
    """Map a parsed number to a probability in [0, 1]."""
    if had_percent or value > 1.0:
        value = value / 100.0
    return max(0.0, min(1.0, value))


def parse_answer_and_confidence(text: str) -> Tuple[str, Optional[float]]:
    """
    Extract (answer, confidence) from free-form model output.

    Confidence is read from an explicit 'Confidence:' field when present, else from
    any percentage in the text, else None. The answer is the 'Answer:' line if present,
    otherwise the first non-empty line with any confidence phrase stripped out.
    """
    confidence: Optional[float] = None
    m = _CONF_LABELED_RE.search(text)
    if m:
        confidence = _coerce_confidence(float(m.group(1)), m.group(2) == "%")
    else:
        m2 = _CONF_PERCENT_RE.search(text)
        if m2:
            confidence = _coerce_confidence(float(m2.group(1)), True)

    answer = ""
    am = _ANSWER_RE.search(text)
    if am:
        answer = am.group(1).strip()
        # Drop a trailing confidence clause that landed on the same line.
        answer = _CONF_LABELED_RE.sub("", answer).strip(" .,-")
    else:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not _CONF_LABELED_RE.match(stripped):
                answer = stripped
                break
    return answer, confidence


def build_prompt(question: Question) -> str:
    """Standard elicitation prompt asking for an answer plus a numeric confidence."""
    return (
        "Answer the following question concisely, then state how confident you are.\n\n"
        f"Question: {question.question}\n\n"
        "Respond in exactly this format:\n"
        "Answer: <your answer>\n"
        "Confidence: <0-100>%"
    )


@dataclass
class AnswerRecord:
    """One answered question with the parsed confidence and scored correctness."""

    question_id: str
    question: str
    answer: str
    confidence: Optional[float]
    correct: bool
    raw_response: str


class ModelSubject:
    """Wraps a real LLM (any harness provider) as a calibration subject."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        temperature: float = 0.0,
        default_confidence: float = 0.5,
        **call_kwargs,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.default_confidence = default_confidence  # used when the model omits one
        self.call_kwargs = call_kwargs

    def answer(self, question: Question) -> AnswerRecord:
        resp = llm_call(
            build_prompt(question),
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            **self.call_kwargs,
        )
        ans, conf = parse_answer_and_confidence(resp.text)
        if conf is None:
            conf = self.default_confidence
        return AnswerRecord(
            question_id=question.id,
            question=question.question,
            answer=ans,
            confidence=conf,
            correct=question.is_correct(ans),
            raw_response=resp.text,
        )


# Calibration profiles supported by the simulated subject.
CALIBRATION_PROFILES = ("calibrated", "overconfident", "underconfident", "random")


class SimulatedSubject:
    """
    Deterministic subject with tunable accuracy and calibration.

    For each question we draw a latent 'true probability of being correct' p around the
    target accuracy, sample correctness ~ Bernoulli(p), and derive a *stated* confidence
    from p according to the chosen profile:

      - "calibrated":    stated = p                 (honest; ECE -> 0 in the limit)
      - "overconfident": stated = p + (1-p)*strength (inflated toward 1)
      - "underconfident":stated = p * (1-strength)   (deflated toward 0)
      - "random":        stated ~ Uniform(0,1), independent of p (no signal)

    All randomness is seeded per question id, so results are fully reproducible. The
    formatted answer is routed through the harness ``sim`` provider so the real
    parse-and-score pipeline runs end to end.
    """

    def __init__(
        self,
        accuracy: float = 0.6,
        profile: str = "calibrated",
        strength: float = 0.4,
        spread: float = 0.5,
        seed: int = 0,
    ):
        if profile not in CALIBRATION_PROFILES:
            raise ValueError(f"Unknown profile {profile!r}; choose from {CALIBRATION_PROFILES}.")
        if not (0.0 <= accuracy <= 1.0):
            raise ValueError("accuracy must be in [0, 1].")
        self.accuracy = accuracy
        self.profile = profile
        self.strength = strength
        self.spread = spread
        self.seed = seed

    def _plan(self, question: Question) -> Tuple[bool, float]:
        """Decide (correct, stated_confidence) deterministically for this question."""
        rng = random.Random(f"{self.seed}:{question.id}")
        # Latent correctness probability, spread around the target accuracy.
        p = self.accuracy + self.spread * (rng.random() - 0.5)
        p = max(0.02, min(0.98, p))
        is_correct = rng.random() < p

        if self.profile == "calibrated":
            stated = p
        elif self.profile == "overconfident":
            stated = p + (1.0 - p) * self.strength
        elif self.profile == "underconfident":
            stated = p * (1.0 - self.strength)
        else:  # random
            stated = rng.random()
        return is_correct, max(0.0, min(1.0, stated))

    def _format_response(self, question: Question, is_correct: bool, confidence: float) -> str:
        # A correct answer must actually satisfy the dataset scorer; a wrong one must not.
        answer_text = question.answers[0] if is_correct else "no-idea-placeholder"
        return f"Answer: {answer_text}\nConfidence: {round(confidence * 100)}%"

    def answer(self, question: Question) -> AnswerRecord:
        is_correct, confidence = self._plan(question)
        canned = self._format_response(question, is_correct, confidence)
        # Route through the harness sim provider so parsing/scoring is exercised for real.
        resp = llm_call(build_prompt(question), provider="sim", sim_response=canned)
        ans, conf = parse_answer_and_confidence(resp.text)
        return AnswerRecord(
            question_id=question.id,
            question=question.question,
            answer=ans,
            confidence=conf,
            correct=question.is_correct(ans),
            raw_response=resp.text,
        )
