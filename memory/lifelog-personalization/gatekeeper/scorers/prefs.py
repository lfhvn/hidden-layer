"""Preference adherence metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd

@dataclass
class PreferenceJudgement:
    """Result returned by the rubric-based judge."""

    score: float
    explanation: str


class PreferenceAdherenceScorer:
    """Evaluate adherence to user preferences using a rubric-based judge."""

    def __init__(self, rubric_prompt: str, judge_fn: Callable[[str], PreferenceJudgement]) -> None:
        self.rubric_prompt = rubric_prompt
        self.judge_fn = judge_fn

    def score_row(self, query: str, response: str, preferences: Dict[str, str]) -> PreferenceJudgement:
        rubric = self._format_rubric(preferences)
        prompt = f"{self.rubric_prompt}\n\nPreferences:\n{rubric}\n\nUser request:\n{query}\n\nModel response:\n{response}"
        return self.judge_fn(prompt)

    def _format_rubric(self, preferences: Dict[str, str]) -> str:
        lines = [f"- {key}: {value}" for key, value in sorted(preferences.items())]
        return "\n".join(lines)

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        scores: List[PreferenceJudgement] = []
        for _, row in df.iterrows():
            preferences = row.get("preferences") or {}
            response = row.get("response") or row.get("prediction") or row.get("model_output") or ""
            query = row.get("query") or row.get("prompt") or ""
            judgement = self.score_row(query, response, preferences)
            scores.append(judgement)
        return pd.DataFrame({"score": [s.score for s in scores], "explanation": [s.explanation for s in scores]})


def preference_drift(adherence_scores: Iterable[float], session_ids: Iterable[str]) -> float:
    """Measure adherence degradation across sessions."""

    paired = sorted(zip(session_ids, adherence_scores), key=lambda x: x[0])
    if not paired:
        return 0.0
    baseline = np.mean([score for _, score in paired[: max(1, len(paired) // 4)]] )
    tail = np.mean([score for _, score in paired[-max(1, len(paired) // 4):]])
    return baseline - tail
