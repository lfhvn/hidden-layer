"""Metrics for factual model editing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class EditOutcome:
    """Represents the evaluation of a single edit."""

    success: bool
    locality: float
    portability: float


def edit_success_rate(outcomes: Sequence[EditOutcome]) -> float:
    return float(np.mean([1.0 if outcome.success else 0.0 for outcome in outcomes])) if outcomes else 0.0


def mean_locality(outcomes: Sequence[EditOutcome]) -> float:
    return float(np.mean([outcome.locality for outcome in outcomes])) if outcomes else 0.0


def mean_portability(outcomes: Sequence[EditOutcome]) -> float:
    return float(np.mean([outcome.portability for outcome in outcomes])) if outcomes else 0.0


def sequential_stability(curves: Iterable[Sequence[EditOutcome]]) -> List[float]:
    """Compute stability curve over sequential edit batches."""

    results: List[float] = []
    for batch in curves:
        results.append(edit_success_rate(batch))
    return results
