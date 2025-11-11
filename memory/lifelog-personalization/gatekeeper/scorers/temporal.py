"""Temporal scoring utilities."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

ISO_FORMATS = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    for fmt in ISO_FORMATS:
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    return None


def temporal_correctness(df: pd.DataFrame, predicted_column: str = "predicted_timestamp", reference_column: str = "timestamp", window_days: int = 1) -> float:
    """Compute temporal correctness within ``window_days``."""

    window = timedelta(days=window_days)
    scores: List[int] = []
    for _, row in df.iterrows():
        predicted = parse_timestamp(row.get(predicted_column))
        reference = parse_timestamp(row.get(reference_column))
        if not predicted or not reference:
            continue
        delta = abs(predicted - reference)
        scores.append(int(delta <= window))
    return float(np.mean(scores)) if scores else 0.0


def forgetting_curve(scores: Iterable[float]) -> List[float]:
    """Return cumulative moving average to illustrate forgetting."""

    history: List[float] = []
    total = 0.0
    count = 0
    for score in scores:
        count += 1
        total += score
        history.append(total / count)
    return history
