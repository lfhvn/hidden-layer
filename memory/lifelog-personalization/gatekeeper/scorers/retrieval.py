"""Retrieval metrics for lifelog evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class RetrievalResult:
    """Container representing the ranked results for a query."""

    query_id: str
    ranked_ids: Sequence[str]
    relevant_ids: Sequence[str]


def precision_at_k(result: RetrievalResult, k: int) -> float:
    ranked = list(result.ranked_ids)[:k]
    if not ranked:
        return 0.0
    rel = set(result.relevant_ids)
    hits = sum(1 for doc_id in ranked if doc_id in rel)
    return hits / len(ranked)


def mean_precision_at_k(results: Iterable[RetrievalResult], k: int) -> float:
    scores = [precision_at_k(result, k) for result in results]
    return float(np.mean(scores)) if scores else 0.0


def dcg_at_k(result: RetrievalResult, k: int) -> float:
    ranked = list(result.ranked_ids)[:k]
    rel = set(result.relevant_ids)
    score = 0.0
    for idx, doc_id in enumerate(ranked, start=1):
        if doc_id in rel:
            score += 1.0 / np.log2(idx + 1)
    return score


def ndcg_at_k(result: RetrievalResult, k: int) -> float:
    ideal = dcg_at_k(RetrievalResult(result.query_id, result.relevant_ids, result.relevant_ids), k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(result, k) / ideal


def mean_ndcg_at_k(results: Iterable[RetrievalResult], k: int) -> float:
    scores = [ndcg_at_k(result, k) for result in results]
    return float(np.mean(scores)) if scores else 0.0


def mean_reciprocal_rank(results: Iterable[RetrievalResult]) -> float:
    reciprocals: List[float] = []
    for result in results:
        rel = set(result.relevant_ids)
        for idx, doc_id in enumerate(result.ranked_ids, start=1):
            if doc_id in rel:
                reciprocals.append(1.0 / idx)
                break
        else:
            reciprocals.append(0.0)
    return float(np.mean(reciprocals)) if reciprocals else 0.0


def entity_grounding_accuracy(df: pd.DataFrame, entity_column: str = "entities", predicted_column: str = "predicted_entities") -> float:
    """Compute entity grounding accuracy.

    Args:
        df: Dataframe with ground-truth and predicted entity annotations.
        entity_column: Column containing reference entity lists.
        predicted_column: Column containing predicted entity lists.
    """

    if entity_column not in df or predicted_column not in df:
        raise ValueError("Dataframe must contain columns for reference and predicted entities")

    scores: List[float] = []
    for _, row in df.iterrows():
        truth = set(row.get(entity_column) or [])
        predicted = set(row.get(predicted_column) or [])
        if not truth:
            continue
        scores.append(len(truth & predicted) / len(truth))
    return float(np.mean(scores)) if scores else 0.0
