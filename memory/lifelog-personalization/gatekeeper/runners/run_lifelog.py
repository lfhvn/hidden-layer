"""Runner for lifelog retrieval evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping

import pandas as pd

from ..scorers.retrieval import (
    RetrievalResult,
    entity_grounding_accuracy,
    mean_ndcg_at_k,
    mean_precision_at_k,
    mean_reciprocal_rank,
)
from ..scorers.temporal import temporal_correctness
from .base import EvaluationResult, load_config


def _to_retrieval_results(df: pd.DataFrame) -> List[RetrievalResult]:
    results: List[RetrievalResult] = []
    for _, row in df.iterrows():
        results.append(
            RetrievalResult(
                query_id=str(row.get("query_id")),
                ranked_ids=row.get("ranked_ids", []),
                relevant_ids=row.get("relevant_ids", []),
            )
        )
    return results


def _compute_metric(metric: str, df: pd.DataFrame) -> float:
    if metric.startswith("precision@"):
        k = int(metric.split("@")[-1])
        return mean_precision_at_k(_to_retrieval_results(df), k)
    if metric.startswith("ndcg@"):
        k = int(metric.split("@")[-1])
        return mean_ndcg_at_k(_to_retrieval_results(df), k)
    if metric == "mrr":
        return mean_reciprocal_rank(_to_retrieval_results(df))
    if metric.startswith("temporal_correctness"):
        window = json.loads(metric.split("@", 1)[1]) if "@" in metric else {"window_days": 1}
        return temporal_correctness(df, window_days=window.get("window_days", 1))
    if metric == "entity_grounding":
        return entity_grounding_accuracy(df)
    raise KeyError(f"Unknown metric '{metric}'")


def evaluate_lifelog(config_path: Path, predictions: Mapping[str, pd.DataFrame]) -> List[EvaluationResult]:
    """Evaluate lifelog predictions according to configuration."""

    config = load_config(config_path)
    metrics = config.get("metrics", [])
    retrievers = [r["name"] if isinstance(r, dict) else r for r in config.get("retrievers", [])]
    results: List[EvaluationResult] = []

    for dataset_cfg in config.get("datasets", []):
        dataset = dataset_cfg["name"]
        for split in dataset_cfg.get("splits", ["test"]):
            for retriever in retrievers:
                key = f"{dataset}:{split}:{retriever}"
                if key not in predictions:
                    raise KeyError(f"Missing predictions for {key}")
                df = predictions[key]
                metric_values: Dict[str, float] = {}
                for metric in metrics:
                    metric_values[metric] = _compute_metric(metric, df)
                results.append(
                    EvaluationResult(
                        name=key,
                        metrics=metric_values,
                    )
                )
    return results
