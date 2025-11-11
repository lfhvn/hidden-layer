"""Runner for long-context evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping

import pandas as pd

from ..scorers.temporal import forgetting_curve
from .base import EvaluationResult, load_config


def _effective_context_utilization(df: pd.DataFrame) -> float:
    if "context_tokens" not in df or "accuracy" not in df:
        return 0.0
    tokens = df["context_tokens"].clip(lower=1)
    return float((df["accuracy"] / tokens).mean())


def _latency_profile(df: pd.DataFrame) -> float:
    if "latency_ms" not in df:
        return 0.0
    return float(df["latency_ms"].mean())


def evaluate_long_context(config_path: Path, predictions: Mapping[str, pd.DataFrame]) -> List[EvaluationResult]:
    config = load_config(config_path)
    metrics = config.get("metrics", [])
    results: List[EvaluationResult] = []

    for dataset_cfg in config.get("datasets", []):
        name = dataset_cfg["name"]
        for split in dataset_cfg.get("splits", ["test"]):
            key = f"{name}:{split}"
            if key not in predictions:
                raise KeyError(f"Missing predictions for {key}")
            df = predictions[key]
            metric_values: Dict[str, float] = {}

            if "task_metrics" in metrics:
                if "accuracy" in df:
                    metric_values["task_metrics"] = float(df["accuracy"].mean())
                elif "f1" in df:
                    metric_values["task_metrics"] = float(df["f1"].mean())
                else:
                    metric_values["task_metrics"] = 0.0

            if "effective_context_utilization" in metrics:
                metric_values["effective_context_utilization"] = _effective_context_utilization(df)

            if "forgetting_curve" in metrics:
                curve = forgetting_curve(df.get("accuracy", []))
                metric_values["forgetting_curve"] = curve[-1] if curve else 0.0

            if "latency_profile" in metrics:
                metric_values["latency_profile"] = _latency_profile(df)

            results.append(EvaluationResult(name=key, metrics=metric_values))

    return results
