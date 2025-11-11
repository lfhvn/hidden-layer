"""Runner for TTL/TTT evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping

import pandas as pd

from ..policies.ttl_lora import build_ttl_policy
from .base import EvaluationResult, load_config


def _accuracy(df: pd.DataFrame) -> float:
    if "accuracy_after" in df:
        return float(df["accuracy_after"].mean())
    if "accuracy" in df:
        return float(df["accuracy"].mean())
    return 0.0


def _loss(df: pd.DataFrame) -> float:
    if "loss_after" in df:
        return float(df["loss_after"].mean())
    if "loss" in df:
        return float(df["loss"].mean())
    return 0.0


def _forgetting_rate(df: pd.DataFrame) -> float:
    if {"replay_accuracy_before", "replay_accuracy_after"}.issubset(df.columns):
        before = float(df["replay_accuracy_before"].mean())
        after = float(df["replay_accuracy_after"].mean())
        if before == 0:
            return 0.0
        return max(0.0, (before - after) / before)
    return 0.0


def _latency(df: pd.DataFrame) -> float:
    if "latency_ms" in df:
        return float(df["latency_ms"].mean())
    return 0.0


def _throughput(df: pd.DataFrame) -> float:
    if {"tokens_processed", "wall_time_s"}.issubset(df.columns):
        total_tokens = df["tokens_processed"].sum()
        total_time = df["wall_time_s"].sum()
        return float(total_tokens / total_time) if total_time else 0.0
    return 0.0


def _replay_regret(df: pd.DataFrame) -> float:
    if {"replay_accuracy_anchor", "replay_accuracy_after"}.issubset(df.columns):
        anchor = float(df["replay_accuracy_anchor"].mean())
        after = float(df["replay_accuracy_after"].mean())
        return max(0.0, anchor - after)
    return 0.0


def evaluate_ttl(config_path: Path, logs: Mapping[str, pd.DataFrame]) -> List[EvaluationResult]:
    config = load_config(config_path)
    build_ttl_policy(config)  # ensure config validation
    metrics = config.get("metrics", [])

    results: List[EvaluationResult] = []
    for dataset_cfg in config.get("datasets", []):
        dataset = dataset_cfg["name"]
        key = dataset
        if key not in logs:
            raise KeyError(f"Missing TTL logs for {key}")
        df = logs[key]
        metric_values: Dict[str, float] = {}

        if "accuracy" in metrics:
            metric_values["accuracy"] = _accuracy(df)
        if "loss" in metrics:
            metric_values["loss"] = _loss(df)
        if "forgetting_rate" in metrics:
            metric_values["forgetting_rate"] = _forgetting_rate(df)
        if "latency" in metrics:
            metric_values["latency"] = _latency(df)
        if "throughput" in metrics:
            metric_values["throughput"] = _throughput(df)
        if "replay_regret" in metrics:
            metric_values["replay_regret"] = _replay_regret(df)

        results.append(EvaluationResult(name=key, metrics=metric_values))

    return results
