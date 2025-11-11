"""Runner for model editing evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import pandas as pd

from ..scorers.editing import EditOutcome, edit_success_rate, mean_locality, mean_portability, sequential_stability
from .base import EvaluationResult, load_config


def _to_outcomes(df: pd.DataFrame) -> List[EditOutcome]:
    outcomes: List[EditOutcome] = []
    for _, row in df.iterrows():
        outcomes.append(
            EditOutcome(
                success=bool(row.get("success")),
                locality=float(row.get("locality", 0.0)),
                portability=float(row.get("portability", 0.0)),
            )
        )
    return outcomes


def evaluate_editing(config_path: Path, results: Mapping[str, pd.DataFrame]) -> List[EvaluationResult]:
    config = load_config(config_path)
    metrics = config.get("metrics", [])
    seq_budget = config.get("sequential_budget", [])

    evaluations: List[EvaluationResult] = []
    for dataset_cfg in config.get("datasets", []):
        dataset = dataset_cfg["name"]
        for split in dataset_cfg.get("splits", ["test"]):
            key = f"{dataset}:{split}"
            if key not in results:
                raise KeyError(f"Missing editing results for {key}")
            df = results[key]
            outcomes = _to_outcomes(df)
            metric_values: Dict[str, float] = {}

            if "edit_success" in metrics:
                metric_values["edit_success"] = edit_success_rate(outcomes)
            if "locality" in metrics:
                metric_values["locality"] = mean_locality(outcomes)
            if "portability" in metrics:
                metric_values["portability"] = mean_portability(outcomes)
            if "sequential_stability" in metrics and seq_budget:
                batches: List[Sequence[EditOutcome]] = []
                cursor = 0
                for budget in seq_budget:
                    batches.append(outcomes[cursor : cursor + budget])
                    cursor += budget
                metric_values["sequential_stability"] = sequential_stability(batches)[-1] if batches else 0.0

            evaluations.append(EvaluationResult(name=key, metrics=metric_values))

    return evaluations
