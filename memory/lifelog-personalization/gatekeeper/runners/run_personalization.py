"""Runner for personalization benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping

import pandas as pd

from ..scorers.prefs import PreferenceAdherenceScorer, PreferenceJudgement, preference_drift
from ..scorers.temporal import forgetting_curve
from .base import EvaluationResult, load_config


def _default_judge(prompt: str) -> PreferenceJudgement:
    raise RuntimeError("LLM judge not configured. Provide a judge_fn when calling evaluate_personalization().")


def evaluate_personalization(
    config_path: Path,
    predictions: Mapping[str, pd.DataFrame],
    judge_fn=_default_judge,
    rubric_prompt: str | None = None,
) -> List[EvaluationResult]:
    config = load_config(config_path)
    metrics = config.get("metrics", [])
    rubric_prompt = rubric_prompt or config.get("judges", {}).get("rubric_prompt", "Provide a binary adherence judgement (0-1)")

    scorer = PreferenceAdherenceScorer(rubric_prompt=rubric_prompt, judge_fn=judge_fn)
    results: List[EvaluationResult] = []

    for dataset_cfg in config.get("datasets", []):
        name = dataset_cfg["name"]
        for split in dataset_cfg.get("splits", ["test"]):
            key = f"{name}:{split}"
            if key not in predictions:
                raise KeyError(f"Missing predictions for {key}")
            df = predictions[key]
            metric_values: Dict[str, float] = {}

            if "preference_adherence" in metrics:
                adherence_df = scorer.evaluate(df)
                metric_values["preference_adherence"] = float(adherence_df["score"].mean())
                df = df.copy()
                df["adherence_score"] = adherence_df["score"]

            if "preference_drift" in metrics:
                if "adherence_score" not in df:
                    raise ValueError("preference_drift requires adherence scores. Enable preference_adherence metric first.")
                metric_values["preference_drift"] = preference_drift(
                    df["adherence_score"],
                    df.get("session_id", range(len(df))),
                )

            if "forgetting_curve" in metrics:
                curve = forgetting_curve(df.get("accuracy", []))
                metric_values["forgetting_curve"] = curve[-1] if curve else 0.0

            if "task_metrics" in metrics:
                metric_values["task_metrics"] = float(df.get("accuracy", [0]).mean()) if "accuracy" in df else float(df.get("f1", [0]).mean() if "f1" in df else 0.0)

            results.append(EvaluationResult(name=key, metrics=metric_values))

    return results
