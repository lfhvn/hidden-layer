"""Metrics and monitoring routes."""

from fastapi import APIRouter
from typing import List

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/adherence/{exp_id}")
async def get_adherence_metrics(exp_id: int):
    return {
        "experiment_id": exp_id,
        "mean_score": 0.82,
        "success_rate": 0.75,
        "total_generations": 100,
    }


@router.get("/summary")
async def get_metrics_summary():
    return {
        "total_experiments": 5,
        "mean_adherence": 0.80,
        "active_vectors": ["positive", "formal"],
    }


@router.get("/timeseries/{exp_id}")
async def get_time_series(exp_id: int):
    return {"timestamps": [], "scores": []}
