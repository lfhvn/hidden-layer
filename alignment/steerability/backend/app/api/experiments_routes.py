"""Experiment management routes."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/experiments", tags=["experiments"])


class ExperimentCreate(BaseModel):
    name: str
    model_name: str
    layer_index: int
    vector_name: str
    strength: float = 1.0


@router.post("/")
async def create_experiment(exp: ExperimentCreate):
    return {"id": 1, **exp.dict(), "status": "active"}


@router.get("/")
async def list_experiments():
    return {"experiments": []}


@router.get("/{exp_id}")
async def get_experiment(exp_id: int):
    return {"id": exp_id, "name": "Test", "status": "active"}
