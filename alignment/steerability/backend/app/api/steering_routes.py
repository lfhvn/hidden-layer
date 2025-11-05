"""Steering control API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter(prefix="/steering", tags=["steering"])


class SteerRequest(BaseModel):
    prompt: str
    vector_name: str
    layer_index: int = 6
    strength: float = 1.0
    max_length: int = 50
    temperature: float = 0.7


class SteerResponse(BaseModel):
    steered_output: str
    unsteered_output: Optional[str] = None
    adherence_score: float
    constraints_satisfied: bool


@router.post("/generate", response_model=SteerResponse)
async def steer_generate(request: SteerRequest):
    """Generate text with steering."""
    # Implementation would use SteeringEngine
    return SteerResponse(
        steered_output=f"Steered: {request.prompt}...",
        unsteered_output=f"Unsteered: {request.prompt}...",
        adherence_score=0.85,
        constraints_satisfied=True,
    )


@router.get("/vectors")
async def list_vectors():
    """List available steering vectors."""
    return {"vectors": ["positive_sentiment", "formal_tone", "concise"]}


@router.post("/vectors")
async def create_vector(name: str, description: str):
    """Create new steering vector."""
    return {"name": name, "description": description, "status": "created"}
