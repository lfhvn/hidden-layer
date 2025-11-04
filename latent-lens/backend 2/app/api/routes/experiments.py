"""Experiment management API routes."""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import select

from ...storage import Experiment, get_session
from ...models.sae import SAETrainingConfig
from ...services import SAEService
from ..dependencies import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/experiments", tags=["experiments"])


class ExperimentCreate(BaseModel):
    """Request model for creating an experiment."""

    name: str
    description: Optional[str] = None
    model_name: str
    layer_name: str
    layer_index: int
    input_dim: int
    hidden_dim: int = 4096
    sparsity_coef: float = 0.01
    learning_rate: float = 1e-3
    num_epochs: int = 10


class ExperimentResponse(BaseModel):
    """Response model for experiment."""

    id: int
    name: str
    description: Optional[str]
    model_name: str
    layer_name: str
    layer_index: int
    input_dim: int
    hidden_dim: int
    sparsity_coef: float
    learning_rate: float
    num_epochs: int
    status: str
    num_samples: int
    train_loss: Optional[float]
    reconstruction_loss: Optional[float]
    sparsity_loss: Optional[float]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    class Config:
        from_attributes = True


@router.post("/", response_model=ExperimentResponse)
async def create_experiment(
    experiment: ExperimentCreate,
    api_key: str = Depends(verify_api_key),
) -> ExperimentResponse:
    """Create a new SAE training experiment.

    This creates a database record for tracking the experiment.
    Use the training endpoints to actually train the SAE.
    """
    config = SAETrainingConfig(
        input_dim=experiment.input_dim,
        hidden_dim=experiment.hidden_dim,
        sparsity_coef=experiment.sparsity_coef,
        learning_rate=experiment.learning_rate,
        num_epochs=experiment.num_epochs,
    )

    service = SAEService()
    exp = service.create_experiment(
        name=experiment.name,
        description=experiment.description,
        model_name=experiment.model_name,
        layer_name=experiment.layer_name,
        layer_index=experiment.layer_index,
        config=config,
    )

    return ExperimentResponse(
        id=exp.id,
        name=exp.name,
        description=exp.description,
        model_name=exp.model_name,
        layer_name=exp.layer_name,
        layer_index=exp.layer_index,
        input_dim=exp.input_dim,
        hidden_dim=exp.hidden_dim,
        sparsity_coef=exp.sparsity_coef,
        learning_rate=exp.learning_rate,
        num_epochs=exp.num_epochs,
        status=exp.status,
        num_samples=exp.num_samples,
        train_loss=exp.train_loss,
        reconstruction_loss=exp.reconstruction_loss,
        sparsity_loss=exp.sparsity_loss,
        created_at=exp.created_at.isoformat(),
        started_at=exp.started_at.isoformat() if exp.started_at else None,
        completed_at=exp.completed_at.isoformat() if exp.completed_at else None,
    )


@router.get("/", response_model=List[ExperimentResponse])
async def list_experiments(
    limit: int = 100,
    api_key: str = Depends(verify_api_key),
) -> List[ExperimentResponse]:
    """List all experiments."""
    with get_session() as session:
        experiments = session.exec(select(Experiment).limit(limit)).all()

        return [
            ExperimentResponse(
                id=exp.id,
                name=exp.name,
                description=exp.description,
                model_name=exp.model_name,
                layer_name=exp.layer_name,
                layer_index=exp.layer_index,
                input_dim=exp.input_dim,
                hidden_dim=exp.hidden_dim,
                sparsity_coef=exp.sparsity_coef,
                learning_rate=exp.learning_rate,
                num_epochs=exp.num_epochs,
                status=exp.status,
                num_samples=exp.num_samples,
                train_loss=exp.train_loss,
                reconstruction_loss=exp.reconstruction_loss,
                sparsity_loss=exp.sparsity_loss,
                created_at=exp.created_at.isoformat(),
                started_at=exp.started_at.isoformat() if exp.started_at else None,
                completed_at=exp.completed_at.isoformat() if exp.completed_at else None,
            )
            for exp in experiments
        ]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: int,
    api_key: str = Depends(verify_api_key),
) -> ExperimentResponse:
    """Get a specific experiment by ID."""
    with get_session() as session:
        exp = session.get(Experiment, experiment_id)

        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")

        return ExperimentResponse(
            id=exp.id,
            name=exp.name,
            description=exp.description,
            model_name=exp.model_name,
            layer_name=exp.layer_name,
            layer_index=exp.layer_index,
            input_dim=exp.input_dim,
            hidden_dim=exp.hidden_dim,
            sparsity_coef=exp.sparsity_coef,
            learning_rate=exp.learning_rate,
            num_epochs=exp.num_epochs,
            status=exp.status,
            num_samples=exp.num_samples,
            train_loss=exp.train_loss,
            reconstruction_loss=exp.reconstruction_loss,
            sparsity_loss=exp.sparsity_loss,
            created_at=exp.created_at.isoformat(),
            started_at=exp.started_at.isoformat() if exp.started_at else None,
            completed_at=exp.completed_at.isoformat() if exp.completed_at else None,
        )


@router.delete("/{experiment_id}")
async def delete_experiment(
    experiment_id: int,
    api_key: str = Depends(verify_api_key),
) -> dict:
    """Delete an experiment."""
    with get_session() as session:
        exp = session.get(Experiment, experiment_id)

        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")

        session.delete(exp)
        session.commit()

    logger.info(f"Deleted experiment {experiment_id}")

    return {"message": "Experiment deleted successfully"}
