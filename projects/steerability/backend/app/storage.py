"""Database storage schemas and management."""

from datetime import datetime
from typing import Optional, List
from enum import Enum

from sqlmodel import Field, SQLModel, Relationship, JSON, Column


class ExperimentStatus(str, Enum):
    """Status of steering experiment."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class SteeringExperiment(SQLModel, table=True):
    """Steering experiment configuration."""

    __tablename__ = "steering_experiments"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: Optional[str] = None

    # Model config
    model_name: str
    layer_index: int

    # Steering config
    vector_name: str
    strength: float = 1.0
    method: str = "add"

    # Status
    status: ExperimentStatus = Field(default=ExperimentStatus.ACTIVE)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    # Relationships
    results: List["SteeringResult"] = Relationship(back_populates="experiment")
    metrics: List["AdherenceRecord"] = Relationship(back_populates="experiment")


class SteeringResult(SQLModel, table=True):
    """Result from steering generation."""

    __tablename__ = "steering_results"

    id: Optional[int] = Field(default=None, primary_key=True)
    experiment_id: int = Field(foreign_key="steering_experiments.id", index=True)

    # Input/Output
    prompt: str
    steered_output: str
    unsteered_output: Optional[str] = None

    # Metrics
    adherence_score: float
    constraints_satisfied: bool

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: str = Field(default="{}", sa_column=Column(JSON))

    # Relationships
    experiment: Optional[SteeringExperiment] = Relationship(back_populates="results")


class AdherenceRecord(SQLModel, table=True):
    """Time-series adherence metrics."""

    __tablename__ = "adherence_records"

    id: Optional[int] = Field(default=None, primary_key=True)
    experiment_id: int = Field(foreign_key="steering_experiments.id", index=True)

    # Metrics
    adherence_score: float
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)

    # Relationships
    experiment: Optional[SteeringExperiment] = Relationship(back_populates="metrics")
