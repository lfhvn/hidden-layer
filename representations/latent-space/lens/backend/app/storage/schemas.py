"""SQLModel schemas for database tables."""

from datetime import datetime
from typing import Optional, List
from enum import Enum

from sqlmodel import Field, SQLModel, Relationship, JSON, Column
from pydantic import ConfigDict


class ExperimentStatus(str, Enum):
    """Status of an experiment."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Experiment(SQLModel, table=True):
    """Experiment tracking for SAE training runs.

    Each experiment represents one training run with specific hyperparameters.
    """

    __tablename__ = "experiments"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: Optional[str] = None

    # Model configuration
    model_name: str
    layer_name: str
    layer_index: int

    # SAE configuration
    input_dim: int
    hidden_dim: int
    sparsity_coef: float
    learning_rate: float
    num_epochs: int

    # Training metadata
    status: ExperimentStatus = Field(default=ExperimentStatus.PENDING)
    num_samples: int = 0
    train_loss: Optional[float] = None
    reconstruction_loss: Optional[float] = None
    sparsity_loss: Optional[float] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Relationships
    features: List["Feature"] = Relationship(back_populates="experiment")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Feature(SQLModel, table=True):
    """Discovered feature from SAE training.

    Each feature represents one learned dimension in the SAE hidden layer.
    """

    __tablename__ = "features"

    id: Optional[int] = Field(default=None, primary_key=True)
    experiment_id: int = Field(foreign_key="experiments.id", index=True)
    feature_index: int = Field(index=True)  # Index in SAE hidden layer

    # Statistics
    activation_mean: float
    activation_max: float
    activation_std: float
    sparsity: float  # Fraction of samples where feature is active
    l0_norm: float = 0.0  # Average number of active positions

    # Interpretability
    top_tokens: str = Field(default="[]", sa_column=Column(JSON))  # JSON list
    top_token_scores: str = Field(default="[]", sa_column=Column(JSON))  # JSON list

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    experiment: Optional[Experiment] = Relationship(back_populates="features")
    labels: List["FeatureLabel"] = Relationship(back_populates="feature")
    activations: List["FeatureActivation"] = Relationship(back_populates="feature")
    groups: List["FeatureGroup"] = Relationship(
        back_populates="features", link_model="feature_group_link"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FeatureLabel(SQLModel, table=True):
    """User-provided labels and annotations for features."""

    __tablename__ = "feature_labels"

    id: Optional[int] = Field(default=None, primary_key=True)
    feature_id: int = Field(foreign_key="features.id", index=True)

    # Label information
    label: str  # Short label/name
    description: Optional[str] = None  # Longer description
    tags: str = Field(default="[]", sa_column=Column(JSON))  # JSON list of tags

    # Metadata
    confidence: float = 1.0  # User confidence in label (0-1)
    created_by: Optional[str] = None  # User ID or name
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    feature: Optional[Feature] = Relationship(back_populates="labels")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FeatureActivation(SQLModel, table=True):
    """Record of when and how strongly a feature activated.

    Used for generating examples and understanding feature behavior.
    """

    __tablename__ = "feature_activations"

    id: Optional[int] = Field(default=None, primary_key=True)
    feature_id: int = Field(foreign_key="features.id", index=True)

    # Context
    text: str  # Text snippet that caused activation
    token_index: int  # Which token in the text
    activation_value: float  # How strongly the feature activated

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    feature: Optional[Feature] = Relationship(back_populates="activations")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FeatureGroupLink(SQLModel, table=True):
    """Many-to-many link between features and groups."""

    __tablename__ = "feature_group_link"

    feature_id: int = Field(foreign_key="features.id", primary_key=True)
    group_id: int = Field(foreign_key="feature_groups.id", primary_key=True)


class FeatureGroup(SQLModel, table=True):
    """Groups of related features.

    Users can group features that seem to represent similar concepts.
    """

    __tablename__ = "feature_groups"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: Optional[str] = None

    # Metadata
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    features: List[Feature] = Relationship(
        back_populates="groups", link_model=FeatureGroupLink
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
