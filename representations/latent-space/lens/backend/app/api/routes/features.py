"""Feature management API routes."""

import logging
import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ...storage import Feature, get_session
from ...services import FeatureService
from ..dependencies import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/features", tags=["features"])


class FeatureResponse(BaseModel):
    """Response model for feature."""

    id: int
    experiment_id: int
    feature_index: int
    activation_mean: float
    activation_max: float
    activation_std: float
    sparsity: float
    top_tokens: List[str]
    top_token_scores: List[float]
    created_at: str

    class Config:
        from_attributes = True


class LabelCreate(BaseModel):
    """Request model for creating a label."""

    label: str
    description: Optional[str] = None
    tags: List[str] = []
    confidence: float = 1.0
    created_by: Optional[str] = None


class LabelResponse(BaseModel):
    """Response model for label."""

    id: int
    feature_id: int
    label: str
    description: Optional[str]
    tags: List[str]
    confidence: float
    created_by: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


class GroupCreate(BaseModel):
    """Request model for creating a feature group."""

    name: str
    description: Optional[str] = None
    feature_ids: List[int]
    created_by: Optional[str] = None


@router.get("/", response_model=List[FeatureResponse])
async def list_features(
    experiment_id: Optional[int] = Query(None),
    min_sparsity: Optional[float] = Query(None),
    max_sparsity: Optional[float] = Query(None),
    limit: int = Query(100, le=1000),
    api_key: str = Depends(verify_api_key),
) -> List[FeatureResponse]:
    """List features with optional filtering.

    Args:
        experiment_id: Filter by experiment
        min_sparsity: Minimum sparsity threshold
        max_sparsity: Maximum sparsity threshold
        limit: Maximum number of results
    """
    service = FeatureService()
    features = service.get_features(
        experiment_id=experiment_id,
        min_sparsity=min_sparsity,
        max_sparsity=max_sparsity,
        limit=limit,
    )

    return [
        FeatureResponse(
            id=f.id,
            experiment_id=f.experiment_id,
            feature_index=f.feature_index,
            activation_mean=f.activation_mean,
            activation_max=f.activation_max,
            activation_std=f.activation_std,
            sparsity=f.sparsity,
            top_tokens=json.loads(f.top_tokens),
            top_token_scores=json.loads(f.top_token_scores),
            created_at=f.created_at.isoformat(),
        )
        for f in features
    ]


@router.get("/{feature_id}", response_model=FeatureResponse)
async def get_feature(
    feature_id: int,
    api_key: str = Depends(verify_api_key),
) -> FeatureResponse:
    """Get a specific feature by ID."""
    with get_session() as session:
        feature = session.get(Feature, feature_id)

        if not feature:
            raise HTTPException(status_code=404, detail="Feature not found")

        return FeatureResponse(
            id=feature.id,
            experiment_id=feature.experiment_id,
            feature_index=feature.feature_index,
            activation_mean=feature.activation_mean,
            activation_max=feature.activation_max,
            activation_std=feature.activation_std,
            sparsity=feature.sparsity,
            top_tokens=json.loads(feature.top_tokens),
            top_token_scores=json.loads(feature.top_token_scores),
            created_at=feature.created_at.isoformat(),
        )


@router.post("/{feature_id}/labels", response_model=LabelResponse)
async def add_label(
    feature_id: int,
    label_data: LabelCreate,
    api_key: str = Depends(verify_api_key),
) -> LabelResponse:
    """Add a label to a feature."""
    service = FeatureService()

    label = service.add_label(
        feature_id=feature_id,
        label=label_data.label,
        description=label_data.description,
        tags=label_data.tags,
        confidence=label_data.confidence,
        created_by=label_data.created_by,
    )

    return LabelResponse(
        id=label.id,
        feature_id=label.feature_id,
        label=label.label,
        description=label.description,
        tags=json.loads(label.tags),
        confidence=label.confidence,
        created_by=label.created_by,
        created_at=label.created_at.isoformat(),
    )


@router.get("/{feature_id}/labels", response_model=List[LabelResponse])
async def get_labels(
    feature_id: int,
    api_key: str = Depends(verify_api_key),
) -> List[LabelResponse]:
    """Get all labels for a feature."""
    service = FeatureService()
    labels = service.get_feature_labels(feature_id)

    return [
        LabelResponse(
            id=label.id,
            feature_id=label.feature_id,
            label=label.label,
            description=label.description,
            tags=json.loads(label.tags),
            confidence=label.confidence,
            created_by=label.created_by,
            created_at=label.created_at.isoformat(),
        )
        for label in labels
    ]


@router.post("/groups")
async def create_group(
    group_data: GroupCreate,
    api_key: str = Depends(verify_api_key),
) -> dict:
    """Create a group of related features."""
    service = FeatureService()

    group = service.create_group(
        name=group_data.name,
        feature_ids=group_data.feature_ids,
        description=group_data.description,
        created_by=group_data.created_by,
    )

    return {
        "id": group.id,
        "name": group.name,
        "description": group.description,
        "num_features": len(group_data.feature_ids),
        "created_at": group.created_at.isoformat(),
    }


@router.get("/experiments/{experiment_id}/export")
async def export_features(
    experiment_id: int,
    api_key: str = Depends(verify_api_key),
) -> dict:
    """Export all labeled features for an experiment."""
    service = FeatureService()
    export_data = service.export_labeled_features(experiment_id)

    return export_data
