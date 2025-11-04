"""Activation analysis API routes."""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ...models.sae import SparseAutoencoder
from ...services import SAEService
from ..dependencies import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/activations", tags=["activations"])


class AnalyzeRequest(BaseModel):
    """Request model for analyzing text."""

    text: str
    experiment_id: int
    top_k: int = 10


class FeatureActivation(BaseModel):
    """Feature activation for a specific token."""

    feature_id: int
    activation_value: float


class TokenActivation(BaseModel):
    """Activations for a single token."""

    token: str
    token_index: int
    features: List[FeatureActivation]


class AnalyzeResponse(BaseModel):
    """Response model for text analysis."""

    text: str
    tokens: List[TokenActivation]
    top_features: List[tuple]


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(
    request: AnalyzeRequest,
    api_key: str = Depends(verify_api_key),
) -> AnalyzeResponse:
    """Analyze text and return which features activate.

    This endpoint:
    1. Loads the trained SAE for the experiment
    2. Processes the text through the model
    3. Captures activations
    4. Runs through SAE to identify active features
    5. Returns token-level feature activations
    """
    try:
        # Load SAE
        service = SAEService()
        checkpoint_path = service.checkpoint_dir / f"sae_exp_{request.experiment_id}.pt"

        if not checkpoint_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No trained SAE found for experiment {request.experiment_id}",
            )

        sae = SparseAutoencoder.load(str(checkpoint_path))

        # TODO: Implement full pipeline
        # For now, return mock response
        return AnalyzeResponse(
            text=request.text,
            tokens=[
                TokenActivation(
                    token="sample",
                    token_index=0,
                    features=[
                        FeatureActivation(feature_id=1, activation_value=0.5)
                    ],
                )
            ],
            top_features=[(1, 0.5)],
        )

    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/top-features")
async def get_top_features(
    experiment_id: int,
    k: int = 20,
    api_key: str = Depends(verify_api_key),
) -> dict:
    """Get top-k most frequently active features for an experiment."""
    # TODO: Implement based on stored activations
    return {
        "experiment_id": experiment_id,
        "top_features": [],
        "message": "Not yet implemented",
    }
