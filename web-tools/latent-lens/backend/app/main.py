"""
Latent Lens Backend - SAE Feature Explorer API

Read-only API for browsing pre-trained sparse autoencoder features.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Add paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "web-tools" / "shared" / "backend"))

from middleware import setup_all_middleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(
    title="Latent Lens API",
    description="Explore sparse autoencoder features",
    version="0.1.0"
)

# Setup middleware
setup_all_middleware(app)


# Models
class SAEFeature(BaseModel):
    """A discovered SAE feature."""
    id: str
    description: str
    category: Optional[str] = None
    activation_examples: List[str]
    statistics: Dict[str, float]
    related_features: List[str] = []


class AnalyzeRequest(BaseModel):
    """Request to analyze text."""
    text: str
    top_k: int = 10


class FeatureActivation(BaseModel):
    """Feature activation in text."""
    feature_id: str
    feature_description: str
    activation_strength: float
    relevant_spans: List[str] = []


# In-memory feature storage (loaded at startup)
FEATURES: Dict[str, SAEFeature] = {}


def load_sample_features():
    """Load sample features for demonstration."""
    return {
        "feat_001": SAEFeature(
            id="feat_001",
            description="City names and geographic locations",
            category="geography",
            activation_examples=[
                "I visited Paris last summer.",
                "New York is a vibrant city.",
                "Tokyo has amazing food culture."
            ],
            statistics={
                "mean_activation": 0.65,
                "max_activation": 0.95,
                "frequency": 0.012
            },
            related_features=["feat_015", "feat_023"]
        ),
        "feat_002": SAEFeature(
            id="feat_002",
            description="Positive sentiment and enthusiastic language",
            category="sentiment",
            activation_examples=[
                "I absolutely love this!",
                "This is amazing and wonderful!",
                "What a fantastic experience!"
            ],
            statistics={
                "mean_activation": 0.72,
                "max_activation": 0.98,
                "frequency": 0.085
            },
            related_features=["feat_005", "feat_019"]
        ),
        "feat_003": SAEFeature(
            id="feat_003",
            description="Technical programming terminology",
            category="technical",
            activation_examples=[
                "The function returns a boolean value.",
                "We need to refactor this code.",
                "The API endpoint accepts JSON payloads."
            ],
            statistics={
                "mean_activation": 0.58,
                "max_activation": 0.89,
                "frequency": 0.034
            },
            related_features=["feat_012", "feat_027"]
        ),
        "feat_004": SAEFeature(
            id="feat_004",
            description="First-person narrative perspective",
            category="perspective",
            activation_examples=[
                "I think that we should consider...",
                "In my experience, this works well.",
                "I've found that..."
            ],
            statistics={
                "mean_activation": 0.81,
                "max_activation": 0.97,
                "frequency": 0.156
            },
            related_features=["feat_008"]
        ),
        "feat_005": SAEFeature(
            id="feat_005",
            description="Temporal references and time expressions",
            category="temporal",
            activation_examples=[
                "Yesterday we went to the park.",
                "Next week I'll start the project.",
                "It happened three years ago."
            ],
            statistics={
                "mean_activation": 0.69,
                "max_activation": 0.93,
                "frequency": 0.091
            },
            related_features=["feat_013"]
        ),
        "feat_006": SAEFeature(
            id="feat_006",
            description="Questions and interrogative structures",
            category="syntax",
            activation_examples=[
                "What do you think about this?",
                "How does this work?",
                "Why did that happen?"
            ],
            statistics={
                "mean_activation": 0.76,
                "max_activation": 0.96,
                "frequency": 0.067
            },
            related_features=["feat_018"]
        ),
        "feat_007": SAEFeature(
            id="feat_007",
            description="Negative sentiment and criticism",
            category="sentiment",
            activation_examples=[
                "This is terrible and disappointing.",
                "I really dislike this approach.",
                "What a waste of time."
            ],
            statistics={
                "mean_activation": 0.68,
                "max_activation": 0.94,
                "frequency": 0.045
            },
            related_features=["feat_002"]
        ),
        "feat_008": SAEFeature(
            id="feat_008",
            description="Numbers and quantitative information",
            category="quantitative",
            activation_examples=[
                "There are 42 items in the list.",
                "The price is $19.99.",
                "We need approximately 1000 samples."
            ],
            statistics={
                "mean_activation": 0.63,
                "max_activation": 0.88,
                "frequency": 0.102
            },
            related_features=["feat_021"]
        )
    }


@app.on_event("startup")
async def startup_event():
    """Load features on startup."""
    global FEATURES

    # Try to load from file, fall back to sample features
    features_path = Path(__file__).parent / "features.json"

    if features_path.exists():
        logger.info(f"Loading features from {features_path}")
        with open(features_path, 'r') as f:
            data = json.load(f)
            FEATURES = {f["id"]: SAEFeature(**f) for f in data}
    else:
        logger.info("Loading sample features")
        FEATURES = load_sample_features()

    logger.info(f"Loaded {len(FEATURES)} features")


@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "service": "latent-lens",
        "version": "0.1.0",
        "features_loaded": len(FEATURES)
    }


@app.get("/api/features", response_model=List[SAEFeature])
async def list_features(
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(100, le=1000, description="Max features to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """List all features with optional filtering."""
    features = list(FEATURES.values())

    # Filter by category if specified
    if category:
        features = [f for f in features if f.category == category]

    # Paginate
    features = features[offset:offset + limit]

    return features


@app.get("/api/features/{feature_id}", response_model=SAEFeature)
async def get_feature(feature_id: str):
    """Get detailed information about a specific feature."""
    if feature_id not in FEATURES:
        raise HTTPException(status_code=404, detail="Feature not found")

    return FEATURES[feature_id]


@app.get("/api/categories")
async def list_categories():
    """List all available feature categories."""
    categories = set(f.category for f in FEATURES.values() if f.category)
    return {"categories": sorted(list(categories))}


@app.get("/api/search")
async def search_features(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, le=100)
):
    """Search features by description."""
    query = q.lower()
    results = []

    for feature in FEATURES.values():
        # Simple text matching
        if query in feature.description.lower():
            results.append(feature)

            if len(results) >= limit:
                break

    return {"query": q, "count": len(results), "features": results}


@app.post("/api/analyze", response_model=List[FeatureActivation])
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze text and return activated features.

    This is a simplified version - in a real implementation,
    we would run the text through the SAE to get actual activations.
    """
    text = request.text.lower()
    activations = []

    # Simple pattern matching for demonstration
    # In real implementation, use actual SAE inference
    for feature in FEATURES.values():
        # Check if any activation examples' patterns appear in text
        activation_score = 0.0

        # Simple heuristic: check for keywords
        keywords = extract_keywords(feature.description)
        for keyword in keywords:
            if keyword.lower() in text:
                activation_score += 0.3

        # Check activation examples
        for example in feature.activation_examples:
            example_words = set(example.lower().split())
            text_words = set(text.split())
            overlap = len(example_words & text_words)
            if overlap > 0:
                activation_score += overlap * 0.1

        if activation_score > 0:
            activations.append(FeatureActivation(
                feature_id=feature.id,
                feature_description=feature.description,
                activation_strength=min(activation_score, 1.0),
                relevant_spans=[]  # Would be filled by actual SAE
            ))

    # Sort by activation strength and return top_k
    activations.sort(key=lambda x: x.activation_strength, reverse=True)
    return activations[:request.top_k]


def extract_keywords(description: str) -> List[str]:
    """Extract keywords from feature description."""
    # Remove common words
    stop_words = {"and", "the", "a", "an", "or", "in", "on", "at", "to", "for"}
    words = description.lower().split()
    return [w for w in words if w not in stop_words and len(w) > 3]


@app.get("/api/stats")
async def get_statistics():
    """Get overall statistics about the feature library."""
    total_features = len(FEATURES)
    categories = {}

    for feature in FEATURES.values():
        if feature.category:
            categories[feature.category] = categories.get(feature.category, 0) + 1

    return {
        "total_features": total_features,
        "categories": categories,
        "avg_examples_per_feature": sum(len(f.activation_examples) for f in FEATURES.values()) / total_features if total_features > 0 else 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
