"""Service for feature management and labeling."""

import logging
import json
from typing import List, Optional, Dict, Any

from sqlmodel import select

from ..storage import Feature, FeatureLabel, FeatureGroup, get_session
from ..models.feature_extractor import ExtractedFeature

logger = logging.getLogger(__name__)


class FeatureService:
    """Service for managing features, labels, and groups."""

    def save_features(
        self, experiment_id: int, extracted_features: Dict[int, ExtractedFeature]
    ) -> List[Feature]:
        """Save extracted features to database.

        Args:
            experiment_id: ID of experiment
            extracted_features: Dictionary of feature_id -> ExtractedFeature

        Returns:
            List of saved Feature records
        """
        features = []

        with get_session() as session:
            for feature_id, extracted in extracted_features.items():
                feature = Feature(
                    experiment_id=experiment_id,
                    feature_index=extracted.feature_id,
                    activation_mean=extracted.activation_mean,
                    activation_max=extracted.activation_max,
                    activation_std=extracted.activation_std,
                    sparsity=extracted.sparsity,
                    top_tokens=json.dumps(extracted.top_tokens),
                    top_token_scores=json.dumps(extracted.top_token_scores),
                )

                session.add(feature)
                features.append(feature)

            session.commit()

            # Refresh to get IDs
            for feature in features:
                session.refresh(feature)

        logger.info(f"Saved {len(features)} features for experiment {experiment_id}")

        return features

    def get_features(
        self,
        experiment_id: Optional[int] = None,
        feature_ids: Optional[List[int]] = None,
        min_sparsity: Optional[float] = None,
        max_sparsity: Optional[float] = None,
        limit: int = 100,
    ) -> List[Feature]:
        """Query features from database.

        Args:
            experiment_id: Filter by experiment
            feature_ids: Filter by specific feature IDs
            min_sparsity: Minimum sparsity threshold
            max_sparsity: Maximum sparsity threshold
            limit: Maximum number of results

        Returns:
            List of Feature records
        """
        with get_session() as session:
            query = select(Feature)

            if experiment_id is not None:
                query = query.where(Feature.experiment_id == experiment_id)

            if feature_ids is not None:
                query = query.where(Feature.id.in_(feature_ids))

            if min_sparsity is not None:
                query = query.where(Feature.sparsity >= min_sparsity)

            if max_sparsity is not None:
                query = query.where(Feature.sparsity <= max_sparsity)

            query = query.limit(limit)

            features = session.exec(query).all()

        return list(features)

    def add_label(
        self,
        feature_id: int,
        label: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        confidence: float = 1.0,
        created_by: Optional[str] = None,
    ) -> FeatureLabel:
        """Add label to a feature.

        Args:
            feature_id: ID of feature
            label: Short label
            description: Longer description
            tags: List of tags
            confidence: Confidence score (0-1)
            created_by: User identifier

        Returns:
            FeatureLabel record
        """
        feature_label = FeatureLabel(
            feature_id=feature_id,
            label=label,
            description=description,
            tags=json.dumps(tags or []),
            confidence=confidence,
            created_by=created_by,
        )

        with get_session() as session:
            session.add(feature_label)
            session.commit()
            session.refresh(feature_label)

        logger.info(f"Added label '{label}' to feature {feature_id}")

        return feature_label

    def create_group(
        self,
        name: str,
        feature_ids: List[int],
        description: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> FeatureGroup:
        """Create a group of related features.

        Args:
            name: Group name
            feature_ids: List of feature IDs to include
            description: Group description
            created_by: User identifier

        Returns:
            FeatureGroup record
        """
        with get_session() as session:
            # Create group
            group = FeatureGroup(
                name=name,
                description=description,
                created_by=created_by,
            )

            session.add(group)
            session.commit()
            session.refresh(group)

            # Add features to group
            features = session.exec(select(Feature).where(Feature.id.in_(feature_ids))).all()

            for feature in features:
                group.features.append(feature)

            session.add(group)
            session.commit()

        logger.info(f"Created group '{name}' with {len(feature_ids)} features")

        return group

    def get_feature_labels(self, feature_id: int) -> List[FeatureLabel]:
        """Get all labels for a feature.

        Args:
            feature_id: ID of feature

        Returns:
            List of FeatureLabel records
        """
        with get_session() as session:
            labels = session.exec(
                select(FeatureLabel).where(FeatureLabel.feature_id == feature_id)
            ).all()

        return list(labels)

    def export_labeled_features(self, experiment_id: int) -> Dict[str, Any]:
        """Export all labeled features for an experiment.

        Args:
            experiment_id: ID of experiment

        Returns:
            Dictionary with labeled features data
        """
        features = self.get_features(experiment_id=experiment_id, limit=10000)

        export_data = {
            "experiment_id": experiment_id,
            "num_features": len(features),
            "features": [],
        }

        for feature in features:
            labels = self.get_feature_labels(feature.id)

            feature_data = {
                "feature_id": feature.id,
                "feature_index": feature.feature_index,
                "statistics": {
                    "activation_mean": feature.activation_mean,
                    "activation_max": feature.activation_max,
                    "sparsity": feature.sparsity,
                },
                "top_tokens": json.loads(feature.top_tokens),
                "labels": [
                    {
                        "label": label.label,
                        "description": label.description,
                        "tags": json.loads(label.tags),
                        "confidence": label.confidence,
                    }
                    for label in labels
                ],
            }

            export_data["features"].append(feature_data)

        return export_data
