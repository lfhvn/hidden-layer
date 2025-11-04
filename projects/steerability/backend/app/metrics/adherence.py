"""Adherence metrics for measuring steering effectiveness."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AdherenceMetrics:
    """Metrics tracking steering adherence over time."""

    experiment_id: int
    steering_vector_name: str

    # Time series of adherence scores
    scores: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Constraint satisfaction tracking
    constraint_satisfaction: Dict[str, List[bool]] = field(default_factory=dict)

    # Summary statistics
    total_generations: int = 0
    successful_generations: int = 0

    def add_score(self, score: float, constraints_met: Optional[Dict[str, bool]] = None):
        """Add an adherence score.

        Args:
            score: Adherence score (0-1)
            constraints_met: Dictionary of constraint names and whether they were met
        """
        self.scores.append(score)
        self.total_generations += 1

        if score >= 0.8:  # Consider 80%+ as successful
            self.successful_generations += 1

        # Track constraint satisfaction
        if constraints_met:
            for constraint_name, met in constraints_met.items():
                if constraint_name not in self.constraint_satisfaction:
                    self.constraint_satisfaction[constraint_name] = []
                self.constraint_satisfaction[constraint_name].append(met)

    @property
    def mean_score(self) -> float:
        """Calculate mean adherence score."""
        if not self.scores:
            return 0.0
        return float(np.mean(self.scores))

    @property
    def std_score(self) -> float:
        """Calculate standard deviation of adherence scores."""
        if not self.scores:
            return 0.0
        return float(np.std(self.scores))

    @property
    def success_rate(self) -> float:
        """Calculate success rate (fraction with score >= 0.8)."""
        if self.total_generations == 0:
            return 0.0
        return self.successful_generations / self.total_generations

    @property
    def recent_scores(self, n: int = 10) -> List[float]:
        """Get most recent n scores."""
        return list(self.scores)[-n:]

    def moving_average(self, window: int = 10) -> List[float]:
        """Calculate moving average of adherence scores.

        Args:
            window: Window size for moving average

        Returns:
            List of moving average values
        """
        if len(self.scores) < window:
            return [self.mean_score]

        scores_array = np.array(self.scores)
        cumsum = np.cumsum(scores_array)
        cumsum[window:] = cumsum[window:] - cumsum[:-window]

        return (cumsum[window - 1 :] / window).tolist()

    def constraint_satisfaction_rate(self, constraint_name: str) -> float:
        """Get satisfaction rate for a specific constraint.

        Args:
            constraint_name: Name of constraint

        Returns:
            Fraction of times constraint was satisfied
        """
        if constraint_name not in self.constraint_satisfaction:
            return 0.0

        satisfactions = self.constraint_satisfaction[constraint_name]
        if not satisfactions:
            return 0.0

        return sum(satisfactions) / len(satisfactions)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "steering_vector_name": self.steering_vector_name,
            "total_generations": self.total_generations,
            "successful_generations": self.successful_generations,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "success_rate": self.success_rate,
            "recent_scores": self.recent_scores(10),
            "constraint_satisfaction_rates": {
                name: self.constraint_satisfaction_rate(name)
                for name in self.constraint_satisfaction.keys()
            },
        }


def calculate_adherence(
    generated_text: str,
    target_attributes: Dict[str, any],
    actual_attributes: Dict[str, any],
) -> float:
    """Calculate adherence score between target and actual attributes.

    Args:
        generated_text: The generated text
        target_attributes: Desired attributes
        actual_attributes: Measured attributes from generated text

    Returns:
        Adherence score (0-1)
    """
    if not target_attributes:
        return 1.0

    scores = []

    for key, target_value in target_attributes.items():
        if key not in actual_attributes:
            continue

        actual_value = actual_attributes[key]

        # Calculate similarity based on type
        if isinstance(target_value, (int, float)):
            # Numerical - use relative difference
            if target_value == 0:
                score = 1.0 if actual_value == 0 else 0.0
            else:
                diff = abs(target_value - actual_value) / abs(target_value)
                score = max(0.0, 1.0 - diff)

        elif isinstance(target_value, bool):
            # Boolean - exact match
            score = 1.0 if target_value == actual_value else 0.0

        elif isinstance(target_value, str):
            # String - contains check (case insensitive)
            score = (
                1.0
                if target_value.lower() in str(actual_value).lower()
                else 0.0
            )

        elif isinstance(target_value, (list, set)):
            # Collection - intersection ratio
            target_set = set(target_value)
            actual_set = set(actual_value) if isinstance(actual_value, (list, set)) else {actual_value}
            intersection = len(target_set & actual_set)
            union = len(target_set | actual_set)
            score = intersection / union if union > 0 else 0.0

        else:
            # Default - exact match
            score = 1.0 if target_value == actual_value else 0.0

        scores.append(score)

    # Average across all attributes
    return float(np.mean(scores)) if scores else 0.0
