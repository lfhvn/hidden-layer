"""Steering effectiveness evaluator."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch

from ..steering.constraints import Constraint, enforce_constraints

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from steering evaluation."""

    experiment_id: int
    vector_name: str
    prompt: str

    # Generated outputs
    steered_output: str
    unsteered_output: str

    # Adherence metrics
    adherence_score: float
    constraints_satisfied: bool
    constraint_violations: List[str]

    # Comparison metrics
    edit_distance: int
    semantic_similarity: Optional[float] = None

    # Metadata
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "vector_name": self.vector_name,
            "prompt": self.prompt,
            "steered_output": self.steered_output,
            "unsteered_output": self.unsteered_output,
            "adherence_score": self.adherence_score,
            "constraints_satisfied": self.constraints_satisfied,
            "constraint_violations": self.constraint_violations,
            "edit_distance": self.edit_distance,
            "semantic_similarity": self.semantic_similarity,
            "metadata": self.metadata,
        }


class SteeringEvaluator:
    """Evaluates steering effectiveness and adherence."""

    def __init__(self):
        """Initialize evaluator."""
        logger.info("SteeringEvaluator initialized")

    def evaluate(
        self,
        experiment_id: int,
        vector_name: str,
        prompt: str,
        steered_output: str,
        unsteered_output: str,
        constraints: Optional[List[Constraint]] = None,
    ) -> EvaluationResult:
        """Evaluate a steering result.

        Args:
            experiment_id: Experiment ID
            vector_name: Steering vector name
            prompt: Input prompt
            steered_output: Steered generation
            unsteered_output: Unsteered generation
            constraints: Constraints to check

        Returns:
            EvaluationResult
        """
        # Check constraint adherence
        if constraints:
            satisfied, adherence_score, violations = enforce_constraints(
                steered_output, constraints
            )
        else:
            satisfied = True
            adherence_score = 1.0
            violations = []

        # Calculate edit distance
        edit_dist = self._levenshtein_distance(steered_output, unsteered_output)

        # Semantic similarity (would use embeddings in production)
        semantic_sim = self._simple_similarity(steered_output, unsteered_output)

        result = EvaluationResult(
            experiment_id=experiment_id,
            vector_name=vector_name,
            prompt=prompt,
            steered_output=steered_output,
            unsteered_output=unsteered_output,
            adherence_score=adherence_score,
            constraints_satisfied=satisfied,
            constraint_violations=violations,
            edit_distance=edit_dist,
            semantic_similarity=semantic_sim,
            metadata={
                "num_constraints": len(constraints) if constraints else 0,
                "steered_length": len(steered_output.split()),
                "unsteered_length": len(unsteered_output.split()),
            },
        )

        logger.info(
            f"Evaluated {vector_name}: adherence={adherence_score:.3f}, "
            f"edit_dist={edit_dist}, satisfied={satisfied}"
        )

        return result

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Edit distance
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row = [i + 1]

            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))

            previous_row = current_row

        return previous_row[-1]

    def _simple_similarity(self, s1: str, s2: str) -> float:
        """Calculate simple word overlap similarity.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score (0-1)
        """
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0
