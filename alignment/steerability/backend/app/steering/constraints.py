"""Constraint enforcement for steering."""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Callable

logger = logging.getLogger(__name__)


class ConstraintType(str, Enum):
    """Types of constraints that can be enforced."""

    KEYWORD_MUST_INCLUDE = "keyword_must_include"
    KEYWORD_MUST_EXCLUDE = "keyword_must_exclude"
    LENGTH_MIN = "length_min"
    LENGTH_MAX = "length_max"
    SENTIMENT_POSITIVE = "sentiment_positive"
    SENTIMENT_NEGATIVE = "sentiment_negative"
    PATTERN_MATCH = "pattern_match"
    PATTERN_AVOID = "pattern_avoid"
    CUSTOM = "custom"


@dataclass
class Constraint:
    """Represents a constraint on model output."""

    type: ConstraintType
    value: any  # Constraint-specific value
    description: Optional[str] = None
    weight: float = 1.0  # Weight for adherence scoring

    def check(self, text: str) -> bool:
        """Check if text satisfies the constraint.

        Args:
            text: Generated text to check

        Returns:
            True if constraint is satisfied
        """
        if self.type == ConstraintType.KEYWORD_MUST_INCLUDE:
            keywords = self.value if isinstance(self.value, list) else [self.value]
            return all(kw.lower() in text.lower() for kw in keywords)

        elif self.type == ConstraintType.KEYWORD_MUST_EXCLUDE:
            keywords = self.value if isinstance(self.value, list) else [self.value]
            return not any(kw.lower() in text.lower() for kw in keywords)

        elif self.type == ConstraintType.LENGTH_MIN:
            return len(text.split()) >= self.value

        elif self.type == ConstraintType.LENGTH_MAX:
            return len(text.split()) <= self.value

        elif self.type == ConstraintType.PATTERN_MATCH:
            pattern = self.value
            return bool(re.search(pattern, text))

        elif self.type == ConstraintType.PATTERN_AVOID:
            pattern = self.value
            return not bool(re.search(pattern, text))

        elif self.type == ConstraintType.SENTIMENT_POSITIVE:
            # Simple heuristic (would use sentiment model in production)
            positive_words = ["good", "great", "excellent", "wonderful", "happy"]
            return any(word in text.lower() for word in positive_words)

        elif self.type == ConstraintType.SENTIMENT_NEGATIVE:
            # Simple heuristic
            negative_words = ["bad", "terrible", "awful", "horrible", "sad"]
            return any(word in text.lower() for word in negative_words)

        elif self.type == ConstraintType.CUSTOM:
            # Value should be a callable
            if callable(self.value):
                return self.value(text)
            return True

        return True


def enforce_constraints(
    text: str, constraints: List[Constraint]
) -> tuple[bool, float, List[str]]:
    """Enforce a list of constraints on generated text.

    Args:
        text: Generated text
        constraints: List of constraints to check

    Returns:
        Tuple of (all_satisfied, adherence_score, violations)
    """
    if not constraints:
        return True, 1.0, []

    total_weight = sum(c.weight for c in constraints)
    satisfied_weight = 0.0
    violations = []

    for constraint in constraints:
        satisfied = constraint.check(text)

        if satisfied:
            satisfied_weight += constraint.weight
        else:
            violation_msg = f"{constraint.type.value}: {constraint.description or constraint.value}"
            violations.append(violation_msg)
            logger.debug(f"Constraint violated: {violation_msg}")

    adherence_score = satisfied_weight / total_weight if total_weight > 0 else 1.0
    all_satisfied = len(violations) == 0

    return all_satisfied, adherence_score, violations


def create_keyword_constraint(
    keywords: List[str], must_include: bool = True, weight: float = 1.0
) -> Constraint:
    """Create a keyword constraint.

    Args:
        keywords: Keywords to check
        must_include: True = must include, False = must exclude
        weight: Constraint weight

    Returns:
        Constraint
    """
    constraint_type = (
        ConstraintType.KEYWORD_MUST_INCLUDE
        if must_include
        else ConstraintType.KEYWORD_MUST_EXCLUDE
    )

    return Constraint(
        type=constraint_type,
        value=keywords,
        description=f"{'Include' if must_include else 'Exclude'}: {', '.join(keywords)}",
        weight=weight,
    )


def create_length_constraint(
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    weight: float = 1.0,
) -> List[Constraint]:
    """Create length constraints.

    Args:
        min_words: Minimum word count
        max_words: Maximum word count
        weight: Constraint weight

    Returns:
        List of Constraint objects
    """
    constraints = []

    if min_words is not None:
        constraints.append(
            Constraint(
                type=ConstraintType.LENGTH_MIN,
                value=min_words,
                description=f"Minimum {min_words} words",
                weight=weight,
            )
        )

    if max_words is not None:
        constraints.append(
            Constraint(
                type=ConstraintType.LENGTH_MAX,
                value=max_words,
                description=f"Maximum {max_words} words",
                weight=weight,
            )
        )

    return constraints
