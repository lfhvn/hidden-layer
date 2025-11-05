"""Tests for steering engine."""

import pytest
import torch

from app.steering.engine import SteeringVector, SteeringMethod


def test_steering_vector_apply():
    """Test steering vector application."""
    vector = SteeringVector(
        name="test",
        vector=torch.ones(10),
        layer_index=0,
        strength=0.5,
        method=SteeringMethod.ADD,
    )

    activations = torch.zeros(1, 1, 10)
    steered = vector.apply(activations)

    assert steered.shape == activations.shape
    assert torch.allclose(steered, torch.ones(1, 1, 10) * 0.5)


def test_constraints():
    """Test constraint enforcement."""
    from app.steering.constraints import Constraint, ConstraintType

    constraint = Constraint(
        type=ConstraintType.KEYWORD_MUST_INCLUDE,
        value=["positive", "happy"],
    )

    assert constraint.check("This is a positive and happy message")
    assert not constraint.check("This is a sad message")
