"""Pytest configuration and fixtures."""

import pytest
import torch

from app.steering.engine import SteeringVector, SteeringMethod


@pytest.fixture
def sample_vector():
    """Sample steering vector for testing."""
    return SteeringVector(
        name="test_vector",
        vector=torch.randn(768),
        layer_index=6,
        strength=1.0,
        method=SteeringMethod.ADD,
    )


@pytest.fixture
def sample_prompt():
    """Sample prompt for testing."""
    return "The quick brown fox jumps over the lazy dog."
