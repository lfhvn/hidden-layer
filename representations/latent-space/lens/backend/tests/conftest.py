"""Pytest configuration and fixtures."""

import pytest
import torch
from sqlmodel import Session, create_engine, SQLModel
from sqlalchemy.pool import StaticPool

from app.config import Settings, get_settings
from app.storage.database import get_engine
from app.models.sae import SparseAutoencoder, SAETrainingConfig


@pytest.fixture
def test_settings():
    """Override settings for tests."""
    return Settings(
        database_url="sqlite:///:memory:",
        api_key="test-api-key",
        device="cpu",
    )


@pytest.fixture
def test_engine(test_settings, monkeypatch):
    """Create test database engine."""
    engine = create_engine(
        test_settings.database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Override get_settings
    monkeypatch.setattr("app.storage.database.get_settings", lambda: test_settings)

    # Create tables
    SQLModel.metadata.create_all(engine)

    yield engine

    # Cleanup
    SQLModel.metadata.drop_all(engine)


@pytest.fixture
def test_session(test_engine):
    """Create test database session."""
    with Session(test_engine) as session:
        yield session


@pytest.fixture
def sample_activations():
    """Generate sample activation tensors."""
    batch_size = 16
    seq_len = 32
    hidden_dim = 64

    return torch.randn(batch_size, seq_len, hidden_dim)


@pytest.fixture
def sample_sae():
    """Create a small SAE for testing."""
    config = SAETrainingConfig(
        input_dim=64,
        hidden_dim=256,
        sparsity_coef=0.01,
        learning_rate=1e-3,
        batch_size=8,
        num_epochs=2,
        device="cpu",
    )

    return SparseAutoencoder(config)


@pytest.fixture
def sample_texts():
    """Sample text data for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neurons.",
    ]
