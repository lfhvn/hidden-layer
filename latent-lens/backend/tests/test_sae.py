"""Tests for Sparse Autoencoder."""

import pytest
import torch

from app.models.sae import SparseAutoencoder, SAETrainingConfig, SAEOutput


def test_sae_initialization():
    """Test SAE initializes with correct dimensions."""
    config = SAETrainingConfig(
        input_dim=128,
        hidden_dim=512,
        sparsity_coef=0.01,
    )

    sae = SparseAutoencoder(config)

    assert sae.config.input_dim == 128
    assert sae.config.hidden_dim == 512
    assert sae.encoder.in_features == 128
    assert sae.encoder.out_features == 512


def test_sae_forward_pass(sample_sae):
    """Test SAE forward pass produces correct output shapes."""
    batch_size = 4
    input_dim = sample_sae.config.input_dim

    x = torch.randn(batch_size, input_dim)
    output = sample_sae(x, return_loss=True)

    assert isinstance(output, SAEOutput)
    assert output.reconstructed.shape == (batch_size, input_dim)
    assert output.hidden.shape == (batch_size, sample_sae.config.hidden_dim)
    assert output.loss is not None
    assert output.reconstruction_loss is not None
    assert output.sparsity_loss is not None


def test_sae_sparsity(sample_sae):
    """Test that SAE produces sparse activations."""
    x = torch.randn(32, sample_sae.config.input_dim)

    with torch.no_grad():
        output = sample_sae(x, return_loss=False)

        # Check that activations are sparse (many zeros due to ReLU)
        num_active = (output.hidden > 0).sum().item()
        total = output.hidden.numel()

        sparsity_ratio = num_active / total

        # Should be less than 50% active
        assert sparsity_ratio < 0.5


def test_sae_reconstruction(sample_sae):
    """Test that SAE can reconstruct inputs reasonably."""
    x = torch.randn(16, sample_sae.config.input_dim)

    with torch.no_grad():
        output = sample_sae(x, return_loss=True)

        # Reconstruction should have same shape
        assert output.reconstructed.shape == x.shape

        # Reconstruction loss should be finite
        assert torch.isfinite(output.reconstruction_loss).all()


def test_sae_encode_decode(sample_sae):
    """Test separate encode and decode operations."""
    x = torch.randn(8, sample_sae.config.input_dim)

    with torch.no_grad():
        # Encode
        hidden = sample_sae.encode(x)
        assert hidden.shape == (8, sample_sae.config.hidden_dim)

        # Decode
        reconstructed = sample_sae.decode(hidden)
        assert reconstructed.shape == x.shape


def test_sae_sparsity_metrics(sample_sae):
    """Test sparsity metrics computation."""
    x = torch.randn(32, sample_sae.config.input_dim)

    with torch.no_grad():
        output = sample_sae(x, return_loss=False)
        metrics = sample_sae.compute_sparsity_metrics(output.hidden)

        assert "l0_sparsity" in metrics
        assert "l1_norm" in metrics
        assert "fraction_active" in metrics

        assert metrics["l0_sparsity"] >= 0
        assert metrics["l1_norm"] >= 0
        assert 0 <= metrics["fraction_active"] <= 1


def test_sae_save_load(sample_sae, tmp_path):
    """Test SAE save and load."""
    save_path = tmp_path / "test_sae.pt"

    # Save
    sample_sae.save(str(save_path))
    assert save_path.exists()

    # Load
    loaded_sae = SparseAutoencoder.load(str(save_path), device="cpu")

    assert loaded_sae.config.input_dim == sample_sae.config.input_dim
    assert loaded_sae.config.hidden_dim == sample_sae.config.hidden_dim

    # Test that loaded model produces same outputs
    x = torch.randn(4, sample_sae.config.input_dim)

    with torch.no_grad():
        orig_output = sample_sae(x, return_loss=False)
        loaded_output = loaded_sae(x, return_loss=False)

        assert torch.allclose(orig_output.reconstructed, loaded_output.reconstructed, atol=1e-5)
