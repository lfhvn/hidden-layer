"""Sample data for testing."""

import torch


def generate_sample_activations(batch_size=16, seq_len=32, hidden_dim=64):
    """Generate random activation tensors for testing.

    Args:
        batch_size: Number of samples
        seq_len: Sequence length
        hidden_dim: Hidden dimension

    Returns:
        Tensor of shape (batch_size, seq_len, hidden_dim)
    """
    return torch.randn(batch_size, seq_len, hidden_dim)


def generate_sparse_activations(batch_size=16, seq_len=32, hidden_dim=64, sparsity=0.9):
    """Generate sparse activation tensors.

    Args:
        batch_size: Number of samples
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        sparsity: Fraction of values to zero out

    Returns:
        Sparse tensor of shape (batch_size, seq_len, hidden_dim)
    """
    activations = torch.randn(batch_size, seq_len, hidden_dim)

    # Zero out fraction of values
    mask = torch.rand_like(activations) > sparsity
    activations = activations * mask

    return activations


SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming artificial intelligence.",
    "Neural networks learn patterns from data.",
    "Sparse autoencoders discover interpretable features.",
    "Language models process text one token at a time.",
]
