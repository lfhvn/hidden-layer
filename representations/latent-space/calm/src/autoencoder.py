"""
Autoencoder for CALM: K tokens ↔ continuous vector

Based on: Continuous Autoregressive Language Models (Shao et al., 2025)
Paper: https://arxiv.org/abs/2510.27688

Key components:
- Variational encoder: K tokens → Gaussian distribution N(μ, σ²I)
- Decoder: l-dimensional vector → K tokens
- Robust latent space via VAE + dropout regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VariationalEncoder(nn.Module):
    """
    Encodes K tokens into a Gaussian distribution over l-dimensional latent vectors.

    Architecture:
    1. Token embeddings (K tokens)
    2. Position-wise FFN
    3. Flatten + Linear compression (K*d → d)
    4. FFN
    5. Linear projection → (μ, σ)

    Args:
        vocab_size: Vocabulary size
        K: Number of tokens per chunk
        d: Hidden dimension (default: 512)
        latent_dim: Latent vector dimension (default: 128 for K=4)
        dropout: Dropout rate (default: 0.15)
    """

    def __init__(
        self,
        vocab_size: int,
        K: int,
        d: int = 512,
        latent_dim: int = 128,
        dropout: float = 0.15
    ):
        super().__init__()
        self.K = K
        self.d = d
        self.latent_dim = latent_dim

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d)

        # TODO: Implement encoder architecture
        # - Position-wise FFN (SwiGLU)
        # - Flatten + Linear (K*d → d)
        # - FFN
        # - Linear → μ, log_σ

        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, apply_dropout: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode K tokens to Gaussian parameters.

        Args:
            tokens: (batch_size, K) token indices
            apply_dropout: Whether to apply dropout (True during training)

        Returns:
            mu: (batch_size, latent_dim) mean vectors
            log_sigma: (batch_size, latent_dim) log standard deviations
        """
        # TODO: Implement forward pass
        # 1. Embed tokens
        # 2. Apply token dropout (mask random tokens during training)
        # 3. Process through encoder
        # 4. Output μ and log_σ

        raise NotImplementedError


class Decoder(nn.Module):
    """
    Decodes l-dimensional latent vector back to K tokens.

    Architecture:
    1. Linear + FFN (l → d)
    2. Linear expansion (d → K*d)
    3. Reshape to K hidden states
    4. Position-wise FFN
    5. Projection to vocab logits (tied with embedding)

    Args:
        vocab_size: Vocabulary size
        K: Number of tokens per chunk
        d: Hidden dimension (default: 512)
        latent_dim: Latent vector dimension (default: 128 for K=4)
    """

    def __init__(
        self,
        vocab_size: int,
        K: int,
        d: int = 512,
        latent_dim: int = 128,
        embedding_weight: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.K = K
        self.d = d
        self.latent_dim = latent_dim

        # TODO: Implement decoder architecture
        # - Linear + FFN (l → d)
        # - Linear expansion (d → K*d)
        # - Position-wise FFN
        # - Tied projection to vocab

        # Tie with encoder embedding if provided
        self.embedding_weight = embedding_weight

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to K token logits.

        Args:
            z: (batch_size, latent_dim) latent vectors

        Returns:
            logits: (batch_size, K, vocab_size) token logits
        """
        # TODO: Implement forward pass
        # 1. Linear + FFN
        # 2. Expand to K hidden states
        # 3. Process through FFN
        # 4. Project to vocab logits

        raise NotImplementedError


class VAEAutoencoder(nn.Module):
    """
    Complete VAE autoencoder for CALM.

    Maps K tokens ↔ l-dimensional continuous vector with:
    - High fidelity: >99.9% reconstruction accuracy
    - Robustness: Tolerates σ≈0.3 Gaussian noise

    Training objectives:
    - Reconstruction loss: Cross-entropy
    - KL divergence loss: With clipping to prevent collapse

    Args:
        vocab_size: Vocabulary size
        K: Number of tokens per chunk
        d: Hidden dimension (default: 512)
        latent_dim: Latent vector dimension (default: 128 for K=4)
        beta: KL divergence weight (default: 0.001)
        kl_floor: KL clipping threshold (default: 0.5)
        dropout: Dropout rate (default: 0.15)
    """

    def __init__(
        self,
        vocab_size: int,
        K: int = 4,
        d: int = 512,
        latent_dim: int = 128,
        beta: float = 0.001,
        kl_floor: float = 0.5,
        dropout: float = 0.15
    ):
        super().__init__()
        self.K = K
        self.latent_dim = latent_dim
        self.beta = beta
        self.kl_floor = kl_floor

        # Encoder and decoder
        self.encoder = VariationalEncoder(vocab_size, K, d, latent_dim, dropout)
        self.decoder = Decoder(vocab_size, K, d, latent_dim, self.encoder.embedding.weight)

        self.dropout = nn.Dropout(dropout)

    def encode(self, tokens: torch.Tensor, apply_dropout: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode tokens to Gaussian parameters."""
        return self.encoder(tokens, apply_dropout)

    def reparameterize(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I)

        Args:
            mu: Mean vectors
            log_sigma: Log standard deviations

        Returns:
            z: Sampled latent vectors
        """
        # TODO: Implement reparameterization trick
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to token logits."""
        return self.decoder(z)

    def forward(
        self,
        tokens: torch.Tensor,
        num_samples: int = 1,
        apply_dropout: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → sample → decode

        Args:
            tokens: (batch_size, K) token indices
            num_samples: Number of latent samples to draw (default: 1)
            apply_dropout: Whether to apply dropout

        Returns:
            logits: (batch_size, num_samples, K, vocab_size) reconstruction logits
            mu: (batch_size, latent_dim) mean vectors
            log_sigma: (batch_size, latent_dim) log standard deviations
        """
        # TODO: Implement forward pass
        # 1. Encode
        # 2. Sample num_samples latent vectors
        # 3. Apply latent dropout if training
        # 4. Decode

        raise NotImplementedError

    def compute_loss(
        self,
        tokens: torch.Tensor,
        logits: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute VAE loss: reconstruction + KL divergence (with clipping)

        Args:
            tokens: (batch_size, K) ground truth tokens
            logits: (batch_size, num_samples, K, vocab_size) reconstruction logits
            mu: (batch_size, latent_dim) mean vectors
            log_sigma: (batch_size, latent_dim) log standard deviations

        Returns:
            loss: Total loss
            metrics: Dict of loss components
        """
        # TODO: Implement loss computation
        # 1. Reconstruction loss (cross-entropy)
        # 2. KL divergence with clipping (per dimension)
        # 3. Combined loss = reconstruction + beta * KL_clipped

        raise NotImplementedError

    def reconstruct(self, tokens: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Reconstruct tokens (for evaluation).

        Args:
            tokens: (batch_size, K) input tokens
            temperature: Sampling temperature

        Returns:
            reconstructed: (batch_size, K) reconstructed tokens
        """
        # TODO: Implement reconstruction
        # 1. Encode (no dropout)
        # 2. Sample single latent vector
        # 3. Decode
        # 4. Sample tokens (argmax or temperature sampling)

        raise NotImplementedError


def create_autoencoder(
    vocab_size: int,
    K: int = 4,
    size: str = "base"
) -> VAEAutoencoder:
    """
    Factory function to create autoencoder with predefined sizes.

    Sizes:
    - base: d=512, latent_dim=128 (paper default for K=4)
    - small: d=256, latent_dim=64
    - large: d=1024, latent_dim=256

    Args:
        vocab_size: Vocabulary size
        K: Number of tokens per chunk
        size: Model size preset

    Returns:
        Autoencoder model
    """
    configs = {
        "small": {"d": 256, "latent_dim": 32 * K},
        "base": {"d": 512, "latent_dim": 32 * K},
        "large": {"d": 1024, "latent_dim": 32 * K},
    }

    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")

    config = configs[size]
    return VAEAutoencoder(vocab_size=vocab_size, K=K, **config)


# TODO: Add helper functions
# - load_pretrained_autoencoder()
# - compute_reconstruction_accuracy()
# - test_robustness_to_noise()
