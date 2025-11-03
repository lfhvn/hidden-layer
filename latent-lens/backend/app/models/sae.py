"""Sparse Autoencoder (SAE) implementation with L1 sparsity penalty.

This module provides a minimal, production-ready SAE that can be trained
on language model activations to discover interpretable features.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

logger = logging.getLogger(__name__)


@dataclass
class SAETrainingConfig:
    """Configuration for SAE training."""

    input_dim: int
    hidden_dim: int
    sparsity_coef: float = 0.01
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 10
    device: str = "cpu"
    l2_coef: float = 1e-4  # Weight decay
    tie_weights: bool = True  # Tie encoder/decoder weights


@dataclass
class SAEOutput:
    """Output from SAE forward pass."""

    reconstructed: torch.Tensor  # Reconstructed activations
    hidden: torch.Tensor  # Hidden layer activations (features)
    loss: Optional[torch.Tensor] = None  # Total loss
    reconstruction_loss: Optional[torch.Tensor] = None  # MSE loss
    sparsity_loss: Optional[torch.Tensor] = None  # L1 penalty
    active_features: Optional[torch.Tensor] = None  # Boolean mask of active features


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder with L1 sparsity penalty.

    Architecture:
        input (d_model) -> encoder -> hidden (d_hidden) -> decoder -> output (d_model)

    The hidden layer learns sparse, interpretable features from model activations.

    Args:
        config: Training configuration containing dimensions and hyperparameters
    """

    def __init__(self, config: SAETrainingConfig):
        super().__init__()
        self.config = config

        # Encoder: input_dim -> hidden_dim
        self.encoder = nn.Linear(config.input_dim, config.hidden_dim, bias=True)

        # Decoder: hidden_dim -> input_dim
        if config.tie_weights:
            # Tied weights: decoder is transpose of encoder
            self.decoder = None
        else:
            self.decoder = nn.Linear(config.hidden_dim, config.input_dim, bias=True)

        # Pre-encoder bias (learned mean to center activations)
        self.pre_bias = nn.Parameter(torch.zeros(config.input_dim))

        # Initialize weights
        self._init_weights()

        logger.info(
            f"Initialized SAE: input_dim={config.input_dim}, "
            f"hidden_dim={config.hidden_dim}, tied_weights={config.tie_weights}"
        )

    def _init_weights(self) -> None:
        """Initialize weights with Xavier/Glorot initialization."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

        if self.decoder is not None:
            nn.init.xavier_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse hidden representation.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            hidden: Sparse features of shape (batch, hidden_dim)
        """
        # Center the input
        x_centered = x - self.pre_bias

        # Encode and apply ReLU for sparsity
        hidden = F.relu(self.encoder(x_centered))

        return hidden

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """Decode hidden representation back to input space.

        Args:
            hidden: Hidden features of shape (batch, hidden_dim)

        Returns:
            reconstructed: Tensor of shape (batch, input_dim)
        """
        if self.decoder is not None:
            # Untied weights
            reconstructed = self.decoder(hidden)
        else:
            # Tied weights: use transpose of encoder
            reconstructed = F.linear(hidden, self.encoder.weight.t())

        # Add back the pre-bias
        reconstructed = reconstructed + self.pre_bias

        return reconstructed

    def forward(
        self, x: torch.Tensor, return_loss: bool = True
    ) -> SAEOutput:
        """Forward pass through SAE.

        Args:
            x: Input activations of shape (batch, input_dim)
            return_loss: Whether to compute and return losses

        Returns:
            SAEOutput containing reconstructed activations, hidden features, and losses
        """
        # Encode
        hidden = self.encode(x)

        # Decode
        reconstructed = self.decode(hidden)

        # Compute losses if requested
        loss = None
        reconstruction_loss = None
        sparsity_loss = None
        active_features = None

        if return_loss:
            # Reconstruction loss (MSE)
            reconstruction_loss = F.mse_loss(reconstructed, x)

            # Sparsity loss (L1 on hidden activations)
            sparsity_loss = self.config.sparsity_coef * torch.mean(torch.abs(hidden))

            # Total loss
            loss = reconstruction_loss + sparsity_loss

            # Track which features are active (> 0)
            active_features = hidden > 0

        return SAEOutput(
            reconstructed=reconstructed,
            hidden=hidden,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss,
            active_features=active_features,
        )

    def get_feature_activations(
        self, x: torch.Tensor, top_k: Optional[int] = None
    ) -> Dict[int, torch.Tensor]:
        """Get top-k most active features for input.

        Args:
            x: Input tensor of shape (batch, input_dim)
            top_k: Number of top features to return per sample (None = all)

        Returns:
            Dictionary mapping feature indices to activation values
        """
        with torch.no_grad():
            hidden = self.encode(x)

            if top_k is None:
                top_k = self.config.hidden_dim

            # Get top-k features per sample
            values, indices = torch.topk(hidden, k=top_k, dim=-1)

            # Convert to dict: {feature_idx: activation_value}
            feature_dict = {}
            for batch_idx in range(hidden.size(0)):
                for k_idx in range(top_k):
                    feat_idx = indices[batch_idx, k_idx].item()
                    feat_val = values[batch_idx, k_idx].item()
                    if feat_val > 0:  # Only include active features
                        if feat_idx not in feature_dict:
                            feature_dict[feat_idx] = []
                        feature_dict[feat_idx].append(feat_val)

        return feature_dict

    def compute_sparsity_metrics(self, hidden: torch.Tensor) -> Dict[str, float]:
        """Compute sparsity statistics for hidden activations.

        Args:
            hidden: Hidden layer activations of shape (batch, hidden_dim)

        Returns:
            Dictionary with sparsity metrics (L0, L1, fraction_active)
        """
        with torch.no_grad():
            # L0: Average number of active features per sample
            l0 = (hidden > 0).float().sum(dim=-1).mean().item()

            # L1: Average L1 norm per sample
            l1 = torch.abs(hidden).sum(dim=-1).mean().item()

            # Fraction of features that are ever active
            fraction_active = (hidden > 0).any(dim=0).float().mean().item()

        return {
            "l0_sparsity": l0,
            "l1_norm": l1,
            "fraction_active": fraction_active,
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config,
            },
            path,
        )
        logger.info(f"Saved SAE to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SparseAutoencoder":
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        logger.info(f"Loaded SAE from {path}")
        return model
