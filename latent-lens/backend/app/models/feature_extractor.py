"""Feature extraction and analysis from trained SAEs."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from transformers import PreTrainedTokenizer

from .sae import SparseAutoencoder

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFeature:
    """Represents a discovered feature from the SAE."""

    feature_id: int
    activation_mean: float
    activation_max: float
    activation_std: float
    sparsity: float  # Fraction of samples where feature is active
    top_tokens: List[str]  # Tokens that most activate this feature
    top_token_scores: List[float]  # Activation scores for top tokens
    example_texts: List[str]  # Example text snippets where feature fires


class FeatureExtractor:
    """Extracts and analyzes features from trained SAE.

    This class provides utilities to:
    - Identify which features activate for given inputs
    - Find top-activating tokens for each feature
    - Compute feature statistics and mutual information
    - Generate human-readable feature summaries
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: str = "cpu",
    ):
        """Initialize feature extractor.

        Args:
            sae: Trained SparseAutoencoder
            tokenizer: Tokenizer for text analysis (optional)
            device: Device to run computations on
        """
        self.sae = sae.to(device)
        self.sae.eval()
        self.tokenizer = tokenizer
        self.device = device

        # Cache for feature statistics
        self._feature_stats: Dict[int, Dict] = {}

        logger.info(f"FeatureExtractor initialized with {sae.config.hidden_dim} features")

    def extract_features(
        self,
        activations: torch.Tensor,
        threshold: float = 0.0,
        top_k: Optional[int] = None,
    ) -> Dict[int, torch.Tensor]:
        """Extract active features from activations.

        Args:
            activations: Input activations of shape (batch, seq_len, hidden_dim)
            threshold: Minimum activation value to consider feature active
            top_k: Return only top-k features by activation (None = all)

        Returns:
            Dictionary mapping feature_id -> activation values
        """
        # Flatten to (batch * seq_len, hidden_dim)
        original_shape = activations.shape
        if len(activations.shape) == 3:
            activations = activations.reshape(-1, activations.shape[-1])

        with torch.no_grad():
            activations = activations.to(self.device)
            hidden = self.sae.encode(activations)

            # Apply threshold
            active_mask = hidden > threshold

            # Get feature activations
            feature_dict = {}

            if top_k is not None:
                # Get top-k features globally
                flat_hidden = hidden.flatten()
                values, indices = torch.topk(flat_hidden, k=min(top_k, len(flat_hidden)))

                for val, idx in zip(values, indices):
                    if val > threshold:
                        feat_id = idx.item() % self.sae.config.hidden_dim
                        if feat_id not in feature_dict:
                            feature_dict[feat_id] = []
                        feature_dict[feat_id].append(val.item())
            else:
                # Get all active features
                for feat_id in range(self.sae.config.hidden_dim):
                    feat_activations = hidden[:, feat_id]
                    active_vals = feat_activations[feat_activations > threshold]

                    if len(active_vals) > 0:
                        feature_dict[feat_id] = active_vals.cpu().tolist()

        return feature_dict

    def get_top_features(
        self, activations: torch.Tensor, k: int = 10
    ) -> List[Tuple[int, float]]:
        """Get top-k most active features for given activations.

        Args:
            activations: Input activations
            k: Number of top features to return

        Returns:
            List of (feature_id, activation_score) tuples
        """
        with torch.no_grad():
            activations = activations.to(self.device)
            if len(activations.shape) == 3:
                activations = activations.reshape(-1, activations.shape[-1])

            hidden = self.sae.encode(activations)

            # Average activation per feature
            avg_activations = hidden.mean(dim=0)

            # Get top-k
            values, indices = torch.topk(avg_activations, k=min(k, len(avg_activations)))

            return [
                (idx.item(), val.item()) for idx, val in zip(indices, values)
            ]

    def compute_feature_statistics(
        self, activations: torch.Tensor, tokens: Optional[torch.Tensor] = None
    ) -> Dict[int, ExtractedFeature]:
        """Compute comprehensive statistics for all features.

        Args:
            activations: Input activations of shape (batch, seq_len, hidden_dim)
            tokens: Optional token IDs for token-level analysis

        Returns:
            Dictionary mapping feature_id -> ExtractedFeature
        """
        with torch.no_grad():
            activations = activations.to(self.device)
            original_shape = activations.shape

            # Flatten
            if len(activations.shape) == 3:
                batch_size, seq_len, hidden_dim = activations.shape
                activations = activations.reshape(-1, hidden_dim)
            else:
                batch_size = activations.shape[0]
                seq_len = 1

            # Encode
            hidden = self.sae.encode(activations)

            # Compute stats per feature
            features = {}

            for feat_id in range(self.sae.config.hidden_dim):
                feat_acts = hidden[:, feat_id].cpu().numpy()

                # Basic statistics
                active_mask = feat_acts > 0
                num_active = active_mask.sum()

                if num_active == 0:
                    continue

                activation_mean = feat_acts[active_mask].mean()
                activation_max = feat_acts.max()
                activation_std = feat_acts[active_mask].std()
                sparsity = num_active / len(feat_acts)

                # Token-level analysis
                top_tokens = []
                top_token_scores = []
                example_texts = []

                if tokens is not None and self.tokenizer is not None:
                    # Find tokens with highest activation
                    top_indices = np.argsort(feat_acts)[-10:][::-1]

                    for idx in top_indices:
                        if feat_acts[idx] > 0:
                            token_id = tokens.flatten()[idx].item()
                            token_str = self.tokenizer.decode([token_id])
                            top_tokens.append(token_str)
                            top_token_scores.append(float(feat_acts[idx]))

                features[feat_id] = ExtractedFeature(
                    feature_id=feat_id,
                    activation_mean=float(activation_mean),
                    activation_max=float(activation_max),
                    activation_std=float(activation_std),
                    sparsity=float(sparsity),
                    top_tokens=top_tokens[:5],
                    top_token_scores=top_token_scores[:5],
                    example_texts=example_texts,
                )

            logger.info(f"Computed statistics for {len(features)} active features")

        return features

    def get_feature_mutual_information(
        self, activations: torch.Tensor, labels: torch.Tensor
    ) -> Dict[int, float]:
        """Compute mutual information between features and labels.

        Args:
            activations: Input activations
            labels: Target labels (e.g., token IDs, sentiment)

        Returns:
            Dictionary mapping feature_id -> mutual information score
        """
        # Simplified MI estimation (can be replaced with sklearn.metrics.mutual_info_score)
        with torch.no_grad():
            activations = activations.to(self.device)
            if len(activations.shape) == 3:
                activations = activations.reshape(-1, activations.shape[-1])

            hidden = self.sae.encode(activations)

            mi_scores = {}

            for feat_id in range(self.sae.config.hidden_dim):
                feat_acts = (hidden[:, feat_id] > 0).cpu().numpy()
                labels_np = labels.cpu().numpy().flatten()

                # Simple correlation-based MI approximation
                if feat_acts.sum() > 0:
                    correlation = np.corrcoef(feat_acts, labels_np)[0, 1]
                    mi_scores[feat_id] = abs(correlation)

        return mi_scores

    def visualize_feature_activation(
        self,
        feature_id: int,
        tokens: torch.Tensor,
        activations: torch.Tensor,
    ) -> List[Tuple[str, float]]:
        """Visualize which tokens activate a specific feature.

        Args:
            feature_id: ID of feature to visualize
            tokens: Token IDs
            activations: Model activations

        Returns:
            List of (token_string, activation_value) tuples
        """
        if self.tokenizer is None:
            logger.warning("No tokenizer provided, cannot visualize tokens")
            return []

        with torch.no_grad():
            activations = activations.to(self.device)
            if len(activations.shape) == 3:
                activations = activations.reshape(-1, activations.shape[-1])

            hidden = self.sae.encode(activations)
            feat_acts = hidden[:, feature_id].cpu().numpy()

            # Get token strings
            tokens_flat = tokens.flatten()
            token_strings = [
                self.tokenizer.decode([tok.item()]) for tok in tokens_flat
            ]

            # Pair with activations
            token_activations = [
                (tok, float(act)) for tok, act in zip(token_strings, feat_acts)
            ]

            # Sort by activation
            token_activations.sort(key=lambda x: x[1], reverse=True)

        return token_activations[:50]  # Return top 50
