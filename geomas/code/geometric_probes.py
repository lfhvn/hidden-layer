"""
Core geometric analysis tools for probing model representations.

This module provides tools to extract and analyze geometric structures
in hidden states, contrasting geometric vs associative memory.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


@dataclass
class GeometricAnalysis:
    """Results of geometric structure analysis"""
    spectral_gap: float
    fiedler_vector: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    cluster_coherence: float
    global_structure_score: float
    quality_score: float
    metadata: Dict[str, Any]


class GeometricProbe:
    """
    Analyze geometric vs associative memory in language models.

    This class provides methods to extract hidden states and compute
    geometric properties like spectral structure, cluster coherence,
    and overall geometric quality.

    Usage:
        probe = GeometricProbe(model="llama3.2:latest", provider="ollama")
        analysis = probe.analyze_task(task_input, labels=concept_labels)
        print(f"Geometric quality: {analysis.quality_score:.3f}")
    """

    def __init__(
        self,
        model: str,
        provider: str = "ollama",
        layer_indices: Optional[List[int]] = None
    ):
        """
        Initialize geometric probe.

        Args:
            model: Model name/path
            provider: LLM provider ("ollama", "mlx", etc.)
            layer_indices: Which layers to extract (default: [-1, -2, -3])
        """
        self.model = model
        self.provider = provider
        self.layer_indices = layer_indices or [-1, -2, -3]

    def extract_hidden_states(
        self,
        inputs: List[str],
        layer_idx: int = -1
    ) -> np.ndarray:
        """
        Extract hidden state activations from model.

        Args:
            inputs: List of input texts to process
            layer_idx: Which layer to extract from

        Returns:
            Hidden states as numpy array (n_inputs, hidden_dim)

        Note:
            Implementation depends on provider. For now, this is a stub
            that will be implemented for specific providers (MLX, Ollama).
        """
        # TODO: Implement provider-specific extraction
        # For now, return placeholder

        if self.provider == "ollama":
            return self._extract_ollama_hidden_states(inputs, layer_idx)
        elif self.provider == "mlx":
            return self._extract_mlx_hidden_states(inputs, layer_idx)
        else:
            raise NotImplementedError(
                f"Hidden state extraction not yet implemented for {self.provider}"
            )

    def _extract_ollama_hidden_states(
        self,
        inputs: List[str],
        layer_idx: int
    ) -> np.ndarray:
        """Extract from Ollama models"""
        # TODO: Implement using Ollama's embedding API or model inspection
        # For validation phase, we can use embeddings as approximation
        from harness import llm_call

        # Use embedding endpoint if available
        # Otherwise, this will need model-specific hooks
        raise NotImplementedError(
            "Ollama hidden state extraction requires embedding API or hooks"
        )

    def _extract_mlx_hidden_states(
        self,
        inputs: List[str],
        layer_idx: int
    ) -> np.ndarray:
        """Extract from MLX models"""
        # TODO: Implement using MLX model hooks
        # Can use mlx.nn.Module hooks to capture intermediate activations
        raise NotImplementedError(
            "MLX hidden state extraction requires model hooks"
        )

    def compute_spectral_structure(
        self,
        hidden_states: np.ndarray,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """
        Compute spectral properties of hidden state manifold.

        This analyzes the geometric structure by computing the eigendecomposition
        of the graph Laplacian derived from pairwise similarities.

        Args:
            hidden_states: Array of shape (n_samples, hidden_dim)
            normalize: Whether to normalize the Laplacian

        Returns:
            Dictionary containing:
                - eigenvalues: All eigenvalues (sorted ascending)
                - eigenvectors: Corresponding eigenvectors
                - spectral_gap: λ₂ - λ₁ (larger = stronger structure)
                - fiedler_vector: 2nd eigenvector (primary geometric axis)
        """
        # Compute pairwise similarity matrix (cosine similarity)
        # S[i,j] = h_i · h_j / (||h_i|| ||h_j||)
        norms = np.linalg.norm(hidden_states, axis=1, keepdims=True)
        normalized_states = hidden_states / (norms + 1e-8)
        similarity = normalized_states @ normalized_states.T

        # Ensure non-negative similarities
        similarity = np.maximum(similarity, 0)

        # Construct graph Laplacian: L = D - S
        degree = np.diag(np.sum(similarity, axis=1))
        laplacian = degree - similarity

        # Normalized Laplacian: L_norm = D^(-1/2) L D^(-1/2)
        if normalize:
            degree_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(degree) + 1e-8))
            laplacian = degree_sqrt_inv @ laplacian @ degree_sqrt_inv

        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(laplacian)

        # Sort by eigenvalue (should already be sorted, but ensure)
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # Spectral gap: difference between 2nd and 1st eigenvalue
        # Larger gap indicates stronger geometric structure
        spectral_gap = eigenvalues[1] - eigenvalues[0]

        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'spectral_gap': spectral_gap,
            'fiedler_vector': eigenvectors[:, 1],  # 2nd smallest eigenvalue
            'similarity_matrix': similarity
        }

    def compute_cluster_coherence(
        self,
        hidden_states: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> float:
        """
        Measure how well-separated and coherent clusters are.

        If labels provided, computes ratio of intra-cluster to inter-cluster distance.
        If no labels, uses Fiedler vector to create binary partition.

        Args:
            hidden_states: Array of shape (n_samples, hidden_dim)
            labels: Optional cluster labels for each sample

        Returns:
            Coherence score (higher = better separation)
        """
        if labels is None:
            # Use Fiedler vector to create binary partition
            spectral = self.compute_spectral_structure(hidden_states)
            fiedler = spectral['fiedler_vector']
            labels = (fiedler > np.median(fiedler)).astype(int)

        # If we have at least 2 clusters, compute silhouette score
        if len(np.unique(labels)) >= 2:
            try:
                score = silhouette_score(hidden_states, labels)
                # Silhouette is in [-1, 1], map to [0, 1]
                return (score + 1) / 2
            except:
                # Fallback if silhouette fails
                return 0.5
        else:
            return 0.5  # Neutral score if only one cluster

    def geometric_quality_score(
        self,
        hidden_states: np.ndarray,
        labels: Optional[np.ndarray] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute overall geometric quality score.

        Combines multiple metrics into single quality score:
        - Spectral gap (normalized)
        - Cluster coherence
        - Dimensionality (effective rank)

        Args:
            hidden_states: Array of shape (n_samples, hidden_dim)
            labels: Optional cluster labels
            weights: Optional custom weights for metrics

        Returns:
            Quality score in [0, 1] (higher = stronger geometric structure)
        """
        if weights is None:
            weights = {
                'spectral_gap': 0.4,
                'cluster_coherence': 0.4,
                'effective_rank': 0.2
            }

        # Compute spectral gap (normalize by max eigenvalue)
        spectral = self.compute_spectral_structure(hidden_states)
        normalized_gap = spectral['spectral_gap'] / (spectral['eigenvalues'][-1] + 1e-8)
        normalized_gap = min(normalized_gap, 1.0)  # Clip to [0, 1]

        # Compute cluster coherence
        coherence = self.compute_cluster_coherence(hidden_states, labels)

        # Compute effective rank (participation ratio of eigenvalues)
        eigenvalues = spectral['eigenvalues']
        eigenvalues = eigenvalues / (np.sum(eigenvalues) + 1e-8)
        entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-8))
        effective_rank = np.exp(entropy) / len(eigenvalues)  # Normalize by n

        # Combine metrics
        quality = (
            weights['spectral_gap'] * normalized_gap +
            weights['cluster_coherence'] * coherence +
            weights['effective_rank'] * effective_rank
        )

        return quality

    def analyze(
        self,
        hidden_states: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> GeometricAnalysis:
        """
        Full geometric analysis of hidden states.

        Args:
            hidden_states: Array of shape (n_samples, hidden_dim)
            labels: Optional cluster labels

        Returns:
            GeometricAnalysis object with all metrics
        """
        spectral = self.compute_spectral_structure(hidden_states)
        coherence = self.compute_cluster_coherence(hidden_states, labels)
        quality = self.geometric_quality_score(hidden_states, labels)

        # Compute global structure score (similar to quality but different weights)
        global_score = 0.6 * quality + 0.4 * coherence

        return GeometricAnalysis(
            spectral_gap=spectral['spectral_gap'],
            fiedler_vector=spectral['fiedler_vector'],
            eigenvalues=spectral['eigenvalues'],
            eigenvectors=spectral['eigenvectors'],
            cluster_coherence=coherence,
            global_structure_score=global_score,
            quality_score=quality,
            metadata={
                'n_samples': hidden_states.shape[0],
                'hidden_dim': hidden_states.shape[1],
                'has_labels': labels is not None,
                'n_clusters': len(np.unique(labels)) if labels is not None else None
            }
        )

    def visualize_geometry(
        self,
        hidden_states: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = 'umap',
        title: str = "Geometric Structure"
    ) -> Tuple[np.ndarray, Any]:
        """
        Project hidden states to 2D for visualization.

        Args:
            hidden_states: Array of shape (n_samples, hidden_dim)
            labels: Optional cluster labels for coloring
            method: Projection method ('umap', 'pca', 'spectral')
            title: Plot title

        Returns:
            Tuple of (2D coordinates, figure object)
        """
        if method == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not available. Install with: pip install umap-learn")
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(hidden_states)

        elif method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            embedding_2d = pca.fit_transform(hidden_states)

        elif method == 'spectral':
            # Use Fiedler vector and 3rd eigenvector for 2D projection
            spectral = self.compute_spectral_structure(hidden_states)
            embedding_2d = np.column_stack([
                spectral['fiedler_vector'],
                spectral['eigenvectors'][:, 2]
            ])

        else:
            raise ValueError(f"Unknown method: {method}")

        # Create visualization (return both coordinates and figure)
        try:
            import plotly.graph_objects as go

            if labels is not None:
                fig = go.Figure(data=go.Scatter(
                    x=embedding_2d[:, 0],
                    y=embedding_2d[:, 1],
                    mode='markers',
                    marker=dict(
                        color=labels,
                        colorscale='Viridis',
                        size=8,
                        line=dict(width=0.5, color='white')
                    ),
                    text=[f"Point {i}" for i in range(len(embedding_2d))]
                ))
            else:
                fig = go.Figure(data=go.Scatter(
                    x=embedding_2d[:, 0],
                    y=embedding_2d[:, 1],
                    mode='markers',
                    marker=dict(size=8, color='blue', line=dict(width=0.5, color='white'))
                ))

            fig.update_layout(
                title=title,
                xaxis_title=f"{method.upper()} 1",
                yaxis_title=f"{method.upper()} 2",
                template='plotly_white'
            )

            return embedding_2d, fig

        except ImportError:
            # If plotly not available, just return coordinates
            return embedding_2d, None


# Standalone functions for convenience

def compute_spectral_structure(
    hidden_states: np.ndarray,
    normalize: bool = True
) -> Dict[str, Any]:
    """Compute spectral structure (standalone function)"""
    probe = GeometricProbe(model="dummy", provider="dummy")
    return probe.compute_spectral_structure(hidden_states, normalize)


def geometric_quality_score(
    hidden_states: np.ndarray,
    labels: Optional[np.ndarray] = None
) -> float:
    """Compute geometric quality score (standalone function)"""
    probe = GeometricProbe(model="dummy", provider="dummy")
    return probe.geometric_quality_score(hidden_states, labels)


def visualize_geometry(
    hidden_states: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = 'umap',
    title: str = "Geometric Structure"
) -> Tuple[np.ndarray, Any]:
    """Visualize geometric structure (standalone function)"""
    probe = GeometricProbe(model="dummy", provider="dummy")
    return probe.visualize_geometry(hidden_states, labels, method, title)
