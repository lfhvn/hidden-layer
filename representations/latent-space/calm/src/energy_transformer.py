"""
Energy Transformer for CALM: Likelihood-free language modeling

Based on: Continuous Autoregressive Language Models (Shao et al., 2025)
Paper: https://arxiv.org/abs/2510.27688

Key components:
- Transformer backbone
- Energy-based generative head (single-step generation)
- Energy loss training (strictly proper scoring rule)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.

    SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊗ (xV + c)
    where Swish(x) = x * sigmoid(x)
    """

    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, intermediate_dim, bias=False)
        self.fc2 = nn.Linear(dim, intermediate_dim, bias=False)
        self.fc3 = nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation."""
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


class EnergyGenerativeHead(nn.Module):
    """
    Energy-based generative head for single-step continuous vector generation.

    Architecture:
    - Input: Hidden state h (from Transformer) + noise vector ε
    - Process: Stack of L residual MLP blocks
    - Output: Latent vector z

    Each MLP block:
    1. Fuse current representation with hidden state (2 linear layers)
    2. SwiGLU layer
    3. Residual connection
    4. Final linear projection to latent_dim

    Args:
        hidden_dim: Transformer hidden dimension
        latent_dim: Output latent vector dimension
        noise_dim: Dimension of noise vector (default: hidden_dim)
        num_blocks: Number of MLP blocks (default: 3, paper uses L = n_layers / 4)
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        noise_dim: Optional[int] = None,
        num_blocks: int = 3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim or hidden_dim
        self.num_blocks = num_blocks

        # TODO: Implement generative head architecture
        # - Noise projection
        # - Hidden state projection
        # - L residual MLP blocks
        # - Final projection to latent_dim

    def forward(self, hidden_state: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Generate latent vectors from hidden state.

        Args:
            hidden_state: (batch_size, hidden_dim) conditioning from Transformer
            num_samples: Number of samples to generate

        Returns:
            z: (batch_size, num_samples, latent_dim) generated latent vectors
        """
        # TODO: Implement forward pass
        # 1. Sample noise vectors ε ~ U[-0.5, 0.5] for each sample
        # 2. Project noise and hidden state
        # 3. Process through MLP blocks
        # 4. Final projection to latent_dim

        raise NotImplementedError


class InputCompression(nn.Module):
    """
    Compress K token embeddings into single input representation for Transformer.

    Architecture: Two-layer MLP

    Args:
        embed_dim: Token embedding dimension
        hidden_dim: Transformer hidden dimension
        K: Number of tokens per chunk
    """

    def __init__(self, embed_dim: int, hidden_dim: int, K: int):
        super().__init__()
        self.K = K

        # TODO: Implement input compression MLP
        # Two-layer MLP: K * embed_dim → hidden_dim

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compress K token embeddings to single representation.

        Args:
            token_embeddings: (batch_size, K, embed_dim) token embeddings

        Returns:
            compressed: (batch_size, hidden_dim) compressed representation
        """
        # TODO: Implement compression
        raise NotImplementedError


class CALMTransformer(nn.Module):
    """
    Complete CALM model: Transformer backbone + Energy generative head

    Architecture:
    1. Input: K tokens from previous step → embeddings → compressed input
    2. Transformer: Processes sequence of compressed inputs
    3. Generative head: Predicts next latent vector from hidden state
    4. Autoencoder decoder: Converts vector to K tokens (for next input)

    Training:
    - Energy loss: Strictly proper scoring rule
    - Multi-sample gradient: N=8 model samples, M=100 target samples

    Args:
        vocab_size: Vocabulary size
        K: Number of tokens per chunk
        hidden_dim: Transformer hidden dimension
        num_layers: Number of Transformer layers
        num_heads: Number of attention heads
        intermediate_dim: FFN intermediate dimension
        latent_dim: Latent vector dimension
        autoencoder: Pretrained frozen autoencoder
        max_seq_len: Maximum sequence length (in steps, not tokens)
    """

    def __init__(
        self,
        vocab_size: int,
        K: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        intermediate_dim: int,
        latent_dim: int,
        autoencoder: nn.Module,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.K = K
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len

        # Freeze autoencoder
        self.autoencoder = autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Input compression
        self.input_compression = InputCompression(hidden_dim, hidden_dim, K)

        # TODO: Implement Transformer backbone
        # - Standard Transformer architecture
        # - RMSNorm, SwiGLU, RoPE (following LLaMA)

        # Energy-based generative head
        num_head_blocks = max(1, num_layers // 4)
        self.generative_head = EnergyGenerativeHead(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_blocks=num_head_blocks
        )

    def forward(
        self,
        input_tokens: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        Forward pass: tokens → Transformer → generative head → latent vectors

        Args:
            input_tokens: (batch_size, seq_len, K) input token sequences
            num_samples: Number of latent samples to generate per position

        Returns:
            latent_vectors: (batch_size, seq_len, num_samples, latent_dim)
        """
        # TODO: Implement forward pass
        # 1. Embed tokens
        # 2. Compress K embeddings per position
        # 3. Process through Transformer
        # 4. Generate latent vectors from hidden states

        raise NotImplementedError

    def compute_energy_loss(
        self,
        model_samples: torch.Tensor,
        target_samples: torch.Tensor,
        alpha: float = 1.0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute energy loss (strictly proper scoring rule).

        Energy score:
        S(P, y) = E[||x' - x''||^α] - 2E[||x - y||^α]
        where x, x' ~ P (model samples), y ~ Q (target samples)

        Args:
            model_samples: (batch_size, seq_len, N, latent_dim) samples from model
            target_samples: (batch_size, seq_len, M, latent_dim) samples from target
            alpha: Energy score exponent (default: 1.0)

        Returns:
            loss: Energy loss
            metrics: Dict of loss components (diversity, fidelity)
        """
        # TODO: Implement energy loss
        # 1. Diversity term: Mean pairwise distance between model samples
        # 2. Fidelity term: Mean distance between model and target samples
        # 3. Energy loss = diversity - 2 * fidelity

        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_steps: int,
        temperature: float = 1.0,
        batch_size: int = 100
    ) -> torch.Tensor:
        """
        Autoregressive generation with temperature sampling.

        Args:
            prompt_tokens: (batch_size, prompt_len, K) prompt tokens
            max_new_steps: Maximum number of steps to generate
            temperature: Temperature for sampling (T ∈ (0, 1])
            batch_size: Batch size for approximate temperature sampling

        Returns:
            generated_tokens: (batch_size, prompt_len + max_new_steps, K) all tokens
        """
        # TODO: Implement generation loop
        # 1. Process prompt
        # 2. For each new step:
        #    - Generate latent vector (with temperature sampling)
        #    - Decode to K tokens
        #    - Use tokens as input for next step

        raise NotImplementedError


def create_calm_model(
    vocab_size: int,
    K: int = 4,
    size: str = "M",
    autoencoder: Optional[nn.Module] = None
) -> CALMTransformer:
    """
    Factory function to create CALM model with predefined sizes.

    Sizes (from paper):
    - S: 12 layers, hidden=768, intermediate=2048
    - M: 16 layers, hidden=1024, intermediate=2752
    - L: 16 layers, hidden=1536, intermediate=4096
    - XL: 16 layers, hidden=2560, intermediate=6880

    Args:
        vocab_size: Vocabulary size
        K: Number of tokens per chunk
        size: Model size preset
        autoencoder: Pretrained autoencoder (required)

    Returns:
        CALM model
    """
    if autoencoder is None:
        raise ValueError("Pretrained autoencoder is required")

    configs = {
        "S": {"num_layers": 12, "hidden_dim": 768, "intermediate_dim": 2048, "num_heads": 12},
        "M": {"num_layers": 16, "hidden_dim": 1024, "intermediate_dim": 2752, "num_heads": 16},
        "L": {"num_layers": 16, "hidden_dim": 1536, "intermediate_dim": 4096, "num_heads": 16},
        "XL": {"num_layers": 16, "hidden_dim": 2560, "intermediate_dim": 6880, "num_heads": 20},
    }

    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")

    config = configs[size]
    latent_dim = autoencoder.latent_dim

    return CALMTransformer(
        vocab_size=vocab_size,
        K=K,
        latent_dim=latent_dim,
        autoencoder=autoencoder,
        **config
    )


# TODO: Add helper functions
# - compute_flops()
# - count_parameters()
# - temperature_sampling() (both exact and approximate algorithms)
