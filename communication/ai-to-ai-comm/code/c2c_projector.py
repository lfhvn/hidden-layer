"""
Cache-to-Cache (C2C) Projector

Implements the core projection network for transforming KV-Caches between different LLM architectures.
Based on the paper: "Cache-to-Cache: Direct Semantic Communication Between Large Language Models"
https://arxiv.org/abs/2510.03215
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch import Tensor


class StandardFFNLayer(nn.Module):
    """
    Pre-norm RMSNorm with standard MLP:
      y = x + Dropout( W2( Act( W1( RMSNorm(x) ) ) ) )
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "relu":
            self.act = nn.ReLU()
        elif activation.lower() == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h = self.act(self.w1(h))
        h = self.w2(h)
        h = self.drop(h)
        return x + h


class RegularMLP(nn.Module):
    """
    Stacked MLP with residual connections.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        intermediate_dim: int = 3072,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.blocks = nn.ModuleList([
            StandardFFNLayer(
                hidden_size=hidden_dim,
                intermediate_size=intermediate_dim,
                dropout=dropout,
                activation=activation,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class C2CProjector(nn.Module):
    """
    Cache-to-Cache Projector for direct semantic communication between LLMs.

    Architecture:
    1. Concatenate source and target KV features
    2. Project to hidden dimension
    3. Compute projected features (for actual transformation)
    4. Compute scalar weights (for blending)
    5. Apply gated residual connection: target + gate * weight * projected

    Args:
        source_dim: Dimension per head of source model
        target_dim: Dimension per head of target model
        source_num_heads: Number of attention heads in source model
        target_num_heads: Number of attention heads in target model
        hidden_dim: Hidden dimension for MLP networks
        intermediate_dim: Intermediate dimension for FFN layers
        num_layers: Number of MLP layers (minimum 3)
        dropout: Dropout rate
        initial_temperature: Initial temperature for annealing
        final_temperature: Final temperature for annealing
        anneal_steps: Number of steps for temperature annealing
        dtype: Data type for parameters
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        source_num_heads: int = 1,
        target_num_heads: int = 1,
        intermediate_dim: int = 1024,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.1,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.001,
        anneal_steps: int = 1929,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        assert num_layers >= 3, "num_layers must be >= 3"

        # Dimensions
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.source_num_heads = source_num_heads
        self.target_num_heads = target_num_heads

        # Sizes
        in_dim = source_dim * source_num_heads
        out_dim = target_dim * target_num_heads

        # 1) Concatenate source and target, project to hidden_dim
        self.key_in = nn.Linear(in_dim + out_dim, hidden_dim, bias=True, dtype=dtype)
        self.value_in = nn.Linear(in_dim + out_dim, hidden_dim, bias=True, dtype=dtype)

        # 2) One-layer common embedding MLP
        self.key_mlp1 = RegularMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=1,
            dropout=dropout,
            dtype=dtype
        )
        self.value_mlp1 = RegularMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=1,
            dropout=dropout,
            dtype=dtype
        )

        # 3a) Scalar weight path
        self.key_scalar_mlp2 = RegularMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=hidden_dim,
            num_layers=1,
            dropout=dropout,
            dtype=dtype
        )
        self.value_scalar_mlp2 = RegularMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=hidden_dim,
            num_layers=1,
            dropout=dropout,
            dtype=dtype
        )
        self.key_scalar_head = nn.Linear(hidden_dim, target_num_heads, dtype=dtype)
        self.value_scalar_head = nn.Linear(hidden_dim, target_num_heads, dtype=dtype)

        # 3b) Projected feature path
        self.key_proj_mlp2 = RegularMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers - 2,
            dropout=dropout,
            dtype=dtype
        )
        self.value_proj_mlp2 = RegularMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers - 2,
            dropout=dropout,
            dtype=dtype
        )
        self.key_proj_out = nn.Linear(hidden_dim, out_dim, bias=True, dtype=dtype)
        self.value_proj_out = nn.Linear(hidden_dim, out_dim, bias=True, dtype=dtype)

        # Scalar gate parameters
        self.key_gate_logit = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.value_gate_logit = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.use_gumbel = True

        # Temperature annealing
        self.register_buffer("gate_temperature", torch.tensor(initial_temperature, dtype=dtype))
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.anneal_steps = anneal_steps
        self.scalar_temperature = 1.0

    def update_temperature(self, step: int):
        """Update temperature using exponential annealing."""
        ratio = min(step / self.anneal_steps, 1.0)
        temp = self.initial_temperature * (self.final_temperature / self.initial_temperature) ** ratio
        self.gate_temperature.fill_(temp)

    def forward(
        self,
        source_kv: Tuple[Tensor, Tensor],
        target_kv: Tuple[Tensor, Tensor],
        position_ids: Optional[Tensor] = None,
        max_pos: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Transform source KV-Cache to target model's semantic space.

        Args:
            source_kv: Tuple of (key, value) tensors, each (B, H_s, N, D_s)
            target_kv: Tuple of (key, value) tensors, each (B, H_t, N, D_t)
            position_ids: Optional position IDs
            max_pos: Optional maximum position

        Returns:
            Tuple of (key, value) tensors, each (B, H_t, N, D_t)
        """
        source_key, source_value = source_kv
        target_key, target_value = target_kv

        B, Hs, N, Ds = source_key.shape
        _, Ht, _, Dt = target_key.shape

        # Flatten heads: (B, H, N, D) -> (B, N, H*D)
        source_key_flat = source_key.transpose(1, 2).contiguous().view(B, N, Hs * Ds)
        source_value_flat = source_value.transpose(1, 2).contiguous().view(B, N, Hs * Ds)
        target_key_flat = target_key.transpose(1, 2).contiguous().view(B, N, Ht * Dt)
        target_value_flat = target_value.transpose(1, 2).contiguous().view(B, N, Ht * Dt)

        # 1) Concatenate source and target features
        key_cat = torch.cat([source_key_flat, target_key_flat], dim=-1)
        value_cat = torch.cat([source_value_flat, target_value_flat], dim=-1)

        # 2) Project to hidden dim
        key_hidden = self.key_in(key_cat)
        value_hidden = self.value_in(value_cat)

        # 3) Common embedding MLP
        key_hidden = self.key_mlp1(key_hidden)
        value_hidden = self.value_mlp1(value_hidden)

        # 4a) Projected feature path
        key_proj_hidden = self.key_proj_out(self.key_proj_mlp2(key_hidden))  # (B, N, Ht * Dt)
        value_proj_hidden = self.value_proj_out(self.value_proj_mlp2(value_hidden))  # (B, N, Ht * Dt)
        projected_key = key_proj_hidden.view(B, N, Ht, Dt).transpose(1, 2)  # (B, Ht, N, Dt)
        projected_value = value_proj_hidden.view(B, N, Ht, Dt).transpose(1, 2)  # (B, Ht, N, Dt)

        # 4b) Scalar weight path
        key_scalar = self.key_scalar_head(self.key_scalar_mlp2(key_hidden))  # (B, N, Ht)
        value_scalar = self.value_scalar_head(self.value_scalar_mlp2(value_hidden))  # (B, N, Ht)
        key_scalar = key_scalar.permute(0, 2, 1).unsqueeze(-1)  # (B, Ht, N, 1)
        value_scalar = value_scalar.permute(0, 2, 1).unsqueeze(-1)  # (B, Ht, N, 1)

        # 5) Compute gates with Gumbel noise
        key_gate_logit = self.key_gate_logit.view(1, 1, 1, 1)
        value_gate_logit = self.value_gate_logit.view(1, 1, 1, 1)

        if self.training and self.use_gumbel:
            # Add Gumbel noise for exploration during training
            u1 = torch.rand(B, Ht, N, 1, device=key_gate_logit.device, dtype=key_gate_logit.dtype)
            u2 = torch.rand(B, Ht, N, 1, device=value_gate_logit.device, dtype=value_gate_logit.dtype)
            g1 = -torch.log(-torch.log(u1 + 1e-20) + 1e-20)
            g2 = -torch.log(-torch.log(u2 + 1e-20) + 1e-20)
            key_gate = torch.sigmoid((key_gate_logit + g1) / self.gate_temperature)
            value_gate = torch.sigmoid((value_gate_logit + g2) / self.gate_temperature)
        else:
            # Deterministic gate during inference
            key_gate = (key_gate_logit > 0).float()
            value_gate = (value_gate_logit > 0).float()

        # 6) Normalize scalar weights
        norm_key_scalar = torch.sigmoid(key_scalar)
        norm_value_scalar = torch.sigmoid(value_scalar)

        # 7) Final combination: target + gate * weight * projected
        output_key = target_key + key_gate * norm_key_scalar * projected_key
        output_value = target_value + value_gate * norm_value_scalar * projected_value

        return output_key, output_value


def create_c2c_projector(
    source_model_config,
    target_model_config,
    **kwargs
) -> C2CProjector:
    """
    Factory function to create a C2CProjector from model configs.

    Args:
        source_model_config: Source model configuration (HuggingFace config)
        target_model_config: Target model configuration (HuggingFace config)
        **kwargs: Additional projector arguments

    Returns:
        C2CProjector instance
    """
    source_dim = source_model_config.hidden_size // source_model_config.num_attention_heads
    target_dim = target_model_config.hidden_size // target_model_config.num_attention_heads

    return C2CProjector(
        source_dim=source_dim,
        target_dim=target_dim,
        source_num_heads=source_model_config.num_attention_heads,
        target_num_heads=target_model_config.num_attention_heads,
        **kwargs
    )
