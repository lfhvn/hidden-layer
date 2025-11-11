"""Policy helpers for TTL/LoRA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TTLBudget:
    max_tokens: int
    steps_per_batch: int
    replay_anchor: int


@dataclass
class TTLPolicy:
    objective: str
    budget: TTLBudget
    sample_strategy: Dict[str, float]


def build_ttl_policy(config: Dict[str, Any]) -> TTLPolicy:
    ttl_cfg = config.get("ttl", {})
    budget = TTLBudget(
        max_tokens=ttl_cfg.get("max_tokens_update", 0),
        steps_per_batch=ttl_cfg.get("steps_per_batch", 1),
        replay_anchor=ttl_cfg.get("replay_anchor", 0),
    )
    return TTLPolicy(
        objective=ttl_cfg.get("objective", "input_perplexity_min"),
        budget=budget,
        sample_strategy=ttl_cfg.get("sample_strategy", {}),
    )
