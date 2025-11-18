"""Policies for GraphRAG ablation runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class AblationConfig:
    name: str
    enabled: bool
    description: str


def graph_ablation_plan() -> List[AblationConfig]:
    """Return the canonical ablation plan for GraphRAG."""

    return [
        AblationConfig(name="graph_on", enabled=True, description="GraphRAG entity graph + community summaries"),
        AblationConfig(name="graph_off", enabled=False, description="Vector-only retrieval without graph context"),
    ]
