"""Policies for RAPTOR hierarchy ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class HierarchyAblation:
    name: str
    enabled: bool
    description: str


def raptor_ablation_plan() -> List[HierarchyAblation]:
    return [
        HierarchyAblation(
            name="raptor_on",
            enabled=True,
            description="Hierarchical week→month→quarter summaries enabled",
        ),
        HierarchyAblation(
            name="raptor_off",
            enabled=False,
            description="Disable hierarchical summaries to measure baseline",
        ),
    ]
