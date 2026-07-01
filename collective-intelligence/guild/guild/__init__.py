"""Guild: an interaction-first environment for studying collective intelligence.

This package contains the Guild kernel — the smallest simulation we could
design that produces emergent coordination (division of labor and stable
groups) without hardcoding agents' roles, teams, or organizations.

Design principles (see collective-intelligence/guild/PRD.md):
- Interactions are the primitive objects; entities are minimal endpoints.
- Organizations must emerge; nothing above the interaction level is scripted.
- Every run produces a structured event log suitable for research analysis.
"""

from __future__ import annotations

from .kernel import Kernel, KernelConfig, Task
from .metrics import (
    emergent_groups,
    partner_concentration,
    specialization_index,
    success_rate,
    team_recurrence,
)

__version__ = "0.1.0"

__all__ = [
    "Kernel",
    "KernelConfig",
    "Task",
    "specialization_index",
    "partner_concentration",
    "success_rate",
    "team_recurrence",
    "emergent_groups",
    "__version__",
]
