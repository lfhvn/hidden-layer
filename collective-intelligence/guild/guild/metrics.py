"""Emergence metrics for the Guild kernel.

Each metric operationalizes one claim about emergent organization. None of
the measured structure exists as a data type in the kernel — specialization,
partnerships, and groups are read off the event log and the adaptive state.
"""

from __future__ import annotations

import math
from .kernel import Event


def specialization_index(skills: list[list[float]]) -> float:
    """Mean specialization across loci, in [0, 1].

    For each locus, take its skill in excess of its own weakest skill,
    normalize that into a distribution, and compute 1 - normalized entropy.
    A locus with a flat skill vector scores 0 (nothing distinguishes it);
    a locus whose competence is concentrated in one skill scores near 1.
    Division of labor shows up as this index rising over a run.
    """
    if not skills:
        return 0.0
    n_skills = len(skills[0])
    if n_skills < 2:
        return 0.0
    total = 0.0
    for vector in skills:
        floor = min(vector)
        excess = [s - floor for s in vector]
        mass = sum(excess)
        if mass <= 1e-9:
            continue
        entropy = 0.0
        for e in excess:
            p = e / mass
            if p > 0:
                entropy -= p * math.log(p)
        total += 1.0 - entropy / math.log(n_skills)
    return total / len(skills)


def partner_concentration(bonds: list[dict[int, float]]) -> float:
    """Mean Herfindahl index of each locus's bond distribution, in [0, 1].

    Measures whether loci interact with everyone a little (low) or with a
    few stable partners a lot (high). Only loci with at least one bond
    contribute; if no bonds exist at all, returns 0.
    """
    indices = []
    for weights in bonds:
        mass = sum(weights.values())
        if mass <= 0:
            continue
        indices.append(sum((w / mass) ** 2 for w in weights.values()))
    if not indices:
        return 0.0
    return sum(indices) / len(indices)


def success_rate(events: list[Event], window: int | None = None) -> float:
    """Fraction of successful events, optionally over the trailing window."""
    if window is not None:
        events = events[-window:]
    if not events:
        return 0.0
    return sum(1 for e in events if e.success) / len(events)


def team_recurrence(events: list[Event], window: int | None = None) -> float:
    """Fraction of (windowed) events whose exact interaction set repeats
    within that window.

    High recurrence means the same sets of loci keep re-forming — persistent
    partnerships emerging from a process with no team memory of its own.
    Under uniform-random recruitment this stays near the birthday-collision
    baseline; under lock-in it approaches 1.
    """
    if window is not None:
        events = events[-window:]
    if not events:
        return 0.0
    counts: dict[tuple[int, ...], int] = {}
    for event in events:
        counts[event.team] = counts.get(event.team, 0) + 1
    repeated = sum(c for c in counts.values() if c >= 2)
    return repeated / len(events)


def emergent_groups(bonds: list[dict[int, float]], threshold: float | None = None) -> list[int]:
    """Sizes of connected components in the strong-bond graph.

    An edge exists between i and j when their symmetrized bond weight is at
    or above ``threshold`` (default: the mean positive symmetrized weight).
    Components of size >= 2 are reported, largest first. These are the
    closest thing the kernel has to organizations — and they are entirely
    an observer-side construct.
    """
    n = len(bonds)
    weights: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j, w in bonds[i].items():
            key = (min(i, j), max(i, j))
            weights[key] = weights.get(key, 0.0) + w
    positive = [w for w in weights.values() if w > 0]
    if not positive:
        return []
    if threshold is None:
        threshold = sum(positive) / len(positive)

    adjacency: dict[int, set[int]] = {}
    for (i, j), w in weights.items():
        if w >= threshold:
            adjacency.setdefault(i, set()).add(j)
            adjacency.setdefault(j, set()).add(i)

    sizes = []
    visited: set[int] = set()
    for start in adjacency:
        if start in visited:
            continue
        stack = [start]
        component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency.get(node, ()) - component)
        visited |= component
        if len(component) >= 2:
            sizes.append(len(component))
    return sorted(sizes, reverse=True)
