"""Minimal-simulation experiment: does coordination emerge, and from what?

Runs the kernel under three conditions across multiple seeds:

- ``full``: bond reinforcement + practice (the hypothesized minimal recipe)
- ``no_reinforcement``: partners chosen uniformly at random every round
- ``no_practice``: skills frozen at their initial values

Hypotheses (see docs/minimal-simulation.md):
- H1: practice + local comparative advantage -> division of labor
      (specialization_index rises only when practice is on).
- H2: shared-outcome bond reinforcement -> persistent partnerships and
      group structure (partner_concentration, team_recurrence, and
      emergent_groups are high only when reinforcement is on).
- H3: the two mechanisms compound — collective performance (late success
      rate) is highest with both, and each ablation costs performance.

Usage::

    python -m guild.experiment [--rounds 600] [--seeds 5] [--json out.json]
    # or, from the repo root:
    python -m collective_intelligence.guild.experiment
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from typing import Any

from .kernel import Kernel, KernelConfig
from .metrics import (
    emergent_groups,
    partner_concentration,
    specialization_index,
    success_rate,
    team_recurrence,
)

CONDITIONS: dict[str, dict[str, bool]] = {
    "full": {},
    "no_reinforcement": {"reinforcement": False},
    "no_practice": {"practice": False},
}

WINDOW = 100  # trailing window for "late" behavior


def run_once(config: KernelConfig, rounds: int) -> dict[str, Any]:
    """Run one kernel and return its emergence metrics."""
    kernel = Kernel(config)
    events = kernel.run(rounds)
    groups = emergent_groups(kernel.bonds)
    return {
        "success_rate_early": success_rate(events[:WINDOW]),
        "success_rate_late": success_rate(events, window=WINDOW),
        "specialization": specialization_index(kernel.skills),
        "partner_concentration": partner_concentration(kernel.bonds),
        "team_recurrence_late": team_recurrence(events, window=WINDOW),
        "n_groups": len(groups),
        "group_sizes": groups,
    }


def run_condition(name: str, rounds: int, seeds: list[int]) -> dict[str, Any]:
    """Run one condition across seeds and aggregate (mean over seeds)."""
    overrides = CONDITIONS[name]
    runs = []
    for seed in seeds:
        config = replace(KernelConfig(seed=seed), **overrides)
        runs.append(run_once(config, rounds))
    numeric = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
    summary: dict[str, Any] = {
        key: sum(r[key] for r in runs) / len(runs) for key in numeric
    }
    summary["group_sizes_by_seed"] = [r["group_sizes"] for r in runs]
    return {"condition": name, "seeds": seeds, "rounds": rounds, "mean": summary, "runs": runs}


def run_experiment(rounds: int = 600, seeds: list[int] | None = None) -> dict[str, Any]:
    seeds = seeds if seeds is not None else list(range(5))
    return {name: run_condition(name, rounds, seeds) for name in CONDITIONS}


def format_table(results: dict[str, Any]) -> str:
    columns = [
        ("success_rate_early", "succ@100"),
        ("success_rate_late", "succ@end"),
        ("specialization", "special."),
        ("partner_concentration", "partner"),
        ("team_recurrence_late", "recur."),
        ("n_groups", "groups"),
    ]
    header = f"{'condition':<18}" + "".join(f"{label:>10}" for _, label in columns)
    lines = [header, "-" * len(header)]
    for name, result in results.items():
        mean = result["mean"]
        row = f"{name:<18}" + "".join(f"{mean[key]:>10.3f}" for key, _ in columns)
        lines.append(row)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds (0..n-1)")
    parser.add_argument("--json", type=str, default=None, help="write full results to this path")
    args = parser.parse_args(argv)

    results = run_experiment(rounds=args.rounds, seeds=list(range(args.seeds)))
    print(format_table(results))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results written to {args.json}")


if __name__ == "__main__":
    main()
