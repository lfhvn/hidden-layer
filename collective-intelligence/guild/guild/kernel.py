"""The Guild kernel: a minimal interaction-first simulation.

The kernel deliberately avoids organizational metaphors. There are no agents
with roles, no teams as data structures, no managers, no messages with
semantics. The ontology is:

- **Loci**: interaction endpoints. A locus is an integer id plus two pieces of
  adaptive state — a skill vector (shaped by practice) and bond weights to
  other loci (shaped by shared outcomes). Loci have no goals, plans, or code.

- **Interaction events**: each round, one joint ACT event occurs — a set of
  loci attempts a task together. The event either succeeds or fails.

- **Plasticity rules**: two local update rules close the loop.
  REINFORCE: shared success strengthens pairwise bonds; all bonds decay.
  PRACTICE: the skill a locus actually exercised improves slightly.

Everything else — specialization, stable partnerships, group structure — must
emerge from repeated events, or it does not exist. That is the experiment.

The kernel is pure standard-library Python, fully seeded, and logs every
event, so runs are cheap, reproducible, and inspectable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    """A task is a demand for distinct skills at a difficulty level.

    Tasks are the environment's only pressure. They carry no instructions
    about who should do them or how work should be divided.
    """

    required_skills: tuple[int, ...]
    difficulty: float


@dataclass
class KernelConfig:
    """Configuration for a kernel run.

    The two boolean switches are the scientific controls:
    - ``reinforcement``: partner selection is biased by bond weights
      (off = partners are chosen uniformly at random every round).
    - ``practice``: exercised skills improve with use
      (off = skill vectors stay at their initial values).
    """

    n_loci: int = 24
    n_skills: int = 10
    team_size: int = 3
    difficulty: float = 0.5
    initial_skill: float = 0.25
    skill_jitter: float = 0.05
    learning_rate: float = 0.2
    skill_decay: float = 0.005
    bond_gain: float = 1.0
    bond_decay: float = 0.01
    exploration: float = 0.05
    reinforcement: bool = True
    practice: bool = True
    seed: int = 0


@dataclass
class Event:
    """One joint ACT event: who interacted, over what, and what happened."""

    round: int
    team: tuple[int, ...]
    required_skills: tuple[int, ...]
    coverage: tuple[tuple[int, int], ...]  # (skill, locus) assignments
    p_success: float
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "team": list(self.team),
            "required_skills": list(self.required_skills),
            "coverage": [list(pair) for pair in self.coverage],
            "p_success": self.p_success,
            "success": self.success,
        }


class Kernel:
    """Minimal interaction-first world.

    Usage::

        kernel = Kernel(KernelConfig(seed=0))
        events = kernel.run(rounds=600)
        # kernel.skills, kernel.bonds, and events are the full record
    """

    def __init__(self, config: KernelConfig | None = None):
        self.config = config or KernelConfig()
        cfg = self.config
        if cfg.team_size > cfg.n_skills:
            raise ValueError("team_size cannot exceed n_skills (tasks need distinct skills)")
        if cfg.team_size > cfg.n_loci:
            raise ValueError("team_size cannot exceed n_loci")

        self.rng = random.Random(cfg.seed)
        self.skills: list[list[float]] = [
            [
                min(1.0, max(0.0, cfg.initial_skill + self.rng.uniform(-cfg.skill_jitter, cfg.skill_jitter)))
                for _ in range(cfg.n_skills)
            ]
            for _ in range(cfg.n_loci)
        ]
        # bonds[i][j] is the (directedly stored, symmetrically updated) weight
        # locus i places on past shared success with locus j.
        self.bonds: list[dict[int, float]] = [dict() for _ in range(cfg.n_loci)]
        self.events: list[Event] = []
        self._round = 0

    # ------------------------------------------------------------------
    # Event generation
    # ------------------------------------------------------------------

    def _sample_task(self) -> Task:
        cfg = self.config
        required = tuple(sorted(self.rng.sample(range(cfg.n_skills), cfg.team_size)))
        return Task(required_skills=required, difficulty=cfg.difficulty)

    def _recruit(self) -> list[int]:
        """Form the round's interaction set, one locus at a time.

        A random convener starts; each subsequent participant is sampled with
        probability proportional to (exploration + summed bond weight to the
        current set). With reinforcement off, sampling is uniform — that is
        the null model for organization.
        """
        cfg = self.config
        team = [self.rng.randrange(cfg.n_loci)]
        while len(team) < cfg.team_size:
            candidates = [j for j in range(cfg.n_loci) if j not in team]
            if cfg.reinforcement:
                weights = [
                    cfg.exploration + sum(self.bonds[m].get(j, 0.0) for m in team)
                    for j in candidates
                ]
            else:
                weights = [1.0] * len(candidates)
            team.append(self.rng.choices(candidates, weights=weights, k=1)[0])
        return team

    def _cover(self, team: list[int], task: Task) -> list[tuple[int, int]]:
        """Assign each required skill to the team member currently best at it.

        This is local comparative advantage, not role assignment: nothing is
        remembered about the mapping, and it is recomputed every event.
        """
        coverage = []
        for skill in task.required_skills:
            best = max(team, key=lambda m: self.skills[m][skill])
            coverage.append((skill, best))
        return coverage

    def step(self) -> Event:
        """Run one round: task arrival, recruitment, joint act, plasticity."""
        cfg = self.config
        task = self._sample_task()
        team = self._recruit()
        coverage = self._cover(team, task)

        p_success = 1.0
        for skill, locus in coverage:
            s = self.skills[locus][skill]
            p_success *= s / (s + task.difficulty)
        success = self.rng.random() < p_success

        # Learning by doing: whatever skill a locus actually exercised
        # improves, success or not. Only shared *success* strengthens bonds.
        if cfg.practice:
            for skill, locus in coverage:
                s = self.skills[locus][skill]
                self.skills[locus][skill] = s + cfg.learning_rate * (1.0 - s)
        if success:
            for i in team:
                for j in team:
                    if i != j:
                        self.bonds[i][j] = self.bonds[i].get(j, 0.0) + cfg.bond_gain

        # Forgetting: unexercised competence relaxes toward baseline. This is
        # what makes specialization a real commitment — no locus can maintain
        # every skill, so who you repeatedly act with determines what the
        # collective can cover.
        if cfg.practice and cfg.skill_decay > 0:
            base = cfg.initial_skill
            keep = 1.0 - cfg.skill_decay
            for vector in self.skills:
                for k in range(cfg.n_skills):
                    if vector[k] > base:
                        vector[k] = base + (vector[k] - base) * keep

        decay = 1.0 - cfg.bond_decay
        for i in range(cfg.n_loci):
            for j in list(self.bonds[i]):
                self.bonds[i][j] *= decay
                if self.bonds[i][j] < 1e-6:
                    del self.bonds[i][j]

        event = Event(
            round=self._round,
            team=tuple(sorted(team)),
            required_skills=task.required_skills,
            coverage=tuple(coverage),
            p_success=p_success,
            success=success,
        )
        self.events.append(event)
        self._round += 1
        return event

    def run(self, rounds: int) -> list[Event]:
        """Run ``rounds`` events and return the events generated by this call."""
        start = len(self.events)
        for _ in range(rounds):
            self.step()
        return self.events[start:]
