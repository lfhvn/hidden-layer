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
    # Social transmission: when a covered skill's coverer is at or above
    # mentor_threshold, teammates absorb a fraction of the gap. Off by
    # default (0.0) — the minimal kernel results hold without it. With it,
    # competence can outlive individual practice (a "learning culture"),
    # which introduces hysteresis: early investment can move the population
    # to a self-sustaining attractor. See docs/minimal-simulation.md §7.
    mentorship: float = 0.0
    mentor_threshold: float = 0.7
    # Cultural ratchet: when tradition_experts loci are simultaneously at or
    # above tradition_expert_level in a skill, that skill becomes a
    # "tradition" (permanently): the baseline that forgetting decays toward
    # rises to tradition_floor for everyone. Off by default (0 = disabled).
    # This is the kernel's model of cumulative culture — competence that
    # outlives the individuals who built it — and the mechanism that makes
    # self-sustaining collectives (rather than steady-state subsistence)
    # reachable. See docs/minimal-simulation.md §7.
    tradition_experts: int = 0
    tradition_expert_level: float = 0.7
    tradition_floor: float = 0.4
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
        self.traditions: set[int] = set()
        # Optional per-skill weights for task arrival (environment shaping);
        # None means uniform. Games/experiments may set this directly.
        self.task_weights: list[float] | None = None
        self._round = 0

    # ------------------------------------------------------------------
    # Event generation
    # ------------------------------------------------------------------

    def _sample_task(self) -> Task:
        cfg = self.config
        if self.task_weights is None:
            required = tuple(sorted(self.rng.sample(range(cfg.n_skills), cfg.team_size)))
        else:
            chosen: list[int] = []
            while len(chosen) < cfg.team_size:
                candidates = [k for k in range(cfg.n_skills) if k not in chosen]
                weights = [self.task_weights[k] for k in candidates]
                chosen.append(self.rng.choices(candidates, weights=weights, k=1)[0])
            required = tuple(sorted(chosen))
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
            if cfg.mentorship > 0:
                for skill, locus in coverage:
                    master = self.skills[locus][skill]
                    if master >= cfg.mentor_threshold:
                        for m in team:
                            s = self.skills[m][skill]
                            if m != locus and s < master:
                                self.skills[m][skill] = s + cfg.mentorship * (master - s)
        if success:
            for i in team:
                for j in team:
                    if i != j:
                        self.bonds[i][j] = self.bonds[i].get(j, 0.0) + cfg.bond_gain

        # Cultural ratchet: a skill held at expert level by enough loci at
        # once becomes a tradition — the forgetting floor rises permanently.
        if cfg.tradition_experts > 0:
            for k in range(cfg.n_skills):
                if k not in self.traditions:
                    experts = sum(1 for v in self.skills if v[k] >= cfg.tradition_expert_level)
                    if experts >= cfg.tradition_experts:
                        self.traditions.add(k)

        # Forgetting: unexercised competence relaxes toward baseline. This is
        # what makes specialization a real commitment — no locus can maintain
        # every skill, so who you repeatedly act with determines what the
        # collective can cover. Traditions raise the baseline itself.
        if cfg.practice and cfg.skill_decay > 0:
            keep = 1.0 - cfg.skill_decay
            for vector in self.skills:
                for k in range(cfg.n_skills):
                    base = cfg.tradition_floor if k in self.traditions else cfg.initial_skill
                    if vector[k] > base:
                        vector[k] = base + (vector[k] - base) * keep
                    elif k in self.traditions and vector[k] < base:
                        # newcomers / the rusty are lifted by living tradition
                        vector[k] = min(base, vector[k] + cfg.skill_decay * (base - vector[k]) * 4)

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
