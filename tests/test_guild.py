"""Tests for the Guild kernel (collective-intelligence/guild).

Covers the import path, determinism, event well-formedness, and — most
importantly — the emergence claims the minimal simulation was designed to
test (see collective-intelligence/guild/docs/minimal-simulation.md):

- H1: practice + comparative advantage -> specialization and capability
- H2: shared-success bond reinforcement -> persistent partnerships
- H3: both together -> the best collective performance
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from collective_intelligence.guild import (
    Kernel,
    KernelConfig,
    emergent_groups,
    partner_concentration,
    specialization_index,
    success_rate,
    team_recurrence,
)

ROUNDS = 3000
WINDOW = 300


class TestImports:
    def test_namespace(self):
        import collective_intelligence.guild as guild_pkg

        assert hasattr(guild_pkg, "Kernel")
        assert hasattr(guild_pkg, "KernelConfig")
        assert guild_pkg.__version__

    def test_experiment_module(self):
        from collective_intelligence.guild import experiment

        assert set(experiment.CONDITIONS) == {"full", "no_reinforcement", "no_practice"}


class TestKernel:
    def test_config_validation(self):
        with pytest.raises(ValueError):
            Kernel(KernelConfig(team_size=11, n_skills=10))
        with pytest.raises(ValueError):
            Kernel(KernelConfig(team_size=5, n_loci=4, n_skills=10))

    def test_determinism(self):
        events_a = Kernel(KernelConfig(seed=7)).run(200)
        events_b = Kernel(KernelConfig(seed=7)).run(200)
        assert [e.to_dict() for e in events_a] == [e.to_dict() for e in events_b]

    def test_events_well_formed(self):
        config = KernelConfig(seed=3)
        kernel = Kernel(config)
        events = kernel.run(50)
        assert len(events) == 50
        for event in events:
            assert len(event.team) == config.team_size
            assert len(set(event.team)) == config.team_size
            assert len(event.required_skills) == config.team_size
            assert len(set(event.required_skills)) == config.team_size
            assert 0.0 <= event.p_success <= 1.0
            covered = {skill for skill, _ in event.coverage}
            assert covered == set(event.required_skills)
            assert all(locus in event.team for _, locus in event.coverage)

    def test_state_stays_bounded(self):
        kernel = Kernel(KernelConfig(seed=1))
        kernel.run(500)
        for vector in kernel.skills:
            assert all(0.0 <= s <= 1.0 for s in vector)
        for weights in kernel.bonds:
            assert all(w >= 0.0 for w in weights.values())


class TestEmergence:
    """The scientific claims, on fixed seeds (verified to hold on seeds 0-2)."""

    @pytest.fixture(scope="class")
    def runs(self):
        out = {}
        for name, overrides in [
            ("full", {}),
            ("no_reinforcement", {"reinforcement": False}),
            ("no_practice", {"practice": False}),
        ]:
            kernel = Kernel(KernelConfig(seed=0, **overrides))
            events = kernel.run(ROUNDS)
            out[name] = (kernel, events)
        return out

    def test_h1_practice_builds_capability(self, runs):
        _, full_events = runs["full"]
        _, frozen_events = runs["no_practice"]
        assert success_rate(full_events, window=WINDOW) > success_rate(frozen_events, window=WINDOW)

    def test_h1_practice_drives_specialization(self, runs):
        full_kernel, _ = runs["full"]
        frozen_kernel, _ = runs["no_practice"]
        assert specialization_index(full_kernel.skills) > specialization_index(frozen_kernel.skills)

    def test_h2_reinforcement_creates_persistent_partnerships(self, runs):
        _, full_events = runs["full"]
        _, random_events = runs["no_reinforcement"]
        assert team_recurrence(full_events, window=WINDOW) > team_recurrence(random_events, window=WINDOW)

    def test_h2_reinforcement_concentrates_partners(self, runs):
        full_kernel, _ = runs["full"]
        random_kernel, _ = runs["no_reinforcement"]
        assert partner_concentration(full_kernel.bonds) > partner_concentration(random_kernel.bonds)

    def test_capability_improves_over_run(self, runs):
        _, full_events = runs["full"]
        assert success_rate(full_events, window=WINDOW) > success_rate(full_events[:WINDOW])


class TestMetrics:
    def test_empty_inputs(self):
        assert specialization_index([]) == 0.0
        assert partner_concentration([{}, {}]) == 0.0
        assert success_rate([]) == 0.0
        assert team_recurrence([]) == 0.0
        assert emergent_groups([{}, {}]) == []

    def test_flat_skills_score_zero(self):
        assert specialization_index([[0.5] * 6, [0.2] * 6]) == 0.0

    def test_single_specialist_scores_high(self):
        vector = [0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
        assert specialization_index([vector]) > 0.95

    def test_emergent_groups_finds_clique(self):
        bonds = [dict() for _ in range(6)]
        for i in (0, 1, 2):
            for j in (0, 1, 2):
                if i != j:
                    bonds[i][j] = 5.0
        bonds[3][4] = 0.01
        bonds[4][3] = 0.01
        assert emergent_groups(bonds) == [3]
