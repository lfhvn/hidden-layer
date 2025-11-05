"""Core functionality tests for Hidden Layer.

These tests verify basic functionality without requiring external
services (no LLM calls) and ensure the reorganized packages import
correctly.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unittest.mock import MagicMock, Mock

import pytest


class TestHarnessCore:
    """Test harness core functionality."""

    def test_strategy_registry(self):
        """Test that strategy registry is properly populated."""
        from communication.multi_agent import STRATEGIES

        expected_strategies = {"single", "debate", "self_consistency", "manager_worker", "consensus"}
        assert expected_strategies.issubset(STRATEGIES.keys())
        for strategy, fn in STRATEGIES.items():
            assert callable(fn), f"Strategy '{strategy}' should be callable"

    def test_eval_functions_registry(self):
        """Test that evaluation functions registry is populated."""
        from harness import EVAL_FUNCTIONS

        # Should have basic eval functions
        expected_evals = {"exact_match", "keyword", "numeric"}
        assert expected_evals.issubset(EVAL_FUNCTIONS.keys())
        for name in expected_evals:
            assert callable(EVAL_FUNCTIONS[name])

    def test_exact_match_eval(self):
        """Test exact match evaluation."""
        from harness import exact_match

        assert exact_match("hello", "hello") == 1.0
        assert exact_match("hello", "Hello") == 1.0  # default case-insensitive
        assert exact_match("hello", "world") == 0.0

    def test_keyword_match_eval(self):
        """Test keyword match evaluation."""
        from harness import keyword_match

        # Test with single keyword
        assert keyword_match("The sky is blue", ["blue"]) == 1.0
        assert keyword_match("The sky is blue", ["red"]) == 0.0

        # Test with multiple keywords
        assert keyword_match("The sky is blue and beautiful", ["blue", "beautiful"], require_all=True) == 1.0
        assert keyword_match("The sky is blue", ["blue", "red"], require_all=True) == 0.0

    def test_numeric_match_eval(self):
        """Test numeric match evaluation."""
        from harness import numeric_match

        # Exact match
        assert numeric_match("42", 42) == 1.0
        assert numeric_match("3.14", 3.14, tolerance=0.01) == 1.0

        # Within tolerance
        assert numeric_match("42.1", 42, tolerance=0.2) == 1.0
        assert numeric_match("42.5", 42, tolerance=0.2) == 0.0

    def test_experiment_config_creation(self):
        """Test creating experiment configuration."""
        from harness import ExperimentConfig

        config = ExperimentConfig(
            experiment_name="test_exp",
            strategy="debate",
            task_type="analysis",
            provider="ollama",
            model="llama3.2:latest",
        )

        assert config.experiment_name == "test_exp"
        assert config.strategy == "debate"
        assert config.provider == "ollama"

    def test_strategy_result_creation(self):
        """Test creating strategy result."""
        from communication.multi_agent import StrategyResult

        result = StrategyResult(
            output="Test output",
            strategy_name="single",
            latency_s=1.5,
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.01,
            metadata={"test": "data"},
        )

        assert result.output == "Test output"
        assert result.latency_s == 1.5
        assert result.metadata["test"] == "data"


class TestCRITCore:
    """Test CRIT core functionality."""

    def test_design_domains(self):
        """Test that design domains are defined."""
        from communication.multi_agent.crit import DesignDomain

        expected_domains = {"UI_UX", "API", "SYSTEM", "DATA", "WORKFLOW"}
        actual_domains = {domain.name for domain in DesignDomain}
        assert expected_domains.issubset(actual_domains)

    def test_design_problems_exist(self):
        """Test that all design problems are defined."""
        from communication.multi_agent.crit import (
            APPROVAL_WORKFLOW,
            CACHING_STRATEGY,
            DASHBOARD_LAYOUT,
            GRAPHQL_SCHEMA,
            MICROSERVICES_SPLIT,
            MOBILE_CHECKOUT,
            PERMISSION_MODEL,
            REST_API_VERSIONING,
        )

        problems = [
            MOBILE_CHECKOUT,
            DASHBOARD_LAYOUT,
            REST_API_VERSIONING,
            GRAPHQL_SCHEMA,
            MICROSERVICES_SPLIT,
            CACHING_STRATEGY,
            PERMISSION_MODEL,
            APPROVAL_WORKFLOW,
        ]

        for problem in problems:
            assert problem.name is not None
            assert problem.description is not None
            assert len(problem.success_criteria) > 0
            assert problem.difficulty in ["easy", "medium", "hard"]

    def test_critique_perspectives(self):
        """Test that critique perspectives are defined."""
        from communication.multi_agent.crit import CritiquePerspective

        expected_perspectives = {
            "USABILITY",
            "ACCESSIBILITY",
            "SECURITY",
            "PERFORMANCE",
            "AESTHETICS",
        }

        actual = {p.name for p in CritiquePerspective}
        assert expected_perspectives.issubset(actual)

    def test_critique_result_creation(self):
        """Test creating critique result."""
        from communication.multi_agent.crit import CritiqueResult

        result = CritiqueResult(
            problem_name="test_problem",
            strategy_name="single",
            critiques=[{"perspective": "general", "critique": "Looks good."}],
            synthesis="Looks good.",
            recommendations=["Ship it"],
            revised_design=None,
            latency_s=2.0,
            total_tokens_in=200,
            total_tokens_out=150,
            total_cost_usd=0.01,
            metadata={"perspectives": ["usability"]},
        )

        assert result.strategy_name == "single"
        assert result.recommendations == ["Ship it"]


class TestSELPHICore:
    """Test SELPHI core functionality."""

    def test_tom_types(self):
        """Test that ToM types are defined."""
        from theory_of_mind.selphi import ToMType

        expected_types = [
            "FALSE_BELIEF",
            "KNOWLEDGE_ATTRIBUTION",
            "PERSPECTIVE_TAKING",
            "BELIEF_UPDATING",
            "SECOND_ORDER_BELIEF",
            "EPISTEMIC_STATE",
            "PRAGMATIC_REASONING",
        ]

        for tom_type in expected_types:
            assert hasattr(ToMType, tom_type), f"ToM type '{tom_type}' not found"

    def test_scenarios_exist(self):
        """Test that all scenarios are defined."""
        from theory_of_mind.selphi import ALL_SCENARIOS

        assert len(ALL_SCENARIOS) >= 6
        for scenario in ALL_SCENARIOS:
            assert scenario.name
            assert scenario.setup
            assert scenario.test_questions
            assert scenario.correct_answers
            assert scenario.difficulty in {"easy", "medium", "hard"}

    def test_get_scenarios_by_difficulty(self):
        """Test filtering scenarios by difficulty."""
        from theory_of_mind.selphi import get_scenarios_by_difficulty

        easy_scenarios = get_scenarios_by_difficulty("easy")
        medium_scenarios = get_scenarios_by_difficulty("medium")
        hard_scenarios = get_scenarios_by_difficulty("hard")

        assert len(easy_scenarios) > 0
        assert len(medium_scenarios) > 0
        assert len(hard_scenarios) > 0

        # Verify all are correct difficulty
        for scenario in easy_scenarios:
            assert scenario.difficulty == "easy"

    def test_get_scenarios_by_type(self):
        """Test filtering scenarios by ToM type."""
        from theory_of_mind.selphi import ToMType, get_scenarios_by_type

        false_belief_scenarios = get_scenarios_by_type(ToMType.FALSE_BELIEF)
        assert len(false_belief_scenarios) > 0

        # Verify all are correct type
        for scenario in false_belief_scenarios:
            assert scenario.tom_type == ToMType.FALSE_BELIEF

    def test_scenario_result_creation(self):
        """Test creating scenario result."""
        from theory_of_mind.selphi import ToMTaskResult

        result = ToMTaskResult(
            scenario_name="sally_anne",
            scenario_type="false_belief",
            difficulty="easy",
            model_response="Basket",
            latency_s=1.2,
            tokens_in=150,
            tokens_out=10,
            cost_usd=0.01,
            provider="ollama",
            model="llama3.2:latest",
            timestamp=0.0,
            metadata={"tom_type": "false_belief"},
        )

        assert result.scenario_name == "sally_anne"
        assert result.model_response == "Basket"
        assert result.scenario_type == "false_belief"


class TestBenchmarkInterface:
    """Test unified benchmark interface."""

    def test_benchmarks_registry(self):
        """Test that BENCHMARKS registry is populated."""
        from harness import BENCHMARKS

        # Should have at least 4 benchmarks
        assert len(BENCHMARKS) >= 4

        # Check specific benchmarks
        assert "uicrit" in BENCHMARKS
        assert "tombench" in BENCHMARKS
        assert "opentom" in BENCHMARKS
        assert "socialiqa" in BENCHMARKS

    def test_benchmark_info_structure(self):
        """Test that benchmark info has correct structure."""
        from harness import BENCHMARKS

        for name, info in BENCHMARKS.items():
            assert hasattr(info, "name")
            assert hasattr(info, "subsystem")
            assert hasattr(info, "size")
            assert hasattr(info, "source")
            assert hasattr(info, "description")
            assert info.subsystem in ["crit", "selphi", "harness"]

    def test_get_baseline_scores(self):
        """Test getting baseline scores."""
        from harness import get_baseline_scores

        # Test with known benchmark
        scores = get_baseline_scores("tombench")

        assert "human_performance" in scores
        assert "gpt4_performance" in scores
        assert "metric" in scores
        assert scores["metric"] == "accuracy"

    def test_baseline_scores_with_model_filter(self):
        """Test getting baseline scores for specific model."""
        from harness import get_baseline_scores

        scores = get_baseline_scores("tombench", model="gpt4")

        assert "score" in scores
        assert "model" in scores
        assert scores["model"] == "gpt4"


class TestModelConfig:
    """Test model configuration system."""

    def test_model_config_creation(self):
        """Test creating a ModelConfig."""
        from harness import ModelConfig

        config = ModelConfig(
            name="test-config", provider="ollama", model="llama3.2:latest", temperature=0.7, max_tokens=1000
        )

        assert config.name == "test-config"
        assert config.provider == "ollama"
        assert config.temperature == 0.7

    def test_model_config_to_dict(self):
        """Test converting config to dict."""
        from harness import ModelConfig

        config = ModelConfig(name="test-config", provider="ollama", model="llama3.2:latest", temperature=0.7)

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["provider"] == "ollama"
        assert config_dict["temperature"] == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
