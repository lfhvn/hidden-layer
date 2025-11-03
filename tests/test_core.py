"""
Core functionality tests for Hidden Layer.

These tests verify basic functionality without requiring
external services (no LLM calls).
"""

import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

import pytest
from unittest.mock import Mock, MagicMock


class TestHarnessCore:
    """Test harness core functionality."""

    def test_strategy_registry(self):
        """Test that strategy registry is properly populated."""
        from harness import STRATEGIES

        # Should have all 5 strategies
        expected_strategies = ['single', 'debate', 'self_consistency', 'manager_worker', 'consensus']
        for strategy in expected_strategies:
            assert strategy in STRATEGIES, f"Strategy '{strategy}' not found in registry"
            assert callable(STRATEGIES[strategy])

    def test_eval_functions_registry(self):
        """Test that evaluation functions registry is populated."""
        from harness import EVAL_FUNCTIONS

        # Should have basic eval functions
        expected_evals = ['exact_match', 'keyword_match', 'numeric_match']
        for eval_func in expected_evals:
            assert eval_func in EVAL_FUNCTIONS, f"Eval function '{eval_func}' not found"
            assert callable(EVAL_FUNCTIONS[eval_func])

    def test_exact_match_eval(self):
        """Test exact match evaluation."""
        from harness import exact_match

        assert exact_match("hello", "hello") == 1.0
        assert exact_match("hello", "Hello") == 0.0  # Case sensitive
        assert exact_match("hello", "world") == 0.0

    def test_keyword_match_eval(self):
        """Test keyword match evaluation."""
        from harness import keyword_match

        # Test with single keyword
        assert keyword_match("The sky is blue", "blue") == 1.0
        assert keyword_match("The sky is blue", "red") == 0.0

        # Test with multiple keywords
        assert keyword_match("The sky is blue and beautiful", ["blue", "beautiful"]) == 1.0
        assert keyword_match("The sky is blue", ["blue", "red"]) == 0.5  # 1 out of 2

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
            provider="ollama",
            model="llama3.2:latest"
        )

        assert config.experiment_name == "test_exp"
        assert config.strategy == "debate"
        assert config.provider == "ollama"

    def test_strategy_result_creation(self):
        """Test creating strategy result."""
        from harness import StrategyResult

        result = StrategyResult(
            output="Test output",
            strategy_name="single",
            latency_s=1.5,
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.01,
            metadata={"test": "data"}
        )

        assert result.output == "Test output"
        assert result.latency_s == 1.5
        assert result.metadata["test"] == "data"


class TestCRITCore:
    """Test CRIT core functionality."""

    def test_design_domains(self):
        """Test that design domains are defined."""
        from crit import DesignDomain

        # Should have all expected domains
        expected_domains = ['UI_UX', 'API_DESIGN', 'SYSTEM_ARCHITECTURE', 'DATA_MODELING', 'WORKFLOW']
        for domain in expected_domains:
            assert hasattr(DesignDomain, domain), f"Domain '{domain}' not found"

    def test_design_problems_exist(self):
        """Test that all design problems are defined."""
        from crit import (
            MOBILE_CHECKOUT,
            DASHBOARD_LAYOUT,
            API_VERSIONING,
            GRAPHQL_SCHEMA,
            MICROSERVICES,
            CACHING_STRATEGY,
            PERMISSION_SYSTEM,
            APPROVAL_WORKFLOW
        )

        problems = [
            MOBILE_CHECKOUT, DASHBOARD_LAYOUT, API_VERSIONING, GRAPHQL_SCHEMA,
            MICROSERVICES, CACHING_STRATEGY, PERMISSION_SYSTEM, APPROVAL_WORKFLOW
        ]

        for problem in problems:
            assert problem.name is not None
            assert problem.description is not None
            assert len(problem.success_criteria) > 0
            assert problem.difficulty in ['easy', 'medium', 'hard']

    def test_critique_perspectives(self):
        """Test that critique perspectives are defined."""
        from crit.strategies import PERSPECTIVES

        expected_perspectives = [
            'usability', 'security', 'accessibility', 'performance',
            'aesthetics', 'scalability', 'maintainability', 'cost_efficiency', 'user_delight'
        ]

        for perspective in expected_perspectives:
            assert perspective in PERSPECTIVES, f"Perspective '{perspective}' not found"

    def test_critique_result_creation(self):
        """Test creating critique result."""
        from crit import CritiqueResult

        result = CritiqueResult(
            critique="Test critique",
            strategy_name="single",
            problem_name="test_problem",
            latency_s=2.0,
            tokens_in=200,
            tokens_out=150,
            metadata={"perspectives": ["usability"]}
        )

        assert result.critique == "Test critique"
        assert result.strategy_name == "single"
        assert "usability" in result.metadata["perspectives"]


class TestSELPHICore:
    """Test SELPHI core functionality."""

    def test_tom_types(self):
        """Test that ToM types are defined."""
        from selphi import ToMType

        expected_types = [
            'FALSE_BELIEF', 'KNOWLEDGE_ATTRIBUTION', 'PERSPECTIVE_TAKING',
            'BELIEF_UPDATING', 'SECOND_ORDER_BELIEF', 'EPISTEMIC_STATE', 'PRAGMATIC_REASONING'
        ]

        for tom_type in expected_types:
            assert hasattr(ToMType, tom_type), f"ToM type '{tom_type}' not found"

    def test_scenarios_exist(self):
        """Test that all scenarios are defined."""
        from selphi import (
            SALLY_ANNE, CHOCOLATE_BAR, BIRTHDAY_PUPPY,
            ICE_CREAM_VAN, MUSEUM_TRIP, PAINTED_ROOM,
            LIBRARY_BOOK, RESTAURANT_BILL, COFFEE_SHOP
        )

        scenarios = [
            SALLY_ANNE, CHOCOLATE_BAR, BIRTHDAY_PUPPY,
            ICE_CREAM_VAN, MUSEUM_TRIP, PAINTED_ROOM,
            LIBRARY_BOOK, RESTAURANT_BILL, COFFEE_SHOP
        ]

        for scenario in scenarios:
            assert scenario.name is not None
            assert scenario.scenario_text is not None
            assert scenario.question is not None
            assert scenario.correct_answer is not None
            assert scenario.difficulty in ['easy', 'medium', 'hard']

    def test_get_scenarios_by_difficulty(self):
        """Test filtering scenarios by difficulty."""
        from selphi import get_scenarios_by_difficulty

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
        from selphi import get_scenarios_by_type, ToMType

        false_belief_scenarios = get_scenarios_by_type(ToMType.FALSE_BELIEF)
        assert len(false_belief_scenarios) > 0

        # Verify all are correct type
        for scenario in false_belief_scenarios:
            assert scenario.tom_type == ToMType.FALSE_BELIEF

    def test_scenario_result_creation(self):
        """Test creating scenario result."""
        from selphi import ScenarioResult

        result = ScenarioResult(
            scenario_name="sally_anne",
            model_response="Basket",
            correct_answer="Basket",
            latency_s=1.2,
            tokens_in=150,
            tokens_out=10,
            metadata={"tom_type": "false_belief"}
        )

        assert result.scenario_name == "sally_anne"
        assert result.model_response == "Basket"
        assert result.correct_answer == "Basket"


class TestBenchmarkInterface:
    """Test unified benchmark interface."""

    def test_benchmarks_registry(self):
        """Test that BENCHMARKS registry is populated."""
        from harness import BENCHMARKS

        # Should have at least 4 benchmarks
        assert len(BENCHMARKS) >= 4

        # Check specific benchmarks
        assert 'uicrit' in BENCHMARKS
        assert 'tombench' in BENCHMARKS
        assert 'opentom' in BENCHMARKS
        assert 'socialiqa' in BENCHMARKS

    def test_benchmark_info_structure(self):
        """Test that benchmark info has correct structure."""
        from harness import BENCHMARKS

        for name, info in BENCHMARKS.items():
            assert hasattr(info, 'name')
            assert hasattr(info, 'subsystem')
            assert hasattr(info, 'size')
            assert hasattr(info, 'source')
            assert hasattr(info, 'description')
            assert info.subsystem in ['crit', 'selphi', 'harness']

    def test_get_baseline_scores(self):
        """Test getting baseline scores."""
        from harness import get_baseline_scores

        # Test with known benchmark
        scores = get_baseline_scores('tombench')

        assert 'human_performance' in scores
        assert 'gpt4_performance' in scores
        assert 'metric' in scores
        assert scores['metric'] == 'accuracy'

    def test_baseline_scores_with_model_filter(self):
        """Test getting baseline scores for specific model."""
        from harness import get_baseline_scores

        scores = get_baseline_scores('tombench', model='gpt4')

        assert 'score' in scores
        assert 'model' in scores
        assert scores['model'] == 'gpt4'


class TestModelConfig:
    """Test model configuration system."""

    def test_model_config_creation(self):
        """Test creating a ModelConfig."""
        from harness import ModelConfig

        config = ModelConfig(
            name="test-config",
            provider="ollama",
            model="llama3.2:latest",
            temperature=0.7,
            max_tokens=1000
        )

        assert config.name == "test-config"
        assert config.provider == "ollama"
        assert config.temperature == 0.7

    def test_model_config_to_dict(self):
        """Test converting config to dict."""
        from harness import ModelConfig

        config = ModelConfig(
            name="test-config",
            provider="ollama",
            model="llama3.2:latest",
            temperature=0.7
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['provider'] == 'ollama'
        assert config_dict['temperature'] == 0.7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
