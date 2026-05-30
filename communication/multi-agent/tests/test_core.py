"""
Core functionality tests for the multi-agent subsystem.

Runs fully offline via the harness ``sim`` provider (no API keys, no network), so it is
CI-safe. Targets the current API: strategies in ``communication.multi_agent``, CRIT in
``communication.multi_agent.crit``, shared eval/config/benchmark utilities in ``harness``.
"""

import pytest

# Core strategies expected in the registry (introspection is also present).
CORE_STRATEGIES = ["single", "debate", "self_consistency", "manager_worker", "consensus"]


class TestStrategies:
    """Strategy registry and execution."""

    def test_strategy_registry(self):
        from communication.multi_agent import STRATEGIES

        for strategy in CORE_STRATEGIES:
            assert strategy in STRATEGIES, f"Strategy '{strategy}' not found in registry"
            assert callable(STRATEGIES[strategy])

    @pytest.mark.parametrize("strategy_id", CORE_STRATEGIES)
    def test_run_strategy_offline(self, strategy_id):
        from communication.multi_agent import StrategyResult, run_strategy

        result = run_strategy(strategy_id, task_input="What is 2 + 2?", provider="sim", model="sim", sim_response="4")
        assert isinstance(result, StrategyResult)
        assert isinstance(result.output, str) and result.output
        assert isinstance(result.strategy_name, str) and result.strategy_name

    def test_run_strategy_unknown_raises(self):
        from communication.multi_agent import run_strategy

        with pytest.raises(ValueError):
            run_strategy("does_not_exist", task_input="x", provider="sim")

    def test_strategy_result_creation(self):
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


class TestEvals:
    """Harness evaluation functions used to score strategy outputs."""

    def test_eval_functions_registry(self):
        from harness import EVAL_FUNCTIONS

        # Registry keys are the short names: exact_match, keyword, numeric.
        for key in ("exact_match", "keyword", "numeric"):
            assert key in EVAL_FUNCTIONS, f"Eval function '{key}' not found"
            assert callable(EVAL_FUNCTIONS[key])

    def test_exact_match(self):
        from harness import exact_match

        assert exact_match("hello", "hello") == 1.0
        assert exact_match("hello", "Hello") == 1.0  # case-insensitive by default
        assert exact_match("hello", "Hello", case_sensitive=True) == 0.0
        assert exact_match("hello", "world") == 0.0

    def test_keyword_match(self):
        from harness import keyword_match

        assert keyword_match("The sky is blue", ["blue"]) == 1.0
        assert keyword_match("The sky is blue", ["red"]) == 0.0
        # require_all=False (default): any keyword present -> 1.0
        assert keyword_match("The sky is blue", ["blue", "red"]) == 1.0
        # require_all=True: every keyword must be present
        assert keyword_match("The sky is blue", ["blue", "red"], require_all=True) == 0.0
        assert keyword_match("blue and beautiful", ["blue", "beautiful"], require_all=True) == 1.0

    def test_numeric_match(self):
        from harness import numeric_match

        assert numeric_match("42", 42) == 1.0
        assert numeric_match("3.14", 3.14, tolerance=0.01) == 1.0
        assert numeric_match("42.1", 42, tolerance=0.2) == 1.0
        assert numeric_match("42.5", 42, tolerance=0.2) == 0.0


class TestExperimentConfig:
    """Experiment configuration (used to log strategy runs)."""

    def test_experiment_config_creation(self):
        from harness import ExperimentConfig

        config = ExperimentConfig(
            experiment_name="test_exp",
            task_type="reasoning",
            strategy="debate",
            provider="ollama",
            model="llama3.2:latest",
        )
        assert config.experiment_name == "test_exp"
        assert config.task_type == "reasoning"
        assert config.strategy == "debate"
        assert config.provider == "ollama"


class TestCRIT:
    """CRIT design-critique core."""

    def test_design_domains(self):
        from communication.multi_agent.crit import DesignDomain

        # Current domain taxonomy.
        for domain in ("UI_UX", "API", "SYSTEM", "DATA", "WORKFLOW"):
            assert hasattr(DesignDomain, domain), f"Domain '{domain}' not found"

    def test_design_problems_exist(self):
        from communication.multi_agent.crit import (
            APPROVAL_WORKFLOW,
            CACHING_STRATEGY,
            DASHBOARD_LAYOUT,
            GRAPHQL_SCHEMA,
            MICROSERVICES_SPLIT,
            MOBILE_CHECKOUT,
            PERMISSION_MODEL,
            REST_API_VERSIONING,
            DesignProblem,
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
            assert isinstance(problem, DesignProblem)
            assert problem.name and problem.description
            assert len(problem.success_criteria) > 0
            assert problem.difficulty in ("easy", "medium", "hard")

    def test_single_critic_offline(self):
        from communication.multi_agent.crit import MOBILE_CHECKOUT, CritiqueResult, single_critic_strategy

        result = single_critic_strategy(
            MOBILE_CHECKOUT,
            provider="sim",
            model="sim",
            sim_response="The checkout flow is too long; reduce steps and add guest checkout.",
        )
        assert isinstance(result, CritiqueResult)
        assert isinstance(result.synthesis, str)
        assert result.problem_name == MOBILE_CHECKOUT.name


class TestBenchmarkInterface:
    """Unified benchmark registry exposed via the harness."""

    def test_benchmarks_registry(self):
        from harness import BENCHMARKS

        for name in ("uicrit", "tombench", "opentom", "socialiqa"):
            assert name in BENCHMARKS

    def test_benchmark_info_structure(self):
        from harness import BENCHMARKS

        for info in BENCHMARKS.values():
            for attr in ("name", "subsystem", "size", "source", "description"):
                assert hasattr(info, attr)
            assert isinstance(info.subsystem, str) and info.subsystem

    def test_get_baseline_scores(self):
        from harness import get_baseline_scores

        scores = get_baseline_scores("tombench")
        assert "human_performance" in scores
        assert "metric" in scores


class TestModelConfig:
    """Model configuration system."""

    def test_model_config_creation(self):
        from harness import ModelConfig

        config = ModelConfig(
            name="test-config", provider="ollama", model="llama3.2:latest", temperature=0.7, max_tokens=1000
        )
        assert config.name == "test-config"
        assert config.provider == "ollama"
        assert config.temperature == 0.7

    def test_model_config_to_dict(self):
        from harness import ModelConfig

        config = ModelConfig(name="test-config", provider="ollama", model="llama3.2:latest", temperature=0.7)
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["provider"] == "ollama"
        assert config_dict["temperature"] == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
