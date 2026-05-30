"""
Import smoke tests for the multi-agent subsystem and its dependencies.

Verifies the *current* public API surface (post-reorg): strategies live in
``communication.multi_agent``, CRIT in ``communication.multi_agent.crit``, and the
shared infrastructure in ``harness`` (which lazily re-exports ``run_strategy``).
No LLM calls are made here.
"""

import pytest


class TestHarnessImports:
    """Shared infrastructure imported from the harness."""

    def test_import_harness(self):
        import harness

        assert isinstance(harness.__version__, str)

    def test_import_llm_provider(self):
        from harness import LLMProvider, LLMResponse, get_provider, llm_call, llm_call_stream

        assert callable(llm_call) and callable(llm_call_stream) and callable(get_provider)
        assert LLMProvider is not None and LLMResponse is not None

    def test_harness_reexports_run_strategy(self):
        # run_strategy lives in communication.multi_agent but is lazily re-exported.
        from harness import run_strategy, run_strategy_with_rationale

        assert callable(run_strategy) and callable(run_strategy_with_rationale)

    def test_import_experiment_tracker(self):
        from harness import ExperimentConfig, ExperimentResult, ExperimentTracker, compare_experiments, get_tracker

        assert callable(get_tracker) and callable(compare_experiments)
        assert ExperimentConfig is not None and ExperimentResult is not None and ExperimentTracker is not None

    def test_import_evals(self):
        from harness import EVAL_FUNCTIONS, evaluate_task, exact_match, keyword_match, numeric_match

        assert callable(exact_match) and callable(keyword_match) and callable(numeric_match)
        assert callable(evaluate_task) and isinstance(EVAL_FUNCTIONS, dict)

    def test_import_model_config(self):
        from harness import ModelConfig, ModelConfigManager, get_config_manager, get_model_config, list_model_configs

        assert callable(get_model_config) and callable(list_model_configs) and callable(get_config_manager)
        assert ModelConfig is not None and ModelConfigManager is not None

    def test_import_benchmarks(self):
        from harness import BENCHMARKS, get_baseline_scores, load_benchmark

        assert callable(load_benchmark) and callable(get_baseline_scores)
        assert isinstance(BENCHMARKS, dict)


class TestMultiAgentImports:
    """Strategy API in communication.multi_agent."""

    def test_import_package(self):
        import communication.multi_agent as ma

        # The namespace wrapper re-exports the strategy API.
        assert hasattr(ma, "STRATEGIES") and hasattr(ma, "run_strategy")

    def test_import_strategies(self):
        from communication.multi_agent import (
            STRATEGIES,
            StrategyResult,
            consensus_strategy,
            debate_strategy,
            manager_worker_strategy,
            run_strategy,
            self_consistency_strategy,
            single_model_strategy,
        )

        assert callable(run_strategy)
        assert isinstance(STRATEGIES, dict) and len(STRATEGIES) >= 5
        for fn in (
            single_model_strategy,
            debate_strategy,
            consensus_strategy,
            self_consistency_strategy,
            manager_worker_strategy,
        ):
            assert callable(fn)
        assert StrategyResult is not None

    def test_import_rationale(self):
        from communication.multi_agent import (
            RationaleResponse,
            ask_with_reasoning,
            extract_rationale_from_result,
            llm_call_with_rationale,
            run_strategy_with_rationale,
        )

        assert callable(llm_call_with_rationale) and callable(ask_with_reasoning)
        assert callable(extract_rationale_from_result) and callable(run_strategy_with_rationale)
        assert RationaleResponse is not None


class TestCRITImports:
    """CRIT design-critique API in communication.multi_agent.crit."""

    def test_import_crit(self):
        from communication.multi_agent import crit

        assert hasattr(crit, "__version__")

    def test_import_problems(self):
        from communication.multi_agent.crit import (
            MOBILE_CHECKOUT,
            REST_API_VERSIONING,
            DesignDomain,
            DesignProblem,
        )

        assert isinstance(MOBILE_CHECKOUT, DesignProblem)
        assert isinstance(REST_API_VERSIONING, DesignProblem)
        assert isinstance(MOBILE_CHECKOUT.domain, DesignDomain)

    def test_import_critique_strategies(self):
        from communication.multi_agent.crit import (
            CritiqueResult,
            adversarial_critique,
            iterative_critique,
            multi_perspective_critique,
            run_critique_strategy,
            single_critic_strategy,
        )

        assert callable(run_critique_strategy) and callable(single_critic_strategy)
        assert callable(multi_perspective_critique) and callable(adversarial_critique)
        assert callable(iterative_critique) and CritiqueResult is not None

    def test_import_evals(self):
        from communication.multi_agent.crit import (
            evaluate_critique,
            evaluate_critique_coverage,
            evaluate_critique_depth,
        )

        assert callable(evaluate_critique)
        assert callable(evaluate_critique_coverage) and callable(evaluate_critique_depth)

    def test_import_benchmarks(self):
        from communication.multi_agent.crit import list_available_benchmarks, load_uicrit, print_benchmark_info

        assert callable(load_uicrit) and callable(list_available_benchmarks) and callable(print_benchmark_info)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
