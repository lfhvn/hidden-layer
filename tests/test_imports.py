"""Smoke tests for importing key Hidden Layer packages.

These tests verify that the reorganized research areas (communication,
theory_of_mind, etc.) can be imported successfully after the migration
away from the legacy ``projects/`` layout.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


class TestHarnessImports:
    """Test harness module imports."""

    def test_import_harness(self):
        """Test basic harness import."""
        import harness

        assert harness.__version__ == "0.2.0"

    def test_import_llm_provider(self):
        """Test LLM provider imports."""
        from harness import LLMProvider, LLMResponse, get_provider, llm_call, llm_call_stream

        assert callable(llm_call)
        assert callable(llm_call_stream)
        assert callable(get_provider)

    def test_import_strategies(self):
        """Test strategy imports."""
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
        assert isinstance(STRATEGIES, dict)
        assert len(STRATEGIES) >= 5  # Should have at least 5 strategies

    def test_import_experiment_tracker(self):
        """Test experiment tracking imports."""
        from harness import ExperimentConfig, ExperimentResult, ExperimentTracker, compare_experiments, get_tracker

        assert callable(get_tracker)
        assert callable(compare_experiments)

    def test_import_evals(self):
        """Test evaluation function imports."""
        from harness import (
            EVAL_FUNCTIONS,
            evaluate_task,
            exact_match,
            keyword_match,
            llm_judge,
            numeric_match,
            win_rate_comparison,
        )

        assert callable(exact_match)
        assert isinstance(EVAL_FUNCTIONS, dict)

    def test_import_model_config(self):
        """Test model configuration imports."""
        from harness import ModelConfig, ModelConfigManager, get_config_manager, get_model_config, list_model_configs

        assert callable(get_model_config)
        assert callable(list_model_configs)

    def test_import_rationale(self):
        """Test rationale extraction imports."""
        from communication.multi_agent import (
            RationaleResponse,
            ask_with_reasoning,
            extract_rationale_from_result,
            llm_call_with_rationale,
            run_strategy_with_rationale,
        )

        assert callable(llm_call_with_rationale)
        assert callable(ask_with_reasoning)

    def test_import_benchmarks(self):
        """Test benchmark imports."""
        from harness import BENCHMARKS, get_baseline_scores, load_benchmark

        assert callable(load_benchmark)
        assert callable(get_baseline_scores)
        assert isinstance(BENCHMARKS, dict)
        assert len(BENCHMARKS) >= 4  # uicrit, tombench, opentom, socialiqa


class TestCRITImports:
    """Test CRIT module imports."""

    def test_import_crit(self):
        """Test basic CRIT import."""
        from communication.multi_agent import crit

        assert hasattr(crit, "__version__")

    def test_import_problems(self):
        """Test problem imports."""
        from communication.multi_agent.crit import (
            APPROVAL_WORKFLOW,
            CACHING_STRATEGY,
            DASHBOARD_LAYOUT,
            GRAPHQL_SCHEMA,
            MICROSERVICES_SPLIT,
            MOBILE_CHECKOUT,
            PERMISSION_MODEL,
            REST_API_VERSIONING,
            DesignDomain,
            DesignProblem,
        )

        assert isinstance(MOBILE_CHECKOUT, DesignProblem)
        assert MOBILE_CHECKOUT.domain == DesignDomain.UI_UX

    def test_import_critique_strategies(self):
        """Test critique strategy imports."""
        from communication.multi_agent.crit import (
            CritiqueResult,
            adversarial_critique,
            iterative_critique,
            multi_perspective_critique,
            run_critique_strategy,
            single_critic_strategy,
        )

        assert callable(run_critique_strategy)
        assert callable(single_critic_strategy)

    def test_import_evals(self):
        """Test CRIT evaluation imports."""
        from communication.multi_agent.crit import (
            evaluate_critique,
            evaluate_critique_coverage,
            evaluate_critique_depth,
        )

        assert callable(evaluate_critique)
        assert callable(evaluate_critique_coverage)
        assert callable(evaluate_critique_depth)

    def test_import_benchmarks(self):
        """Test CRIT benchmark imports."""
        from communication.multi_agent.crit.benchmarks import (
            BenchmarkDataset,
            list_available_benchmarks,
            load_uicrit,
            print_benchmark_info,
        )

        assert callable(load_uicrit)


class TestSELPHIImports:
    """Test SELPHI module imports."""

    def test_import_selphi(self):
        """Test basic SELPHI import."""
        from theory_of_mind import selphi

        assert hasattr(selphi, "__version__")

    def test_import_scenarios(self):
        """Test scenario imports."""
        from theory_of_mind.selphi import CHOCOLATE_BAR, SALLY_ANNE, SURPRISE_PARTY, ToMScenario, ToMType

        assert isinstance(SALLY_ANNE, ToMScenario)
        assert isinstance(CHOCOLATE_BAR, ToMScenario)
        assert SURPRISE_PARTY.tom_type == ToMType.KNOWLEDGE_ATTRIBUTION

    def test_import_scenario_functions(self):
        """Test scenario function imports."""
        from theory_of_mind.selphi import (
            get_scenarios_by_difficulty,
            get_scenarios_by_type,
            run_multiple_scenarios,
            run_scenario,
            ToMTaskResult,
        )

        assert callable(run_scenario)
        assert callable(run_multiple_scenarios)
        assert ToMTaskResult.__module__.endswith("tasks")

    def test_import_evals(self):
        """Test SELPHI evaluation imports."""
        from theory_of_mind.selphi import compare_models, evaluate_batch, evaluate_scenario

        assert callable(evaluate_scenario)

    def test_import_benchmarks(self):
        """Test SELPHI benchmark imports."""
        from theory_of_mind.selphi.benchmarks import list_available_benchmarks, load_opentom, load_socialiqa, load_tombench

        assert callable(load_tombench)
        assert callable(load_opentom)
        assert callable(load_socialiqa)
        assert callable(list_available_benchmarks)


class TestCrossSubsystemIntegration:
    """Test integration between subsystems."""

    def test_all_subsystems_importable(self):
        """Test that all three subsystems can be imported together."""
        from communication.multi_agent import crit
        import harness
        from theory_of_mind import selphi

        # All should have versions
        assert hasattr(harness, "__version__")
        assert hasattr(crit, "__version__")
        assert hasattr(selphi, "__version__")

    def test_unified_benchmark_interface(self):
        """Test that harness.benchmarks provides access to all benchmarks."""
        from harness import BENCHMARKS

        # Should include benchmarks from all subsystems
        assert "uicrit" in BENCHMARKS  # CRIT
        assert "tombench" in BENCHMARKS  # SELPHI
        assert "opentom" in BENCHMARKS  # SELPHI
        assert "socialiqa" in BENCHMARKS  # SELPHI

    def test_benchmark_loaders(self):
        """Test that benchmark loaders are accessible."""
        from harness import load_benchmark

        # Should be able to call load_benchmark (will fail if files don't exist, but callable)
        assert callable(load_benchmark)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
