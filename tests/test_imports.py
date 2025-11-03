"""
Test that all modules can be imported successfully.

This is a critical smoke test to ensure there are no import errors
across all three subsystems (harness, CRIT, SELPHI).
"""

import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

import pytest


class TestHarnessImports:
    """Test harness module imports."""

    def test_import_harness(self):
        """Test basic harness import."""
        import harness
        assert harness.__version__ == "0.1.0"

    def test_import_llm_provider(self):
        """Test LLM provider imports."""
        from harness import llm_call, llm_call_stream, get_provider, LLMProvider, LLMResponse
        assert callable(llm_call)
        assert callable(llm_call_stream)
        assert callable(get_provider)

    def test_import_strategies(self):
        """Test strategy imports."""
        from harness import (
            run_strategy,
            single_model_strategy,
            debate_strategy,
            consensus_strategy,
            self_consistency_strategy,
            manager_worker_strategy,
            StrategyResult,
            STRATEGIES
        )
        assert callable(run_strategy)
        assert isinstance(STRATEGIES, dict)
        assert len(STRATEGIES) >= 5  # Should have at least 5 strategies

    def test_import_experiment_tracker(self):
        """Test experiment tracking imports."""
        from harness import (
            ExperimentTracker,
            ExperimentConfig,
            ExperimentResult,
            get_tracker,
            compare_experiments
        )
        assert callable(get_tracker)
        assert callable(compare_experiments)

    def test_import_evals(self):
        """Test evaluation function imports."""
        from harness import (
            exact_match,
            keyword_match,
            numeric_match,
            llm_judge,
            win_rate_comparison,
            evaluate_task,
            EVAL_FUNCTIONS
        )
        assert callable(exact_match)
        assert isinstance(EVAL_FUNCTIONS, dict)

    def test_import_model_config(self):
        """Test model configuration imports."""
        from harness import (
            ModelConfig,
            ModelConfigManager,
            get_config_manager,
            get_model_config,
            list_model_configs
        )
        assert callable(get_model_config)
        assert callable(list_model_configs)

    def test_import_rationale(self):
        """Test rationale extraction imports."""
        from harness import (
            llm_call_with_rationale,
            extract_rationale_from_result,
            run_strategy_with_rationale,
            ask_with_reasoning,
            RationaleResponse
        )
        assert callable(llm_call_with_rationale)
        assert callable(ask_with_reasoning)

    def test_import_benchmarks(self):
        """Test benchmark imports."""
        from harness import load_benchmark, get_baseline_scores, BENCHMARKS
        assert callable(load_benchmark)
        assert callable(get_baseline_scores)
        assert isinstance(BENCHMARKS, dict)
        assert len(BENCHMARKS) >= 4  # uicrit, tombench, opentom, socialiqa


class TestCRITImports:
    """Test CRIT module imports."""

    def test_import_crit(self):
        """Test basic CRIT import."""
        import crit
        assert hasattr(crit, '__version__')

    def test_import_problems(self):
        """Test problem imports."""
        from crit import (
            DesignProblem,
            DesignDomain,
            MOBILE_CHECKOUT,
            DASHBOARD_LAYOUT,
            API_VERSIONING,
            GRAPHQL_SCHEMA,
            MICROSERVICES,
            CACHING_STRATEGY,
            PERMISSION_SYSTEM,
            APPROVAL_WORKFLOW
        )
        assert isinstance(MOBILE_CHECKOUT, DesignProblem)
        assert MOBILE_CHECKOUT.domain == DesignDomain.UI_UX

    def test_import_critique_strategies(self):
        """Test critique strategy imports."""
        from crit import (
            run_critique_strategy,
            single_critic,
            multi_perspective_critique,
            iterative_critique,
            adversarial_critique,
            CritiqueResult
        )
        assert callable(run_critique_strategy)
        assert callable(single_critic)

    def test_import_evals(self):
        """Test CRIT evaluation imports."""
        from crit import evaluate_critique, critique_coverage, critique_depth
        assert callable(evaluate_critique)

    def test_import_benchmarks(self):
        """Test CRIT benchmark imports."""
        from crit.benchmarks import (
            BenchmarkDataset,
            load_uicrit,
            list_available_benchmarks,
            print_benchmark_info
        )
        assert callable(load_uicrit)


class TestSELPHIImports:
    """Test SELPHI module imports."""

    def test_import_selphi(self):
        """Test basic SELPHI import."""
        import selphi
        assert hasattr(selphi, '__version__')

    def test_import_scenarios(self):
        """Test scenario imports."""
        from selphi import (
            ToMScenario,
            ToMType,
            SALLY_ANNE,
            CHOCOLATE_BAR,
            BIRTHDAY_PUPPY,
            ICE_CREAM_VAN,
            MUSEUM_TRIP,
            PAINTED_ROOM,
            LIBRARY_BOOK,
            RESTAURANT_BILL,
            COFFEE_SHOP
        )
        assert isinstance(SALLY_ANNE, ToMScenario)
        assert SALLY_ANNE.tom_type == ToMType.FALSE_BELIEF

    def test_import_scenario_functions(self):
        """Test scenario function imports."""
        from selphi import (
            run_scenario,
            run_multiple_scenarios,
            get_scenarios_by_type,
            get_scenarios_by_difficulty,
            ScenarioResult
        )
        assert callable(run_scenario)
        assert callable(run_multiple_scenarios)

    def test_import_evals(self):
        """Test SELPHI evaluation imports."""
        from selphi import (
            evaluate_scenario,
            evaluate_batch,
            compare_models
        )
        assert callable(evaluate_scenario)

    def test_import_benchmarks(self):
        """Test SELPHI benchmark imports."""
        from selphi.benchmarks import (
            BenchmarkDataset,
            load_tombench,
            load_opentom,
            load_socialiqa,
            list_available_benchmarks,
            print_benchmark_info
        )
        assert callable(load_tombench)
        assert callable(load_opentom)
        assert callable(load_socialiqa)


class TestCrossSubsystemIntegration:
    """Test integration between subsystems."""

    def test_all_subsystems_importable(self):
        """Test that all three subsystems can be imported together."""
        import harness
        import crit
        import selphi

        # All should have versions
        assert hasattr(harness, '__version__')
        assert hasattr(crit, '__version__')
        assert hasattr(selphi, '__version__')

    def test_unified_benchmark_interface(self):
        """Test that harness.benchmarks provides access to all benchmarks."""
        from harness import BENCHMARKS

        # Should include benchmarks from all subsystems
        assert 'uicrit' in BENCHMARKS  # CRIT
        assert 'tombench' in BENCHMARKS  # SELPHI
        assert 'opentom' in BENCHMARKS  # SELPHI
        assert 'socialiqa' in BENCHMARKS  # SELPHI

    def test_benchmark_loaders(self):
        """Test that benchmark loaders are accessible."""
        from harness import load_benchmark

        # Should be able to call load_benchmark (will fail if files don't exist, but callable)
        assert callable(load_benchmark)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
