"""
SELPHI - Study of Epistemic and Logical Processing in Heuristic Intelligence

A research toolkit for testing theory of mind and epistemology in language models.

This package provides:
- Pre-defined ToM scenarios (Sally-Anne, false belief, knowledge attribution, etc.)
- Task execution functions for running scenarios with different models
- Evaluation metrics for assessing ToM understanding
- Integration with the hidden-layer harness for experiment tracking

Quick Start:
    >>> from selphi import run_scenario, SALLY_ANNE, evaluate_scenario
    >>> result = run_scenario(SALLY_ANNE, provider="ollama")
    >>> eval_result = evaluate_scenario(SALLY_ANNE, result.model_response)
    >>> print(f"Score: {eval_result['average_score']:.2f}")
"""

from .scenarios import (
    ToMScenario,
    ToMType,
    ALL_SCENARIOS,
    SCENARIOS_BY_TYPE,
    SCENARIOS_BY_DIFFICULTY,
    SCENARIOS_BY_NAME,
    get_scenario,
    get_scenarios_by_type,
    get_scenarios_by_difficulty,
    # Pre-defined scenarios
    SALLY_ANNE,
    CHOCOLATE_BAR,
    SURPRISE_PARTY,
    BROKEN_VASE,
    MOVIE_OPINIONS,
    WEATHER_UPDATE,
    GIFT_SURPRISE,
    COIN_FLIP,
    DOOR_LOCKED,
)

from .tasks import (
    ToMTaskResult,
    run_scenario,
    run_multiple_scenarios,
    run_all_scenarios,
    run_scenarios_by_type,
    run_scenarios_by_difficulty,
    compare_models_on_scenarios,
    results_to_dict_list,
)

from .evals import (
    parse_multi_answer_response,
    semantic_match_score,
    llm_judge_tom,
    evaluate_scenario,
    evaluate_batch,
    compare_models,
)

from .benchmarks import (
    BenchmarkDataset,
    load_tombench,
    load_opentom,
    load_socialiqa,
    list_available_benchmarks,
    print_benchmark_info,
)

__version__ = "0.1.0"

__all__ = [
    # Core types
    "ToMScenario",
    "ToMType",
    "ToMTaskResult",

    # Scenario access
    "ALL_SCENARIOS",
    "SCENARIOS_BY_TYPE",
    "SCENARIOS_BY_DIFFICULTY",
    "SCENARIOS_BY_NAME",
    "get_scenario",
    "get_scenarios_by_type",
    "get_scenarios_by_difficulty",

    # Pre-defined scenarios
    "SALLY_ANNE",
    "CHOCOLATE_BAR",
    "SURPRISE_PARTY",
    "BROKEN_VASE",
    "MOVIE_OPINIONS",
    "WEATHER_UPDATE",
    "GIFT_SURPRISE",
    "COIN_FLIP",
    "DOOR_LOCKED",

    # Task execution
    "run_scenario",
    "run_multiple_scenarios",
    "run_all_scenarios",
    "run_scenarios_by_type",
    "run_scenarios_by_difficulty",
    "compare_models_on_scenarios",
    "results_to_dict_list",

    # Evaluation
    "parse_multi_answer_response",
    "semantic_match_score",
    "llm_judge_tom",
    "evaluate_scenario",
    "evaluate_batch",
    "compare_models",

    # Benchmarks
    "BenchmarkDataset",
    "load_tombench",
    "load_opentom",
    "load_socialiqa",
    "list_available_benchmarks",
    "print_benchmark_info",
]
