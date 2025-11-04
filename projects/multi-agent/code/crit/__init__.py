"""
CRIT - Collective Reasoning for Iterative Testing

A research toolkit for testing collective design critique reasoning agents
on challenging design problems.

This package provides:
- Pre-defined design problems across multiple domains (UI/UX, API, System, Data, Workflow)
- Collective critique strategies (multi-perspective, iterative, adversarial)
- Evaluation metrics for critique quality, coverage, and depth
- Integration with the hidden-layer harness for experiment tracking

Quick Start:
    >>> from crit import run_critique_strategy, MOBILE_CHECKOUT, evaluate_critique
    >>> result = run_critique_strategy("multi_perspective", MOBILE_CHECKOUT, provider="ollama")
    >>> eval_result = evaluate_critique(MOBILE_CHECKOUT, result)
    >>> print(f"Quality Score: {eval_result['quality']['overall_quality']:.2f}")
"""

from .benchmarks import (
    BenchmarkDataset,
    compare_to_experts,
    list_available_benchmarks,
    load_uicrit,
    load_uicrit_for_comparison,
    print_benchmark_info,
)
from .evals import (
    batch_evaluate,
    compare_strategies,
    evaluate_critique,
    evaluate_critique_coverage,
    evaluate_critique_depth,
    evaluate_recommendation_quality,
)
from .problems import (  # Pre-defined problems
    ALL_PROBLEMS,
    APPROVAL_WORKFLOW,
    CACHING_STRATEGY,
    DASHBOARD_LAYOUT,
    GRAPHQL_SCHEMA,
    MICROSERVICES_SPLIT,
    MOBILE_CHECKOUT,
    PERMISSION_MODEL,
    PROBLEMS_BY_DIFFICULTY,
    PROBLEMS_BY_DOMAIN,
    PROBLEMS_BY_NAME,
    REST_API_VERSIONING,
    CritiquePerspective,
    DesignDomain,
    DesignProblem,
    get_problem,
    get_problems_by_difficulty,
    get_problems_by_domain,
)
from .strategies import (
    STRATEGIES,
    CritiqueResult,
    adversarial_critique,
    iterative_critique,
    multi_perspective_critique,
    run_critique_strategy,
    single_critic_strategy,
)

__version__ = "0.1.0"

__all__ = [
    # Core types
    "DesignProblem",
    "DesignDomain",
    "CritiquePerspective",
    "CritiqueResult",
    # Problem access
    "ALL_PROBLEMS",
    "PROBLEMS_BY_DOMAIN",
    "PROBLEMS_BY_DIFFICULTY",
    "PROBLEMS_BY_NAME",
    "get_problem",
    "get_problems_by_domain",
    "get_problems_by_difficulty",
    # Pre-defined problems
    "MOBILE_CHECKOUT",
    "DASHBOARD_LAYOUT",
    "REST_API_VERSIONING",
    "GRAPHQL_SCHEMA",
    "MICROSERVICES_SPLIT",
    "CACHING_STRATEGY",
    "PERMISSION_MODEL",
    "APPROVAL_WORKFLOW",
    # Critique strategies
    "single_critic_strategy",
    "multi_perspective_critique",
    "iterative_critique",
    "adversarial_critique",
    "run_critique_strategy",
    "STRATEGIES",
    # Evaluation
    "evaluate_critique_coverage",
    "evaluate_recommendation_quality",
    "evaluate_critique_depth",
    "evaluate_critique",
    "compare_strategies",
    "batch_evaluate",
    # Benchmarks
    "BenchmarkDataset",
    "load_uicrit",
    "load_uicrit_for_comparison",
    "list_available_benchmarks",
    "print_benchmark_info",
    "compare_to_experts",
]
