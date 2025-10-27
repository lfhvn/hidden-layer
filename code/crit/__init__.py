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

from .problems import (
    DesignProblem,
    DesignDomain,
    CritiquePerspective,
    ALL_PROBLEMS,
    PROBLEMS_BY_DOMAIN,
    PROBLEMS_BY_DIFFICULTY,
    PROBLEMS_BY_NAME,
    get_problem,
    get_problems_by_domain,
    get_problems_by_difficulty,
    # Pre-defined problems
    MOBILE_CHECKOUT,
    DASHBOARD_LAYOUT,
    REST_API_VERSIONING,
    GRAPHQL_SCHEMA,
    MICROSERVICES_SPLIT,
    CACHING_STRATEGY,
    PERMISSION_MODEL,
    APPROVAL_WORKFLOW,
)

from .strategies import (
    CritiqueResult,
    single_critic_strategy,
    multi_perspective_critique,
    iterative_critique,
    adversarial_critique,
    run_critique_strategy,
    STRATEGIES,
)

from .evals import (
    evaluate_critique_coverage,
    evaluate_recommendation_quality,
    evaluate_critique_depth,
    evaluate_critique,
    compare_strategies,
    batch_evaluate,
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
]
