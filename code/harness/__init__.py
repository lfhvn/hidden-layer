"""
Agentic Simulation Harness

A toolkit for running and comparing single-model and multi-agent strategies.
Supports local (MLX, Ollama) and API (Anthropic, OpenAI) providers.
"""

from . import defaults
from .benchmarks import BENCHMARKS, get_baseline_scores, load_benchmark
from .evals import (
    EVAL_FUNCTIONS,
    evaluate_task,
    exact_match,
    keyword_match,
    llm_judge,
    numeric_match,
    win_rate_comparison,
)
from .experiment_tracker import ExperimentConfig, ExperimentResult, ExperimentTracker, compare_experiments, get_tracker
from .llm_provider import LLMProvider, LLMResponse, get_provider, llm_call, llm_call_stream
from .model_config import ModelConfig, ModelConfigManager, get_config_manager, get_model_config, list_model_configs
from .rationale import (
    RationaleResponse,
    ask_with_reasoning,
    extract_rationale_from_result,
    llm_call_with_rationale,
    run_strategy_with_rationale,
)
from .strategies import (
    STRATEGIES,
    StrategyResult,
    consensus_strategy,
    debate_strategy,
    introspection_strategy,
    manager_worker_strategy,
    run_strategy,
    self_consistency_strategy,
    single_model_strategy,
)
from .system_prompts import (
    SystemPromptMetadata,
    get_system_prompt_info,
    list_system_prompts,
    load_system_prompt,
    load_system_prompt_metadata,
    show_prompt,
)

# Introspection modules (optional - only if MLX is available)
try:
    from .activation_steering import ActivationCache, ActivationSteerer, SteeringConfig
    from .concept_vectors import ConceptLibrary, ConceptVector, build_emotion_library, build_topic_library
    from .introspection_tasks import (
        IntrospectionEvaluator,
        IntrospectionResult,
        IntrospectionTask,
        IntrospectionTaskGenerator,
        IntrospectionTaskType,
    )

    _has_introspection = True
except ImportError:
    _has_introspection = False

# API Introspection (always available - doesn't require MLX)
try:
    from .introspection_api import (
        NATURAL_INTROSPECTION_PROMPTS,
        APIIntrospectionTester,
        PromptSteerer,
        PromptSteeringConfig,
    )

    _has_api_introspection = True
except ImportError:
    _has_api_introspection = False

__version__ = "0.1.0"

__all__ = [
    # LLM Provider
    "llm_call",
    "llm_call_stream",
    "get_provider",
    "LLMProvider",
    "LLMResponse",
    # Strategies
    "run_strategy",
    "single_model_strategy",
    "debate_strategy",
    "consensus_strategy",
    "self_consistency_strategy",
    "manager_worker_strategy",
    "introspection_strategy",
    "StrategyResult",
    "STRATEGIES",
    # Defaults
    "defaults",
    # Experiment Tracking
    "ExperimentTracker",
    "ExperimentConfig",
    "ExperimentResult",
    "get_tracker",
    "compare_experiments",
    # Evaluation
    "exact_match",
    "keyword_match",
    "numeric_match",
    "llm_judge",
    "win_rate_comparison",
    "evaluate_task",
    "EVAL_FUNCTIONS",
    # Model Configuration
    "ModelConfig",
    "ModelConfigManager",
    "get_config_manager",
    "get_model_config",
    "list_model_configs",
    # Rationale Extraction
    "llm_call_with_rationale",
    "extract_rationale_from_result",
    "run_strategy_with_rationale",
    "ask_with_reasoning",
    "RationaleResponse",
    # Benchmarks
    "load_benchmark",
    "get_baseline_scores",
    "BENCHMARKS",
    # System Prompts
    "load_system_prompt",
    "load_system_prompt_metadata",
    "list_system_prompts",
    "get_system_prompt_info",
    "show_prompt",
    "SystemPromptMetadata",
]

# Add introspection exports if available
if _has_introspection:
    __all__.extend(
        [
            # Activation Steering
            "ActivationSteerer",
            "SteeringConfig",
            "ActivationCache",
            # Concept Vectors
            "ConceptLibrary",
            "ConceptVector",
            "build_emotion_library",
            "build_topic_library",
            # Introspection Tasks
            "IntrospectionTask",
            "IntrospectionResult",
            "IntrospectionTaskType",
            "IntrospectionTaskGenerator",
            "IntrospectionEvaluator",
        ]
    )

# Add API introspection exports if available
if _has_api_introspection:
    __all__.extend(
        [
            # API Introspection
            "APIIntrospectionTester",
            "PromptSteerer",
            "PromptSteeringConfig",
            "NATURAL_INTROSPECTION_PROMPTS",
        ]
    )
