"""
Hidden Layer Harness - Core Infrastructure

A toolkit for LLM research providing:
- Unified LLM provider abstraction (Ollama, MLX, Claude, GPT, etc.)
- Experiment tracking and reproducibility
- Evaluation utilities and benchmarks
- Model configuration management
- System prompt management

This is the core infrastructure used by all Hidden Layer research projects.
Can be used standalone or as part of the Hidden Layer research lab.
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
from .experiment_tracker import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentTracker,
    compare_experiments,
    get_tracker,
)
from .llm_provider import (
    LLMProvider,
    LLMResponse,
    get_provider,
    llm_call,
    llm_call_stream,
)
from .model_config import (
    ModelConfig,
    ModelConfigManager,
    get_config_manager,
    get_model_config,
    list_model_configs,
)
from .system_prompts import (
    SystemPromptMetadata,
    get_system_prompt_info,
    list_system_prompts,
    load_system_prompt,
    load_system_prompt_metadata,
    show_prompt,
)

__version__ = "0.2.0"

__all__ = [
    # LLM Provider
    "llm_call",
    "llm_call_stream",
    "get_provider",
    "LLMProvider",
    "LLMResponse",
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
