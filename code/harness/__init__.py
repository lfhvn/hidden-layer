"""
Agentic Simulation Harness

A toolkit for running and comparing single-model and multi-agent strategies.
Supports local (MLX, Ollama) and API (Anthropic, OpenAI) providers.
"""

from .llm_provider import llm_call, llm_call_stream, get_provider, LLMProvider, LLMResponse
from .strategies import (
    run_strategy,
    single_model_strategy,
    debate_strategy,
    consensus_strategy,
    self_consistency_strategy,
    manager_worker_strategy,
    StrategyResult,
    STRATEGIES
)
from . import defaults
from .experiment_tracker import (
    ExperimentTracker,
    ExperimentConfig,
    ExperimentResult,
    get_tracker,
    compare_experiments
)
from .evals import (
    exact_match,
    keyword_match,
    numeric_match,
    llm_judge,
    win_rate_comparison,
    evaluate_task,
    EVAL_FUNCTIONS
)
from .model_config import (
    ModelConfig,
    ModelConfigManager,
    get_config_manager,
    get_model_config,
    list_model_configs
)

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
]
