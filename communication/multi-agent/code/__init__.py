"""
Multi-Agent Research Project

Multi-agent coordination strategies including:
- Debate
- CRIT (design critique)
- XFN team strategies
- Self-consistency
- Manager-worker
- Consensus

Uses the harness for LLM provider abstraction and experiment tracking.
"""

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
    manager_worker_strategy,
    run_strategy,
    self_consistency_strategy,
    single_model_strategy,
)

__version__ = "0.1.0"

__all__ = [
    # Strategies
    "run_strategy",
    "single_model_strategy",
    "debate_strategy",
    "consensus_strategy",
    "self_consistency_strategy",
    "manager_worker_strategy",
    "StrategyResult",
    "STRATEGIES",
    # Rationale Extraction
    "llm_call_with_rationale",
    "extract_rationale_from_result",
    "run_strategy_with_rationale",
    "ask_with_reasoning",
    "RationaleResponse",
]
