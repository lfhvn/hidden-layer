"""
Concrete strategy nodes that wrap Hidden Layer multi-agent strategies.

Each node is a thin wrapper around a harness strategy.
When research adds a new strategy, add a new class here.
"""

from agentmesh.core.nodes.base import StrategyNode


class DebateNode(StrategyNode):
    """
    Wraps Hidden Layer debate strategy.

    Multi-agent debate with judge.

    Config:
        n_debaters: Number of debating agents (default 3)
        n_rounds: Number of debate rounds (default 2)

    Research: /communication/multi-agent/strategies.py::debate()
    """
    strategy_id = "debate"


class CRITNode(StrategyNode):
    """
    Wraps Hidden Layer CRIT strategy.

    Multi-perspective design critique.

    Config:
        perspectives: List of perspective IDs to use
        design_problem: Problem type (if using CRIT framework)

    Research: /communication/multi-agent/strategies.py::crit()
    """
    strategy_id = "crit"


class ConsensusNode(StrategyNode):
    """
    Wraps Hidden Layer consensus strategy.

    Multiple agents find agreement.

    Config:
        n_agents: Number of agents (default 3)
        n_rounds: Consensus rounds (default 2)

    Research: /communication/multi-agent/strategies.py::consensus()
    """
    strategy_id = "consensus"


class ManagerWorkerNode(StrategyNode):
    """
    Wraps Hidden Layer manager-worker strategy.

    Manager decomposes task, workers execute in parallel, manager synthesizes.

    Config:
        n_workers: Number of worker agents (default 3)

    Research: /communication/multi-agent/strategies.py::manager_worker()
    """
    strategy_id = "manager_worker"


class SelfConsistencyNode(StrategyNode):
    """
    Wraps Hidden Layer self-consistency strategy.

    Sample multiple times, aggregate via majority voting.

    Config:
        n_samples: Number of samples (default 5)

    Research: /communication/multi-agent/strategies.py::self_consistency()
    """
    strategy_id = "self_consistency"


class SingleNode(StrategyNode):
    """
    Wraps Hidden Layer single-agent strategy (baseline).

    Simple single LLM call.

    Research: /communication/multi-agent/strategies.py::single()
    """
    strategy_id = "single"


# Registry: Map strategy IDs to node classes
NODE_REGISTRY = {
    "debate": DebateNode,
    "crit": CRITNode,
    "consensus": ConsensusNode,
    "manager_worker": ManagerWorkerNode,
    "self_consistency": SelfConsistencyNode,
    "single": SingleNode,
}


def get_strategy_node(strategy_id: str, config: dict) -> StrategyNode:
    """
    Factory function to create strategy node by ID.

    Args:
        strategy_id: Strategy identifier (e.g., "debate")
        config: Node configuration dict

    Returns:
        Instantiated strategy node

    Raises:
        ValueError: If strategy_id not found
    """
    node_class = NODE_REGISTRY.get(strategy_id)
    if not node_class:
        available = ", ".join(NODE_REGISTRY.keys())
        raise ValueError(
            f"Unknown strategy_id: {strategy_id}. "
            f"Available strategies: {available}"
        )
    return node_class(config)
