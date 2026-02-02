"""
Workflow node types.

Each node wraps Hidden Layer functionality.
"""

from agentmesh.core.nodes.base import StrategyNode, WorkflowNodeExecutor
from agentmesh.core.nodes.strategy_nodes import (
    NODE_REGISTRY,
    CRITNode,
    ConsensusNode,
    DebateNode,
    ManagerWorkerNode,
    SelfConsistencyNode,
    SingleNode,
    get_strategy_node,
)

__all__ = [
    # Base classes
    "WorkflowNodeExecutor",
    "StrategyNode",
    # Concrete nodes
    "DebateNode",
    "CRITNode",
    "ConsensusNode",
    "ManagerWorkerNode",
    "SelfConsistencyNode",
    "SingleNode",
    # Registry
    "NODE_REGISTRY",
    "get_strategy_node",
]
