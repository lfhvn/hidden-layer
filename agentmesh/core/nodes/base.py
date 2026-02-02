"""
Base classes for workflow nodes.

Each node type wraps Hidden Layer harness functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from agentmesh.core.models import ExecutionContext, StepResult


class WorkflowNodeExecutor(ABC):
    """
    Base class for workflow node executors.

    Each node type implements execute() to perform its action.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize node with configuration.

        Args:
            config: Node-specific configuration dict
        """
        self.config = config

    @abstractmethod
    async def execute(self, input_data: Any, context: ExecutionContext) -> StepResult:
        """
        Execute the node action.

        Args:
            input_data: Input to the node
            context: Execution context (provider, model, etc.)

        Returns:
            StepResult with output and metadata
        """
        pass


class StrategyNode(WorkflowNodeExecutor):
    """
    Base class for nodes that wrap Hidden Layer strategies.

    Subclasses specify which strategy_id to use from harness.
    """

    # Subclasses override this
    strategy_id: str = None

    async def execute(self, input_data: Any, context: ExecutionContext) -> StepResult:
        """
        Execute Hidden Layer strategy.

        This wraps harness.run_strategy() with AgentMesh product features:
        - Async execution
        - Error handling
        - Metrics collection
        """
        import time
        from harness import run_strategy

        if not self.strategy_id:
            raise ValueError(f"{self.__class__.__name__} must set strategy_id")

        # Extract task input
        task_input = input_data.get('task') if isinstance(input_data, dict) else str(input_data)

        start_time = time.time()

        # Call Hidden Layer harness (research code!)
        result = run_strategy(
            self.strategy_id,
            task_input=task_input,
            **self.config,  # Pass node config (n_debaters, etc.)
            **context.to_kwargs()  # Pass execution context (provider, model)
        )

        latency = time.time() - start_time

        # Return product-wrapped result
        return StepResult(
            output=result.output,
            metadata={
                'strategy': result.strategy_name,
                'strategy_metadata': result.metadata
            },
            latency_s=latency,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd
        )
