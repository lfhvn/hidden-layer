"""
Workflow orchestration engine.

Executes workflow graphs using Hidden Layer strategies.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from agentmesh.core.models import (
    ExecutionContext,
    NodeType,
    RunStatus,
    StepStatus,
    Workflow,
    WorkflowGraph,
    WorkflowNode,
    WorkflowRun,
    WorkflowStep,
)
from agentmesh.core.nodes.strategy_nodes import get_strategy_node


class OrchestrationError(Exception):
    """Raised when workflow orchestration fails"""
    pass


class WorkflowOrchestrator:
    """
    Orchestrates workflow execution.

    Responsibilities:
    - Execute workflow DAG
    - Manage step dependencies
    - Update run/step status
    - Handle errors and retries

    Delegates to:
    - Hidden Layer harness for strategy execution
    - Strategy nodes for wrapping harness calls
    """

    def __init__(self, workflow: Workflow, run: WorkflowRun, db_repo):
        """
        Initialize orchestrator.

        Args:
            workflow: Workflow definition
            run: Run instance to execute
            db_repo: Database repository for persistence
        """
        self.workflow = workflow
        self.run = run
        self.db = db_repo
        self.graph = workflow.graph

    async def execute(self) -> WorkflowRun:
        """
        Execute the workflow.

        Returns:
            Updated run with results

        Raises:
            OrchestrationError: If execution fails
        """
        # Update run status
        self.run.status = RunStatus.RUNNING
        self.run.started_at = datetime.utcnow()
        await self.db.update_run(self.run)

        try:
            # Find start node
            start_node = self._find_start_node()
            if not start_node:
                raise OrchestrationError("No start node found in workflow")

            # Execute from start node
            result = await self._execute_from_node(start_node, self.run.input)

            # Update run success
            self.run.status = RunStatus.SUCCEEDED
            self.run.output = result
            self.run.finished_at = datetime.utcnow()

        except Exception as e:
            # Update run failure
            self.run.status = RunStatus.FAILED
            self.run.error = str(e)
            self.run.finished_at = datetime.utcnow()
            raise OrchestrationError(f"Workflow execution failed: {e}") from e

        finally:
            await self.db.update_run(self.run)

        return self.run

    async def _execute_from_node(self, node: WorkflowNode, input_data: Any) -> Any:
        """
        Execute starting from a specific node.

        Args:
            node: Node to execute
            input_data: Input to the node

        Returns:
            Output from node (or end node if reached)
        """
        # Base case: end node
        if node.type == NodeType.END:
            return input_data

        # Execute current node
        output = await self._execute_node(node, input_data)

        # Find next nodes
        outgoing_edges = self.graph.get_outgoing_edges(node.id)

        if not outgoing_edges:
            # No outgoing edges, this is terminal
            return output

        # For now, simple linear execution (take first edge)
        # TODO: Support branching, parallel execution
        next_edge = outgoing_edges[0]
        next_node = self.graph.get_node(next_edge.to_node_id)

        if not next_node:
            raise OrchestrationError(f"Next node not found: {next_edge.to_node_id}")

        # Recursively execute next node
        return await self._execute_from_node(next_node, output)

    async def _execute_node(self, node: WorkflowNode, input_data: Any) -> Any:
        """
        Execute a single node.

        Args:
            node: Node to execute
            input_data: Input to the node

        Returns:
            Node output
        """
        # Create step record
        step = WorkflowStep.create(
            run_id=self.run.id,
            workflow_id=self.workflow.id,
            node_id=node.id,
            node_type=node.type,
            input=input_data
        )
        step = await self.db.create_step(step)

        # Update step status to running
        step.status = StepStatus.RUNNING
        step.started_at = datetime.utcnow()
        await self.db.update_step(step)

        try:
            # Execute based on node type
            if node.type == NodeType.STRATEGY:
                result = await self._execute_strategy_node(node, input_data)
            elif node.type == NodeType.HUMAN:
                result = await self._execute_human_node(node, input_data, step)
            elif node.type == NodeType.START:
                # Start node just passes input through
                result = input_data
            else:
                raise OrchestrationError(f"Unsupported node type: {node.type}")

            # Update step success
            step.status = StepStatus.SUCCEEDED
            step.output = result.output if hasattr(result, 'output') else result
            step.finished_at = datetime.utcnow()

            # Store metrics if available (from harness)
            if hasattr(result, 'latency_s'):
                step.latency_s = result.latency_s
                step.tokens_in = result.tokens_in
                step.tokens_out = result.tokens_out
                step.cost_usd = result.cost_usd

            await self.db.update_step(step)

            # Return output for next node
            return step.output

        except Exception as e:
            # Update step failure
            step.status = StepStatus.FAILED
            step.error = str(e)
            step.finished_at = datetime.utcnow()
            await self.db.update_step(step)
            raise

    async def _execute_strategy_node(self, node: WorkflowNode, input_data: Any):
        """
        Execute a strategy node (wraps Hidden Layer strategy).

        Args:
            node: Strategy node
            input_data: Input data

        Returns:
            StepResult from strategy execution
        """
        if not node.strategy_id:
            raise OrchestrationError(f"Strategy node {node.id} missing strategy_id")

        # Get strategy node executor
        strategy_node = get_strategy_node(node.strategy_id, node.config)

        # Execute strategy (calls Hidden Layer harness!)
        result = await strategy_node.execute(input_data, self.run.context)

        return result

    async def _execute_human_node(self, node: WorkflowNode, input_data: Any, step: WorkflowStep):
        """
        Execute a human-in-the-loop node.

        This pauses workflow execution until human completes the step.

        Args:
            node: Human node
            input_data: Input data
            step: Step instance (already created)

        Returns:
            Human approval result
        """
        from agentmesh.core.nodes.human_nodes import get_human_node

        # Get human node executor
        human_node = get_human_node(node.config.get("node_type", "approval"))

        # Execute - this returns immediately with "waiting" status
        result = await human_node.execute(input_data, self.run.context)

        # Check if this requires human intervention
        if result.metadata and result.metadata.get("awaiting_human"):
            # Update step to WAITING_HUMAN status
            step.status = StepStatus.WAITING_HUMAN
            step.output = result.output
            await self.db.update_step(step)

            # Raise special exception to pause workflow
            # NOTE: Workflow will be resumed by human_steps.py endpoint
            raise OrchestrationError(
                f"Workflow paused: waiting for human approval on step {step.id}"
            )

        return result

    def _find_start_node(self) -> Optional[WorkflowNode]:
        """Find the start node in the workflow graph"""
        for node in self.graph.nodes:
            if node.type == NodeType.START:
                return node
        return None
