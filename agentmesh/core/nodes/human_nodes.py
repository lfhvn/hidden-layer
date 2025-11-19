"""
Human-in-the-loop workflow nodes.
"""

from typing import Any
from datetime import datetime

from agentmesh.core.models import ExecutionContext, StepResult
from agentmesh.core.nodes.base import WorkflowNodeExecutor


class HumanApprovalNode(WorkflowNodeExecutor):
    """
    Human approval gate node.

    Pauses workflow execution until human approves or rejects.

    Config:
        instructions: What to ask the human
        timeout_seconds: Max wait time (optional)
        allow_edit: Allow human to edit the input (default: False)
    """

    async def execute(self, input_data: Any, context: ExecutionContext) -> StepResult:
        """
        Execute human approval node.

        This node:
        1. Stores the input
        2. Sets step status to 'waiting_human'
        3. Waits for human response via API

        The orchestrator will handle the waiting state.
        """
        instructions = self.config.get("instructions", "Please review and approve")
        allow_edit = self.config.get("allow_edit", False)

        # Return a special result that tells orchestrator to wait
        return StepResult(
            output={
                "status": "waiting_approval",
                "instructions": instructions,
                "input": input_data,
                "allow_edit": allow_edit,
                "requested_at": datetime.utcnow().isoformat(),
            },
            metadata={
                "node_type": "human_approval",
                "awaiting_human": True,
            }
        )


class HumanInputNode(WorkflowNodeExecutor):
    """
    Request input from human.

    Similar to approval, but human provides new data rather than approve/reject.

    Config:
        prompt: What to ask for
        input_schema: JSON schema for expected input (optional)
        timeout_seconds: Max wait time (optional)
    """

    async def execute(self, input_data: Any, context: ExecutionContext) -> StepResult:
        """Execute human input node"""
        prompt = self.config.get("prompt", "Please provide input")
        input_schema = self.config.get("input_schema")

        return StepResult(
            output={
                "status": "waiting_input",
                "prompt": prompt,
                "input_schema": input_schema,
                "context": input_data,  # Previous step data as context
                "requested_at": datetime.utcnow().isoformat(),
            },
            metadata={
                "node_type": "human_input",
                "awaiting_human": True,
            }
        )
