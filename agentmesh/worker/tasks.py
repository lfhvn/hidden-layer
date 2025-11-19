"""
Celery tasks for async workflow execution.
"""

import asyncio
from uuid import UUID

from agentmesh.worker.celery_app import celery_app
from agentmesh.core.orchestrator.engine import WorkflowOrchestrator
from agentmesh.db.session import AsyncSessionLocal
from agentmesh.db.repository import AgentMeshRepository


@celery_app.task(bind=True, name="agentmesh.execute_workflow")
def execute_workflow_async(self, workflow_id: str, run_id: str):
    """
    Execute a workflow asynchronously.

    Args:
        workflow_id: Workflow ID
        run_id: Run ID

    Returns:
        Run status and output
    """
    # Update task state
    self.update_state(state="PROGRESS", meta={"status": "Starting workflow execution"})

    # Run async code in event loop
    return asyncio.run(_execute_workflow_async_impl(workflow_id, run_id))


async def _execute_workflow_async_impl(workflow_id: str, run_id: str):
    """
    Async implementation of workflow execution.
    """
    async with AsyncSessionLocal() as session:
        repo = AgentMeshRepository(session)

        # Get workflow and run
        workflow = await repo.get_workflow(UUID(workflow_id))
        run = await repo.get_run(UUID(run_id))

        if not workflow or not run:
            return {"status": "error", "message": "Workflow or run not found"}

        # Execute workflow
        orchestrator = WorkflowOrchestrator(workflow, run, repo)

        try:
            run = await orchestrator.execute()
            return {
                "status": run.status.value,
                "output": run.output,
                "error": run.error,
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
            }
