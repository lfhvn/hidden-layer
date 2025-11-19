"""
Async run execution API endpoints (with Celery).
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agentmesh.core.models import ExecutionContext, WorkflowRun
from agentmesh.db.repository import AgentMeshRepository
from agentmesh.db.session import get_db_session
from agentmesh.worker.tasks import execute_workflow_async


router = APIRouter(tags=["async-runs"])


class CreateAsyncRunRequest(BaseModel):
    """Request to create and execute a run asynchronously"""
    input: dict = Field(..., description="Input data for the workflow")
    context: dict | None = Field(None, description="Execution context")


class AsyncRunResponse(BaseModel):
    """Async run response"""
    run_id: str
    task_id: str
    status: str


# Dependency
async def get_repo():
    async with get_db_session() as session:
        yield AgentMeshRepository(session)


@router.post("/workflows/{workflow_id}/runs/async", response_model=AsyncRunResponse)
async def create_async_run(
    workflow_id: str,
    request: CreateAsyncRunRequest,
    repo: AgentMeshRepository = Depends(get_repo)
):
    """
    Execute a workflow asynchronously using Celery.

    This returns immediately with a task ID. Use /tasks/:id to check status.
    """
    # Get workflow
    workflow = await repo.get_workflow(UUID(workflow_id))
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Parse execution context
    context_data = request.context or {}
    context = ExecutionContext(
        provider=context_data.get("provider", "anthropic"),
        model=context_data.get("model", "claude-3-5-sonnet-20241022"),
        temperature=context_data.get("temperature", 0.7),
        max_tokens=context_data.get("max_tokens"),
    )

    # Create run
    run = WorkflowRun.create(
        workflow_id=workflow.id,
        org_id=workflow.org_id,
        input=request.input,
        context=context
    )
    run = await repo.create_run(run)

    # Queue workflow execution as Celery task
    task = execute_workflow_async.delay(str(workflow.id), str(run.id))

    return AsyncRunResponse(
        run_id=str(run.id),
        task_id=task.id,
        status="queued"
    )


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    Get Celery task status.
    """
    from celery.result import AsyncResult
    from agentmesh.worker.celery_app import celery_app

    task_result = AsyncResult(task_id, app=celery_app)

    return {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else None,
        "info": task_result.info if not task_result.ready() else None,
    }
