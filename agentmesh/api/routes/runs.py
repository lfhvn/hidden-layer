"""
Run execution API endpoints.
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agentmesh.core.models import ExecutionContext, WorkflowRun
from agentmesh.core.orchestrator.engine import WorkflowOrchestrator
from agentmesh.db.repository import AgentMeshRepository
from agentmesh.db.session import get_db_session


router = APIRouter(tags=["runs"])


# Pydantic schemas

class CreateRunRequest(BaseModel):
    """Request to create and execute a run"""
    input: dict = Field(..., description="Input data for the workflow")
    context: dict | None = Field(
        None,
        description="Execution context (provider, model, etc.)",
        example={
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.7
        }
    )


class RunResponse(BaseModel):
    """Run response"""
    id: str
    workflow_id: str
    org_id: str
    status: str
    input: dict
    output: dict | None
    error: str | None
    started_at: str | None
    finished_at: str | None
    created_at: str


class StepResponse(BaseModel):
    """Step response"""
    id: str
    run_id: str
    node_id: str
    node_type: str
    status: str
    input: dict | None
    output: dict | None
    error: str | None
    latency_s: float | None
    tokens_in: int | None
    tokens_out: int | None
    cost_usd: float | None
    started_at: str | None
    finished_at: str | None


# Dependency
async def get_repo():
    async with get_db_session() as session:
        yield AgentMeshRepository(session)


# Endpoints

@router.post("/workflows/{workflow_id}/runs", response_model=RunResponse)
async def create_run(
    workflow_id: str,
    request: CreateRunRequest,
    repo: AgentMeshRepository = Depends(get_repo)
):
    """
    Execute a workflow.

    This creates a run and executes it using Hidden Layer strategies.

    Example request body:
    ```json
    {
      "input": {
        "task": "Should we invest in renewable energy?"
      },
      "context": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022"
      }
    }
    ```
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

    # Execute workflow (uses Hidden Layer harness!)
    orchestrator = WorkflowOrchestrator(workflow, run, repo)

    try:
        run = await orchestrator.execute()
    except Exception as e:
        # Run status already updated by orchestrator
        pass

    return RunResponse(
        id=str(run.id),
        workflow_id=str(run.workflow_id),
        org_id=str(run.org_id),
        status=run.status.value,
        input=run.input,
        output=run.output,
        error=run.error,
        started_at=run.started_at.isoformat() if run.started_at else None,
        finished_at=run.finished_at.isoformat() if run.finished_at else None,
        created_at=run.created_at.isoformat(),
    )


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    repo: AgentMeshRepository = Depends(get_repo)
):
    """Get a run by ID"""
    run = await repo.get_run(UUID(run_id))

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return RunResponse(
        id=str(run.id),
        workflow_id=str(run.workflow_id),
        org_id=str(run.org_id),
        status=run.status.value,
        input=run.input,
        output=run.output,
        error=run.error,
        started_at=run.started_at.isoformat() if run.started_at else None,
        finished_at=run.finished_at.isoformat() if run.finished_at else None,
        created_at=run.created_at.isoformat(),
    )


@router.get("/runs/{run_id}/steps", response_model=List[StepResponse])
async def get_run_steps(
    run_id: str,
    repo: AgentMeshRepository = Depends(get_repo)
):
    """Get all steps for a run"""
    steps = await repo.list_steps(UUID(run_id))

    return [
        StepResponse(
            id=str(s.id),
            run_id=str(s.run_id),
            node_id=s.node_id,
            node_type=s.node_type.value,
            status=s.status.value,
            input=s.input,
            output=s.output,
            error=s.error,
            latency_s=s.latency_s,
            tokens_in=s.tokens_in,
            tokens_out=s.tokens_out,
            cost_usd=s.cost_usd,
            started_at=s.started_at.isoformat() if s.started_at else None,
            finished_at=s.finished_at.isoformat() if s.finished_at else None,
        )
        for s in steps
    ]
