"""
Human-in-the-loop step completion API endpoints.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agentmesh.core.models import StepStatus
from agentmesh.db.repository import AgentMeshRepository
from agentmesh.db.session import get_db_session


router = APIRouter(tags=["human-steps"])


class HumanCompleteRequest(BaseModel):
    """Request to complete a human step"""
    approved: bool
    output: dict | None = None
    comment: str | None = None


# Dependency
async def get_repo():
    async with get_db_session() as session:
        yield AgentMeshRepository(session)


@router.post("/steps/{step_id}/human-complete")
async def complete_human_step(
    step_id: str,
    request: HumanCompleteRequest,
    repo: AgentMeshRepository = Depends(get_repo)
):
    """
    Complete a human-in-the-loop step.

    This resumes workflow execution after human review.
    """
    # Get step
    step = await repo.get_step(UUID(step_id))
    if not step:
        raise HTTPException(status_code=404, detail="Step not found")

    # Verify step is waiting for human
    if step.status != StepStatus.WAITING_HUMAN:
        raise HTTPException(
            status_code=400,
            detail=f"Step is not waiting for human (status: {step.status})"
        )

    # Update step based on human response
    if request.approved:
        step.status = StepStatus.SUCCEEDED
        step.output = request.output if request.output else step.output
    else:
        step.status = StepStatus.FAILED
        step.error = f"Rejected by human: {request.comment or 'No reason provided'}"

    step = await repo.update_step(step)

    # TODO: Resume workflow execution from this step
    # This would require workflow resumption logic

    return {
        "step_id": str(step.id),
        "status": step.status.value,
        "message": "Step completed. Workflow resumption not yet implemented."
    }


@router.get("/steps/{step_id}")
async def get_step_detail(
    step_id: str,
    repo: AgentMeshRepository = Depends(get_repo)
):
    """Get step detail (including human-waiting steps)"""
    step = await repo.get_step(UUID(step_id))
    if not step:
        raise HTTPException(status_code=404, detail="Step not found")

    return {
        "id": str(step.id),
        "run_id": str(step.run_id),
        "node_id": step.node_id,
        "node_type": step.node_type.value,
        "status": step.status.value,
        "input": step.input,
        "output": step.output,
        "error": step.error,
        "created_at": step.created_at.isoformat() if step.created_at else None,
        "started_at": step.started_at.isoformat() if step.started_at else None,
        "finished_at": step.finished_at.isoformat() if step.finished_at else None,
    }
