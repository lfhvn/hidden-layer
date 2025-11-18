"""
Workflow API endpoints.
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agentmesh.core.models import Workflow, WorkflowGraph
from agentmesh.db.repository import AgentMeshRepository
from agentmesh.db.session import get_db_session


router = APIRouter(tags=["workflows"])


# Pydantic schemas for API

class CreateWorkflowRequest(BaseModel):
    """Request to create a workflow"""
    name: str = Field(..., description="Workflow name")
    description: str | None = Field(None, description="Workflow description")
    graph: dict = Field(..., description="Workflow graph (nodes and edges)")
    org_id: str = Field(..., description="Organization ID")


class WorkflowResponse(BaseModel):
    """Workflow response"""
    id: str
    org_id: str
    name: str
    description: str | None
    graph: dict
    created_at: str
    updated_at: str


# Dependency: Get database repository
async def get_repo():
    async with get_db_session() as session:
        yield AgentMeshRepository(session)


# Endpoints

@router.post("/workflows", response_model=WorkflowResponse)
async def create_workflow(
    request: CreateWorkflowRequest,
    repo: AgentMeshRepository = Depends(get_repo)
):
    """
    Create a new workflow.

    Example request body:
    ```json
    {
      "name": "Debate Analysis",
      "org_id": "123e4567-e89b-12d3-a456-426614174000",
      "graph": {
        "nodes": [
          {
            "id": "start",
            "type": "start",
            "label": "Start"
          },
          {
            "id": "debate1",
            "type": "strategy",
            "label": "Debate",
            "strategy_id": "debate",
            "config": {"n_debaters": 3}
          },
          {
            "id": "end",
            "type": "end",
            "label": "End"
          }
        ],
        "edges": [
          {"id": "e1", "from_node_id": "start", "to_node_id": "debate1"},
          {"id": "e2", "from_node_id": "debate1", "to_node_id": "end"}
        ]
      }
    }
    ```
    """
    # Create domain model
    from agentmesh.core.models import WorkflowNode, WorkflowEdge, NodeType

    # Parse graph
    nodes = [
        WorkflowNode(
            id=n["id"],
            type=NodeType(n["type"]),
            label=n["label"],
            config=n.get("config", {}),
            strategy_id=n.get("strategy_id"),
            tool_id=n.get("tool_id"),
            condition=n.get("condition"),
        )
        for n in request.graph["nodes"]
    ]

    edges = [
        WorkflowEdge(
            id=e["id"],
            from_node_id=e["from_node_id"],
            to_node_id=e["to_node_id"],
            condition=e.get("condition"),
        )
        for e in request.graph["edges"]
    ]

    graph = WorkflowGraph(nodes=nodes, edges=edges)

    # Create workflow
    workflow = Workflow.create(
        org_id=UUID(request.org_id),
        name=request.name,
        graph=graph,
        description=request.description
    )

    # Save to DB
    workflow = await repo.create_workflow(workflow)

    return WorkflowResponse(
        id=str(workflow.id),
        org_id=str(workflow.org_id),
        name=workflow.name,
        description=workflow.description,
        graph=repo._graph_to_dict(workflow.graph),
        created_at=workflow.created_at.isoformat(),
        updated_at=workflow.updated_at.isoformat(),
    )


@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    repo: AgentMeshRepository = Depends(get_repo)
):
    """Get a workflow by ID"""
    workflow = await repo.get_workflow(UUID(workflow_id))

    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return WorkflowResponse(
        id=str(workflow.id),
        org_id=str(workflow.org_id),
        name=workflow.name,
        description=workflow.description,
        graph=repo._graph_to_dict(workflow.graph),
        created_at=workflow.created_at.isoformat(),
        updated_at=workflow.updated_at.isoformat(),
    )


@router.get("/workflows", response_model=List[WorkflowResponse])
async def list_workflows(
    org_id: str,
    repo: AgentMeshRepository = Depends(get_repo)
):
    """List all workflows for an organization"""
    workflows = await repo.list_workflows(UUID(org_id))

    return [
        WorkflowResponse(
            id=str(w.id),
            org_id=str(w.org_id),
            name=w.name,
            description=w.description,
            graph=repo._graph_to_dict(w.graph),
            created_at=w.created_at.isoformat(),
            updated_at=w.updated_at.isoformat(),
        )
        for w in workflows
    ]
