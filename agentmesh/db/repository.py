"""
Database repository for AgentMesh.

Handles CRUD operations and converts between domain models and DB models.
"""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentmesh.core.models import (
    ExecutionContext,
    Workflow,
    WorkflowEdge,
    WorkflowGraph,
    WorkflowNode,
    WorkflowRun,
    WorkflowStep,
    NodeType,
)
from agentmesh.db.models import OrganizationModel, RunModel, StepModel, WorkflowModel


class AgentMeshRepository:
    """Database repository for AgentMesh entities"""

    def __init__(self, session: AsyncSession):
        self.session = session

    # Workflows

    async def create_workflow(self, workflow: Workflow) -> Workflow:
        """Create a new workflow"""
        db_workflow = WorkflowModel(
            id=str(workflow.id),
            org_id=str(workflow.org_id),
            name=workflow.name,
            description=workflow.description,
            graph=self._graph_to_dict(workflow.graph),
        )
        self.session.add(db_workflow)
        await self.session.commit()
        return workflow

    async def get_workflow(self, workflow_id: UUID) -> Optional[Workflow]:
        """Get workflow by ID"""
        result = await self.session.execute(
            select(WorkflowModel).where(WorkflowModel.id == str(workflow_id))
        )
        db_workflow = result.scalar_one_or_none()
        return self._workflow_from_db(db_workflow) if db_workflow else None

    async def list_workflows(self, org_id: UUID) -> List[Workflow]:
        """List all workflows for an organization"""
        result = await self.session.execute(
            select(WorkflowModel).where(WorkflowModel.org_id == str(org_id))
        )
        return [self._workflow_from_db(w) for w in result.scalars().all()]

    # Runs

    async def create_run(self, run: WorkflowRun) -> WorkflowRun:
        """Create a new run"""
        db_run = RunModel(
            id=str(run.id),
            workflow_id=str(run.workflow_id),
            org_id=str(run.org_id),
            status=run.status,
            input=run.input,
            output=run.output,
            context=self._context_to_dict(run.context),
            error=run.error,
            started_at=run.started_at,
            finished_at=run.finished_at,
        )
        self.session.add(db_run)
        await self.session.commit()
        return run

    async def update_run(self, run: WorkflowRun) -> WorkflowRun:
        """Update existing run"""
        result = await self.session.execute(
            select(RunModel).where(RunModel.id == str(run.id))
        )
        db_run = result.scalar_one()

        db_run.status = run.status
        db_run.output = run.output
        db_run.error = run.error
        db_run.started_at = run.started_at
        db_run.finished_at = run.finished_at

        await self.session.commit()
        return run

    async def get_run(self, run_id: UUID) -> Optional[WorkflowRun]:
        """Get run by ID"""
        result = await self.session.execute(
            select(RunModel).where(RunModel.id == str(run_id))
        )
        db_run = result.scalar_one_or_none()
        return self._run_from_db(db_run) if db_run else None

    # Steps

    async def create_step(self, step: WorkflowStep) -> WorkflowStep:
        """Create a new step"""
        db_step = StepModel(
            id=str(step.id),
            run_id=str(step.run_id),
            workflow_id=str(step.workflow_id),
            node_id=step.node_id,
            node_type=step.node_type,
            status=step.status,
            input=step.input,
            output=step.output,
            error=step.error,
            latency_s=step.latency_s,
            tokens_in=step.tokens_in,
            tokens_out=step.tokens_out,
            cost_usd=step.cost_usd,
            started_at=step.started_at,
            finished_at=step.finished_at,
        )
        self.session.add(db_step)
        await self.session.commit()
        return step

    async def update_step(self, step: WorkflowStep) -> WorkflowStep:
        """Update existing step"""
        result = await self.session.execute(
            select(StepModel).where(StepModel.id == str(step.id))
        )
        db_step = result.scalar_one()

        db_step.status = step.status
        db_step.output = step.output
        db_step.error = step.error
        db_step.latency_s = step.latency_s
        db_step.tokens_in = step.tokens_in
        db_step.tokens_out = step.tokens_out
        db_step.cost_usd = step.cost_usd
        db_step.started_at = step.started_at
        db_step.finished_at = step.finished_at

        await self.session.commit()
        return step

    async def list_steps(self, run_id: UUID) -> List[WorkflowStep]:
        """List all steps for a run"""
        result = await self.session.execute(
            select(StepModel).where(StepModel.run_id == str(run_id))
            .order_by(StepModel.created_at)
        )
        return [self._step_from_db(s) for s in result.scalars().all()]

    # Conversion helpers

    def _workflow_from_db(self, db: WorkflowModel) -> Workflow:
        """Convert DB model to domain model"""
        return Workflow(
            id=UUID(db.id),
            org_id=UUID(db.org_id),
            name=db.name,
            description=db.description,
            graph=self._graph_from_dict(db.graph),
            created_at=db.created_at,
            updated_at=db.updated_at,
        )

    def _run_from_db(self, db: RunModel) -> WorkflowRun:
        """Convert DB model to domain model"""
        return WorkflowRun(
            id=UUID(db.id),
            workflow_id=UUID(db.workflow_id),
            org_id=UUID(db.org_id),
            status=db.status,
            input=db.input,
            output=db.output,
            context=self._context_from_dict(db.context),
            error=db.error,
            started_at=db.started_at,
            finished_at=db.finished_at,
            created_at=db.created_at,
        )

    def _step_from_db(self, db: StepModel) -> WorkflowStep:
        """Convert DB model to domain model"""
        return WorkflowStep(
            id=UUID(db.id),
            run_id=UUID(db.run_id),
            workflow_id=UUID(db.workflow_id),
            node_id=db.node_id,
            node_type=db.node_type,
            status=db.status,
            input=db.input,
            output=db.output,
            error=db.error,
            latency_s=db.latency_s,
            tokens_in=db.tokens_in,
            tokens_out=db.tokens_out,
            cost_usd=db.cost_usd,
            started_at=db.started_at,
            finished_at=db.finished_at,
            created_at=db.created_at,
        )

    def _graph_to_dict(self, graph: WorkflowGraph) -> dict:
        """Convert WorkflowGraph to JSON dict"""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type.value,
                    "label": n.label,
                    "config": n.config,
                    "strategy_id": n.strategy_id,
                    "tool_id": n.tool_id,
                    "condition": n.condition,
                }
                for n in graph.nodes
            ],
            "edges": [
                {
                    "id": e.id,
                    "from_node_id": e.from_node_id,
                    "to_node_id": e.to_node_id,
                    "condition": e.condition,
                }
                for e in graph.edges
            ],
        }

    def _graph_from_dict(self, data: dict) -> WorkflowGraph:
        """Convert JSON dict to WorkflowGraph"""
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
            for n in data["nodes"]
        ]
        edges = [
            WorkflowEdge(
                id=e["id"],
                from_node_id=e["from_node_id"],
                to_node_id=e["to_node_id"],
                condition=e.get("condition"),
            )
            for e in data["edges"]
        ]
        return WorkflowGraph(nodes=nodes, edges=edges)

    def _context_to_dict(self, context: ExecutionContext) -> dict:
        """Convert ExecutionContext to JSON dict"""
        return {
            "provider": context.provider,
            "model": context.model,
            "temperature": context.temperature,
            "max_tokens": context.max_tokens,
        }

    def _context_from_dict(self, data: dict) -> ExecutionContext:
        """Convert JSON dict to ExecutionContext"""
        return ExecutionContext(
            provider=data.get("provider", "anthropic"),
            model=data.get("model", "claude-3-5-sonnet-20241022"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens"),
        )
