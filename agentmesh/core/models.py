"""
Core domain models for AgentMesh.

These are business logic models, separate from DB models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class RunStatus(str, Enum):
    """Status of a workflow run"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class StepStatus(str, Enum):
    """Status of a workflow step"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    WAITING_HUMAN = "waiting_human"


class NodeType(str, Enum):
    """Type of workflow node"""
    START = "start"
    END = "end"
    STRATEGY = "strategy"  # Wraps Hidden Layer strategy
    TOOL = "tool"
    HUMAN_APPROVAL = "human_approval"
    BRANCH = "branch"


@dataclass
class WorkflowNode:
    """A node in a workflow graph"""
    id: str
    type: NodeType
    label: str
    config: Dict[str, Any] = field(default_factory=dict)

    # For strategy nodes
    strategy_id: Optional[str] = None  # e.g., "debate", "crit"

    # For tool nodes
    tool_id: Optional[str] = None

    # For branch nodes
    condition: Optional[str] = None


@dataclass
class WorkflowEdge:
    """An edge connecting workflow nodes"""
    id: str
    from_node_id: str
    to_node_id: str
    condition: Optional[str] = None  # For conditional branches


@dataclass
class WorkflowGraph:
    """The graph structure of a workflow"""
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]

    def get_node(self, node_id: str) -> Optional[WorkflowNode]:
        """Get node by ID"""
        return next((n for n in self.nodes if n.id == node_id), None)

    def get_outgoing_edges(self, node_id: str) -> List[WorkflowEdge]:
        """Get edges leaving a node"""
        return [e for e in self.edges if e.from_node_id == node_id]

    def get_incoming_edges(self, node_id: str) -> List[WorkflowEdge]:
        """Get edges entering a node"""
        return [e for e in self.edges if e.to_node_id == node_id]


@dataclass
class Workflow:
    """A workflow definition"""
    id: UUID
    org_id: UUID
    name: str
    description: Optional[str]
    graph: WorkflowGraph
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(cls, org_id: UUID, name: str, graph: WorkflowGraph, description: Optional[str] = None):
        """Create new workflow"""
        now = datetime.utcnow()
        return cls(
            id=uuid4(),
            org_id=org_id,
            name=name,
            description=description,
            graph=graph,
            created_at=now,
            updated_at=now
        )


@dataclass
class ExecutionContext:
    """Context passed through workflow execution"""
    provider: str = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    def to_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for harness calls"""
        kwargs = {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        return kwargs


@dataclass
class WorkflowRun:
    """An execution of a workflow"""
    id: UUID
    workflow_id: UUID
    org_id: UUID
    status: RunStatus
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]]
    context: ExecutionContext
    error: Optional[str]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    created_at: datetime

    @classmethod
    def create(cls, workflow_id: UUID, org_id: UUID, input: Dict[str, Any], context: ExecutionContext):
        """Create new run"""
        return cls(
            id=uuid4(),
            workflow_id=workflow_id,
            org_id=org_id,
            status=RunStatus.PENDING,
            input=input,
            output=None,
            context=context,
            error=None,
            started_at=None,
            finished_at=None,
            created_at=datetime.utcnow()
        )


@dataclass
class StepResult:
    """Result from executing a step"""
    output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Metrics from harness (if strategy step)
    latency_s: Optional[float] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_usd: Optional[float] = None


@dataclass
class WorkflowStep:
    """A single step in a workflow run"""
    id: UUID
    run_id: UUID
    workflow_id: UUID
    node_id: str
    node_type: NodeType
    status: StepStatus
    input: Any
    output: Optional[Any]
    error: Optional[str]

    # Metrics (from harness for strategy nodes)
    latency_s: Optional[float]
    tokens_in: Optional[int]
    tokens_out: Optional[int]
    cost_usd: Optional[float]

    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    created_at: datetime

    parent_step_ids: List[UUID] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        run_id: UUID,
        workflow_id: UUID,
        node_id: str,
        node_type: NodeType,
        input: Any,
        parent_step_ids: List[UUID] = None
    ):
        """Create new step"""
        return cls(
            id=uuid4(),
            run_id=run_id,
            workflow_id=workflow_id,
            node_id=node_id,
            node_type=node_type,
            status=StepStatus.PENDING,
            input=input,
            output=None,
            error=None,
            latency_s=None,
            tokens_in=None,
            tokens_out=None,
            cost_usd=None,
            started_at=None,
            finished_at=None,
            created_at=datetime.utcnow(),
            parent_step_ids=parent_step_ids or []
        )
