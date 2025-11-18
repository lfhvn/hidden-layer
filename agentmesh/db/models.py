"""
SQLAlchemy database models for AgentMesh.

Maps domain models to database tables.
"""

import json
from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, Enum, Float, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from agentmesh.core.models import NodeType, RunStatus, StepStatus


Base = declarative_base()


def generate_uuid():
    """Generate UUID for primary keys"""
    return str(uuid4())


class OrganizationModel(Base):
    """Organization/tenant"""
    __tablename__ = "organizations"

    id = Column(UUID, primary_key=True, default=generate_uuid)
    name = Column(Text, nullable=False)
    plan = Column(Text, default="free")  # free, pro, team, enterprise
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    workflows = relationship("WorkflowModel", back_populates="organization")
    runs = relationship("RunModel", back_populates="organization")


class WorkflowModel(Base):
    """Workflow definition"""
    __tablename__ = "workflows"

    id = Column(UUID, primary_key=True, default=generate_uuid)
    org_id = Column(UUID, ForeignKey("organizations.id"), nullable=False)
    name = Column(Text, nullable=False)
    description = Column(Text)
    graph = Column(JSONB, nullable=False)  # WorkflowGraph as JSON
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    organization = relationship("OrganizationModel", back_populates="workflows")
    runs = relationship("RunModel", back_populates="workflow")


class RunModel(Base):
    """Workflow run execution"""
    __tablename__ = "runs"

    id = Column(UUID, primary_key=True, default=generate_uuid)
    workflow_id = Column(UUID, ForeignKey("workflows.id"), nullable=False)
    org_id = Column(UUID, ForeignKey("organizations.id"), nullable=False)

    status = Column(Enum(RunStatus), default=RunStatus.PENDING, nullable=False)

    input = Column(JSONB, nullable=False)  # Run input payload
    output = Column(JSONB)  # Run output (when succeeded)
    context = Column(JSONB, nullable=False)  # ExecutionContext as JSON

    error = Column(Text)  # Error message if failed

    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    workflow = relationship("WorkflowModel", back_populates="runs")
    organization = relationship("OrganizationModel", back_populates="runs")
    steps = relationship("StepModel", back_populates="run")


class StepModel(Base):
    """Individual step in a workflow run"""
    __tablename__ = "steps"

    id = Column(UUID, primary_key=True, default=generate_uuid)
    run_id = Column(UUID, ForeignKey("runs.id"), nullable=False)
    workflow_id = Column(UUID, ForeignKey("workflows.id"), nullable=False)

    node_id = Column(Text, nullable=False)  # ID of node in workflow graph
    node_type = Column(Enum(NodeType), nullable=False)

    status = Column(Enum(StepStatus), default=StepStatus.PENDING, nullable=False)

    input = Column(JSONB)  # Step input
    output = Column(JSONB)  # Step output
    error = Column(Text)  # Error message if failed

    # Metrics from Hidden Layer harness (for strategy nodes)
    latency_s = Column(Float)
    tokens_in = Column(Integer)
    tokens_out = Column(Integer)
    cost_usd = Column(Float)

    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    run = relationship("RunModel", back_populates="steps")


# Indexes for common queries
from sqlalchemy import Index

Index('idx_workflows_org', WorkflowModel.org_id)
Index('idx_runs_workflow', RunModel.workflow_id)
Index('idx_runs_org', RunModel.org_id)
Index('idx_runs_status', RunModel.status)
Index('idx_steps_run', StepModel.run_id)
Index('idx_steps_status', StepModel.status)
