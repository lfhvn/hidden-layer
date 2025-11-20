"""
Context management and playbook structures for ACE.

Implements the structured context format that prevents collapse
and maintains detailed knowledge over iterations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import yaml


@dataclass
class Example:
    """Example of strategy application."""
    description: str
    input: str
    output: str
    success: bool


@dataclass
class Strategy:
    """A learned strategy in the playbook."""
    id: str
    category: str
    description: str
    when_to_use: str
    examples: List[Example] = field(default_factory=list)
    success_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "examples": [
                {
                    "description": ex.description,
                    "input": ex.input,
                    "output": ex.output,
                    "success": ex.success
                }
                for ex in self.examples
            ],
            "success_rate": self.success_rate,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Strategy":
        examples = [
            Example(**ex) for ex in data.get("examples", [])
        ]
        return cls(
            id=data["id"],
            category=data["category"],
            description=data["description"],
            when_to_use=data["when_to_use"],
            examples=examples,
            success_rate=data.get("success_rate", 0.0),
            metadata=data.get("metadata", {})
        )


@dataclass
class Pitfall:
    """A common mistake or failure pattern."""
    id: str
    description: str
    how_to_avoid: str
    related_strategies: List[str] = field(default_factory=list)
    frequency: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "how_to_avoid": self.how_to_avoid,
            "related_strategies": self.related_strategies,
            "frequency": self.frequency,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pitfall":
        return cls(
            id=data["id"],
            description=data["description"],
            how_to_avoid=data["how_to_avoid"],
            related_strategies=data.get("related_strategies", []),
            frequency=data.get("frequency", 0.0),
            metadata=data.get("metadata", {})
        )


@dataclass
class Context:
    """
    Structured context representing an evolving playbook.

    Maintains strategies, pitfalls, and history of updates.
    Designed to prevent context collapse through structured format.
    """
    version: int
    domain: str
    base_prompt: str
    strategies: List[Strategy] = field(default_factory=list)
    pitfalls: List[Pitfall] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: Optional[datetime] = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

    def to_prompt(self) -> str:
        """Convert context to a prompt string for LLM use."""
        sections = [self.base_prompt, ""]

        # Add strategies by category
        if self.strategies:
            sections.append("# Strategies")
            strategies_by_category = {}
            for strategy in self.strategies:
                if strategy.category not in strategies_by_category:
                    strategies_by_category[strategy.category] = []
                strategies_by_category[strategy.category].append(strategy)

            for category, strategies in strategies_by_category.items():
                sections.append(f"\n## {category}")
                for strategy in strategies:
                    sections.append(f"\n### {strategy.description}")
                    sections.append(f"**When to use**: {strategy.when_to_use}")
                    if strategy.examples:
                        sections.append("**Examples**:")
                        for i, ex in enumerate(strategy.examples[:2], 1):  # Limit to 2 examples
                            sections.append(f"{i}. {ex.description}")
                    sections.append("")

        # Add pitfalls
        if self.pitfalls:
            sections.append("\n# Common Pitfalls to Avoid")
            for pitfall in self.pitfalls:
                sections.append(f"\n- **{pitfall.description}**")
                sections.append(f"  How to avoid: {pitfall.how_to_avoid}")

        return "\n".join(sections)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "domain": self.domain,
            "base_prompt": self.base_prompt,
            "strategies": [s.to_dict() for s in self.strategies],
            "pitfalls": [p.to_dict() for p in self.pitfalls],
            "history": self.history,
            "metadata": self.metadata,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }

    def to_yaml(self) -> str:
        """Export to YAML format."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def to_json(self) -> str:
        """Export to JSON format."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Context":
        """Load from dictionary."""
        strategies = [Strategy.from_dict(s) for s in data.get("strategies", [])]
        pitfalls = [Pitfall.from_dict(p) for p in data.get("pitfalls", [])]
        last_updated = None
        if data.get("last_updated"):
            last_updated = datetime.fromisoformat(data["last_updated"])

        return cls(
            version=data["version"],
            domain=data["domain"],
            base_prompt=data["base_prompt"],
            strategies=strategies,
            pitfalls=pitfalls,
            history=data.get("history", []),
            metadata=data.get("metadata", {}),
            last_updated=last_updated
        )

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "Context":
        """Load from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, json_str: str) -> "Context":
        """Load from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """Get strategy by ID."""
        for strategy in self.strategies:
            if strategy.id == strategy_id:
                return strategy
        return None

    def get_pitfall(self, pitfall_id: str) -> Optional[Pitfall]:
        """Get pitfall by ID."""
        for pitfall in self.pitfalls:
            if pitfall.id == pitfall_id:
                return pitfall
        return None

    def get_strategies_by_category(self, category: str) -> List[Strategy]:
        """Get all strategies in a category."""
        return [s for s in self.strategies if s.category == category]


@dataclass
class Insight:
    """An insight extracted from reflection."""
    type: str  # "strategy" or "pitfall"
    description: str
    category: Optional[str] = None
    when_to_use: Optional[str] = None
    how_to_avoid: Optional[str] = None
    examples: List[Example] = field(default_factory=list)
    confidence: float = 1.0
    source_trajectory_ids: List[str] = field(default_factory=list)


@dataclass
class ContextDelta:
    """
    A delta representing changes to apply to a context.

    Deltas are synthesized by the Curator and merged deterministically.
    """
    delta_id: str
    timestamp: datetime
    source_trajectory_ids: List[str]
    insights: List[Insight]

    # Changes to apply
    add_strategies: List[Strategy] = field(default_factory=list)
    add_pitfalls: List[Pitfall] = field(default_factory=list)
    modify_strategies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    remove_strategies: List[str] = field(default_factory=list)
    remove_pitfalls: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta_id": self.delta_id,
            "timestamp": self.timestamp.isoformat(),
            "source_trajectory_ids": self.source_trajectory_ids,
            "insights": [
                {
                    "type": i.type,
                    "description": i.description,
                    "category": i.category,
                    "when_to_use": i.when_to_use,
                    "how_to_avoid": i.how_to_avoid,
                    "confidence": i.confidence
                }
                for i in self.insights
            ],
            "add_strategies": [s.to_dict() for s in self.add_strategies],
            "add_pitfalls": [p.to_dict() for p in self.add_pitfalls],
            "modify_strategies": self.modify_strategies,
            "remove_strategies": self.remove_strategies,
            "remove_pitfalls": self.remove_pitfalls,
            "metadata": self.metadata
        }


@dataclass
class Trajectory:
    """
    A reasoning trajectory from task execution.

    Contains the task, reasoning steps, result, and feedback.
    """
    trajectory_id: str
    task: str
    context_version: int
    steps: List[str]
    result: str
    feedback: Optional[str] = None
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "task": self.task,
            "context_version": self.context_version,
            "steps": self.steps,
            "result": self.result,
            "feedback": self.feedback,
            "success": self.success,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
