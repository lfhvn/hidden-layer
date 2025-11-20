"""
Base classes for benchmarks.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod


@dataclass
class Task:
    """A single task in a benchmark."""
    id: str
    description: str
    expected_output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class BenchmarkResult:
    """Results from evaluating on a benchmark."""
    benchmark_name: str
    num_tasks: int
    num_correct: int
    accuracy: float
    details: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Benchmark: {self.benchmark_name}\n"
            f"Tasks: {self.num_tasks}\n"
            f"Correct: {self.num_correct}\n"
            f"Accuracy: {self.accuracy:.1%}"
        )


class Benchmark(ABC):
    """
    Base class for benchmarks.

    A benchmark provides:
    - A set of tasks to evaluate on
    - An evaluation function to check correctness
    - Metadata about the benchmark
    """

    def __init__(self, name: str):
        """
        Initialize the benchmark.

        Args:
            name: Name of the benchmark
        """
        self.name = name
        self.tasks: List[Task] = []
        self._loaded = False

    @abstractmethod
    def load_tasks(self) -> List[Task]:
        """
        Load tasks for this benchmark.

        Returns:
            List of tasks
        """
        pass

    @abstractmethod
    def evaluate(self, task: Task, output: str) -> Dict[str, Any]:
        """
        Evaluate an output for a task.

        Args:
            task: The task
            output: The model output

        Returns:
            Dictionary with 'correct' (bool) and 'feedback' (str)
        """
        pass

    def get_tasks(self, split: str = "test", num_tasks: Optional[int] = None) -> List[Task]:
        """
        Get tasks from the benchmark.

        Args:
            split: Data split ("train", "test", "all")
            num_tasks: Number of tasks to return (None = all)

        Returns:
            List of tasks
        """
        if not self._loaded:
            self.tasks = self.load_tasks()
            self._loaded = True

        # Filter by split if needed
        tasks = self.tasks
        if split != "all":
            tasks = [t for t in tasks if t.metadata.get("split", "test") == split]

        # Limit number of tasks
        if num_tasks is not None:
            tasks = tasks[:num_tasks]

        return tasks

    def evaluate_batch(
        self,
        tasks: List[Task],
        outputs: List[str]
    ) -> BenchmarkResult:
        """
        Evaluate a batch of outputs.

        Args:
            tasks: List of tasks
            outputs: List of outputs (same order as tasks)

        Returns:
            BenchmarkResult with aggregated metrics
        """
        assert len(tasks) == len(outputs), "Tasks and outputs must have same length"

        results = []
        num_correct = 0

        for task, output in zip(tasks, outputs):
            eval_result = self.evaluate(task, output)
            correct = eval_result.get("correct", False)

            if correct:
                num_correct += 1

            results.append({
                "task_id": task.id,
                "correct": correct,
                "output": output,
                "feedback": eval_result.get("feedback", ""),
                "expected": task.expected_output
            })

        accuracy = num_correct / len(tasks) if tasks else 0.0

        return BenchmarkResult(
            benchmark_name=self.name,
            num_tasks=len(tasks),
            num_correct=num_correct,
            accuracy=accuracy,
            details=results
        )

    def create_evaluator(self) -> Callable:
        """
        Create an evaluator function compatible with ACE.

        Returns:
            Function that takes (task_str, result_str) and returns evaluation dict
        """
        # Build a mapping from task description to Task object
        task_map = {}
        for task in self.get_tasks():
            task_map[task.description] = task

        def evaluator(task_str: str, result_str: str) -> Dict[str, Any]:
            """Evaluator function for ACE."""
            task = task_map.get(task_str)
            if task is None:
                return {
                    "success": False,
                    "feedback": "Task not found in benchmark"
                }

            eval_result = self.evaluate(task, result_str)
            return {
                "success": eval_result.get("correct", False),
                "feedback": eval_result.get("feedback", "")
            }

        return evaluator

    def get_metadata(self) -> Dict[str, Any]:
        """Get benchmark metadata."""
        return {
            "name": self.name,
            "num_tasks": len(self.tasks),
            "splits": list(set(t.metadata.get("split", "test") for t in self.tasks)),
            "difficulties": list(set(t.difficulty for t in self.tasks))
        }
