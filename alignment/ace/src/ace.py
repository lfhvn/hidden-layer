"""
Main ACE Framework orchestration.

Coordinates Generator, Reflector, and Curator for context optimization.
"""

from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import json

from .context import Context, Trajectory, ContextDelta
from .generator import Generator
from .reflector import Reflector
from .curator import Curator


class ACEFramework:
    """
    Main ACE (Agentic Context Engineering) framework.

    Orchestrates the three-role architecture:
    - Generator: Produces reasoning trajectories
    - Reflector: Extracts insights from trajectories
    - Curator: Integrates insights into structured contexts
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-3-5-sonnet-20241022",
        generator_config: Optional[Dict[str, Any]] = None,
        reflector_config: Optional[Dict[str, Any]] = None,
        curator_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ACE framework.

        Args:
            provider: LLM provider (via harness)
            model: Model name
            generator_config: Configuration for Generator
            reflector_config: Configuration for Reflector
            curator_config: Configuration for Curator
        """
        self.provider = provider
        self.model = model

        # Initialize components
        gen_cfg = generator_config or {}
        self.generator = Generator(
            provider=provider,
            model=model,
            **gen_cfg
        )

        ref_cfg = reflector_config or {}
        self.reflector = Reflector(
            provider=provider,
            model=model,
            **ref_cfg
        )

        cur_cfg = curator_config or {}
        self.curator = Curator(**cur_cfg)

        # Tracking
        self.iteration_history = []

    def run_offline(
        self,
        tasks: List[str],
        initial_context: Context,
        num_iterations: int = 5,
        evaluator: Optional[Callable] = None,
        save_path: Optional[Path] = None
    ) -> Context:
        """
        Run offline ACE optimization.

        Iteratively improves context by:
        1. Generating trajectories on tasks
        2. Reflecting on trajectories
        3. Curating insights into context updates

        Args:
            tasks: Training tasks
            initial_context: Initial context to optimize
            num_iterations: Number of optimization iterations
            evaluator: Optional function to evaluate task results
            save_path: Optional path to save intermediate results

        Returns:
            Optimized context
        """
        context = initial_context

        for iteration in range(num_iterations):
            print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

            # 1. Generate trajectories
            print(f"Generating trajectories for {len(tasks)} tasks...")
            trajectories = []
            for task in tasks:
                if evaluator:
                    trajectory = self.generator.execute_with_feedback(
                        task, context, evaluator
                    )
                else:
                    trajectory = self.generator.generate_trajectory(task, context)
                trajectories.append(trajectory)

            # Calculate success rate
            success_rate = sum(1 for t in trajectories if t.success) / len(trajectories)
            print(f"Success rate: {success_rate:.1%}")

            # 2. Reflect on trajectories
            print("Reflecting on trajectories...")
            insights = self.reflector.reflect(trajectories)
            print(f"Extracted {len(insights)} insights")

            # 3. Curate delta and merge
            print("Curating context update...")
            delta = self.curator.synthesize_delta(insights, context)
            context = self.curator.merge_delta(context, delta)

            # Organize strategies
            context = self.curator.organize_strategies(context)

            print(f"Context updated to v{context.version}")
            print(f"  Strategies: {len(context.strategies)}")
            print(f"  Pitfalls: {len(context.pitfalls)}")

            # Track iteration
            self.iteration_history.append({
                "iteration": iteration + 1,
                "success_rate": success_rate,
                "num_insights": len(insights),
                "num_strategies": len(context.strategies),
                "num_pitfalls": len(context.pitfalls),
                "context_version": context.version
            })

            # Save intermediate results
            if save_path:
                self._save_iteration(
                    save_path, iteration, context, trajectories, insights, delta
                )

        return context

    def run_online(
        self,
        task_stream: List[str],
        initial_context: Context,
        buffer_size: int = 10,
        update_frequency: int = 5,
        evaluator: Optional[Callable] = None,
        save_path: Optional[Path] = None
    ) -> Context:
        """
        Run online ACE optimization.

        Continuously adapts context during task execution by:
        1. Buffering recent trajectories
        2. Periodically reflecting and updating context

        Args:
            task_stream: Stream of tasks to execute
            initial_context: Initial context
            buffer_size: Number of trajectories to buffer
            update_frequency: Update context every N tasks
            evaluator: Optional function to evaluate task results
            save_path: Optional path to save updates

        Returns:
            Final adapted context
        """
        context = initial_context
        trajectory_buffer = []
        update_count = 0

        for i, task in enumerate(task_stream, 1):
            print(f"\n=== Task {i}/{len(task_stream)} ===")

            # Generate trajectory with current context
            if evaluator:
                trajectory = self.generator.execute_with_feedback(
                    task, context, evaluator
                )
            else:
                trajectory = self.generator.generate_trajectory(task, context)

            trajectory_buffer.append(trajectory)
            print(f"Success: {trajectory.success}")

            # Check if it's time to update context
            if len(trajectory_buffer) >= buffer_size or i % update_frequency == 0:
                print(f"\nUpdating context (buffer size: {len(trajectory_buffer)})...")

                # Reflect on buffered trajectories
                insights = self.reflector.reflect(trajectory_buffer)
                print(f"Extracted {len(insights)} insights")

                # Curate and merge delta
                delta = self.curator.synthesize_delta(insights, context)
                context = self.curator.merge_delta(context, delta)
                context = self.curator.organize_strategies(context)

                print(f"Context updated to v{context.version}")

                # Track update
                update_count += 1
                success_rate = sum(1 for t in trajectory_buffer if t.success) / len(trajectory_buffer)
                self.iteration_history.append({
                    "update": update_count,
                    "task_index": i,
                    "success_rate": success_rate,
                    "buffer_size": len(trajectory_buffer),
                    "num_insights": len(insights),
                    "context_version": context.version
                })

                # Save update
                if save_path:
                    self._save_online_update(
                        save_path, update_count, context, trajectory_buffer, insights, delta
                    )

                # Clear buffer
                trajectory_buffer = []

        return context

    def evaluate(
        self,
        tasks: List[str],
        context: Context,
        evaluator: Callable
    ) -> Dict[str, Any]:
        """
        Evaluate a context on a set of tasks.

        Args:
            tasks: Evaluation tasks
            context: Context to evaluate
            evaluator: Function to evaluate task results

        Returns:
            Evaluation metrics
        """
        trajectories = []
        for task in tasks:
            trajectory = self.generator.execute_with_feedback(
                task, context, evaluator
            )
            trajectories.append(trajectory)

        # Calculate metrics
        success_rate = sum(1 for t in trajectories if t.success) / len(trajectories)

        # Average tokens per task (rough estimate)
        avg_steps = sum(len(t.steps) for t in trajectories) / len(trajectories)

        return {
            "success_rate": success_rate,
            "num_tasks": len(tasks),
            "num_successes": sum(1 for t in trajectories if t.success),
            "avg_steps": avg_steps,
            "trajectories": trajectories
        }

    def compare_contexts(
        self,
        contexts: Dict[str, Context],
        tasks: List[str],
        evaluator: Callable
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple contexts on the same tasks.

        Args:
            contexts: Dictionary of {name: context}
            tasks: Evaluation tasks
            evaluator: Function to evaluate task results

        Returns:
            Dictionary of {name: metrics}
        """
        results = {}

        for name, context in contexts.items():
            print(f"\nEvaluating {name}...")
            metrics = self.evaluate(tasks, context, evaluator)
            results[name] = metrics
            print(f"  Success rate: {metrics['success_rate']:.1%}")

        return results

    def _save_iteration(
        self,
        save_path: Path,
        iteration: int,
        context: Context,
        trajectories: List[Trajectory],
        insights: List,
        delta: ContextDelta
    ):
        """Save intermediate results from an iteration."""
        save_path.mkdir(parents=True, exist_ok=True)

        # Save context
        context_file = save_path / f"context_iter_{iteration}.json"
        with open(context_file, "w") as f:
            f.write(context.to_json())

        # Save trajectories
        trajectories_file = save_path / f"trajectories_iter_{iteration}.json"
        with open(trajectories_file, "w") as f:
            json.dump(
                [t.to_dict() for t in trajectories],
                f,
                indent=2
            )

        # Save delta
        delta_file = save_path / f"delta_iter_{iteration}.json"
        with open(delta_file, "w") as f:
            json.dump(delta.to_dict(), f, indent=2)

    def _save_online_update(
        self,
        save_path: Path,
        update_num: int,
        context: Context,
        trajectories: List[Trajectory],
        insights: List,
        delta: ContextDelta
    ):
        """Save results from an online update."""
        save_path.mkdir(parents=True, exist_ok=True)

        # Save context
        context_file = save_path / f"context_update_{update_num}.json"
        with open(context_file, "w") as f:
            f.write(context.to_json())

        # Save delta
        delta_file = save_path / f"delta_update_{update_num}.json"
        with open(delta_file, "w") as f:
            json.dump(delta.to_dict(), f, indent=2)

    def get_history(self) -> List[Dict[str, Any]]:
        """Get iteration/update history."""
        return self.iteration_history

    def save_history(self, path: Path):
        """Save iteration/update history to file."""
        with open(path, "w") as f:
            json.dump(self.iteration_history, f, indent=2)
