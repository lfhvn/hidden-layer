"""
Online ACE optimization experiments.

Tests ACE with online (continuous) adaptation during task execution.
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from alignment.ace.src import ACEFramework, Context
from alignment.ace.experiments.benchmarks import (
    SimpleAgentBenchmark,
    MathBenchmark,
    FinanceBenchmark,
    CodeBenchmark
)


def create_initial_context(domain: str) -> Context:
    """Create initial context for a domain."""
    base_prompts = {
        "agent": "You are an intelligent task planning assistant.",
        "math": "You are an expert at solving mathematical problems.",
        "finance": "You are a financial analyst.",
        "code": "You are a software engineering expert."
    }

    return Context(
        version=0,
        domain=domain,
        base_prompt=base_prompts.get(domain, "You are a helpful assistant.")
    )


def run_online_experiment(
    benchmark_name: str,
    buffer_size: int = 10,
    update_frequency: int = 5,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-20241022",
    save_dir: str = "results/online"
):
    """
    Run online ACE experiment.

    Args:
        benchmark_name: Name of benchmark to use
        buffer_size: Size of trajectory buffer
        update_frequency: Update context every N tasks
        provider: LLM provider
        model: Model name
        save_dir: Directory to save results
    """
    print("=" * 80)
    print(f"Online ACE Experiment: {benchmark_name}")
    print("=" * 80)

    # Load benchmark
    benchmarks = {
        "agent": SimpleAgentBenchmark(),
        "math": MathBenchmark(),
        "finance": FinanceBenchmark(),
        "code": CodeBenchmark()
    }

    if benchmark_name not in benchmarks:
        print(f"Error: Unknown benchmark '{benchmark_name}'")
        return

    benchmark = benchmarks[benchmark_name]
    print(f"\nLoaded benchmark: {benchmark.name}")

    # Get all tasks (simulating a task stream)
    all_tasks = benchmark.get_tasks(split="all")
    print(f"Total tasks in stream: {len(all_tasks)}")

    # Create initial context
    initial_context = create_initial_context(benchmark_name)
    print(f"\nInitial context (v{initial_context.version}): {initial_context.domain}")

    # Initialize ACE
    print(f"\nInitializing ACE with {provider}/{model}...")
    ace = ACEFramework(
        provider=provider,
        model=model
    )

    # Create evaluator
    evaluator = benchmark.create_evaluator()

    # Run online ACE
    print("\n" + "=" * 80)
    print("Running Online ACE (Continuous Adaptation)")
    print("=" * 80)
    print(f"Buffer size: {buffer_size}")
    print(f"Update frequency: every {update_frequency} tasks")

    task_descriptions = [t.description for t in all_tasks]
    save_path = Path(save_dir) / benchmark_name

    final_context = ace.run_online(
        task_stream=task_descriptions,
        initial_context=initial_context,
        buffer_size=buffer_size,
        update_frequency=update_frequency,
        evaluator=evaluator,
        save_path=save_path
    )

    # Analyze performance over time
    print("\n" + "=" * 80)
    print("Performance Analysis")
    print("=" * 80)

    history = ace.get_history()

    if history:
        print("\nContext updates:")
        for i, update in enumerate(history, 1):
            print(f"\nUpdate {i} (task {update.get('task_index', i*update_frequency)}):")
            print(f"  Success rate: {update.get('success_rate', 0):.1%}")
            print(f"  Insights extracted: {update.get('num_insights', 0)}")
            print(f"  Context version: {update.get('context_version', 0)}")

        # Calculate improvement
        if len(history) > 1:
            first_success = history[0].get('success_rate', 0)
            last_success = history[-1].get('success_rate', 0)
            improvement = last_success - first_success

            print(f"\nOverall improvement: {improvement:+.1%}")
            print(f"Initial success rate: {first_success:.1%}")
            print(f"Final success rate: {last_success:.1%}")

    # Show final context
    print("\n" + "=" * 80)
    print("Final Context")
    print("=" * 80)

    print(f"Version: {final_context.version}")
    print(f"Strategies: {len(final_context.strategies)}")
    print(f"Pitfalls: {len(final_context.pitfalls)}")

    if final_context.strategies:
        print("\nTop strategies:")
        for i, strategy in enumerate(final_context.strategies[:3], 1):
            print(f"{i}. {strategy.description}")

    # Save results
    final_context_path = save_path / "final_context.yaml"
    save_path.mkdir(parents=True, exist_ok=True)
    with open(final_context_path, "w") as f:
        f.write(final_context.to_yaml())
    print(f"\nFinal context saved to: {final_context_path}")

    history_path = save_path / "history.json"
    ace.save_history(history_path)
    print(f"History saved to: {history_path}")

    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run online ACE experiments")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="math",
        choices=["agent", "math", "finance", "code"],
        help="Benchmark to use"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=10,
        help="Trajectory buffer size"
    )
    parser.add_argument(
        "--update-frequency",
        type=int,
        default=5,
        help="Update context every N tasks"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="anthropic",
        help="LLM provider"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Model name"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/online",
        help="Directory to save results"
    )

    args = parser.parse_args()

    run_online_experiment(
        benchmark_name=args.benchmark,
        buffer_size=args.buffer_size,
        update_frequency=args.update_frequency,
        provider=args.provider,
        model=args.model,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
