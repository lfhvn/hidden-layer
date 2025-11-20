"""
Offline ACE optimization experiments.

Tests ACE on various benchmarks with offline (pre-deployment) optimization.
"""

import sys
from pathlib import Path
import argparse
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from alignment.ace.src import ACEFramework, Context
from alignment.ace.experiments.benchmarks import (
    SimpleAgentBenchmark,
    ToolUseBenchmark,
    ReasoningBenchmark,
    MathBenchmark,
    FinanceBenchmark,
    CodeBenchmark
)


def create_initial_context(domain: str) -> Context:
    """
    Create initial context for a domain.

    Args:
        domain: Domain name (e.g., "agent", "math", "finance")

    Returns:
        Initial Context
    """
    base_prompts = {
        "agent": (
            "You are an intelligent assistant that helps with task planning "
            "and execution. Break down complex tasks into steps and reason "
            "through each decision carefully."
        ),
        "math": (
            "You are an expert at solving mathematical problems. "
            "Show your work step-by-step and clearly explain your reasoning."
        ),
        "finance": (
            "You are a financial analyst. Provide accurate calculations "
            "and clear explanations of financial concepts."
        ),
        "code": (
            "You are a software engineering expert. Analyze code carefully, "
            "identify issues, and provide clear solutions."
        ),
        "reasoning": (
            "You are an expert at logical reasoning. Think through problems "
            "systematically and justify your conclusions."
        ),
        "tools": (
            "You are an assistant that can use various tools. Choose the right "
            "tool for each task and use it correctly."
        )
    }

    return Context(
        version=0,
        domain=domain,
        base_prompt=base_prompts.get(domain, "You are a helpful assistant.")
    )


def run_offline_experiment(
    benchmark_name: str,
    num_iterations: int = 3,
    tasks_per_iteration: int = None,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-20241022",
    save_dir: str = "results/offline"
):
    """
    Run offline ACE experiment on a benchmark.

    Args:
        benchmark_name: Name of benchmark to use
        num_iterations: Number of optimization iterations
        tasks_per_iteration: Tasks per iteration (None = all training tasks)
        provider: LLM provider
        model: Model name
        save_dir: Directory to save results
    """
    print("=" * 80)
    print(f"Offline ACE Experiment: {benchmark_name}")
    print("=" * 80)

    # Load benchmark
    benchmarks = {
        "agent": SimpleAgentBenchmark(),
        "tools": ToolUseBenchmark(),
        "reasoning": ReasoningBenchmark(),
        "math": MathBenchmark(),
        "finance": FinanceBenchmark(),
        "code": CodeBenchmark()
    }

    if benchmark_name not in benchmarks:
        print(f"Error: Unknown benchmark '{benchmark_name}'")
        print(f"Available: {list(benchmarks.keys())}")
        return

    benchmark = benchmarks[benchmark_name]
    print(f"\nLoaded benchmark: {benchmark.name}")

    # Get training and test tasks
    train_tasks = benchmark.get_tasks(split="train")
    test_tasks = benchmark.get_tasks(split="test")

    if not train_tasks:
        print("Warning: No training tasks found. Using test tasks for training.")
        train_tasks = test_tasks[:len(test_tasks)//2]
        test_tasks = test_tasks[len(test_tasks)//2:]

    if tasks_per_iteration:
        train_tasks = train_tasks[:tasks_per_iteration]

    print(f"Training tasks: {len(train_tasks)}")
    print(f"Test tasks: {len(test_tasks)}")

    # Create initial context
    initial_context = create_initial_context(benchmark_name)
    print(f"\nInitial context (v{initial_context.version}): {initial_context.domain}")

    # Initialize ACE
    print(f"\nInitializing ACE with {provider}/{model}...")
    ace = ACEFramework(
        provider=provider,
        model=model
    )

    # Create evaluator from benchmark
    evaluator = benchmark.create_evaluator()

    # Evaluate baseline (initial context)
    print("\n" + "=" * 80)
    print("Baseline Evaluation (Initial Context)")
    print("=" * 80)

    train_task_descriptions = [t.description for t in train_tasks]
    baseline_train = ace.evaluate(train_task_descriptions, initial_context, evaluator)
    print(f"Training accuracy: {baseline_train['success_rate']:.1%}")

    test_task_descriptions = [t.description for t in test_tasks]
    baseline_test = ace.evaluate(test_task_descriptions, initial_context, evaluator)
    print(f"Test accuracy: {baseline_test['success_rate']:.1%}")

    # Run offline ACE
    print("\n" + "=" * 80)
    print("Running Offline ACE Optimization")
    print("=" * 80)

    save_path = Path(save_dir) / benchmark_name
    optimized_context = ace.run_offline(
        tasks=train_task_descriptions,
        initial_context=initial_context,
        num_iterations=num_iterations,
        evaluator=evaluator,
        save_path=save_path
    )

    # Evaluate optimized context
    print("\n" + "=" * 80)
    print("Final Evaluation (Optimized Context)")
    print("=" * 80)

    final_train = ace.evaluate(train_task_descriptions, optimized_context, evaluator)
    print(f"Training accuracy: {final_train['success_rate']:.1%}")

    final_test = ace.evaluate(test_task_descriptions, optimized_context, evaluator)
    print(f"Test accuracy: {final_test['success_rate']:.1%}")

    # Show improvements
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    train_improvement = final_train['success_rate'] - baseline_train['success_rate']
    test_improvement = final_test['success_rate'] - baseline_test['success_rate']

    print(f"\nTraining improvement: {train_improvement:+.1%}")
    print(f"Test improvement: {test_improvement:+.1%}")

    # Show learned strategies
    if optimized_context.strategies:
        print("\n" + "=" * 80)
        print("Learned Strategies")
        print("=" * 80)

        for i, strategy in enumerate(optimized_context.strategies[:5], 1):
            print(f"\n{i}. {strategy.description}")
            print(f"   Category: {strategy.category}")
            print(f"   When to use: {strategy.when_to_use}")
            print(f"   Success rate: {strategy.success_rate:.1%}")

    # Show learned pitfalls
    if optimized_context.pitfalls:
        print("\n" + "=" * 80)
        print("Common Pitfalls")
        print("=" * 80)

        for i, pitfall in enumerate(optimized_context.pitfalls[:5], 1):
            print(f"\n{i}. {pitfall.description}")
            print(f"   How to avoid: {pitfall.how_to_avoid}")

    # Save final context
    final_context_path = save_path / "final_context.yaml"
    save_path.mkdir(parents=True, exist_ok=True)
    with open(final_context_path, "w") as f:
        f.write(optimized_context.to_yaml())
    print(f"\nFinal context saved to: {final_context_path}")

    # Save history
    history_path = save_path / "history.json"
    ace.save_history(history_path)
    print(f"History saved to: {history_path}")

    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)

    return {
        "benchmark": benchmark_name,
        "baseline_train": baseline_train['success_rate'],
        "baseline_test": baseline_test['success_rate'],
        "final_train": final_train['success_rate'],
        "final_test": final_test['success_rate'],
        "train_improvement": train_improvement,
        "test_improvement": test_improvement,
        "num_strategies": len(optimized_context.strategies),
        "num_pitfalls": len(optimized_context.pitfalls)
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run offline ACE experiments")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="math",
        choices=["agent", "tools", "reasoning", "math", "finance", "code"],
        help="Benchmark to use"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of optimization iterations"
    )
    parser.add_argument(
        "--tasks-per-iteration",
        type=int,
        default=None,
        help="Number of tasks per iteration (None = all)"
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
        default="results/offline",
        help="Directory to save results"
    )

    args = parser.parse_args()

    run_offline_experiment(
        benchmark_name=args.benchmark,
        num_iterations=args.iterations,
        tasks_per_iteration=args.tasks_per_iteration,
        provider=args.provider,
        model=args.model,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
