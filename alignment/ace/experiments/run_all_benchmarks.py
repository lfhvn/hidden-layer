"""
Run ACE on all benchmarks and generate comparison report.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from alignment.ace.experiments.offline_ace import run_offline_experiment


def run_all_benchmarks(
    num_iterations: int = 3,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-20241022",
    save_dir: str = "results/all_benchmarks"
):
    """
    Run ACE on all available benchmarks.

    Args:
        num_iterations: Number of optimization iterations
        provider: LLM provider
        model: Model name
        save_dir: Directory to save results
    """
    benchmarks = ["agent", "tools", "reasoning", "math", "finance", "code"]

    print("=" * 80)
    print("Running ACE on All Benchmarks")
    print("=" * 80)
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"Iterations: {num_iterations}")
    print(f"Model: {provider}/{model}")
    print("=" * 80)

    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(save_dir) / timestamp

    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\n[{i}/{len(benchmarks)}] Running {benchmark} benchmark...")

        try:
            result = run_offline_experiment(
                benchmark_name=benchmark,
                num_iterations=num_iterations,
                provider=provider,
                model=model,
                save_dir=str(run_dir)
            )

            results[benchmark] = result

        except Exception as e:
            print(f"Error running {benchmark}: {e}")
            results[benchmark] = {"error": str(e)}

        print(f"\n{benchmark} complete!")
        print("-" * 80)

    # Generate summary report
    print("\n" + "=" * 80)
    print("Summary Report")
    print("=" * 80)

    print("\n{:<15} {:<12} {:<12} {:<15} {:<15}".format(
        "Benchmark", "Base Test", "Final Test", "Improvement", "Strategies"
    ))
    print("-" * 80)

    for benchmark, result in results.items():
        if "error" in result:
            print(f"{benchmark:<15} ERROR: {result['error']}")
        else:
            print("{:<15} {:<12.1%} {:<12.1%} {:<15.1%} {:<15}".format(
                benchmark,
                result.get('baseline_test', 0),
                result.get('final_test', 0),
                result.get('test_improvement', 0),
                result.get('num_strategies', 0)
            ))

    # Save summary
    summary_path = run_dir / "summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    # Calculate average improvement
    improvements = [
        r.get('test_improvement', 0)
        for r in results.values()
        if 'error' not in r
    ]

    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        print(f"\nAverage improvement: {avg_improvement:+.1%}")

    print("\n" + "=" * 80)
    print("All Benchmarks Complete!")
    print("=" * 80)

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run all ACE benchmarks")
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of optimization iterations"
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
        default="results/all_benchmarks",
        help="Directory to save results"
    )

    args = parser.parse_args()

    run_all_benchmarks(
        num_iterations=args.iterations,
        provider=args.provider,
        model=args.model,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
