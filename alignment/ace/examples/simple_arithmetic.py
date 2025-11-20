"""
Simple arithmetic example for ACE.

Tests the ACE framework on basic arithmetic word problems.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from alignment.ace.src import ACEFramework, Context


def create_initial_context() -> Context:
    """Create initial context for arithmetic tasks."""
    return Context(
        version=0,
        domain="arithmetic",
        base_prompt=(
            "You are an expert at solving arithmetic word problems. "
            "Break down problems step-by-step and show your work clearly."
        )
    )


def evaluate_arithmetic(task: str, result: str) -> dict:
    """
    Simple evaluator for arithmetic tasks.

    In a real implementation, this would parse the result and check correctness.
    For now, we'll use simple heuristics.
    """
    # Extract expected answer from task (assumes format "... The answer is X.")
    if "answer is" in task.lower():
        expected = task.lower().split("answer is")[-1].strip().rstrip(".")
    else:
        # Can't evaluate without expected answer
        return {"success": False, "feedback": "No expected answer provided"}

    # Check if result contains the expected answer
    result_lower = result.lower()
    success = expected in result_lower

    feedback = "Correct!" if success else f"Incorrect. Expected: {expected}"

    return {
        "success": success,
        "feedback": feedback
    }


def generate_arithmetic_tasks():
    """Generate simple arithmetic tasks."""
    tasks = [
        # Addition
        "Sarah has 5 apples. John gives her 3 more apples. How many apples does Sarah have now? The answer is 8.",
        "A store sold 12 books on Monday and 8 books on Tuesday. How many books did they sell in total? The answer is 20.",
        "There are 15 students in class A and 17 students in class B. How many students are there in total? The answer is 32.",

        # Subtraction
        "Mike had 20 dollars. He spent 7 dollars on lunch. How much money does he have left? The answer is 13.",
        "A library has 50 books. 12 books were checked out. How many books remain? The answer is 38.",

        # Multiplication
        "Each box contains 6 oranges. If there are 4 boxes, how many oranges are there in total? The answer is 24.",
        "A classroom has 5 rows of desks with 6 desks in each row. How many desks are there? The answer is 30.",

        # Division
        "24 cookies are shared equally among 6 children. How many cookies does each child get? The answer is 4.",
        "A rope 36 meters long is cut into 9 equal pieces. How long is each piece? The answer is 4.",

        # Mixed operations
        "Tom has 10 marbles. He buys 5 more and then gives 3 to his friend. How many marbles does Tom have now? The answer is 12.",
    ]
    return tasks


def main():
    """Run simple arithmetic example."""
    print("=" * 70)
    print("ACE Framework - Simple Arithmetic Example")
    print("=" * 70)

    # Create initial context
    print("\nCreating initial context...")
    context = create_initial_context()
    print(f"Initial context (v{context.version}): {context.domain}")

    # Generate tasks
    tasks = generate_arithmetic_tasks()
    print(f"\nGenerated {len(tasks)} arithmetic tasks")

    # Split into train and test
    train_tasks = tasks[:7]
    test_tasks = tasks[7:]

    # Initialize ACE
    print("\nInitializing ACE framework...")
    ace = ACEFramework(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022"
    )

    # Run offline ACE
    print("\n" + "=" * 70)
    print("Running Offline ACE Optimization")
    print("=" * 70)

    optimized_context = ace.run_offline(
        tasks=train_tasks,
        initial_context=context,
        num_iterations=3,
        evaluator=evaluate_arithmetic,
        save_path=Path("results/simple_arithmetic")
    )

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluation on Test Set")
    print("=" * 70)

    print("\nBaseline (initial context):")
    baseline_results = ace.evaluate(test_tasks, context, evaluate_arithmetic)
    print(f"  Success rate: {baseline_results['success_rate']:.1%}")

    print("\nOptimized context:")
    optimized_results = ace.evaluate(test_tasks, optimized_context, evaluate_arithmetic)
    print(f"  Success rate: {optimized_results['success_rate']:.1%}")

    improvement = optimized_results['success_rate'] - baseline_results['success_rate']
    print(f"\nImprovement: {improvement:+.1%}")

    # Show learned strategies
    print("\n" + "=" * 70)
    print("Learned Strategies")
    print("=" * 70)

    if optimized_context.strategies:
        for strategy in optimized_context.strategies[:3]:  # Show top 3
            print(f"\n- {strategy.description}")
            print(f"  Category: {strategy.category}")
            print(f"  When to use: {strategy.when_to_use}")
    else:
        print("No strategies learned yet.")

    # Show learned pitfalls
    if optimized_context.pitfalls:
        print("\n" + "=" * 70)
        print("Common Pitfalls")
        print("=" * 70)

        for pitfall in optimized_context.pitfalls[:3]:  # Show top 3
            print(f"\n- {pitfall.description}")
            print(f"  How to avoid: {pitfall.how_to_avoid}")

    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
