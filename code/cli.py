#!/usr/bin/env python3
"""
Simple CLI for testing the harness.

Usage:
    python cli.py --help
    python cli.py "What is 2+2?" --strategy single --provider ollama
    python cli.py "Should we invest in AI?" --strategy debate --n-debaters 3
"""
import argparse
import json
import sys
from pathlib import Path

# Add harness to path
sys.path.insert(0, str(Path(__file__).parent))

from harness import run_strategy, STRATEGIES


def main():
    parser = argparse.ArgumentParser(
        description="Test the agentic simulation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model (local)
  python cli.py "What is the capital of France?" --strategy single --provider ollama
  
  # Debate strategy
  python cli.py "Should we use nuclear energy?" --strategy debate --n-debaters 3
  
  # Self-consistency
  python cli.py "What is 17 * 24?" --strategy self_consistency --n-samples 5
  
  # Manager-worker
  python cli.py "Plan a week-long vacation to Japan" --strategy manager_worker --n-workers 3
  
  # Use API for comparison
  python cli.py "Explain quantum computing" --provider anthropic --model claude-3-5-haiku-20241022
        """
    )
    
    parser.add_argument(
        "input",
        help="Task input / question"
    )
    
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        default="single",
        help="Strategy to use"
    )
    
    parser.add_argument(
        "--provider",
        choices=["ollama", "mlx", "anthropic", "openai"],
        default="ollama",
        help="LLM provider"
    )
    
    parser.add_argument(
        "--model",
        default="llama3.2:latest",
        help="Model identifier"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )

    # Reasoning parameters
    parser.add_argument(
        "--thinking-budget",
        type=int,
        help="Thinking budget for reasoning models (number of tokens)"
    )

    parser.add_argument(
        "--num-ctx",
        type=int,
        help="Context window size"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        help="Top-K sampling parameter"
    )

    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-P (nucleus) sampling parameter"
    )

    # Strategy-specific args
    parser.add_argument(
        "--n-debaters",
        type=int,
        default=2,
        help="Number of debaters (for debate strategy)"
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of samples (for self_consistency strategy)"
    )

    parser.add_argument(
        "--n-workers",
        type=int,
        default=3,
        help="Number of workers (for manager_worker strategy)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Stream thinking/debating in real-time (default: True)"
    )

    parser.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="Disable streaming, return complete response only"
    )
    
    args = parser.parse_args()

    # Build kwargs based on strategy
    kwargs = {
        "provider": args.provider,
        "model": args.model,
        "temperature": args.temperature,
        "verbose": args.verbose if not args.json else False  # Disable verbose in JSON mode
    }

    # Add reasoning parameters if provided
    if args.thinking_budget:
        kwargs["thinking_budget"] = args.thinking_budget
    if args.num_ctx:
        kwargs["num_ctx"] = args.num_ctx
    if args.top_k:
        kwargs["top_k"] = args.top_k
    if args.top_p:
        kwargs["top_p"] = args.top_p

    # Strategy-specific parameters
    if args.strategy == "debate":
        kwargs["n_debaters"] = args.n_debaters
    elif args.strategy == "self_consistency":
        kwargs["n_samples"] = args.n_samples
    elif args.strategy == "manager_worker":
        kwargs["n_workers"] = args.n_workers
    
    # Run strategy
    if not args.json and not args.verbose:
        print(f"Running {args.strategy} strategy...")
        print(f"Provider: {args.provider}, Model: {args.model}")
        print(f"Input: {args.input}\n")
        print("-" * 80)

    try:
        result = run_strategy(args.strategy, args.input, **kwargs)

        if args.json:
            # JSON output
            output = {
                "strategy": args.strategy,
                "output": result.output,
                "latency_s": result.latency_s,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "cost_usd": result.cost_usd,
                "metadata": result.metadata
            }
            print(json.dumps(output, indent=2))
        else:
            # Human-readable output
            if not args.verbose:
                # Only print output if not already streamed
                print(f"\nOutput:\n{result.output}\n")

            print("=" * 60)
            print(f"✅ Complete!")
            print("=" * 60)
            print(f"Latency: {result.latency_s:.2f}s")
            print(f"Tokens: {result.tokens_in} in, {result.tokens_out} out")
            if result.cost_usd > 0:
                print(f"Cost: ${result.cost_usd:.4f}")

            if result.metadata and (args.verbose or 'thinking_budget' in result.metadata):
                print("\nMetadata:")
                if 'thinking_budget' in result.metadata:
                    print(f"  Thinking Budget: {result.metadata['thinking_budget']}")
                if args.verbose:
                    print(json.dumps(result.metadata, indent=2))
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
