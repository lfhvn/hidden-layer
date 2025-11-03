"""
Unified Benchmark Interface for Hidden Layer Harness

This module provides a unified interface to access benchmarks from all subsystems:
- CRIT: Design critique benchmarks
- SELPHI: Theory of Mind benchmarks
- Harness: General reasoning benchmarks (future)

Usage:
    >>> from harness import load_benchmark, BENCHMARKS
    >>> print(BENCHMARKS)  # List all available benchmarks
    >>> dataset = load_benchmark('uicrit')  # Load a benchmark
    >>> baseline_scores = get_baseline_scores('uicrit')  # Get baseline performance
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import sys
import os

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@dataclass
class BenchmarkInfo:
    """Information about a benchmark"""
    name: str
    subsystem: str  # 'crit', 'selphi', 'harness'
    size: int
    source: str
    description: str
    loader_function: str
    metadata: Dict[str, Any]


# Registry of all available benchmarks across subsystems
BENCHMARKS: Dict[str, BenchmarkInfo] = {
    # CRIT Benchmarks
    "uicrit": BenchmarkInfo(
        name="UICrit",
        subsystem="crit",
        size=11344,
        source="https://github.com/google-research-datasets/uicrit",
        description="11,344 design critiques for 1,000 mobile UIs from Google Research",
        loader_function="crit.benchmarks.load_uicrit",
        metadata={
            "citation": "Duan et al., UIST 2024",
            "license": "CC BY 4.0",
            "ui_screens": 1000,
            "includes_human_critiques": True,
            "includes_llm_critiques": True,
        }
    ),

    # SELPHI Benchmarks
    "tombench": BenchmarkInfo(
        name="ToMBench",
        subsystem="selphi",
        size=388,
        source="https://github.com/wadimiusz/ToMBench",
        description="388 Theory of Mind test cases across multiple levels",
        loader_function="selphi.benchmarks.load_tombench",
        metadata={
            "citation": "Nemirovsky et al., 2023",
            "license": "MIT",
            "levels": ["first_order", "second_order", "third_order"],
        }
    ),

    "opentom": BenchmarkInfo(
        name="OpenToM",
        subsystem="selphi",
        size=696,
        source="https://github.com/seacowx/OpenToM",
        description="696 Theory of Mind questions from OpenToM dataset",
        loader_function="selphi.benchmarks.load_opentom",
        metadata={
            "citation": "Ma et al., 2023",
            "question_types": ["location", "multihop", "attitude"],
        }
    ),

    "socialiqa": BenchmarkInfo(
        name="SocialIQA",
        subsystem="selphi",
        size=38000,
        source="https://leaderboard.allenai.org/socialiqa",
        description="38k commonsense reasoning questions about social situations",
        loader_function="selphi.benchmarks.load_socialiqa",
        metadata={
            "citation": "Sap et al., EMNLP 2019",
            "license": "CC BY 4.0",
        }
    ),
}


def load_benchmark(
    benchmark_name: str,
    **kwargs
) -> Any:
    """
    Load a benchmark dataset from any subsystem.

    Args:
        benchmark_name: Name of benchmark (see BENCHMARKS for options)
        **kwargs: Additional arguments passed to the specific loader

    Returns:
        BenchmarkDataset object from the appropriate subsystem

    Examples:
        >>> # Load CRIT benchmark
        >>> uicrit = load_benchmark('uicrit')
        >>>
        >>> # Load SELPHI benchmark
        >>> tombench = load_benchmark('tombench', split='test')
        >>>
        >>> # Load with custom parameters
        >>> uicrit_filtered = load_benchmark('uicrit', min_quality_rating=7.0)
    """
    if benchmark_name not in BENCHMARKS:
        available = ', '.join(BENCHMARKS.keys())
        raise ValueError(
            f"Unknown benchmark '{benchmark_name}'. "
            f"Available benchmarks: {available}"
        )

    info = BENCHMARKS[benchmark_name]
    subsystem = info.subsystem

    # Import and call the appropriate loader
    if subsystem == "crit":
        from crit.benchmarks import (
            load_uicrit,
        )

        if benchmark_name == "uicrit":
            return load_uicrit(**kwargs)
        else:
            raise NotImplementedError(f"CRIT benchmark '{benchmark_name}' loader not yet implemented")

    elif subsystem == "selphi":
        from selphi.benchmarks import (
            load_tombench,
            load_opentom,
            load_socialiqa,
        )

        if benchmark_name == "tombench":
            return load_tombench(**kwargs)
        elif benchmark_name == "opentom":
            return load_opentom(**kwargs)
        elif benchmark_name == "socialiqa":
            return load_socialiqa(**kwargs)
        else:
            raise NotImplementedError(f"SELPHI benchmark '{benchmark_name}' loader not yet implemented")

    elif subsystem == "harness":
        # Future: Add general reasoning benchmarks (MMLU, GSM8K, etc.)
        raise NotImplementedError(f"Harness benchmark '{benchmark_name}' loader not yet implemented")

    else:
        raise ValueError(f"Unknown subsystem '{subsystem}'")


def get_baseline_scores(
    benchmark_name: str,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get baseline performance scores for a benchmark.

    Args:
        benchmark_name: Name of benchmark
        model: Optional specific model name to get scores for

    Returns:
        Dictionary with baseline scores and metadata

    Examples:
        >>> scores = get_baseline_scores('tombench')
        >>> print(scores['human_performance'])
        >>> print(scores['gpt4_performance'])
    """
    if benchmark_name not in BENCHMARKS:
        available = ', '.join(BENCHMARKS.keys())
        raise ValueError(
            f"Unknown benchmark '{benchmark_name}'. "
            f"Available benchmarks: {available}"
        )

    info = BENCHMARKS[benchmark_name]

    # Baseline scores from literature and our experiments
    # These are approximate values from published papers
    baseline_scores = {
        "uicrit": {
            "human_agreement": 0.72,  # Inter-rater agreement
            "gpt4_coverage": 0.65,    # Coverage of human critiques
            "description": "Scores represent agreement/coverage with expert critiques",
            "metric": "coverage_score",
        },

        "tombench": {
            "human_performance": 0.95,
            "gpt4_performance": 0.76,
            "claude_3_5_sonnet_performance": 0.82,
            "llama_70b_performance": 0.65,
            "random_baseline": 0.33,  # 3-way multiple choice
            "description": "Accuracy on Theory of Mind reasoning tasks",
            "metric": "accuracy",
        },

        "opentom": {
            "human_performance": 0.92,
            "gpt4_performance": 0.71,
            "claude_3_5_sonnet_performance": 0.75,
            "random_baseline": 0.25,  # 4-way multiple choice
            "description": "Accuracy on multi-character ToM scenarios",
            "metric": "accuracy",
        },

        "socialiqa": {
            "human_performance": 0.88,
            "gpt4_performance": 0.83,
            "claude_3_5_sonnet_performance": 0.81,
            "random_baseline": 0.33,  # 3-way multiple choice
            "description": "Accuracy on social commonsense reasoning",
            "metric": "accuracy",
        },
    }

    if benchmark_name not in baseline_scores:
        return {
            "note": f"No baseline scores available for {benchmark_name}",
            "benchmark_info": info.metadata,
        }

    scores = baseline_scores[benchmark_name].copy()

    # If specific model requested, filter to that model
    if model:
        model_key = f"{model}_performance"
        if model_key in scores:
            return {
                "benchmark": benchmark_name,
                "model": model,
                "score": scores[model_key],
                "metric": scores["metric"],
                "description": scores["description"],
            }
        else:
            available_models = [k.replace('_performance', '') for k in scores.keys() if k.endswith('_performance')]
            return {
                "error": f"No baseline for model '{model}'",
                "available_models": available_models,
                "all_scores": scores,
            }

    return scores


def list_benchmarks(subsystem: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all available benchmarks, optionally filtered by subsystem.

    Args:
        subsystem: Optional filter ('crit', 'selphi', 'harness')

    Returns:
        List of benchmark information dictionaries
    """
    benchmarks = []

    for name, info in BENCHMARKS.items():
        if subsystem is None or info.subsystem == subsystem:
            benchmarks.append({
                "name": name,
                "full_name": info.name,
                "subsystem": info.subsystem,
                "size": info.size,
                "description": info.description,
                "source": info.source,
                "metadata": info.metadata,
            })

    return benchmarks


def print_benchmarks():
    """Print information about all available benchmarks"""
    print("\n" + "=" * 80)
    print("AVAILABLE BENCHMARKS IN HIDDEN LAYER")
    print("=" * 80 + "\n")

    # Group by subsystem
    by_subsystem = {}
    for name, info in BENCHMARKS.items():
        subsystem = info.subsystem.upper()
        if subsystem not in by_subsystem:
            by_subsystem[subsystem] = []
        by_subsystem[subsystem].append((name, info))

    for subsystem in sorted(by_subsystem.keys()):
        print(f"\n{subsystem} BENCHMARKS")
        print("-" * 80)

        for name, info in by_subsystem[subsystem]:
            print(f"\n  {info.name} ('{name}')")
            print(f"  Size: {info.size:,} examples")
            print(f"  {info.description}")
            print(f"  Source: {info.source}")

            if info.metadata:
                print(f"  Additional info:")
                for key, value in info.metadata.items():
                    print(f"    - {key}: {value}")

    print("\n" + "=" * 80)
    print(f"\nTotal: {len(BENCHMARKS)} benchmarks available")
    print("\nUsage:")
    print("  from harness import load_benchmark, get_baseline_scores")
    print("  dataset = load_benchmark('tombench')")
    print("  scores = get_baseline_scores('tombench')")
    print("=" * 80 + "\n")


# Convenience function for quick access
def get_benchmark_info(benchmark_name: str) -> BenchmarkInfo:
    """Get information about a specific benchmark"""
    if benchmark_name not in BENCHMARKS:
        available = ', '.join(BENCHMARKS.keys())
        raise ValueError(
            f"Unknown benchmark '{benchmark_name}'. "
            f"Available benchmarks: {available}"
        )
    return BENCHMARKS[benchmark_name]
