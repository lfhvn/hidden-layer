"""
Benchmark Dataset Loaders for CRIT

This module provides loaders for design critique benchmark datasets:
- UICrit: 11,344 design critiques for 1,000 mobile UIs from Google Research

Usage:
    >>> from crit.benchmarks import load_uicrit
    >>> problems = load_uicrit()
    >>> # Use with existing CRIT evaluation pipeline
"""

import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from crit.problems import DesignDomain, DesignProblem


@dataclass
class BenchmarkDataset:
    """A benchmark dataset with metadata"""

    name: str
    source: str
    problems: List[DesignProblem]
    critiques: List[Dict[str, Any]]  # For datasets that include critiques
    total_count: int
    metadata: Dict[str, Any]


def load_uicrit(
    data_path: Optional[str] = None, min_quality_rating: Optional[float] = None, include_llm_critiques: bool = True
) -> BenchmarkDataset:
    """
    Load UICrit dataset.

    UICrit (UIST 2024) contains 11,344 design critiques for 1,000 mobile UIs
    collected from seven experienced designers plus LLM-generated critiques.

    Installation:
        git clone https://github.com/google-research-datasets/uicrit.git
        # Data will be in ./uicrit/uicrit_public.csv

    Or download directly:
        wget https://raw.githubusercontent.com/google-research-datasets/uicrit/main/uicrit_public.csv

    Args:
        data_path: Path to uicrit_public.csv (default: ./uicrit/uicrit_public.csv)
        min_quality_rating: Optional filter for minimum design quality (1-10)
        include_llm_critiques: Include LLM-generated critiques (default: True)

    Returns:
        BenchmarkDataset with DesignProblem objects and critique data

    Note:
        Licensed under Creative Commons Attribution 4.0 International License
    """
    if data_path is None:
        data_path = "./uicrit/uicrit_public.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"UICrit data not found at {data_path}. "
            "Please clone the repository:\n"
            "git clone https://github.com/google-research-datasets/uicrit.git\n"
            "Or download the CSV:\n"
            "wget https://raw.githubusercontent.com/google-research-datasets/uicrit/main/uicrit_public.csv"
        )

    # Load CSV data
    critiques = []
    ui_screens = {}  # Group by rico_id

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rico_id = row["rico_id"]

            # Filter by quality if requested
            if min_quality_rating:
                quality = float(row.get("design_quality_rating", 0))
                if quality < min_quality_rating:
                    continue

            # Parse comments
            comments_source = _parse_list_field(row.get("comments_source", "[]"))
            comments = _parse_list_field(row.get("comments", "[]"))

            # Filter LLM critiques if requested
            if not include_llm_critiques:
                comments = [comment for comment, source in zip(comments, comments_source) if source == "human"]
                comments_source = [s for s in comments_source if s == "human"]

            # Store critique data
            critique_data = {
                "rico_id": rico_id,
                "task": row.get("task", ""),
                "aesthetics_rating": float(row.get("aesthetics_rating", 0)),
                "learnability": float(row.get("learnability", 0)),
                "efficiency": float(row.get("efficency", 0)),  # Note: typo in original
                "usability_rating": float(row.get("usability_rating", 0)),
                "design_quality_rating": float(row.get("design_quality_rating", 0)),
                "comments": comments,
                "comments_source": comments_source,
            }

            critiques.append(critique_data)

            # Group by UI screen
            if rico_id not in ui_screens:
                ui_screens[rico_id] = {
                    "task": row.get("task", ""),
                    "critiques": [],
                    "avg_quality": 0,
                    "avg_aesthetics": 0,
                    "avg_usability": 0,
                }

            ui_screens[rico_id]["critiques"].append(critique_data)

    # Calculate averages per screen
    for rico_id, screen_data in ui_screens.items():
        screen_critiques = screen_data["critiques"]
        if screen_critiques:
            screen_data["avg_quality"] = sum(c["design_quality_rating"] for c in screen_critiques) / len(
                screen_critiques
            )
            screen_data["avg_aesthetics"] = sum(c["aesthetics_rating"] for c in screen_critiques) / len(
                screen_critiques
            )
            screen_data["avg_usability"] = sum(c["usability_rating"] for c in screen_critiques) / len(screen_critiques)

    # Convert to DesignProblem objects
    problems = []
    for rico_id, screen_data in ui_screens.items():
        problem = _uicrit_to_design_problem(rico_id, screen_data)
        problems.append(problem)

    return BenchmarkDataset(
        name="UICrit",
        source="https://github.com/google-research-datasets/uicrit",
        problems=problems,
        critiques=critiques,
        total_count=len(critiques),
        metadata={
            "citation": "Duan et al., UIST 2024",
            "ui_screens": len(problems),
            "total_critiques": len(critiques),
            "min_quality_rating": min_quality_rating,
            "include_llm_critiques": include_llm_critiques,
            "license": "CC BY 4.0",
        },
    )


def load_uicrit_for_comparison(
    data_path: Optional[str] = None, sample_size: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load UICrit in format suitable for comparing with your critique strategies.

    Returns a list of items, each containing:
    - problem: DesignProblem object
    - expert_critiques: List of human critiques
    - llm_critiques: List of LLM critiques (if available)
    - quality_ratings: Quality metrics

    This allows you to:
    1. Run your strategies on the same UIs
    2. Compare your critiques to expert critiques
    3. Evaluate against quality ratings

    Args:
        data_path: Path to uicrit_public.csv
        sample_size: Optional limit on number of UIs to load

    Returns:
        List of comparison items
    """
    benchmark = load_uicrit(data_path, include_llm_critiques=True)

    comparison_items = []
    count = 0

    for problem in benchmark.problems:
        if sample_size and count >= sample_size:
            break

        rico_id = problem.metadata["rico_id"]

        # Get all critiques for this UI
        ui_critiques = [c for c in benchmark.critiques if c["rico_id"] == rico_id]

        # Separate human and LLM critiques
        expert_critiques = []
        llm_critiques = []

        for critique in ui_critiques:
            for comment, source in zip(critique["comments"], critique["comments_source"]):
                if source == "human":
                    expert_critiques.append(comment)
                elif source == "llm":
                    llm_critiques.append(comment)

        # Get quality ratings
        quality_ratings = {
            "design_quality": problem.metadata["avg_quality"],
            "aesthetics": problem.metadata["avg_aesthetics"],
            "usability": problem.metadata["avg_usability"],
        }

        comparison_items.append(
            {
                "rico_id": rico_id,
                "problem": problem,
                "expert_critiques": expert_critiques,
                "llm_critiques": llm_critiques,
                "quality_ratings": quality_ratings,
                "task": problem.description,
            }
        )

        count += 1

    return comparison_items


# ============================================================================
# Conversion Helpers
# ============================================================================


def _uicrit_to_design_problem(rico_id: str, screen_data: Dict[str, Any]) -> DesignProblem:
    """Convert UICrit screen data to DesignProblem"""

    task = screen_data["task"]
    avg_quality = screen_data["avg_quality"]
    critiques = screen_data["critiques"]

    # Aggregate all comments as "known issues"
    all_comments = []
    for critique in critiques:
        all_comments.extend(
            [
                c
                for c, s in zip(critique["comments"], critique["comments_source"])
                if s == "human"  # Only use human critiques as known issues
            ]
        )

    # Create a generic UI design problem
    return DesignProblem(
        name=f"uicrit_{rico_id}",
        domain=DesignDomain.UI_UX,
        description=f"Mobile UI for {task}",
        current_design=f"[UI Screenshot - RICO ID: {rico_id}]\n"
        f"Task: {task}\n"
        f"Note: This is a mobile UI from the RICO dataset. "
        f"You can view the screenshot at rico_id {rico_id}.",
        context=f"Mobile application UI for {task}.\n"
        f"Current design quality rating: {avg_quality:.1f}/10\n"
        f"Evaluated by {len(critiques)} reviewers.",
        success_criteria=[
            "High visual appeal (aesthetics)",
            "Easy to learn and use (learnability)",
            "Efficient task completion",
            "Good overall usability",
        ],
        known_issues=all_comments[:5],  # Limit to top 5 issues
        difficulty="medium",
        metadata={
            "source": "UICrit",
            "rico_id": rico_id,
            "task": task,
            "avg_quality": avg_quality,
            "avg_aesthetics": screen_data["avg_aesthetics"],
            "avg_usability": screen_data["avg_usability"],
            "num_critiques": len(critiques),
            "all_human_critiques": all_comments,
        },
    )


def _parse_list_field(field_str: str) -> List[Any]:
    """Parse string representation of list from CSV"""
    try:
        return json.loads(field_str.replace("'", '"'))
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fallback: try to split by common delimiters
        if field_str.startswith("[") and field_str.endswith("]"):
            field_str = field_str[1:-1]
        return [item.strip().strip('"').strip("'") for item in field_str.split(",") if item.strip()]


# ============================================================================
# Utility Functions
# ============================================================================


def list_available_benchmarks() -> Dict[str, Dict[str, Any]]:
    """List all available benchmarks with information"""
    return {
        "UICrit": {
            "size": 11344,
            "ui_screens": 1000,
            "source": "https://github.com/google-research-datasets/uicrit",
            "citation": "Duan et al., UIST 2024",
            "installation": "git clone https://github.com/google-research-datasets/uicrit.git",
            "usage": "load_uicrit()",
            "license": "CC BY 4.0",
            "note": "Includes both human and LLM-generated critiques",
        },
    }


def print_benchmark_info():
    """Print information about all available benchmarks"""
    benchmarks = list_available_benchmarks()

    print("Available Design Critique Benchmarks for CRIT:\n")
    print("=" * 70)

    for name, info in benchmarks.items():
        print(f"\n{name}")
        print("-" * 70)
        for key, value in info.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)


def compare_to_experts(
    your_critiques: List[str], expert_critiques: List[str], method: str = "overlap"
) -> Dict[str, Any]:
    """
    Compare your critiques to expert critiques.

    Args:
        your_critiques: Your generated critiques
        expert_critiques: Expert human critiques
        method: Comparison method ("overlap", "coverage")

    Returns:
        Comparison metrics
    """
    if method == "overlap":
        # Simple keyword overlap
        your_keywords = set()
        for critique in your_critiques:
            words = critique.lower().split()
            your_keywords.update([w for w in words if len(w) > 4])

        expert_keywords = set()
        for critique in expert_critiques:
            words = critique.lower().split()
            expert_keywords.update([w for w in words if len(w) > 4])

        overlap = your_keywords & expert_keywords
        overlap_score = len(overlap) / len(expert_keywords) if expert_keywords else 0

        return {
            "method": "overlap",
            "overlap_score": overlap_score,
            "your_unique_keywords": len(your_keywords - expert_keywords),
            "expert_unique_keywords": len(expert_keywords - your_keywords),
            "shared_keywords": len(overlap),
        }

    elif method == "coverage":
        # Check if your critiques cover expert critiques
        covered = 0
        for expert_critique in expert_critiques:
            expert_lower = expert_critique.lower()
            # Check if any of your critiques has keyword overlap
            for your_critique in your_critiques:
                your_lower = your_critique.lower()
                expert_words = [w for w in expert_lower.split() if len(w) > 4]
                matches = sum(1 for w in expert_words if w in your_lower)
                if matches >= len(expert_words) * 0.3:  # 30% keyword match
                    covered += 1
                    break

        coverage_score = covered / len(expert_critiques) if expert_critiques else 0

        return {
            "method": "coverage",
            "coverage_score": coverage_score,
            "expert_critiques_covered": covered,
            "expert_critiques_total": len(expert_critiques),
        }

    else:
        raise ValueError(f"Unknown method: {method}")
