"""
Evaluation Functions for Design Critiques

This module provides evaluation methods for assessing the quality
of design critiques and recommendations.
"""

import os
import sys
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from crit.problems import DesignProblem
from crit.strategies import CritiqueResult
from harness import llm_call


def evaluate_critique_coverage(problem: DesignProblem, critique_result: CritiqueResult) -> Dict[str, Any]:
    """
    Evaluate how well the critique covers important aspects.

    Checks coverage of:
    - Known issues
    - Success criteria
    - Multiple perspectives

    Args:
        problem: The design problem
        critique_result: The critique to evaluate

    Returns:
        Dictionary with coverage scores
    """
    # Combine all critique text
    all_critique_text = ""
    for critique in critique_result.critiques:
        if "critique" in critique:
            all_critique_text += critique["critique"] + "\n"
        elif "feedback" in critique:
            all_critique_text += critique["feedback"] + "\n"
        elif "content" in critique:
            all_critique_text += critique["content"] + "\n"

    if critique_result.synthesis:
        all_critique_text += critique_result.synthesis

    all_critique_lower = all_critique_text.lower()

    # Check coverage of known issues
    known_issues_mentioned = 0
    if problem.known_issues:
        for issue in problem.known_issues:
            # Simple keyword matching (could be improved with semantic similarity)
            issue_keywords = [w.lower() for w in issue.split() if len(w) > 4]
            if any(keyword in all_critique_lower for keyword in issue_keywords):
                known_issues_mentioned += 1

        known_issues_coverage = known_issues_mentioned / len(problem.known_issues)
    else:
        known_issues_coverage = None

    # Check coverage of success criteria
    criteria_mentioned = 0
    for criterion in problem.success_criteria:
        criterion_keywords = [w.lower() for w in criterion.split() if len(w) > 4]
        if any(keyword in all_critique_lower for keyword in criterion_keywords):
            criteria_mentioned += 1

    criteria_coverage = criteria_mentioned / len(problem.success_criteria)

    # Count unique perspectives
    perspectives_used = set()
    for critique in critique_result.critiques:
        if "perspective" in critique:
            perspectives_used.add(critique["perspective"])

    return {
        "known_issues_coverage": known_issues_coverage,
        "known_issues_mentioned": known_issues_mentioned,
        "known_issues_total": len(problem.known_issues) if problem.known_issues else 0,
        "criteria_coverage": criteria_coverage,
        "criteria_mentioned": criteria_mentioned,
        "criteria_total": len(problem.success_criteria),
        "perspectives_count": len(perspectives_used),
        "perspectives_used": list(perspectives_used),
        "overall_coverage": criteria_coverage,  # Primary metric
    }


def evaluate_recommendation_quality(
    problem: DesignProblem,
    critique_result: CritiqueResult,
    judge_provider: str = "ollama",
    judge_model: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Use LLM judge to evaluate recommendation quality.

    Evaluates on:
    - Specificity (concrete vs vague)
    - Actionability (can be implemented)
    - Relevance (addresses the problem)
    - Feasibility (practical to implement)

    Args:
        problem: The design problem
        critique_result: The critique with recommendations
        judge_provider: Provider for judge model
        judge_model: Model for judging
        **kwargs: Additional llm_call arguments

    Returns:
        Dictionary with quality scores
    """
    if not critique_result.recommendations:
        return {
            "error": "No recommendations to evaluate",
            "specificity": 0.0,
            "actionability": 0.0,
            "relevance": 0.0,
            "feasibility": 0.0,
            "overall_quality": 0.0,
        }

    judge_prompt = f"""Evaluate these design recommendations for quality.

DESIGN PROBLEM:
{problem.description}

CONTEXT:
{problem.context}

CURRENT DESIGN:
{problem.current_design}

RECOMMENDATIONS:
"""

    for i, rec in enumerate(critique_result.recommendations, 1):
        judge_prompt += f"\n{i}. {rec}"

    judge_prompt += """

Evaluate each recommendation on these dimensions (0-10 scale):

1. SPECIFICITY: How concrete and specific is it? (10 = very specific, 0 = vague)
2. ACTIONABILITY: Can it be clearly implemented? (10 = clear action, 0 = unclear)
3. RELEVANCE: Does it address the problem? (10 = highly relevant, 0 = off-topic)
4. FEASIBILITY: Is it practical to implement? (10 = very feasible, 0 = impractical)

Provide your evaluation in this format:
Recommendation 1:
- Specificity: X/10
- Actionability: X/10
- Relevance: X/10
- Feasibility: X/10

Recommendation 2:
...

Overall Assessment: [summary]
"""

    judge_response = llm_call(judge_prompt, provider=judge_provider, model=judge_model, **kwargs)

    # Parse scores
    import re

    specificity_scores = []
    actionability_scores = []
    relevance_scores = []
    feasibility_scores = []

    specificity_pattern = r"Specificity:\s*(\d+(?:\.\d+)?)/10"
    actionability_pattern = r"Actionability:\s*(\d+(?:\.\d+)?)/10"
    relevance_pattern = r"Relevance:\s*(\d+(?:\.\d+)?)/10"
    feasibility_pattern = r"Feasibility:\s*(\d+(?:\.\d+)?)/10"

    specificity_scores = [float(s) for s in re.findall(specificity_pattern, judge_response.text)]
    actionability_scores = [float(s) for s in re.findall(actionability_pattern, judge_response.text)]
    relevance_scores = [float(s) for s in re.findall(relevance_pattern, judge_response.text)]
    feasibility_scores = [float(s) for s in re.findall(feasibility_pattern, judge_response.text)]

    def avg(scores):
        return sum(scores) / len(scores) if scores else 0.0

    specificity_avg = avg(specificity_scores)
    actionability_avg = avg(actionability_scores)
    relevance_avg = avg(relevance_scores)
    feasibility_avg = avg(feasibility_scores)

    overall_quality = (specificity_avg + actionability_avg + relevance_avg + feasibility_avg) / 4.0

    # Extract overall assessment
    overall_match = re.search(r"Overall Assessment:\s*(.+)", judge_response.text, re.DOTALL)
    overall_assessment = overall_match.group(1).strip() if overall_match else "No assessment"

    return {
        "specificity": specificity_avg / 10.0,  # Normalize to 0-1
        "actionability": actionability_avg / 10.0,
        "relevance": relevance_avg / 10.0,
        "feasibility": feasibility_avg / 10.0,
        "overall_quality": overall_quality / 10.0,
        "overall_assessment": overall_assessment,
        "recommendations_count": len(critique_result.recommendations),
        "judge_response": judge_response.text,
        "judge_tokens_in": judge_response.tokens_in,
        "judge_tokens_out": judge_response.tokens_out,
        "judge_cost_usd": judge_response.cost_usd,
    }


def evaluate_critique_depth(critique_result: CritiqueResult) -> Dict[str, Any]:
    """
    Evaluate the depth and thoroughness of critique.

    Metrics:
    - Length of critiques
    - Number of recommendations
    - Number of perspectives
    - Presence of synthesis

    Args:
        critique_result: The critique to evaluate

    Returns:
        Dictionary with depth metrics
    """
    total_critique_length = 0
    for critique in critique_result.critiques:
        if "critique" in critique:
            total_critique_length += len(critique["critique"])
        elif "feedback" in critique:
            total_critique_length += len(critique["feedback"])
        elif "content" in critique:
            total_critique_length += len(critique["content"])

    avg_critique_length = total_critique_length / len(critique_result.critiques) if critique_result.critiques else 0

    # Simple depth score based on heuristics
    # More critiques, longer text, more recommendations = deeper analysis
    depth_score = 0.0

    # Critique count (0-0.3 points)
    critique_count = len(critique_result.critiques)
    depth_score += min(0.3, critique_count * 0.1)

    # Average length (0-0.3 points, normalize to typical range)
    depth_score += min(0.3, avg_critique_length / 3000)

    # Recommendations count (0-0.2 points)
    rec_count = len(critique_result.recommendations)
    depth_score += min(0.2, rec_count * 0.04)

    # Has synthesis (0-0.2 points)
    if critique_result.synthesis:
        depth_score += 0.2

    return {
        "depth_score": min(1.0, depth_score),
        "total_critique_length": total_critique_length,
        "avg_critique_length": avg_critique_length,
        "critique_count": critique_count,
        "recommendations_count": rec_count,
        "has_synthesis": critique_result.synthesis is not None,
    }


def evaluate_critique(
    problem: DesignProblem,
    critique_result: CritiqueResult,
    method: str = "combined",
    judge_provider: str = "ollama",
    judge_model: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a design critique.

    Args:
        problem: The design problem
        critique_result: The critique to evaluate
        method: "coverage", "quality", "depth", or "combined"
        judge_provider: Provider for LLM judge (if using quality eval)
        judge_model: Model for LLM judge
        **kwargs: Additional arguments for LLM judge

    Returns:
        Evaluation results dictionary
    """
    results = {
        "problem_name": problem.name,
        "strategy_name": critique_result.strategy_name,
        "method": method,
    }

    if method in ["coverage", "combined"]:
        coverage = evaluate_critique_coverage(problem, critique_result)
        results["coverage"] = coverage

    if method in ["quality", "combined"]:
        quality = evaluate_recommendation_quality(
            problem, critique_result, judge_provider=judge_provider, judge_model=judge_model, **kwargs
        )
        results["quality"] = quality

    if method in ["depth", "combined"]:
        depth = evaluate_critique_depth(critique_result)
        results["depth"] = depth

    # Combined score
    if method == "combined":
        combined_score = (
            coverage.get("overall_coverage", 0) * 0.3
            + quality.get("overall_quality", 0) * 0.5
            + depth.get("depth_score", 0) * 0.2
        )
        results["combined_score"] = combined_score

    return results


def compare_strategies(
    problem: DesignProblem,
    results: Dict[str, CritiqueResult],
    judge_provider: str = "ollama",
    judge_model: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compare multiple critique strategies on the same problem.

    Args:
        problem: The design problem
        results: Dictionary mapping strategy names to CritiqueResults
        judge_provider: Provider for LLM judge
        judge_model: Model for LLM judge
        **kwargs: Additional arguments for LLM judge

    Returns:
        Comparison dictionary
    """
    evaluations = {}

    for strategy_name, critique_result in results.items():
        eval_result = evaluate_critique(
            problem,
            critique_result,
            method="combined",
            judge_provider=judge_provider,
            judge_model=judge_model,
            **kwargs,
        )
        evaluations[strategy_name] = eval_result

    # Create comparison summary
    comparison = {
        "problem_name": problem.name,
        "strategies": list(results.keys()),
        "evaluations": evaluations,
        "rankings": {},
    }

    # Rank by combined score
    if all("combined_score" in ev for ev in evaluations.values()):
        ranked = sorted(evaluations.items(), key=lambda x: x[1]["combined_score"], reverse=True)
        comparison["rankings"]["combined"] = [{"strategy": name, "score": ev["combined_score"]} for name, ev in ranked]

    # Rank by coverage
    if all("coverage" in ev for ev in evaluations.values()):
        ranked = sorted(evaluations.items(), key=lambda x: x[1]["coverage"]["overall_coverage"], reverse=True)
        comparison["rankings"]["coverage"] = [
            {"strategy": name, "score": ev["coverage"]["overall_coverage"]} for name, ev in ranked
        ]

    # Rank by quality
    if all("quality" in ev for ev in evaluations.values()):
        ranked = sorted(evaluations.items(), key=lambda x: x[1]["quality"]["overall_quality"], reverse=True)
        comparison["rankings"]["quality"] = [
            {"strategy": name, "score": ev["quality"]["overall_quality"]} for name, ev in ranked
        ]

    # Cost and latency comparison
    comparison["performance"] = {
        strategy_name: {
            "latency_s": result.latency_s,
            "total_cost_usd": result.total_cost_usd,
            "total_tokens": result.total_tokens_in + result.total_tokens_out,
        }
        for strategy_name, result in results.items()
    }

    return comparison


def batch_evaluate(
    results: List[Dict[str, Any]], judge_provider: str = "ollama", judge_model: Optional[str] = None, **kwargs
) -> Dict[str, Any]:
    """
    Evaluate multiple critique results.

    Args:
        results: List of dicts with 'problem' and 'critique_result'
        judge_provider: Provider for LLM judge
        judge_model: Model for LLM judge
        **kwargs: Additional arguments

    Returns:
        Aggregated evaluation results
    """
    evaluations = []

    for item in results:
        problem = item["problem"]
        critique_result = item["critique_result"]

        eval_result = evaluate_critique(
            problem,
            critique_result,
            method="combined",
            judge_provider=judge_provider,
            judge_model=judge_model,
            **kwargs,
        )

        evaluations.append(eval_result)

    # Aggregate
    combined_scores = [ev["combined_score"] for ev in evaluations if "combined_score" in ev]

    coverage_scores = [ev["coverage"]["overall_coverage"] for ev in evaluations if "coverage" in ev]

    quality_scores = [ev["quality"]["overall_quality"] for ev in evaluations if "quality" in ev]

    depth_scores = [ev["depth"]["depth_score"] for ev in evaluations if "depth" in ev]

    return {
        "evaluations": evaluations,
        "aggregates": {
            "combined_avg": sum(combined_scores) / len(combined_scores) if combined_scores else 0,
            "coverage_avg": sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0,
            "quality_avg": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "depth_avg": sum(depth_scores) / len(depth_scores) if depth_scores else 0,
        },
        "count": len(evaluations),
    }
