"""
Evaluation Functions for Theory of Mind Tests

This module provides specialized evaluation functions for assessing
theory of mind and epistemology understanding in language models.
"""

import os
import re
import sys
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from harness import llm_call
from selphi.scenarios import ToMScenario


def parse_multi_answer_response(response: str, num_questions: int) -> List[str]:
    """
    Parse a response that contains answers to multiple questions.

    Tries multiple parsing strategies:
    1. Numbered list format (1. answer, 2. answer, ...)
    2. Line-by-line (one answer per line)
    3. Question-answer pairs

    Args:
        response: The model's response text
        num_questions: Expected number of questions

    Returns:
        List of individual answers
    """
    response = response.strip()
    _answers = []  # noqa: F841

    # Strategy 1: Try numbered format (1. answer, 2. answer, etc.)
    numbered_pattern = r"^\s*(\d+)[\.\)]\s*(.+?)(?=^\s*\d+[\.\)]|\Z)"
    matches = re.findall(numbered_pattern, response, re.MULTILINE | re.DOTALL)

    if matches and len(matches) == num_questions:
        return [match[1].strip() for match in matches]

    # Strategy 2: Try question-answer format (Q: ... A: ...)
    qa_pattern = r"(?:Question|Q)\s*\d*[\.\):]?\s*[^\n]*\n\s*(?:Answer|A)[\.\):]?\s*(.+?)(?=(?:Question|Q)|\Z)"
    qa_matches = re.findall(qa_pattern, response, re.IGNORECASE | re.DOTALL)

    if qa_matches and len(qa_matches) == num_questions:
        return [match.strip() for match in qa_matches]

    # Strategy 3: Split by lines (filter out empty and question lines)
    lines = [line.strip() for line in response.split("\n") if line.strip()]
    # Filter out lines that are just questions (contain '?')
    answer_lines = [line for line in lines if not line.endswith("?")]

    if len(answer_lines) >= num_questions:
        return answer_lines[:num_questions]

    # Fallback: return the whole response for each question
    return [response] * num_questions


def semantic_match_score(answer: str, correct: str) -> float:
    """
    Simple semantic matching using keyword overlap.

    Returns a score between 0 and 1 based on how many important
    words from the correct answer appear in the model's answer.

    Args:
        answer: Model's answer
        correct: Correct answer

    Returns:
        Score between 0 and 1
    """
    # Normalize
    answer_lower = answer.lower()
    correct_lower = correct.lower()

    # Extract important words (ignore common words)
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "from",
        "by",
        "about",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "that",
        "this",
        "these",
        "those",
    }

    # Get important words from correct answer
    correct_words = [w for w in re.findall(r"\w+", correct_lower) if w not in stop_words and len(w) > 2]

    if not correct_words:
        return 1.0 if answer_lower == correct_lower else 0.0

    # Count matches
    matches = sum(1 for word in correct_words if word in answer_lower)

    return matches / len(correct_words)


def llm_judge_tom(
    scenario: ToMScenario,
    model_response: str,
    judge_provider: str = "ollama",
    judge_model: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Use an LLM as a judge to evaluate theory of mind responses.

    Args:
        scenario: The ToM scenario that was tested
        model_response: The model's response to evaluate
        judge_provider: Provider to use for judging
        judge_model: Model to use for judging
        **kwargs: Additional arguments for llm_call

    Returns:
        Dictionary with scores and reasoning
    """
    # Parse the response
    answers = parse_multi_answer_response(model_response, len(scenario.test_questions))

    # Build judging prompt
    judge_prompt = f"""You are evaluating a language model's understanding of theory of mind and epistemology.

Scenario Type: {scenario.tom_type.value}
Difficulty: {scenario.difficulty}

SCENARIO:
{scenario.to_prompt(include_context=False)}

CORRECT ANSWERS:
"""
    for i, correct in enumerate(scenario.correct_answers, 1):
        judge_prompt += f"{i}. {correct}\n"

    judge_prompt += f"\nMODEL'S ANSWERS:\n"
    for i, answer in enumerate(answers, 1):
        judge_prompt += f"{i}. {answer}\n"

    judge_prompt += """
EVALUATION INSTRUCTIONS:
For each answer, assess whether the model demonstrates understanding of the mental states involved.
Give a score from 0-10 for each answer where:
- 0-3: Incorrect or shows no understanding of the mental state
- 4-6: Partially correct, shows some understanding but misses key aspects
- 7-9: Mostly correct, demonstrates good understanding
- 10: Perfect understanding and explanation

Provide your evaluation in this exact format:
Question 1 Score: [0-10]
Question 1 Reasoning: [brief explanation]
Question 2 Score: [0-10]
Question 2 Reasoning: [brief explanation]
...
Overall Assessment: [overall evaluation of ToM understanding]
"""

    # Call judge model
    judge_response = llm_call(judge_prompt, provider=judge_provider, model=judge_model, **kwargs)

    # Parse judge response
    scores = []
    reasonings = []

    score_pattern = r"Question \d+ Score:\s*(\d+(?:\.\d+)?)"
    reasoning_pattern = r"Question \d+ Reasoning:\s*(.+?)(?=Question \d+|Overall Assessment|$)"

    score_matches = re.findall(score_pattern, judge_response.text)
    reasoning_matches = re.findall(reasoning_pattern, judge_response.text, re.DOTALL)

    scores = [float(s) for s in score_matches]
    reasonings = [r.strip() for r in reasoning_matches]

    # Extract overall assessment
    overall_match = re.search(r"Overall Assessment:\s*(.+)", judge_response.text, re.DOTALL)
    overall_assessment = overall_match.group(1).strip() if overall_match else "No assessment provided"

    # Calculate average score
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "individual_scores": scores,
        "individual_reasonings": reasonings,
        "average_score": avg_score,
        "normalized_score": avg_score / 10.0,  # Normalize to 0-1
        "overall_assessment": overall_assessment,
        "judge_response": judge_response.text,
        "judge_tokens_in": judge_response.tokens_in,
        "judge_tokens_out": judge_response.tokens_out,
        "judge_cost_usd": judge_response.cost_usd,
    }


def evaluate_scenario(scenario: ToMScenario, model_response: str, method: str = "semantic", **kwargs) -> Dict[str, Any]:
    """
    Evaluate a model's response to a ToM scenario.

    Args:
        scenario: The scenario that was tested
        model_response: The model's response
        method: Evaluation method ("semantic" or "llm_judge")
        **kwargs: Additional arguments (for llm_judge)

    Returns:
        Evaluation results dictionary
    """
    if method == "llm_judge":
        return llm_judge_tom(scenario, model_response, **kwargs)

    elif method == "semantic":
        # Parse answers
        answers = parse_multi_answer_response(model_response, len(scenario.test_questions))

        # Score each answer
        scores = []
        for answer, correct in zip(answers, scenario.correct_answers):
            score = semantic_match_score(answer, correct)
            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "individual_scores": scores,
            "average_score": avg_score,
            "normalized_score": avg_score,
            "method": "semantic_match",
            "parsed_answers": answers,
        }

    else:
        raise ValueError(f"Unknown evaluation method: {method}")


def evaluate_batch(results: List[Dict[str, Any]], method: str = "semantic", **kwargs) -> Dict[str, Any]:
    """
    Evaluate a batch of scenario results.

    Args:
        results: List of result dictionaries, each containing 'scenario' and 'response'
        method: Evaluation method
        **kwargs: Additional arguments for evaluation

    Returns:
        Aggregated evaluation results
    """
    evaluations = []

    for result in results:
        scenario = result["scenario"]
        response = result["response"]

        eval_result = evaluate_scenario(scenario, response, method=method, **kwargs)
        eval_result["scenario_name"] = scenario.name
        eval_result["scenario_type"] = scenario.tom_type.value
        eval_result["difficulty"] = scenario.difficulty

        evaluations.append(eval_result)

    # Aggregate scores
    all_scores = [e["average_score"] for e in evaluations]

    # Group by difficulty
    by_difficulty = {}
    for eval_result in evaluations:
        diff = eval_result["difficulty"]
        if diff not in by_difficulty:
            by_difficulty[diff] = []
        by_difficulty[diff].append(eval_result["average_score"])

    # Group by ToM type
    by_type = {}
    for eval_result in evaluations:
        tom_type = eval_result["scenario_type"]
        if tom_type not in by_type:
            by_type[tom_type] = []
        by_type[tom_type].append(eval_result["average_score"])

    return {
        "evaluations": evaluations,
        "overall_average": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "by_difficulty": {diff: sum(scores) / len(scores) if scores else 0.0 for diff, scores in by_difficulty.items()},
        "by_type": {tom_type: sum(scores) / len(scores) if scores else 0.0 for tom_type, scores in by_type.items()},
        "total_scenarios": len(evaluations),
    }


def compare_models(
    model_results: Dict[str, List[Dict[str, Any]]], method: str = "semantic", **kwargs
) -> Dict[str, Any]:
    """
    Compare multiple models on the same scenarios.

    Args:
        model_results: Dictionary mapping model names to their results
        method: Evaluation method
        **kwargs: Additional arguments for evaluation

    Returns:
        Comparison dictionary
    """
    model_evaluations = {}

    for model_name, results in model_results.items():
        eval_results = evaluate_batch(results, method=method, **kwargs)
        model_evaluations[model_name] = eval_results

    # Create comparison summary
    comparison = {
        "models": list(model_results.keys()),
        "overall_scores": {name: eval_results["overall_average"] for name, eval_results in model_evaluations.items()},
        "by_difficulty": {},
        "by_type": {},
        "detailed_evaluations": model_evaluations,
    }

    # Aggregate by difficulty across models
    all_difficulties = set()
    for eval_results in model_evaluations.values():
        all_difficulties.update(eval_results["by_difficulty"].keys())

    for diff in all_difficulties:
        comparison["by_difficulty"][diff] = {
            name: eval_results["by_difficulty"].get(diff, 0.0) for name, eval_results in model_evaluations.items()
        }

    # Aggregate by type across models
    all_types = set()
    for eval_results in model_evaluations.values():
        all_types.update(eval_results["by_type"].keys())

    for tom_type in all_types:
        comparison["by_type"][tom_type] = {
            name: eval_results["by_type"].get(tom_type, 0.0) for name, eval_results in model_evaluations.items()
        }

    return comparison
