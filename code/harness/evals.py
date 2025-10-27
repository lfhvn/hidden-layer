"""
Evaluation functions for assessing task performance.
Start simple, add complexity as needed.
"""
import re
from typing import Dict, Any, Optional
from .llm_provider import llm_call
from .defaults import DEFAULT_PROVIDER, DEFAULT_MODEL


def exact_match(output: str, expected: str, case_sensitive: bool = False) -> float:
    """
    Check if output exactly matches expected answer.
    
    Returns 1.0 for match, 0.0 for no match.
    """
    output_clean = output.strip()
    expected_clean = expected.strip()
    
    if not case_sensitive:
        output_clean = output_clean.lower()
        expected_clean = expected_clean.lower()
    
    return 1.0 if output_clean == expected_clean else 0.0


def keyword_match(output: str, keywords: list[str], require_all: bool = False) -> float:
    """
    Check if output contains specific keywords.
    
    Args:
        output: Model output to check
        keywords: List of keywords to look for
        require_all: If True, all keywords must be present. If False, any keyword is enough.
    
    Returns:
        1.0 if criteria met, 0.0 otherwise
    """
    output_lower = output.lower()
    
    if require_all:
        return 1.0 if all(kw.lower() in output_lower for kw in keywords) else 0.0
    else:
        return 1.0 if any(kw.lower() in output_lower for kw in keywords) else 0.0


def extract_number(text: str) -> Optional[float]:
    """
    Extract first number from text.
    Handles formats like: 42, 3.14, -7, "the answer is 42"
    """
    # Try to find a number pattern
    patterns = [
        r'-?\d+\.?\d*',  # Basic number
        r'-?\d+/\d+',    # Fraction
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                # Handle fractions
                if '/' in match.group():
                    num, denom = match.group().split('/')
                    return float(num) / float(denom)
                return float(match.group())
            except ValueError:
                continue
    
    return None


def numeric_match(output: str, expected: float, tolerance: float = 0.01) -> float:
    """
    Check if output contains expected number (within tolerance).
    
    Args:
        output: Model output
        expected: Expected number
        tolerance: Absolute tolerance for match
    
    Returns:
        1.0 if match, 0.0 otherwise
    """
    extracted = extract_number(output)
    
    if extracted is None:
        return 0.0
    
    return 1.0 if abs(extracted - expected) <= tolerance else 0.0


def llm_judge(
    task_input: str,
    output: str,
    criteria: str = "overall quality",
    provider: str = None,
    model: str = None,
    scale: int = 10
) -> Dict[str, Any]:
    """
    Use an LLM to judge the output quality.
    
    Args:
        task_input: Original task/question
        output: Model's output to judge
        criteria: What to judge (e.g., "correctness", "creativity", "helpfulness")
        provider: LLM provider for judge
        model: Model to use as judge
        scale: Rating scale (e.g., 10 = 1-10 scale)
    
    Returns:
        Dict with 'score' (normalized 0-1) and 'reasoning'
    """
    judge_prompt = f"""You are an expert evaluator. Rate the following response on a scale of 1-{scale}.

Task: {task_input}

Response to evaluate:
{output}

Criteria: {criteria}

Provide your evaluation in this format:
Score: [number from 1-{scale}]
Reasoning: [brief explanation]
"""
    
    response = llm_call(judge_prompt, provider=provider, model=model, temperature=0.3)
    
    # Parse score
    score_match = re.search(r'Score:\s*(\d+)', response.text)
    score_raw = int(score_match.group(1)) if score_match else scale // 2
    
    # Normalize to 0-1
    score_normalized = (score_raw - 1) / (scale - 1)
    
    # Extract reasoning
    reasoning_match = re.search(r'Reasoning:\s*(.+)', response.text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else response.text
    
    return {
        'score': score_normalized,
        'score_raw': score_raw,
        'reasoning': reasoning,
        'judge_provider': provider,
        'judge_model': model
    }


def win_rate_comparison(
    task_input: str,
    output_a: str,
    output_b: str,
    provider: str = None,
    model: str = None
) -> Dict[str, Any]:
    """
    Compare two outputs and determine winner.
    
    Useful for comparing strategies (e.g., single vs debate).
    
    Returns:
        Dict with 'winner' ('A', 'B', or 'tie'), 'reasoning', and 'confidence'
    """
    judge_prompt = f"""You are comparing two responses to the same task. Choose which is better.

Task: {task_input}

Response A:
{output_a}

Response B:
{output_b}

Which response is better? Respond in this format:
Winner: [A/B/Tie]
Confidence: [High/Medium/Low]
Reasoning: [brief explanation]
"""
    
    response = llm_call(judge_prompt, provider=provider, model=model, temperature=0.3)
    
    # Parse winner
    winner_match = re.search(r'Winner:\s*(A|B|Tie)', response.text, re.IGNORECASE)
    winner = winner_match.group(1).upper() if winner_match else 'TIE'
    
    # Parse confidence
    conf_match = re.search(r'Confidence:\s*(High|Medium|Low)', response.text, re.IGNORECASE)
    confidence = conf_match.group(1).lower() if conf_match else 'medium'
    
    # Extract reasoning
    reasoning_match = re.search(r'Reasoning:\s*(.+)', response.text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else response.text
    
    return {
        'winner': winner,
        'confidence': confidence,
        'reasoning': reasoning
    }


def coherence_score(output: str) -> float:
    """
    Simple heuristic for coherence based on length and structure.
    
    NOT a deep analysis, just catches obvious issues.
    """
    if not output or len(output.strip()) < 10:
        return 0.0
    
    # Check for repetition (simple heuristic)
    words = output.split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:  # Very repetitive
            return 0.3
    
    # Check for reasonable length
    if len(output) > 5000:  # Too long might indicate rambling
        return 0.7
    
    # Otherwise assume coherent
    return 1.0


# Registry of eval functions
EVAL_FUNCTIONS = {
    'exact_match': exact_match,
    'keyword': keyword_match,
    'numeric': numeric_match,
    'llm_judge': llm_judge,
    'win_rate': win_rate_comparison,
    'coherence': coherence_score
}


def evaluate_task(
    task: Dict[str, Any],
    output: str,
    eval_type: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate a task output using appropriate eval function.
    
    Args:
        task: Task dict with 'input', 'expected', 'eval_type', etc.
        output: Model output to evaluate
        eval_type: Override task's eval_type if provided
    
    Returns:
        Dict of scores, e.g., {'accuracy': 1.0, 'coherence': 0.9}
    """
    eval_type = eval_type or task.get('eval_type', 'llm_judge')
    scores = {}
    
    # Always compute coherence as a basic check
    scores['coherence'] = coherence_score(output)
    
    # Task-specific evaluation
    if eval_type == 'exact_match' and 'expected' in task:
        scores['accuracy'] = exact_match(output, task['expected'])
    
    elif eval_type == 'keyword' and 'expected' in task:
        keywords = [task['expected']] if isinstance(task['expected'], str) else task['expected']
        scores['accuracy'] = keyword_match(output, keywords)
    
    elif eval_type == 'numeric' and 'expected' in task:
        scores['accuracy'] = numeric_match(output, float(task['expected']))
    
    elif eval_type == 'llm_judge':
        judge_result = llm_judge(
            task['input'],
            output,
            criteria=task.get('criteria', 'overall quality')
        )
        scores['quality'] = judge_result['score']
        scores['_judge_reasoning'] = judge_result['reasoning']
    
    return scores
