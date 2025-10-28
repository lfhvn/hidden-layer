"""
Rationale extraction utilities.

Provides tools to get models to explain their reasoning before providing final answers.
Useful for:
- Understanding model decision-making
- Debugging incorrect answers
- Building transparent AI systems
- Improving answer quality through explicit reasoning
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import re

from .llm_provider import llm_call, LLMResponse
from .strategies import StrategyResult


@dataclass
class RationaleResponse:
    """Response containing both reasoning and final answer"""
    rationale: str  # The model's step-by-step reasoning
    answer: str     # The final answer
    raw_response: str  # Full response text
    llm_response: LLMResponse  # Underlying LLM response with metadata


def llm_call_with_rationale(
    prompt: str,
    provider: str = "ollama",
    model: str = "gpt-oss:20b",
    temperature: float = 0.7,
    thinking_budget: Optional[int] = None,
    **kwargs
) -> RationaleResponse:
    """
    Call LLM and extract both reasoning and final answer.

    The model is prompted to show its step-by-step reasoning before
    providing a final answer.

    Args:
        prompt: The question or task
        provider: LLM provider (ollama, anthropic, openai)
        model: Model name
        temperature: Sampling temperature
        thinking_budget: Optional token budget for extended reasoning
        **kwargs: Additional parameters passed to llm_call

    Returns:
        RationaleResponse with rationale and answer separated

    Example:
        >>> result = llm_call_with_rationale(
        ...     "What is 15% of 240?",
        ...     provider="ollama",
        ...     model="gpt-oss:20b",
        ...     thinking_budget=1000
        ... )
        >>> print(result.rationale)
        To find 15% of 240:
        1. Convert 15% to decimal: 15/100 = 0.15
        2. Multiply: 0.15 * 240 = 36
        >>> print(result.answer)
        36
    """

    # Create structured prompt that asks for reasoning
    structured_prompt = f"""Think through this problem step-by-step and show your reasoning.

Problem: {prompt}

Please format your response as:

REASONING:
[Explain your step-by-step thinking, analysis, considerations, trade-offs, etc. Be thorough and explicit about your reasoning process.]

ANSWER:
[Provide your final, concise answer here]

Now solve the problem:"""

    # Call the LLM
    response = llm_call(
        structured_prompt,
        provider=provider,
        model=model,
        temperature=temperature,
        thinking_budget=thinking_budget,
        **kwargs
    )

    # Parse the response
    parsed = _parse_rationale_response(response.text)

    return RationaleResponse(
        rationale=parsed['rationale'],
        answer=parsed['answer'],
        raw_response=response.text,
        llm_response=response
    )


def _parse_rationale_response(text: str) -> Dict[str, str]:
    """
    Parse structured response into rationale and answer.

    Handles various formats:
    - Explicit REASONING:/ANSWER: sections
    - Implicit structure (explanation followed by conclusion)
    - Fallback for unstructured responses
    """

    # Try to find explicit REASONING: and ANSWER: sections
    reasoning_pattern = r'REASONING:\s*(.*?)(?=ANSWER:|$)'
    answer_pattern = r'ANSWER:\s*(.*?)$'

    reasoning_match = re.search(reasoning_pattern, text, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(answer_pattern, text, re.DOTALL | re.IGNORECASE)

    if reasoning_match and answer_match:
        # Found explicit structure
        return {
            'rationale': reasoning_match.group(1).strip(),
            'answer': answer_match.group(1).strip()
        }

    # Try alternative format with "Therefore" or "Thus" or "So" as separator
    conclusion_pattern = r'(.*?)(?:Therefore|Thus|So|In conclusion)[,:]?\s*(.*?)$'
    conclusion_match = re.search(conclusion_pattern, text, re.DOTALL | re.IGNORECASE)

    if conclusion_match:
        return {
            'rationale': conclusion_match.group(1).strip(),
            'answer': conclusion_match.group(2).strip()
        }

    # Fallback: treat last paragraph as answer, rest as rationale
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    if len(paragraphs) >= 2:
        return {
            'rationale': '\n\n'.join(paragraphs[:-1]),
            'answer': paragraphs[-1]
        }

    # Ultimate fallback: entire text is both
    return {
        'rationale': text.strip(),
        'answer': text.strip()
    }


def extract_rationale_from_result(result: StrategyResult) -> RationaleResponse:
    """
    Extract rationale from a StrategyResult.

    Useful for post-hoc analysis of strategy outputs.

    Args:
        result: A StrategyResult from run_strategy

    Returns:
        RationaleResponse with parsed rationale and answer

    Example:
        >>> from harness import run_strategy
        >>> result = run_strategy("single", "Complex question...")
        >>> rationale_result = extract_rationale_from_result(result)
        >>> print(rationale_result.rationale)
    """

    parsed = _parse_rationale_response(result.output)

    # Create a mock LLMResponse for consistency
    mock_response = LLMResponse(
        text=result.output,
        model=result.metadata.get('model', 'unknown'),
        provider=result.metadata.get('provider', 'unknown'),
        latency_s=result.latency_s,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        metadata=result.metadata
    )

    return RationaleResponse(
        rationale=parsed['rationale'],
        answer=parsed['answer'],
        raw_response=result.output,
        llm_response=mock_response
    )


def run_strategy_with_rationale(
    strategy: str,
    task_input: str,
    **kwargs
) -> RationaleResponse:
    """
    Run a multi-agent strategy and extract rationale from the result.

    This wraps run_strategy and automatically parses the output to
    separate reasoning from final answer.

    Args:
        strategy: Strategy name (e.g., "adaptive_team", "design_critique")
        task_input: The task/question to solve
        **kwargs: Strategy-specific parameters

    Returns:
        RationaleResponse with rationale and answer

    Example:
        >>> result = run_strategy_with_rationale(
        ...     "adaptive_team",
        ...     "Should we use React or Vue?",
        ...     n_experts=3,
        ...     provider="ollama",
        ...     model="gpt-oss:20b"
        ... )
        >>> print(result.rationale)
        [Expert analyses and synthesis reasoning...]
        >>> print(result.answer)
        React is recommended because...
    """

    from .strategies import run_strategy

    # Add instruction to provide reasoning
    enhanced_input = f"""{task_input}

IMPORTANT: In your final response, clearly separate your reasoning from your answer using this format:

REASONING:
[Your analysis, considerations, trade-offs, etc.]

ANSWER:
[Your final recommendation/answer]
"""

    result = run_strategy(strategy, enhanced_input, **kwargs)

    return extract_rationale_from_result(result)


# Convenience function for common use case
def ask_with_reasoning(
    question: str,
    provider: str = "ollama",
    model: str = "gpt-oss:20b",
    thinking_budget: int = 1500,
    temperature: float = 0.7,
    **kwargs
) -> RationaleResponse:
    """
    Convenience function: Ask a question and get reasoning + answer.

    This is the simplest way to get rationale extraction.

    Args:
        question: Your question
        provider: LLM provider
        model: Model name
        thinking_budget: Tokens allocated for reasoning (default: 1500)
        temperature: Sampling temperature
        **kwargs: Additional llm_call parameters

    Returns:
        RationaleResponse with rationale and answer

    Example:
        >>> result = ask_with_reasoning("What is 234 * 567?")
        >>> print(f"Reasoning: {result.rationale}")
        >>> print(f"Answer: {result.answer}")
    """

    return llm_call_with_rationale(
        question,
        provider=provider,
        model=model,
        temperature=temperature,
        thinking_budget=thinking_budget,
        **kwargs
    )
