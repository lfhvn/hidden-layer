"""
Collective Design Critique Strategies

This module implements different approaches to design critique:
- Single critic (baseline)
- Multi-perspective critique (different expert viewpoints)
- Iterative critique (rounds of refinement)
- Synthesis critique (combining perspectives)
- Adversarial critique (challenge assumptions)
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from crit.problems import CritiquePerspective, DesignProblem
from harness import llm_call


@dataclass
class CritiqueResult:
    """Result of a design critique"""

    problem_name: str
    strategy_name: str
    critiques: List[Dict[str, Any]]  # List of individual critiques
    synthesis: Optional[str]  # Combined/synthesized feedback
    recommendations: List[str]  # Actionable recommendations
    revised_design: Optional[str]  # If strategy includes revision
    latency_s: float
    total_tokens_in: int
    total_tokens_out: int
    total_cost_usd: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "problem_name": self.problem_name,
            "strategy_name": self.strategy_name,
            "critiques": self.critiques,
            "synthesis": self.synthesis,
            "recommendations": self.recommendations,
            "revised_design": self.revised_design,
            "latency_s": self.latency_s,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_cost_usd": self.total_cost_usd,
            "metadata": self.metadata,
        }


def single_critic_strategy(
    problem: DesignProblem, provider: str = "ollama", model: Optional[str] = None, **kwargs
) -> CritiqueResult:
    """
    Single expert provides comprehensive critique.

    This is the baseline strategy - one model reviews the design
    and provides feedback.

    Args:
        problem: The design problem to critique
        provider: LLM provider
        model: Model name
        **kwargs: Additional llm_call arguments

    Returns:
        CritiqueResult with single critique
    """
    start_time = time.time()

    # Generate critique prompt
    prompt = problem.to_critique_prompt(perspective=None)

    # Get critique
    response = llm_call(prompt, provider=provider, model=model, **kwargs)

    # Parse recommendations (look for numbered lists or bullet points)
    recommendations = _extract_recommendations(response.text)

    critique = {
        "perspective": "general",
        "critique": response.text,
        "tokens_in": response.tokens_in,
        "tokens_out": response.tokens_out,
        "cost_usd": response.cost_usd,
    }

    return CritiqueResult(
        problem_name=problem.name,
        strategy_name="single_critic",
        critiques=[critique],
        synthesis=response.text,
        recommendations=recommendations,
        revised_design=None,
        latency_s=time.time() - start_time,
        total_tokens_in=response.tokens_in,
        total_tokens_out=response.tokens_out,
        total_cost_usd=response.cost_usd,
        metadata={"problem": problem.to_dict()},
    )


def multi_perspective_critique(
    problem: DesignProblem,
    perspectives: Optional[List[CritiquePerspective]] = None,
    synthesize: bool = True,
    provider: str = "ollama",
    model: Optional[str] = None,
    **kwargs,
) -> CritiqueResult:
    """
    Multiple experts from different perspectives critique the design.

    Each expert reviews from their specific viewpoint (usability, security, etc.)
    and optionally synthesizes the feedback.

    Args:
        problem: The design problem to critique
        perspectives: List of perspectives to use (default: common set)
        synthesize: Whether to synthesize critiques into unified feedback
        provider: LLM provider
        model: Model name
        **kwargs: Additional llm_call arguments

    Returns:
        CritiqueResult with multiple perspective critiques
    """
    start_time = time.time()

    # Default perspectives if none provided
    if perspectives is None:
        perspectives = _get_default_perspectives(problem.domain)

    critiques = []
    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0

    # Get critique from each perspective
    for perspective in perspectives:
        prompt = problem.to_critique_prompt(perspective=perspective)
        response = llm_call(prompt, provider=provider, model=model, **kwargs)

        critiques.append(
            {
                "perspective": perspective.value,
                "critique": response.text,
                "tokens_in": response.tokens_in,
                "tokens_out": response.tokens_out,
                "cost_usd": response.cost_usd,
            }
        )

        total_tokens_in += response.tokens_in
        total_tokens_out += response.tokens_out
        total_cost += response.cost_usd

    # Synthesize if requested
    synthesis_text = None
    recommendations = []

    if synthesize:
        synthesis_prompt = _build_synthesis_prompt(problem, critiques)
        synthesis_response = llm_call(synthesis_prompt, provider=provider, model=model, **kwargs)

        synthesis_text = synthesis_response.text
        recommendations = _extract_recommendations(synthesis_text)

        total_tokens_in += synthesis_response.tokens_in
        total_tokens_out += synthesis_response.tokens_out
        total_cost += synthesis_response.cost_usd
    else:
        # Extract recommendations from individual critiques
        for critique in critiques:
            recommendations.extend(_extract_recommendations(critique["critique"]))

    return CritiqueResult(
        problem_name=problem.name,
        strategy_name="multi_perspective",
        critiques=critiques,
        synthesis=synthesis_text,
        recommendations=recommendations,
        revised_design=None,
        latency_s=time.time() - start_time,
        total_tokens_in=total_tokens_in,
        total_tokens_out=total_tokens_out,
        total_cost_usd=total_cost,
        metadata={
            "problem": problem.to_dict(),
            "perspectives": [p.value for p in perspectives],
            "synthesized": synthesize,
        },
    )


def iterative_critique(
    problem: DesignProblem, iterations: int = 2, provider: str = "ollama", model: Optional[str] = None, **kwargs
) -> CritiqueResult:
    """
    Iterative design critique with refinement.

    1. Initial critique of design
    2. Propose improvements
    3. Critique the improvements
    4. Repeat for N iterations

    Args:
        problem: The design problem to critique
        iterations: Number of critique-refine cycles
        provider: LLM provider
        model: Model name
        **kwargs: Additional llm_call arguments

    Returns:
        CritiqueResult with iterative critiques and final revised design
    """
    start_time = time.time()

    critiques = []
    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0

    current_design = problem.current_design

    for i in range(iterations):
        # Critique current design
        critique_prompt = f"""Review this design and provide constructive critique.

DESIGN PROBLEM: {problem.description}

CONTEXT:
{problem.context}

CURRENT DESIGN:
{current_design}

SUCCESS CRITERIA:
{chr(10).join(f"{i+1}. {c}" for i, c in enumerate(problem.success_criteria))}

Provide:
1. What works well
2. What could be improved
3. Specific recommendations for improvement
"""
        critique_response = llm_call(critique_prompt, provider=provider, model=model, **kwargs)

        critiques.append(
            {
                "iteration": i + 1,
                "type": "critique",
                "design": current_design,
                "feedback": critique_response.text,
                "tokens_in": critique_response.tokens_in,
                "tokens_out": critique_response.tokens_out,
                "cost_usd": critique_response.cost_usd,
            }
        )

        total_tokens_in += critique_response.tokens_in
        total_tokens_out += critique_response.tokens_out
        total_cost += critique_response.cost_usd

        # Generate improved design
        if i < iterations - 1:  # Don't revise on last iteration
            revision_prompt = f"""Based on this critique, propose an improved design.

ORIGINAL DESIGN:
{current_design}

CRITIQUE:
{critique_response.text}

Provide the complete revised design, incorporating the feedback.
"""
            revision_response = llm_call(revision_prompt, provider=provider, model=model, **kwargs)

            current_design = revision_response.text

            critiques.append(
                {
                    "iteration": i + 1,
                    "type": "revision",
                    "revised_design": current_design,
                    "tokens_in": revision_response.tokens_in,
                    "tokens_out": revision_response.tokens_out,
                    "cost_usd": revision_response.cost_usd,
                }
            )

            total_tokens_in += revision_response.tokens_in
            total_tokens_out += revision_response.tokens_out
            total_cost += revision_response.cost_usd

    # Extract final recommendations
    final_critique = critiques[-1]["feedback"]
    recommendations = _extract_recommendations(final_critique)

    return CritiqueResult(
        problem_name=problem.name,
        strategy_name="iterative",
        critiques=critiques,
        synthesis=final_critique,
        recommendations=recommendations,
        revised_design=current_design,
        latency_s=time.time() - start_time,
        total_tokens_in=total_tokens_in,
        total_tokens_out=total_tokens_out,
        total_cost_usd=total_cost,
        metadata={"problem": problem.to_dict(), "iterations": iterations},
    )


def adversarial_critique(
    problem: DesignProblem, provider: str = "ollama", model: Optional[str] = None, **kwargs
) -> CritiqueResult:
    """
    Two-agent adversarial critique.

    - Agent 1: Proposes improvements
    - Agent 2: Challenges and finds flaws
    - Agent 1: Responds to challenges
    - Synthesis of the debate

    Args:
        problem: The design problem to critique
        provider: LLM provider
        model: Model name
        **kwargs: Additional llm_call arguments

    Returns:
        CritiqueResult with adversarial exchange
    """
    start_time = time.time()

    critiques = []
    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0

    # Agent 1: Propose improvements
    improvement_prompt = f"""You are a design expert proposing improvements to this design.

DESIGN PROBLEM: {problem.description}

CONTEXT:
{problem.context}

CURRENT DESIGN:
{problem.current_design}

SUCCESS CRITERIA:
{chr(10).join(f"{i+1}. {c}" for i, c in enumerate(problem.success_criteria))}

Propose specific improvements to this design. Be bold and innovative.
"""
    improvement_response = llm_call(improvement_prompt, provider=provider, model=model, **kwargs)

    critiques.append(
        {
            "agent": "proposer",
            "content": improvement_response.text,
            "tokens_in": improvement_response.tokens_in,
            "tokens_out": improvement_response.tokens_out,
            "cost_usd": improvement_response.cost_usd,
        }
    )

    total_tokens_in += improvement_response.tokens_in
    total_tokens_out += improvement_response.tokens_out
    total_cost += improvement_response.cost_usd

    # Agent 2: Challenge the proposals
    challenge_prompt = f"""You are a critical reviewer challenging these design proposals.

ORIGINAL DESIGN:
{problem.current_design}

PROPOSED IMPROVEMENTS:
{improvement_response.text}

CONTEXT:
{problem.context}

Challenge these proposals. What could go wrong? What are the trade-offs?
What edge cases or problems might arise? Be skeptical and thorough.
"""
    challenge_response = llm_call(challenge_prompt, provider=provider, model=model, **kwargs)

    critiques.append(
        {
            "agent": "challenger",
            "content": challenge_response.text,
            "tokens_in": challenge_response.tokens_in,
            "tokens_out": challenge_response.tokens_out,
            "cost_usd": challenge_response.cost_usd,
        }
    )

    total_tokens_in += challenge_response.tokens_in
    total_tokens_out += challenge_response.tokens_out
    total_cost += challenge_response.cost_usd

    # Agent 1: Respond to challenges
    response_prompt = f"""Respond to these challenges of your design proposals.

YOUR PROPOSALS:
{improvement_response.text}

CHALLENGES:
{challenge_response.text}

Address the concerns raised. Refine your proposals based on valid critiques.
"""
    response_response = llm_call(response_prompt, provider=provider, model=model, **kwargs)

    critiques.append(
        {
            "agent": "proposer_response",
            "content": response_response.text,
            "tokens_in": response_response.tokens_in,
            "tokens_out": response_response.tokens_out,
            "cost_usd": response_response.cost_usd,
        }
    )

    total_tokens_in += response_response.tokens_in
    total_tokens_out += response_response.tokens_out
    total_cost += response_response.cost_usd

    # Synthesize the debate
    synthesis_prompt = f"""Synthesize this design debate into actionable recommendations.

PROPOSALS:
{improvement_response.text}

CHALLENGES:
{challenge_response.text}

RESPONSES:
{response_response.text}

Provide a balanced synthesis that incorporates valid points from both sides.
Give specific, actionable recommendations.
"""
    synthesis_response = llm_call(synthesis_prompt, provider=provider, model=model, **kwargs)

    total_tokens_in += synthesis_response.tokens_in
    total_tokens_out += synthesis_response.tokens_out
    total_cost += synthesis_response.cost_usd

    recommendations = _extract_recommendations(synthesis_response.text)

    return CritiqueResult(
        problem_name=problem.name,
        strategy_name="adversarial",
        critiques=critiques,
        synthesis=synthesis_response.text,
        recommendations=recommendations,
        revised_design=None,
        latency_s=time.time() - start_time,
        total_tokens_in=total_tokens_in,
        total_tokens_out=total_tokens_out,
        total_cost_usd=total_cost,
        metadata={"problem": problem.to_dict()},
    )


# ============================================================================
# Helper Functions
# ============================================================================


def _get_default_perspectives(domain) -> List[CritiquePerspective]:
    """Get default perspectives for a design domain"""
    from crit.problems import DesignDomain

    # Common perspectives for all domains
    common = [
        CritiquePerspective.USABILITY,
        CritiquePerspective.MAINTAINABILITY,
    ]

    # Domain-specific perspectives
    domain_specific = {
        DesignDomain.UI_UX: [
            CritiquePerspective.ACCESSIBILITY,
            CritiquePerspective.AESTHETICS,
            CritiquePerspective.USER_ADVOCACY,
        ],
        DesignDomain.API: [
            CritiquePerspective.CONSISTENCY,
            CritiquePerspective.PERFORMANCE,
        ],
        DesignDomain.SYSTEM: [
            CritiquePerspective.SCALABILITY,
            CritiquePerspective.PERFORMANCE,
            CritiquePerspective.SECURITY,
        ],
        DesignDomain.DATA: [
            CritiquePerspective.SCALABILITY,
            CritiquePerspective.SECURITY,
            CritiquePerspective.CONSISTENCY,
        ],
        DesignDomain.WORKFLOW: [
            CritiquePerspective.USER_ADVOCACY,
            CritiquePerspective.CONSISTENCY,
        ],
    }

    return common + domain_specific.get(domain, [])


def _build_synthesis_prompt(problem: DesignProblem, critiques: List[Dict[str, Any]]) -> str:
    """Build prompt for synthesizing multiple critiques"""
    prompt_parts = [
        "Synthesize these expert critiques into unified, actionable feedback.\n",
        f"DESIGN PROBLEM: {problem.description}\n",
        f"\nCURRENT DESIGN:\n{problem.current_design}\n",
        "\nEXPERT CRITIQUES:\n",
    ]

    for critique in critiques:
        perspective = critique["perspective"]
        content = critique["critique"]
        prompt_parts.append(f"\n{perspective.upper()} PERSPECTIVE:")
        prompt_parts.append(content)
        prompt_parts.append("")

    prompt_parts.append(
        "\nProvide:\n"
        "1. Key themes across critiques\n"
        "2. Prioritized list of issues (most important first)\n"
        "3. Specific, actionable recommendations\n"
        "4. Any trade-offs or conflicts between perspectives\n"
    )

    return "\n".join(prompt_parts)


def _extract_recommendations(text: str) -> List[str]:
    """Extract recommendations from critique text"""
    import re

    recommendations = []

    # Look for numbered lists
    numbered_pattern = r"^\s*(\d+)[\.\)]\s*(.+?)(?=^\s*\d+[\.\)]|\Z)"
    matches = re.findall(numbered_pattern, text, re.MULTILINE | re.DOTALL)

    if matches:
        recommendations.extend([match[1].strip() for match in matches])

    # Look for bullet points
    bullet_pattern = r"^\s*[-\*]\s*(.+?)$"
    bullet_matches = re.findall(bullet_pattern, text, re.MULTILINE)

    if bullet_matches and not recommendations:
        recommendations.extend([match.strip() for match in bullet_matches])

    # If no structured recommendations found, look for "recommend" keywords
    if not recommendations:
        recommend_pattern = r"(?:recommend|suggest|propose)[^.!?]*[.!?]"
        recommend_matches = re.findall(recommend_pattern, text, re.IGNORECASE | re.MULTILINE)
        recommendations.extend([match.strip() for match in recommend_matches[:5]])

    return recommendations[:10]  # Limit to 10 recommendations


# ============================================================================
# Strategy Registry
# ============================================================================

STRATEGIES = {
    "single": single_critic_strategy,
    "multi_perspective": multi_perspective_critique,
    "iterative": iterative_critique,
    "adversarial": adversarial_critique,
}


def run_critique_strategy(strategy_name: str, problem: DesignProblem, **kwargs) -> CritiqueResult:
    """
    Run a critique strategy by name.

    Args:
        strategy_name: Name of the strategy to run
        problem: The design problem
        **kwargs: Strategy-specific arguments

    Returns:
        CritiqueResult

    Raises:
        ValueError: If strategy name is unknown
    """
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. " f"Available: {list(STRATEGIES.keys())}")

    strategy_func = STRATEGIES[strategy_name]
    return strategy_func(problem, **kwargs)
