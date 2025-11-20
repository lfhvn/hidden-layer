"""
Reflector component for ACE.

Extracts insights from execution trajectories through critique and analysis.
"""

import json
from typing import List, Dict, Any, Optional

from .context import Trajectory, Insight, Example


class Reflector:
    """
    Reflector extracts insights from execution trajectories.

    The reflector critiques reasoning traces to identify what worked,
    what didn't, and extracts actionable lessons.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.3,  # Lower temperature for more focused reflection
        num_refinement_iterations: int = 1
    ):
        """
        Initialize the Reflector.

        Args:
            provider: LLM provider (via harness)
            model: Model name
            temperature: Sampling temperature
            num_refinement_iterations: Number of refinement passes
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.num_refinement_iterations = num_refinement_iterations

    def reflect(self, trajectories: List[Trajectory]) -> List[Insight]:
        """
        Reflect on trajectories to extract insights.

        Args:
            trajectories: List of trajectories to analyze

        Returns:
            List of extracted insights
        """
        # Group trajectories by success
        successful = [t for t in trajectories if t.success]
        failed = [t for t in trajectories if not t.success]

        insights = []

        # Extract insights from successful trajectories
        if successful:
            success_insights = self._extract_success_patterns(successful)
            insights.extend(success_insights)

        # Extract insights from failures
        if failed:
            failure_insights = self._extract_failure_patterns(failed)
            insights.extend(failure_insights)

        # Refine insights if requested
        if self.num_refinement_iterations > 0:
            insights = self._refine_insights(insights, trajectories)

        return insights

    def critique(self, trajectory: Trajectory) -> Dict[str, Any]:
        """
        Generate a detailed critique of a single trajectory.

        Args:
            trajectory: Trajectory to critique

        Returns:
            Critique dictionary with strengths, weaknesses, suggestions
        """
        # Import here to avoid circular dependencies
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))
        from harness import llm_call

        prompt = self._build_critique_prompt(trajectory)

        response = llm_call(
            prompt,
            provider=self.provider,
            model=self.model,
            temperature=self.temperature
        )

        # Parse critique
        critique = self._parse_critique(response.text)
        return critique

    def _extract_success_patterns(
        self,
        trajectories: List[Trajectory]
    ) -> List[Insight]:
        """Extract patterns from successful trajectories."""
        # Import here to avoid circular dependencies
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))
        from harness import llm_call

        prompt = self._build_success_pattern_prompt(trajectories)

        response = llm_call(
            prompt,
            provider=self.provider,
            model=self.model,
            temperature=self.temperature
        )

        # Parse insights
        insights = self._parse_insights(response.text, "strategy", trajectories)
        return insights

    def _extract_failure_patterns(
        self,
        trajectories: List[Trajectory]
    ) -> List[Insight]:
        """Extract patterns from failed trajectories."""
        # Import here to avoid circular dependencies
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))
        from harness import llm_call

        prompt = self._build_failure_pattern_prompt(trajectories)

        response = llm_call(
            prompt,
            provider=self.provider,
            model=self.model,
            temperature=self.temperature
        )

        # Parse insights
        insights = self._parse_insights(response.text, "pitfall", trajectories)
        return insights

    def _refine_insights(
        self,
        insights: List[Insight],
        trajectories: List[Trajectory]
    ) -> List[Insight]:
        """Refine insights over multiple iterations."""
        refined_insights = insights

        for i in range(self.num_refinement_iterations):
            refined_insights = self._refine_iteration(refined_insights, trajectories)

        return refined_insights

    def _refine_iteration(
        self,
        insights: List[Insight],
        trajectories: List[Trajectory]
    ) -> List[Insight]:
        """Single refinement iteration."""
        # Import here to avoid circular dependencies
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))
        from harness import llm_call

        prompt = self._build_refinement_prompt(insights, trajectories)

        response = llm_call(
            prompt,
            provider=self.provider,
            model=self.model,
            temperature=self.temperature
        )

        # Parse refined insights
        refined = self._parse_insights(response.text, None, trajectories)
        return refined

    def _build_critique_prompt(self, trajectory: Trajectory) -> str:
        """Build prompt for critiquing a trajectory."""
        return f"""Please critique the following reasoning trajectory.

Task: {trajectory.task}

Reasoning Steps:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(trajectory.steps))}

Result: {trajectory.result}
Success: {trajectory.success}
{f"Feedback: {trajectory.feedback}" if trajectory.feedback else ""}

Please provide:
1. **Strengths**: What was done well?
2. **Weaknesses**: What could be improved?
3. **Suggestions**: Specific actionable improvements

Format as JSON:
{{
  "strengths": ["strength 1", "strength 2", ...],
  "weaknesses": ["weakness 1", "weakness 2", ...],
  "suggestions": ["suggestion 1", "suggestion 2", ...]
}}
"""

    def _build_success_pattern_prompt(
        self,
        trajectories: List[Trajectory]
    ) -> str:
        """Build prompt for extracting success patterns."""
        trajectory_summaries = []
        for i, t in enumerate(trajectories[:5], 1):  # Limit to 5 examples
            summary = f"**Example {i}**\nTask: {t.task}\nSteps: {len(t.steps)}\n"
            if t.steps:
                summary += f"Key step: {t.steps[0][:200]}...\n"
            trajectory_summaries.append(summary)

        return f"""Analyze these successful reasoning trajectories and extract effective strategies.

{chr(10).join(trajectory_summaries)}

Please identify:
1. **Common patterns**: What approaches consistently led to success?
2. **Effective techniques**: What specific techniques were used?
3. **When to apply**: In what situations should these strategies be used?

Format each insight as JSON:
[
  {{
    "type": "strategy",
    "description": "Brief description of the strategy",
    "category": "Category name (e.g., 'reasoning', 'decomposition', 'verification')",
    "when_to_use": "When to apply this strategy",
    "confidence": 0.0-1.0
  }},
  ...
]

Provide 2-5 high-quality insights.
"""

    def _build_failure_pattern_prompt(
        self,
        trajectories: List[Trajectory]
    ) -> str:
        """Build prompt for extracting failure patterns."""
        trajectory_summaries = []
        for i, t in enumerate(trajectories[:5], 1):  # Limit to 5 examples
            summary = f"**Example {i}**\nTask: {t.task}\n"
            if t.feedback:
                summary += f"Feedback: {t.feedback[:200]}...\n"
            trajectory_summaries.append(summary)

        return f"""Analyze these failed reasoning trajectories and identify common pitfalls.

{chr(10).join(trajectory_summaries)}

Please identify:
1. **Common mistakes**: What errors frequently occurred?
2. **Root causes**: Why did these failures happen?
3. **How to avoid**: How can these mistakes be prevented?

Format each insight as JSON:
[
  {{
    "type": "pitfall",
    "description": "Brief description of the pitfall",
    "how_to_avoid": "How to avoid this mistake",
    "confidence": 0.0-1.0
  }},
  ...
]

Provide 2-5 high-quality insights.
"""

    def _build_refinement_prompt(
        self,
        insights: List[Insight],
        trajectories: List[Trajectory]
    ) -> str:
        """Build prompt for refining insights."""
        insight_summaries = []
        for insight in insights:
            summary = f"- **{insight.type}**: {insight.description}"
            if insight.when_to_use:
                summary += f" (when: {insight.when_to_use})"
            if insight.how_to_avoid:
                summary += f" (avoid by: {insight.how_to_avoid})"
            insight_summaries.append(summary)

        return f"""Refine these insights to make them more concrete and actionable.

Current Insights:
{chr(10).join(insight_summaries)}

Based on {len(trajectories)} trajectories.

Please:
1. Make descriptions more specific and concrete
2. Add clear applicability conditions
3. Merge redundant insights
4. Remove vague or low-confidence insights

Format as JSON (same structure as before):
[
  {{
    "type": "strategy" or "pitfall",
    "description": "...",
    "category": "..." (for strategies),
    "when_to_use": "..." (for strategies),
    "how_to_avoid": "..." (for pitfalls),
    "confidence": 0.0-1.0
  }},
  ...
]
"""

    def _parse_critique(self, response: str) -> Dict[str, Any]:
        """Parse critique response."""
        try:
            # Try to find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: basic parsing
        return {
            "strengths": [],
            "weaknesses": [],
            "suggestions": []
        }

    def _parse_insights(
        self,
        response: str,
        default_type: Optional[str],
        trajectories: List[Trajectory]
    ) -> List[Insight]:
        """Parse insights from response."""
        insights = []

        try:
            # Try to find JSON array in response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                insights_data = json.loads(json_str)

                for data in insights_data:
                    insight = Insight(
                        type=data.get("type", default_type or "strategy"),
                        description=data.get("description", ""),
                        category=data.get("category"),
                        when_to_use=data.get("when_to_use"),
                        how_to_avoid=data.get("how_to_avoid"),
                        confidence=data.get("confidence", 0.8),
                        source_trajectory_ids=[t.trajectory_id for t in trajectories]
                    )
                    insights.append(insight)

        except (json.JSONDecodeError, ValueError):
            pass

        return insights
