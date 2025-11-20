"""
Generator component for ACE.

Produces reasoning trajectories for tasks using the current context.
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

from .context import Context, Trajectory


class Generator:
    """
    Generator produces reasoning trajectories for tasks.

    The generator executes tasks using the current context and captures
    the reasoning process, including successes and failures.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize the Generator.

        Args:
            provider: LLM provider (via harness)
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_trajectory(
        self,
        task: str,
        context: Context,
        include_reasoning: bool = True
    ) -> Trajectory:
        """
        Generate a reasoning trajectory for a task.

        Args:
            task: The task to perform
            context: Current context (playbook)
            include_reasoning: Whether to request explicit reasoning steps

        Returns:
            Trajectory containing reasoning and result
        """
        # Import here to avoid circular dependencies
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))
        from harness import llm_call

        # Build prompt with context
        prompt = self._build_prompt(task, context, include_reasoning)

        # Execute task
        response = llm_call(
            prompt,
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # Parse response into trajectory
        trajectory = self._parse_response(
            response.text,
            task,
            context.version
        )

        return trajectory

    def generate_batch(
        self,
        tasks: List[str],
        context: Context,
        include_reasoning: bool = True
    ) -> List[Trajectory]:
        """
        Generate trajectories for a batch of tasks.

        Args:
            tasks: List of tasks to perform
            context: Current context (playbook)
            include_reasoning: Whether to request explicit reasoning steps

        Returns:
            List of trajectories
        """
        trajectories = []
        for task in tasks:
            trajectory = self.generate_trajectory(task, context, include_reasoning)
            trajectories.append(trajectory)
        return trajectories

    def execute_with_feedback(
        self,
        task: str,
        context: Context,
        evaluator: Optional[callable] = None
    ) -> Trajectory:
        """
        Execute task and add feedback to trajectory.

        Args:
            task: The task to perform
            context: Current context (playbook)
            evaluator: Optional function to evaluate result and provide feedback

        Returns:
            Trajectory with feedback
        """
        trajectory = self.generate_trajectory(task, context)

        # Add feedback if evaluator provided
        if evaluator:
            feedback = evaluator(task, trajectory.result)
            trajectory.feedback = feedback.get("feedback", "")
            trajectory.success = feedback.get("success", False)
            trajectory.metadata["evaluation"] = feedback

        return trajectory

    def _build_prompt(
        self,
        task: str,
        context: Context,
        include_reasoning: bool
    ) -> str:
        """Build the prompt for task execution."""
        sections = []

        # Add context (playbook)
        sections.append(context.to_prompt())
        sections.append("\n---\n")

        # Add task
        sections.append("# Task")
        sections.append(f"\n{task}\n")

        # Add reasoning instructions if requested
        if include_reasoning:
            sections.append("\n# Instructions")
            sections.append(
                "Please solve this task step-by-step. "
                "For each step, explain your reasoning clearly. "
                "Apply relevant strategies from the playbook above when appropriate. "
                "Format your response as:\n\n"
                "**Step 1**: [reasoning]\n"
                "**Step 2**: [reasoning]\n"
                "...\n\n"
                "**Final Answer**: [your answer]\n"
            )

        return "\n".join(sections)

    def _parse_response(
        self,
        response: str,
        task: str,
        context_version: int
    ) -> Trajectory:
        """
        Parse LLM response into a Trajectory.

        Extracts reasoning steps and final answer.
        """
        trajectory_id = str(uuid.uuid4())

        # Parse steps
        steps = []
        lines = response.split("\n")
        current_step = []

        for line in lines:
            line = line.strip()
            if line.startswith("**Step") or line.startswith("**Final"):
                if current_step:
                    steps.append("\n".join(current_step))
                current_step = [line]
            elif current_step:
                current_step.append(line)

        if current_step:
            steps.append("\n".join(current_step))

        # Extract final answer
        result = response
        for step in steps:
            if "**Final Answer**" in step or "**Result**" in step:
                # Extract text after the marker
                parts = step.split(":", 1)
                if len(parts) > 1:
                    result = parts[1].strip()
                break

        # If no explicit final answer, use last step or full response
        if result == response and steps:
            result = steps[-1]

        return Trajectory(
            trajectory_id=trajectory_id,
            task=task,
            context_version=context_version,
            steps=steps if steps else [response],
            result=result,
            timestamp=datetime.now()
        )

    def estimate_cost(self, task: str, context: Context) -> Dict[str, float]:
        """
        Estimate token cost for generating a trajectory.

        Args:
            task: The task
            context: Current context

        Returns:
            Dictionary with estimated input/output tokens
        """
        prompt = self._build_prompt(task, context, include_reasoning=True)

        # Rough token estimation (1 token ~= 4 characters)
        input_tokens = len(prompt) / 4
        output_tokens = self.max_tokens / 2  # Assume average usage

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
