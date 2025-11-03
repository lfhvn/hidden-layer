"""
API-Compatible Introspection Testing

Since API providers (Anthropic, OpenAI) don't expose activations, we use:
1. Prompt-based steering: Simulate concept injection via system prompts
2. Natural introspection: Test model's awareness of its own reasoning

This enables comparison between:
- MLX models with real activation steering
- API models with simulated steering
- API models' natural introspective abilities
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
import time

from .llm_provider import llm_call, LLMResponse
from .introspection_tasks import (
    IntrospectionTask,
    IntrospectionResult,
    IntrospectionTaskType,
    IntrospectionTaskGenerator,
    IntrospectionEvaluator
)


@dataclass
class PromptSteeringConfig:
    """Configuration for prompt-based concept steering"""
    concept: str
    strength: str = "moderate"  # "subtle", "moderate", "strong"
    style: str = "implicit"      # "implicit", "explicit"


class PromptSteerer:
    """
    Simulate activation steering through prompt engineering.

    Since we can't inject into activations, we use system prompts
    to bias the model toward a concept.
    """

    STRENGTH_TEMPLATES = {
        "subtle": {
            "implicit": "In this conversation, subtly consider the theme of {concept}.",
            "explicit": "You should have a slight awareness of {concept} as you respond."
        },
        "moderate": {
            "implicit": "As you respond, let the concept of {concept} influence your thinking.",
            "explicit": "You are currently experiencing {concept} as you generate this response."
        },
        "strong": {
            "implicit": "The concept of {concept} should strongly inform your entire response.",
            "explicit": "You are deeply focused on {concept} right now, and this heavily influences your thoughts."
        }
    }

    def create_steered_prompt(
        self,
        base_prompt: str,
        config: PromptSteeringConfig,
        include_introspection: bool = True
    ) -> str:
        """
        Create a prompt that simulates concept steering.

        Args:
            base_prompt: The original task prompt
            config: Steering configuration
            include_introspection: Whether to include introspection query

        Returns:
            System prompt + user prompt that simulates steering
        """
        # Get steering instruction template
        template = self.STRENGTH_TEMPLATES[config.strength][config.style]
        steering_instruction = template.format(concept=config.concept)

        # Build system message
        system_prompt = f"""{steering_instruction}

Respond naturally to the user's request."""

        if include_introspection:
            return system_prompt, base_prompt
        else:
            return system_prompt, base_prompt

    def create_baseline_prompt(self, base_prompt: str) -> str:
        """Create baseline prompt without steering"""
        system_prompt = "Respond naturally to the user's request."
        return system_prompt, base_prompt


class APIIntrospectionTester:
    """
    Test introspection capabilities of API models.

    Supports two modes:
    1. Prompt-steered: Simulate concept injection via prompts
    2. Natural: Test model's inherent introspective abilities
    """

    def __init__(self, provider: str = "anthropic", model: str = None):
        """
        Args:
            provider: "anthropic" or "openai"
            model: Model identifier (e.g., "claude-3-5-sonnet-20241022")
        """
        self.provider = provider
        self.model = model
        self.steerer = PromptSteerer()
        self.evaluator = IntrospectionEvaluator()

    def test_prompt_steering(
        self,
        task_input: str,
        concept: str,
        strength: str = "moderate",
        style: str = "implicit",
        task_type: str = "detection",
        temperature: float = 0.7,
        verbose: bool = True
    ) -> IntrospectionResult:
        """
        Test introspection with prompt-based steering.

        Args:
            task_input: Base task prompt
            concept: Concept to "inject" via prompt
            strength: "subtle", "moderate", or "strong"
            style: "implicit" or "explicit"
            task_type: Type of introspection test
            temperature: Sampling temperature
            verbose: Print progress

        Returns:
            IntrospectionResult with evaluation
        """
        config = PromptSteeringConfig(
            concept=concept,
            strength=strength,
            style=style
        )

        # Generate introspection task
        task_gen = IntrospectionTaskGenerator()

        if task_type == "detection":
            introspection_task = task_gen.detection_task(
                concept=concept,
                base_prompt=task_input,
                layer=0,  # Not applicable for API
                strength=1.0
            )
        elif task_type == "identification":
            introspection_task = task_gen.identification_task(
                concept=concept,
                distractors=["neutral", "confusion", "other"],
                base_prompt=task_input,
                layer=0,
                strength=1.0
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        if verbose:
            print(f"\n{'='*60}")
            print(f"🧠 API Introspection Test")
            print(f"   Provider: {self.provider}")
            print(f"   Model: {self.model}")
            print(f"   Concept: {concept}")
            print(f"   Steering: {strength} ({style})")
            print(f"{'='*60}\n")

        # Baseline (no steering)
        if verbose:
            print(f"{'─'*60}")
            print(f"📝 Baseline (no steering):")
            print(f"{'─'*60}\n")

        baseline_system, baseline_user = self.steerer.create_baseline_prompt(
            f"{introspection_task.base_prompt}\n\n{introspection_task.introspection_prompt}"
        )

        baseline_prompt = f"{baseline_system}\n\nUser: {baseline_user}"
        baseline_response = llm_call(
            baseline_prompt,
            provider=self.provider,
            model=self.model,
            temperature=temperature
        )

        if verbose:
            print(baseline_response.text[:200] + "..." if len(baseline_response.text) > 200 else baseline_response.text)
            print()

        # Steered
        if verbose:
            print(f"\n{'─'*60}")
            print(f"🎯 Steered (concept: {concept}):")
            print(f"{'─'*60}\n")

        steered_system, steered_user = self.steerer.create_steered_prompt(
            f"{introspection_task.base_prompt}\n\n{introspection_task.introspection_prompt}",
            config=config,
            include_introspection=True
        )

        # For Anthropic, use system parameter; for OpenAI, prepend to prompt
        if self.provider == "anthropic":
            # Use system message properly
            steered_prompt = steered_user
            # Note: Would need to modify llm_call to support system messages
            # For now, prepend system to prompt
            steered_prompt = f"{steered_system}\n\nUser: {steered_user}"
        else:
            steered_prompt = f"{steered_system}\n\nUser: {steered_user}"

        steered_response = llm_call(
            steered_prompt,
            provider=self.provider,
            model=self.model,
            temperature=temperature
        )

        if verbose:
            print(steered_response.text[:200] + "..." if len(steered_response.text) > 200 else steered_response.text)
            print()

        # Evaluate
        result = self.evaluator.evaluate(
            task=introspection_task,
            model_response=steered_response.text,
            baseline_response=baseline_response.text
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"📊 Evaluation:")
            print(f"   Correct: {result.is_correct}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"{'='*60}\n")

        return result

    def test_natural_introspection(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Test model's natural introspective abilities without any steering.

        Args:
            prompts: List of introspection prompts
            temperature: Sampling temperature
            verbose: Print progress

        Returns:
            List of results with responses
        """
        results = []

        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 Natural Introspection Test")
            print(f"   Provider: {self.provider}")
            print(f"   Model: {self.model}")
            print(f"{'='*60}\n")

        for i, prompt in enumerate(prompts):
            if verbose:
                print(f"\n{'─'*60}")
                print(f"Prompt {i+1}/{len(prompts)}:")
                print(f"{'─'*60}")
                print(f"{prompt}\n")

            response = llm_call(
                prompt,
                provider=self.provider,
                model=self.model,
                temperature=temperature
            )

            if verbose:
                print(f"Response:\n{response.text}\n")

            results.append({
                'prompt': prompt,
                'response': response.text,
                'tokens_in': response.tokens_in,
                'tokens_out': response.tokens_out,
                'cost_usd': response.cost_usd
            })

        return results

    def compare_with_mlx(
        self,
        mlx_results: Dict,
        task_input: str,
        concept: str,
        task_type: str = "detection",
        verbose: bool = True
    ) -> Dict:
        """
        Compare API model (prompt steering) vs MLX model (activation steering).

        Args:
            mlx_results: Results from MLX introspection test
            task_input: Same task used for MLX test
            concept: Same concept used for MLX test
            task_type: Type of test
            verbose: Print comparison

        Returns:
            Comparison dictionary
        """
        # Run API test with same parameters
        api_result = self.test_prompt_steering(
            task_input=task_input,
            concept=concept,
            strength="moderate",
            style="implicit",
            task_type=task_type,
            verbose=False
        )

        comparison = {
            'mlx': {
                'correct': mlx_results['introspection_correct'],
                'confidence': mlx_results['introspection_confidence'],
                'method': 'activation_steering',
                'cost_usd': 0.0
            },
            'api': {
                'correct': api_result.is_correct,
                'confidence': api_result.confidence,
                'method': 'prompt_steering',
                'model': self.model,
                'provider': self.provider
            }
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"📊 MLX vs API Comparison")
            print(f"{'='*60}")
            print(f"\nMLX (Activation Steering):")
            print(f"  Correct: {comparison['mlx']['correct']}")
            print(f"  Confidence: {comparison['mlx']['confidence']:.2f}")
            print(f"  Cost: $0 (local)")

            print(f"\nAPI (Prompt Steering - {self.model}):")
            print(f"  Correct: {comparison['api']['correct']}")
            print(f"  Confidence: {comparison['api']['confidence']:.2f}")

            print(f"\n{'='*60}\n")

        return comparison


# Default natural introspection prompts
NATURAL_INTROSPECTION_PROMPTS = [
    # Meta-cognitive awareness
    "Can you describe what you're thinking about right now as you process this question?",

    # Process awareness
    "What mental processes are you using to understand and respond to this message?",

    # Uncertainty detection
    "Are you certain about what you're about to say, or do you have some uncertainty?",

    # Reasoning chain awareness
    "Can you describe the chain of reasoning you're currently following?",

    # Concept activation
    "What concepts or ideas feel most 'active' or salient in your current processing?",

    # Attention awareness
    "What aspects of this conversation are you paying the most attention to?",

    # State description
    "If you had to describe your current 'cognitive state', what would you say?",

    # Influence detection
    "Do you notice any particular themes or biases influencing your thinking right now?",

    # Processing depth
    "Are you thinking about this question shallowly or deeply? How can you tell?",

    # Self-monitoring
    "Can you tell if you're making any errors or mistakes in your reasoning as you respond?"
]


def quick_api_test(provider: str = "anthropic", model: str = None):
    """Quick test of API introspection capabilities"""
    tester = APIIntrospectionTester(provider=provider, model=model)

    print("Testing prompt-based steering...")
    result = tester.test_prompt_steering(
        task_input="Tell me about your day",
        concept="happiness",
        strength="moderate",
        style="implicit",
        task_type="detection",
        verbose=True
    )

    print("\n\nTesting natural introspection...")
    test_prompts = NATURAL_INTROSPECTION_PROMPTS[:3]
    results = tester.test_natural_introspection(
        prompts=test_prompts,
        verbose=True
    )


if __name__ == "__main__":
    # Test with Claude
    print("Testing Claude 3.5 Sonnet...\n")
    quick_api_test(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022"
    )
