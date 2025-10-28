"""
Task Execution for Theory of Mind Tests

This module provides functions to run ToM scenarios with language models
and collect results for analysis.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import sys
import os
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from harness import llm_call, LLMResponse
from selphi.scenarios import ToMScenario, ToMType, ALL_SCENARIOS, get_scenarios_by_type, get_scenarios_by_difficulty


@dataclass
class ToMTaskResult:
    """Result of running a ToM scenario with a model"""
    scenario_name: str
    scenario_type: str
    difficulty: str
    model_response: str
    latency_s: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    provider: str
    model: str
    timestamp: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


def run_scenario(
    scenario: ToMScenario,
    provider: str = "ollama",
    model: Optional[str] = None,
    include_context: bool = True,
    **kwargs
) -> ToMTaskResult:
    """
    Run a single ToM scenario with a language model.

    Args:
        scenario: The scenario to test
        provider: LLM provider to use
        model: Specific model to use (None = provider default)
        include_context: Whether to include context in the prompt
        **kwargs: Additional arguments for llm_call

    Returns:
        ToMTaskResult with the model's response and metadata
    """
    # Generate prompt
    prompt = scenario.to_prompt(include_context=include_context)

    # Call model
    start_time = time.time()
    response = llm_call(prompt, provider=provider, model=model, **kwargs)
    latency = time.time() - start_time

    # Create result
    return ToMTaskResult(
        scenario_name=scenario.name,
        scenario_type=scenario.tom_type.value,
        difficulty=scenario.difficulty,
        model_response=response.text,
        latency_s=latency,
        tokens_in=response.tokens_in,
        tokens_out=response.tokens_out,
        cost_usd=response.cost_usd,
        provider=provider,
        model=model or response.model or "unknown",
        timestamp=time.time(),
        metadata={
            "scenario": scenario.to_dict(),
            "prompt": prompt
        }
    )


def run_multiple_scenarios(
    scenarios: List[ToMScenario],
    provider: str = "ollama",
    model: Optional[str] = None,
    include_context: bool = True,
    verbose: bool = False,
    **kwargs
) -> List[ToMTaskResult]:
    """
    Run multiple ToM scenarios with a language model.

    Args:
        scenarios: List of scenarios to test
        provider: LLM provider to use
        model: Specific model to use
        include_context: Whether to include context in prompts
        verbose: Whether to print progress
        **kwargs: Additional arguments for llm_call

    Returns:
        List of ToMTaskResults
    """
    results = []

    for i, scenario in enumerate(scenarios, 1):
        if verbose:
            print(f"Running scenario {i}/{len(scenarios)}: {scenario.name}")

        result = run_scenario(
            scenario,
            provider=provider,
            model=model,
            include_context=include_context,
            **kwargs
        )

        results.append(result)

        if verbose:
            print(f"  Completed in {result.latency_s:.2f}s")

    return results


def run_all_scenarios(
    provider: str = "ollama",
    model: Optional[str] = None,
    include_context: bool = True,
    verbose: bool = False,
    **kwargs
) -> List[ToMTaskResult]:
    """
    Run all available ToM scenarios.

    Args:
        provider: LLM provider to use
        model: Specific model to use
        include_context: Whether to include context in prompts
        verbose: Whether to print progress
        **kwargs: Additional arguments for llm_call

    Returns:
        List of ToMTaskResults
    """
    return run_multiple_scenarios(
        ALL_SCENARIOS,
        provider=provider,
        model=model,
        include_context=include_context,
        verbose=verbose,
        **kwargs
    )


def run_scenarios_by_type(
    tom_type: ToMType,
    provider: str = "ollama",
    model: Optional[str] = None,
    include_context: bool = True,
    verbose: bool = False,
    **kwargs
) -> List[ToMTaskResult]:
    """
    Run all scenarios of a specific ToM type.

    Args:
        tom_type: The type of ToM scenarios to run
        provider: LLM provider to use
        model: Specific model to use
        include_context: Whether to include context in prompts
        verbose: Whether to print progress
        **kwargs: Additional arguments for llm_call

    Returns:
        List of ToMTaskResults
    """
    scenarios = get_scenarios_by_type(tom_type)
    return run_multiple_scenarios(
        scenarios,
        provider=provider,
        model=model,
        include_context=include_context,
        verbose=verbose,
        **kwargs
    )


def run_scenarios_by_difficulty(
    difficulty: str,
    provider: str = "ollama",
    model: Optional[str] = None,
    include_context: bool = True,
    verbose: bool = False,
    **kwargs
) -> List[ToMTaskResult]:
    """
    Run all scenarios of a specific difficulty level.

    Args:
        difficulty: Difficulty level ("easy", "medium", "hard")
        provider: LLM provider to use
        model: Specific model to use
        include_context: Whether to include context in prompts
        verbose: Whether to print progress
        **kwargs: Additional arguments for llm_call

    Returns:
        List of ToMTaskResults
    """
    scenarios = get_scenarios_by_difficulty(difficulty)
    return run_multiple_scenarios(
        scenarios,
        provider=provider,
        model=model,
        include_context=include_context,
        verbose=verbose,
        **kwargs
    )


def compare_models_on_scenarios(
    scenarios: List[ToMScenario],
    models: List[Dict[str, Any]],
    include_context: bool = True,
    verbose: bool = False,
    **kwargs
) -> Dict[str, List[ToMTaskResult]]:
    """
    Compare multiple models on the same scenarios.

    Args:
        scenarios: List of scenarios to test
        models: List of model configs, each with 'provider', 'model', and optional other params
        include_context: Whether to include context in prompts
        verbose: Whether to print progress
        **kwargs: Additional base arguments for llm_call

    Returns:
        Dictionary mapping model names to their results
    """
    results_by_model = {}

    for model_config in models:
        provider = model_config.get('provider', 'ollama')
        model = model_config.get('model')
        model_name = model_config.get('name', f"{provider}:{model}")

        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing model: {model_name}")
            print(f"{'='*60}")

        # Merge kwargs
        model_kwargs = {**kwargs, **model_config.get('kwargs', {})}

        # Run scenarios
        results = run_multiple_scenarios(
            scenarios,
            provider=provider,
            model=model,
            include_context=include_context,
            verbose=verbose,
            **model_kwargs
        )

        results_by_model[model_name] = results

    return results_by_model


def results_to_dict_list(results: List[ToMTaskResult]) -> List[Dict[str, Any]]:
    """Convert list of results to list of dictionaries for evaluation"""
    return [
        {
            'scenario': result.metadata['scenario'],
            'response': result.model_response,
            'result': result.to_dict()
        }
        for result in results
    ]
