"""
Multi-agent strategies for the simulation harness.
Each strategy is a function that takes task input and returns output + metadata.
"""
from typing import Dict, Any, List
from dataclasses import dataclass
from collections import Counter
import time
import re

from .llm_provider import llm_call, LLMResponse


def _extract_answer(text: str) -> str:
    """
    Try to extract the core answer from a response.
    Handles common patterns like "The answer is X" or numbers.
    """
    text = text.strip()

    # Try to find "answer is X" patterns
    patterns = [
        r'(?:answer|result|solution) is:?\s*([^\n.]+)',
        r'(?:therefore|thus|so),?\s*(?:the answer is)?\s*([^\n.]+)',
        r'^\s*([0-9]+(?:\.[0-9]+)?)\s*$',  # Just a number
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # If no pattern found, return first sentence or first 100 chars
    sentences = text.split('.')
    if sentences:
        return sentences[0].strip()

    return text[:100].strip()


def _aggregate_samples(samples: List[str]) -> str:
    """
    Aggregate multiple samples using majority voting.

    Extracts answers, finds most common one, and returns
    the full response that contained that answer.
    """
    if not samples:
        return ""

    if len(samples) == 1:
        return samples[0]

    # Extract answers from each sample
    answers = [_extract_answer(s) for s in samples]

    # Normalize for comparison (lowercase, strip whitespace)
    normalized = [a.lower().strip() for a in answers]

    # Find most common answer
    counter = Counter(normalized)
    most_common_normalized, count = counter.most_common(1)[0]

    # Find the original sample that had this answer
    for i, norm in enumerate(normalized):
        if norm == most_common_normalized:
            return samples[i]

    # Fallback: return first sample
    return samples[0]


@dataclass
class StrategyResult:
    """Result from a strategy execution"""
    output: str
    strategy_name: str
    latency_s: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    metadata: Dict[str, Any]
    
    @classmethod
    def from_llm_response(cls, response: LLMResponse, strategy_name: str, metadata: Dict[str, Any] = None):
        """Create StrategyResult from LLMResponse"""
        return cls(
            output=response.text,
            strategy_name=strategy_name,
            latency_s=response.latency_s,
            tokens_in=response.tokens_in or 0,
            tokens_out=response.tokens_out or 0,
            cost_usd=response.cost_usd or 0.0,
            metadata=metadata or {}
        )


def single_model_strategy(
    task_input: str,
    provider: str = "ollama",
    model: str = "llama3.2:latest",
    temperature: float = 0.7,
    system_prompt: str = None,
    **kwargs
) -> StrategyResult:
    """
    Single model baseline.
    
    Args:
        task_input: The task/question
        provider: LLM provider
        model: Model identifier
        temperature: Sampling temperature
        system_prompt: Optional system prompt
    """
    start = time.time()
    
    # Construct prompt
    if system_prompt:
        prompt = f"{system_prompt}\n\n{task_input}"
    else:
        prompt = task_input
    
    response = llm_call(prompt, provider=provider, model=model, temperature=temperature, **kwargs)
    
    return StrategyResult.from_llm_response(
        response,
        strategy_name="single",
        metadata={"provider": provider, "model": model}
    )


def debate_strategy(
    task_input: str,
    n_debaters: int = 2,
    n_rounds: int = 1,
    provider: str = "ollama",
    model: str = "llama3.2:latest",
    judge_provider: str = None,
    judge_model: str = None,
    temperature: float = 0.7,
    **kwargs
) -> StrategyResult:
    """
    Multi-agent debate strategy.
    
    n_debaters agents each provide an answer, then a judge selects the best.
    Optionally run multiple rounds of debate.
    
    Args:
        task_input: The task/question
        n_debaters: Number of debating agents
        n_rounds: Number of debate rounds
        provider/model: For debaters
        judge_provider/judge_model: For judge (defaults to same as debaters)
        temperature: Sampling temperature
    """
    start = time.time()
    
    if judge_provider is None:
        judge_provider = provider
    if judge_model is None:
        judge_model = model
    
    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0
    
    # Initial arguments
    arguments = []
    for i in range(n_debaters):
        debater_prompt = f"""You are Debater {i+1} in a debate. Provide your best answer to the following question.

Question: {task_input}

Your answer:"""
        
        response = llm_call(debater_prompt, provider=provider, model=model, temperature=temperature, **kwargs)
        arguments.append(response.text)
        
        total_tokens_in += response.tokens_in or 0
        total_tokens_out += response.tokens_out or 0
        total_cost += response.cost_usd or 0.0
    
    # Optional: Multiple rounds (each debater sees others' arguments)
    for round_num in range(1, n_rounds):
        new_arguments = []
        for i in range(n_debaters):
            # Show other arguments
            other_args = "\n\n".join([
                f"Debater {j+1}: {arg}" 
                for j, arg in enumerate(arguments) if j != i
            ])
            
            rebuttal_prompt = f"""You are Debater {i+1}. You've seen other debaters' arguments. Refine or defend your position.

Question: {task_input}

Other arguments:
{other_args}

Your argument:
{arguments[i]}

Your refined answer:"""
            
            response = llm_call(rebuttal_prompt, provider=provider, model=model, temperature=temperature, **kwargs)
            new_arguments.append(response.text)
            
            total_tokens_in += response.tokens_in or 0
            total_tokens_out += response.tokens_out or 0
            total_cost += response.cost_usd or 0.0
        
        arguments = new_arguments
    
    # Judge selects best
    judge_prompt = f"""You are a judge evaluating {n_debaters} answers to a question. Choose the best answer and explain why.

Question: {task_input}

Answers:
"""
    for i, arg in enumerate(arguments):
        judge_prompt += f"\nAnswer {i+1}:\n{arg}\n"
    
    judge_prompt += "\nWhich answer is best? Respond with: 'Answer X is best because...'"
    
    judge_response = llm_call(judge_prompt, provider=judge_provider, model=judge_model, temperature=0.3, **kwargs)
    
    total_tokens_in += judge_response.tokens_in or 0
    total_tokens_out += judge_response.tokens_out or 0
    total_cost += judge_response.cost_usd or 0.0
    
    latency = time.time() - start
    
    return StrategyResult(
        output=judge_response.text,
        strategy_name="debate",
        latency_s=latency,
        tokens_in=total_tokens_in,
        tokens_out=total_tokens_out,
        cost_usd=total_cost,
        metadata={
            "n_debaters": n_debaters,
            "n_rounds": n_rounds,
            "arguments": arguments,
            "provider": provider,
            "model": model,
            "judge_provider": judge_provider,
            "judge_model": judge_model
        }
    )


def self_consistency_strategy(
    task_input: str,
    n_samples: int = 5,
    provider: str = "ollama",
    model: str = "llama3.2:latest",
    temperature: float = 0.8,
    **kwargs
) -> StrategyResult:
    """
    Self-consistency strategy: sample N times and take majority vote or aggregate.
    
    Good for reasoning tasks with clear right/wrong answers.
    
    Args:
        task_input: The task/question
        n_samples: Number of samples to generate
        provider/model: LLM config
        temperature: Higher temperature for diversity
    """
    start = time.time()
    
    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0
    
    samples = []
    for i in range(n_samples):
        response = llm_call(task_input, provider=provider, model=model, temperature=temperature, **kwargs)
        samples.append(response.text)
        
        total_tokens_in += response.tokens_in or 0
        total_tokens_out += response.tokens_out or 0
        total_cost += response.cost_usd or 0.0
    
    # Aggregate: majority vote on answers
    # Try to extract answers and find most common one
    aggregated_output = _aggregate_samples(samples)
    
    latency = time.time() - start
    
    return StrategyResult(
        output=aggregated_output,
        strategy_name="self_consistency",
        latency_s=latency,
        tokens_in=total_tokens_in,
        tokens_out=total_tokens_out,
        cost_usd=total_cost,
        metadata={
            "n_samples": n_samples,
            "all_samples": samples,
            "provider": provider,
            "model": model
        }
    )


def manager_worker_strategy(
    task_input: str,
    n_workers: int = 3,
    provider: str = "ollama",
    model: str = "llama3.2:latest",
    manager_provider: str = None,
    manager_model: str = None,
    temperature: float = 0.7,
    **kwargs
) -> StrategyResult:
    """
    Manager-worker strategy: Manager decomposes task, workers execute subtasks, manager synthesizes.
    
    Args:
        task_input: The task/question
        n_workers: Number of worker agents
        provider/model: For workers
        manager_provider/manager_model: For manager (defaults to same)
        temperature: Sampling temperature
    """
    start = time.time()
    
    if manager_provider is None:
        manager_provider = provider
    if manager_model is None:
        manager_model = model
    
    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0
    
    # Manager: decompose task
    decompose_prompt = f"""You are a task manager. Break down the following task into {n_workers} subtasks that can be worked on independently.

Task: {task_input}

List {n_workers} subtasks (numbered):"""
    
    manager_response = llm_call(decompose_prompt, provider=manager_provider, model=manager_model, temperature=0.3, **kwargs)
    subtasks_text = manager_response.text
    
    total_tokens_in += manager_response.tokens_in or 0
    total_tokens_out += manager_response.tokens_out or 0
    total_cost += manager_response.cost_usd or 0.0
    
    # Parse subtasks (simple line-based parsing)
    subtasks = [line.strip() for line in subtasks_text.split('\n') if line.strip()][:n_workers]
    
    # Workers: execute subtasks
    worker_results = []
    for i, subtask in enumerate(subtasks):
        worker_prompt = f"""You are Worker {i+1}. Complete this subtask:

{subtask}

Your result:"""
        
        worker_response = llm_call(worker_prompt, provider=provider, model=model, temperature=temperature, **kwargs)
        worker_results.append(worker_response.text)
        
        total_tokens_in += worker_response.tokens_in or 0
        total_tokens_out += worker_response.tokens_out or 0
        total_cost += worker_response.cost_usd or 0.0
    
    # Manager: synthesize
    synthesize_prompt = f"""You are a task manager. Synthesize the workers' results into a final answer.

Original task: {task_input}

Worker results:
"""
    for i, result in enumerate(worker_results):
        synthesize_prompt += f"\nWorker {i+1}: {result}\n"
    
    synthesize_prompt += "\nFinal synthesized answer:"
    
    final_response = llm_call(synthesize_prompt, provider=manager_provider, model=manager_model, temperature=0.3, **kwargs)
    
    total_tokens_in += final_response.tokens_in or 0
    total_tokens_out += final_response.tokens_out or 0
    total_cost += final_response.cost_usd or 0.0
    
    latency = time.time() - start
    
    return StrategyResult(
        output=final_response.text,
        strategy_name="manager_worker",
        latency_s=latency,
        tokens_in=total_tokens_in,
        tokens_out=total_tokens_out,
        cost_usd=total_cost,
        metadata={
            "n_workers": n_workers,
            "subtasks": subtasks,
            "worker_results": worker_results,
            "provider": provider,
            "model": model,
            "manager_provider": manager_provider,
            "manager_model": manager_model
        }
    )


# Registry of strategies
STRATEGIES = {
    "single": single_model_strategy,
    "debate": debate_strategy,
    "self_consistency": self_consistency_strategy,
    "manager_worker": manager_worker_strategy,
}


def run_strategy(strategy_name: str, task_input: str, **kwargs) -> StrategyResult:
    """
    Run a strategy by name.
    
    Args:
        strategy_name: One of: "single", "debate", "self_consistency", "manager_worker"
        task_input: The task/question
        **kwargs: Strategy-specific parameters
    """
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGIES.keys())}")
    
    return STRATEGIES[strategy_name](task_input, **kwargs)
