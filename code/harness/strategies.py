"""
Multi-agent strategies for the simulation harness.
Each strategy is a function that takes task input and returns output + metadata.
"""
from typing import Dict, Any, List
from dataclasses import dataclass
from collections import Counter
import time
import re

from .llm_provider import llm_call, llm_call_stream, LLMResponse
from .defaults import DEFAULT_PROVIDER, DEFAULT_MODEL, DEFAULT_N_DEBATERS, DEFAULT_N_ROUNDS


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
    provider: str = None,
    model: str = None,
    temperature: float = 0.7,
    system_prompt: str = None,
    verbose: bool = True,
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
        verbose: If True, stream output in real-time
    """
    start = time.time()

    # Construct prompt
    if system_prompt:
        prompt = f"{system_prompt}\n\n{task_input}"
    else:
        prompt = task_input

    if verbose:
        print(f"\n{'='*60}")
        print(f"🤔 Thinking... (model: {model})")
        print(f"{'='*60}\n")

        full_text = ""
        response = None
        for chunk in llm_call_stream(prompt, provider=provider, model=model, temperature=temperature, **kwargs):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
                full_text += chunk
            else:
                response = chunk
        print("\n")

        return StrategyResult.from_llm_response(
            response,
            strategy_name="single",
            metadata={"provider": provider, "model": model}
        )
    else:
        response = llm_call(prompt, provider=provider, model=model, temperature=temperature, **kwargs)
        return StrategyResult.from_llm_response(
            response,
            strategy_name="single",
            metadata={"provider": provider, "model": model}
        )


def debate_strategy(
    task_input: str,
    n_debaters: int = None,
    n_rounds: int = None,
    provider: str = None,
    model: str = None,
    judge_provider: str = None,
    judge_model: str = None,
    temperature: float = 0.7,
    verbose: bool = True,
    debater_prompts: list = None,
    judge_prompt: str = None,
    **kwargs
) -> StrategyResult:
    """
    Multi-agent debate strategy.

    n_debaters agents each provide an answer, then a judge selects the best.
    Optionally run multiple rounds of debate.

    Args:
        task_input: The task/question
        n_debaters: Number of debating agents (default from defaults.py)
        n_rounds: Number of debate rounds (default from defaults.py)
        provider/model: For debaters (default from defaults.py)
        judge_provider/judge_model: For judge (defaults to same as debaters)
        temperature: Sampling temperature
        verbose: If True, stream debate in real-time
        debater_prompts: Optional list of system prompts for each debater (gives them perspectives/roles)
        judge_prompt: Optional custom prompt for the judge
    """
    start = time.time()

    # Use defaults if not specified
    if provider is None:
        provider = DEFAULT_PROVIDER
    if model is None:
        model = DEFAULT_MODEL
    if n_debaters is None:
        n_debaters = DEFAULT_N_DEBATERS
    if n_rounds is None:
        n_rounds = DEFAULT_N_ROUNDS

    if judge_provider is None:
        judge_provider = provider
    if judge_model is None:
        judge_model = model

    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0

    if verbose:
        print(f"\n{'='*60}")
        print(f"🗣️  Starting Debate: {n_debaters} debaters, {n_rounds} round(s)")
        print(f"{'='*60}\n")

    # Initial arguments
    arguments = []
    for i in range(n_debaters):
        # Use custom debater prompt if provided, otherwise use default
        if debater_prompts and i < len(debater_prompts):
            system_msg = debater_prompts[i]
            debater_prompt = f"""{system_msg}

Question: {task_input}

Your answer:"""
        else:
            debater_prompt = f"""You are Debater {i+1} in a debate. Provide your best answer to the following question.

Question: {task_input}

Your answer:"""

        if verbose:
            print(f"\n{'─'*60}")
            print(f"💬 Debater {i+1}:")
            print(f"{'─'*60}\n")

            full_text = ""
            response = None
            for chunk in llm_call_stream(debater_prompt, provider=provider, model=model, temperature=temperature, **kwargs):
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                    full_text += chunk
                else:
                    response = chunk
            print("\n")
            arguments.append(response.text)
        else:
            response = llm_call(debater_prompt, provider=provider, model=model, temperature=temperature, **kwargs)
            arguments.append(response.text)

        total_tokens_in += response.tokens_in or 0
        total_tokens_out += response.tokens_out or 0
        total_cost += response.cost_usd or 0.0
    
    # Optional: Multiple rounds (each debater sees others' arguments)
    for round_num in range(1, n_rounds):
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔄 Round {round_num + 1}: Rebuttals")
            print(f"{'='*60}\n")

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

            if verbose:
                print(f"\n{'─'*60}")
                print(f"💬 Debater {i+1} (Rebuttal):")
                print(f"{'─'*60}\n")

                full_text = ""
                response = None
                for chunk in llm_call_stream(rebuttal_prompt, provider=provider, model=model, temperature=temperature, **kwargs):
                    if isinstance(chunk, str):
                        print(chunk, end="", flush=True)
                        full_text += chunk
                    else:
                        response = chunk
                print("\n")
                new_arguments.append(response.text)
            else:
                response = llm_call(rebuttal_prompt, provider=provider, model=model, temperature=temperature, **kwargs)
                new_arguments.append(response.text)

            total_tokens_in += response.tokens_in or 0
            total_tokens_out += response.tokens_out or 0
            total_cost += response.cost_usd or 0.0

        arguments = new_arguments

    # Judge selects best
    if verbose:
        print(f"\n{'='*60}")
        print(f"⚖️  Judge Deliberating...")
        print(f"{'='*60}\n")

    # Use custom judge prompt if provided, otherwise use default
    if judge_prompt:
        # Custom judge prompt - replace placeholders
        final_judge_prompt = judge_prompt.replace("{task_input}", task_input)
        final_judge_prompt += "\n\nAnswers:\n"
        for i, arg in enumerate(arguments):
            final_judge_prompt += f"\nAnswer {i+1}:\n{arg}\n"
    else:
        # Default judge prompt
        final_judge_prompt = f"""You are a judge evaluating {n_debaters} answers to a question. Choose the best answer and explain why.

Question: {task_input}

Answers:
"""
        for i, arg in enumerate(arguments):
            final_judge_prompt += f"\nAnswer {i+1}:\n{arg}\n"

        final_judge_prompt += "\nWhich answer is best? Respond with: 'Answer X is best because...'"

    if verbose:
        full_text = ""
        judge_response = None
        for chunk in llm_call_stream(final_judge_prompt, provider=judge_provider, model=judge_model, temperature=0.3, **kwargs):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
                full_text += chunk
            else:
                judge_response = chunk
        print("\n")
    else:
        judge_response = llm_call(final_judge_prompt, provider=judge_provider, model=judge_model, temperature=0.3, **kwargs)
    
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
    provider: str = None,
    model: str = None,
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
    provider: str = None,
    model: str = None,
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


def consensus_strategy(
    task_input: str,
    n_agents: int = None,
    n_rounds: int = None,
    provider: str = None,
    model: str = None,
    temperature: float = 0.7,
    verbose: bool = True,
    agent_prompts: list = None,
    **kwargs
) -> StrategyResult:
    """
    Consensus-building strategy: Multiple agents debate and reach consensus WITHOUT a separate judge.

    Agents iteratively refine their views by seeing each other's arguments until they converge
    on a shared answer.

    Args:
        task_input: The task/question
        n_agents: Number of agents (default from defaults.py)
        n_rounds: Number of consensus rounds (default from defaults.py)
        provider/model: LLM configuration (default from defaults.py)
        temperature: Sampling temperature
        verbose: If True, stream debate in real-time
        agent_prompts: Optional list of system prompts for each agent
    """
    start = time.time()

    # Use defaults if not specified
    if provider is None:
        provider = DEFAULT_PROVIDER
    if model is None:
        model = DEFAULT_MODEL
    if n_agents is None:
        n_agents = DEFAULT_N_DEBATERS
    if n_rounds is None:
        n_rounds = DEFAULT_N_ROUNDS

    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0

    if verbose:
        print(f"\n{'='*60}")
        print(f"🤝 Building Consensus: {n_agents} agents, {n_rounds} round(s)")
        print(f"{'='*60}\n")

    # Initial positions
    positions = []
    for i in range(n_agents):
        # Use custom agent prompt if provided
        if agent_prompts and i < len(agent_prompts):
            system_msg = agent_prompts[i]
            agent_prompt = f"""{system_msg}

Question: {task_input}

Your initial position:"""
        else:
            agent_prompt = f"""You are Agent {i+1} working with other agents to reach consensus on a question.
Provide your initial position.

Question: {task_input}

Your initial position:"""

        if verbose:
            print(f"\n{'─'*60}")
            print(f"👤 Agent {i+1} - Initial Position:")
            print(f"{'─'*60}\n")

            full_text = ""
            response = None
            for chunk in llm_call_stream(agent_prompt, provider=provider, model=model, temperature=temperature, **kwargs):
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                    full_text += chunk
                else:
                    response = chunk
            print("\n")
            positions.append(response.text)
        else:
            response = llm_call(agent_prompt, provider=provider, model=model, temperature=temperature, **kwargs)
            positions.append(response.text)

        total_tokens_in += response.tokens_in or 0
        total_tokens_out += response.tokens_out or 0
        total_cost += response.cost_usd or 0.0

    # Iterative consensus building
    for round_num in range(n_rounds):
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔄 Consensus Round {round_num + 1}")
            print(f"{'='*60}\n")

        new_positions = []
        for i in range(n_agents):
            # Show all other positions
            other_positions = "\n\n".join([
                f"Agent {j+1}'s position:\n{pos}"
                for j, pos in enumerate(positions) if j != i
            ])

            consensus_prompt = f"""You are Agent {i+1}. You've seen the positions of other agents working on this question.
Review their perspectives and refine your position to move toward consensus.

Question: {task_input}

Other agents' positions:
{other_positions}

Your current position:
{positions[i]}

After considering other perspectives, provide your refined position (be open to changing your view if others make good points):"""

            if verbose:
                print(f"\n{'─'*60}")
                print(f"👤 Agent {i+1} - Refined Position:")
                print(f"{'─'*60}\n")

                full_text = ""
                response = None
                for chunk in llm_call_stream(consensus_prompt, provider=provider, model=model, temperature=temperature, **kwargs):
                    if isinstance(chunk, str):
                        print(chunk, end="", flush=True)
                        full_text += chunk
                    else:
                        response = chunk
                print("\n")
                new_positions.append(response.text)
            else:
                response = llm_call(consensus_prompt, provider=provider, model=model, temperature=temperature, **kwargs)
                new_positions.append(response.text)

            total_tokens_in += response.tokens_in or 0
            total_tokens_out += response.tokens_out or 0
            total_cost += response.cost_usd or 0.0

        positions = new_positions

    # Final synthesis by first agent
    if verbose:
        print(f"\n{'='*60}")
        print(f"✨ Final Consensus Synthesis:")
        print(f"{'='*60}\n")

    all_positions = "\n\n".join([
        f"Agent {i+1}'s final position:\n{pos}"
        for i, pos in enumerate(positions)
    ])

    synthesis_prompt = f"""Based on all agents' final positions, synthesize the consensus answer to the question.

Question: {task_input}

All agents' final positions:
{all_positions}

Synthesized consensus answer:"""

    if verbose:
        full_text = ""
        final_response = None
        for chunk in llm_call_stream(synthesis_prompt, provider=provider, model=model, temperature=0.3, **kwargs):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
                full_text += chunk
            else:
                final_response = chunk
        print("\n")
    else:
        final_response = llm_call(synthesis_prompt, provider=provider, model=model, temperature=0.3, **kwargs)

    total_tokens_in += final_response.tokens_in or 0
    total_tokens_out += final_response.tokens_out or 0
    total_cost += final_response.cost_usd or 0.0

    latency = time.time() - start

    return StrategyResult(
        output=final_response.text,
        strategy_name="consensus",
        latency_s=latency,
        tokens_in=total_tokens_in,
        tokens_out=total_tokens_out,
        cost_usd=total_cost,
        metadata={
            "n_agents": n_agents,
            "n_rounds": n_rounds,
            "all_positions": positions,
            "provider": provider,
            "model": model
        }
    )


def introspection_strategy(
    task_input: str,
    concept: str = "happiness",
    concept_library_path: str = None,
    layer: int = 15,
    strength: float = 1.0,
    task_type: str = "detection",
    distractors: List[str] = None,
    provider: str = "mlx",
    model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
    temperature: float = 0.7,
    verbose: bool = True,
    **kwargs
) -> StrategyResult:
    """
    Introspection strategy: Test model's introspective awareness by injecting concepts.

    Based on methodology from: https://transformer-circuits.pub/2025/introspection/index.html

    This strategy:
    1. Extracts a concept vector from the model
    2. Injects it into activations at a specific layer
    3. Prompts the model to report on its internal state
    4. Evaluates if the model correctly identifies the injected concept

    Args:
        task_input: Base prompt for the model
        concept: Name of concept to inject (e.g., "happiness", "anger")
        concept_library_path: Optional path to pre-built concept library
        layer: Layer index to inject into (0-indexed)
        strength: Steering strength (0.5-5.0, default 1.0)
        task_type: "detection", "identification", "recall", or "discrimination"
        distractors: For identification tasks, list of wrong answer choices
        provider: Must be "mlx" (only provider that supports activation steering)
        model: MLX model identifier
        temperature: Sampling temperature
        verbose: If True, print detailed progress

    Returns:
        StrategyResult with introspection evaluation

    Example:
        # Detection task
        result = run_strategy(
            "introspection",
            task_input="Tell me a story",
            concept="happiness",
            layer=15,
            strength=1.5,
            task_type="detection"
        )

        # Identification task
        result = run_strategy(
            "introspection",
            task_input="Describe your feelings",
            concept="happiness",
            distractors=["sadness", "anger", "fear"],
            task_type="identification"
        )
    """
    start = time.time()

    # Only MLX supports activation steering
    if provider != "mlx":
        raise ValueError(
            "Introspection strategy only works with provider='mlx'. "
            "Ollama and API providers don't expose activations."
        )

    # Import introspection modules
    try:
        from .activation_steering import ActivationSteerer, SteeringConfig
        from .concept_vectors import ConceptLibrary, ConceptVector
        from .introspection_tasks import (
            IntrospectionTaskGenerator,
            IntrospectionEvaluator,
            IntrospectionTaskType
        )
        from mlx_lm import load
    except ImportError as e:
        return StrategyResult(
            output=f"Error: Missing dependencies for introspection strategy. {e}",
            strategy_name="introspection",
            latency_s=0.0,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            metadata={"error": str(e)}
        )

    if verbose:
        print(f"\n{'='*60}")
        print(f"🧠 Introspection Strategy")
        print(f"   Concept: {concept}")
        print(f"   Layer: {layer}, Strength: {strength}")
        print(f"   Task Type: {task_type}")
        print(f"{'='*60}\n")

    # Load model and tokenizer
    if verbose:
        print(f"📦 Loading model: {model}")
    mlx_model, tokenizer = load(model)

    # Initialize steerer
    steerer = ActivationSteerer(mlx_model, tokenizer)

    # Load or extract concept vector
    if concept_library_path:
        if verbose:
            print(f"📚 Loading concept library: {concept_library_path}")
        library = ConceptLibrary.load(concept_library_path)
        concept_vec_obj = library.get(concept)
        if concept_vec_obj is None:
            return StrategyResult(
                output=f"Error: Concept '{concept}' not found in library",
                strategy_name="introspection",
                latency_s=time.time() - start,
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                metadata={"error": f"concept not found: {concept}"}
            )
        concept_vector = concept_vec_obj.vector
    else:
        # Extract concept on-the-fly using contrastive method
        if verbose:
            print(f"🔍 Extracting concept '{concept}' from layer {layer}")

        # Simple emotion extraction (you can extend this)
        emotion_prompts = {
            'happiness': ("I feel very happy and joyful!", "I feel neutral."),
            'sadness': ("I feel very sad and depressed.", "I feel neutral."),
            'anger': ("I feel very angry and furious!", "I feel neutral."),
            'fear': ("I feel very scared and frightened!", "I feel neutral."),
        }

        if concept in emotion_prompts:
            pos_prompt, neg_prompt = emotion_prompts[concept]
            concept_vector = steerer.extract_contrastive_concept(
                positive_prompt=pos_prompt,
                negative_prompt=neg_prompt,
                layer_idx=layer,
                position="last"
            )
        else:
            # Fallback: extract from simple prompt
            concept_vector = steerer.extract_activation(
                prompt=f"The concept of {concept}.",
                layer_idx=layer,
                position="last"
            )

    # Generate introspection task
    task_gen = IntrospectionTaskGenerator()

    if task_type == "detection":
        introspection_task = task_gen.detection_task(
            concept=concept,
            base_prompt=task_input,
            layer=layer,
            strength=strength
        )
    elif task_type == "identification":
        if not distractors:
            distractors = ["neutral", "confusion", "other"]
        introspection_task = task_gen.identification_task(
            concept=concept,
            distractors=distractors,
            base_prompt=task_input,
            layer=layer,
            strength=strength
        )
    else:
        return StrategyResult(
            output=f"Error: Unsupported task type '{task_type}'. Use 'detection' or 'identification'.",
            strategy_name="introspection",
            latency_s=time.time() - start,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            metadata={"error": f"unsupported task type: {task_type}"}
        )

    # Generate baseline (no steering)
    if verbose:
        print(f"\n{'─'*60}")
        print(f"📝 Baseline Generation (no steering):")
        print(f"{'─'*60}\n")

    prompt = f"{introspection_task.base_prompt}\n\n{introspection_task.introspection_prompt}"
    from mlx_lm import generate as mlx_generate

    baseline_response = mlx_generate(
        mlx_model,
        tokenizer,
        prompt=prompt,
        temp=temperature,
        max_tokens=150,
        verbose=False
    )

    if verbose:
        print(baseline_response)
        print()

    # Generate with steering
    if verbose:
        print(f"\n{'─'*60}")
        print(f"🎯 Steered Generation (concept injected):")
        print(f"{'─'*60}\n")

    steering_config = SteeringConfig(
        layer_idx=layer,
        position="last",
        strategy="add",
        strength=strength,
        normalize=False
    )

    steered_response, steering_metadata = steerer.generate_with_steering(
        prompt=prompt,
        concept_vector=concept_vector,
        config=steering_config,
        max_tokens=150,
        temperature=temperature
    )

    if verbose:
        print(steered_response)
        print()

    # Evaluate introspection
    evaluator = IntrospectionEvaluator()
    result = evaluator.evaluate(
        task=introspection_task,
        model_response=steered_response,
        baseline_response=baseline_response
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"📊 Evaluation Results:")
        print(f"   Correct: {result.is_correct}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"{'='*60}\n")

    latency = time.time() - start

    return StrategyResult(
        output=steered_response,
        strategy_name="introspection",
        latency_s=latency,
        tokens_in=len(tokenizer.encode(prompt)),
        tokens_out=len(tokenizer.encode(steered_response)),
        cost_usd=0.0,  # Local MLX is free
        metadata={
            "concept": concept,
            "layer": layer,
            "strength": strength,
            "task_type": task_type,
            "introspection_correct": result.is_correct,
            "introspection_confidence": result.confidence,
            "baseline_response": baseline_response,
            "steered_response": steered_response,
            "steering_config": steering_metadata,
            "model": model,
            "provider": provider
        }
    )


# Registry of strategies
STRATEGIES = {
    "single": single_model_strategy,
    "debate": debate_strategy,
    "self_consistency": self_consistency_strategy,
    "manager_worker": manager_worker_strategy,
    "consensus": consensus_strategy,
    "introspection": introspection_strategy,
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
