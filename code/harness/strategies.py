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


def design_critique_strategy(
    task_input: str,
    n_iterations: int = 2,
    critique_panel: List[Dict[str, str]] = None,
    provider: str = None,
    model: str = None,
    temperature: float = 0.7,
    verbose: bool = False,
    **kwargs
) -> StrategyResult:
    """
    Design critique strategy: Generate draft, critique from multiple perspectives, revise iteratively.

    Args:
        task_input: The task/question
        n_iterations: Number of critique/revision cycles
        critique_panel: List of critics with 'name', 'focus', 'criteria' (uses defaults if None)
        provider: LLM provider
        model: Model name
        temperature: Sampling temperature
        verbose: Print intermediate steps
    """
    start = time.time()
    provider = provider or DEFAULT_PROVIDER
    model = model or DEFAULT_MODEL

    # Default critique panel if not provided - Designer archetypes
    if critique_panel is None:
        critique_panel = [
            {
                "name": "Systems Designer",
                "focus": "System architecture and holistic design",
                "criteria": """You are a Systems Designer who thinks about the big picture and interconnections.
Evaluate:
- Does the solution consider the whole system and its parts?
- Are relationships between components clear?
- Is there coherence between different elements?
- Does the design scale and adapt to different contexts?
Focus on holistic thinking, interconnections, and systemic coherence."""
            },
            {
                "name": "Visual Craft Specialist",
                "focus": "Visual clarity, aesthetics, and presentation",
                "criteria": """You are a Visual Craft Specialist focused on how information is presented and perceived.
Evaluate:
- Is information presented clearly and visually comprehensible?
- Is there good hierarchy and structure in the presentation?
- Are concepts illustrated or explained in ways that are easy to visualize?
- Does the format enhance understanding?
Focus on clarity of presentation, visual thinking, and aesthetic quality."""
            },
            {
                "name": "AI Specialist",
                "focus": "AI/ML capabilities, limitations, and best practices",
                "criteria": """You are an AI Specialist with deep knowledge of AI systems, capabilities, and limitations.
Evaluate:
- Are claims about AI accurate and grounded in current capabilities?
- Are limitations and potential issues with AI acknowledged?
- Are AI-related recommendations practical and informed?
- Is the approach aligned with AI best practices?
Focus on technical accuracy regarding AI/ML, practical feasibility, and responsible AI considerations."""
            },
            {
                "name": "Human-Computer Interaction Expert",
                "focus": "User experience, usability, and human factors",
                "criteria": """You are an HCI Expert focused on how humans interact with systems and information.
Evaluate:
- Is the solution usable and accessible to the target audience?
- Are user needs and cognitive limitations considered?
- Is interaction intuitive and aligned with mental models?
- Are there potential usability issues or barriers?
Focus on user-centered design, accessibility, cognitive ergonomics, and interaction patterns."""
            },
            {
                "name": "IDEO Design Thinking Facilitator",
                "focus": "Human-centered innovation and creative problem-solving",
                "criteria": """You are an IDEO-trained Design Thinking practitioner emphasizing empathy, ideation, and iteration.
Evaluate:
- Does the solution demonstrate empathy for user needs and pain points?
- Is there creative thinking and exploration of possibilities?
- Are assumptions tested or validated?
- Is the approach iterative and open to refinement?
Focus on empathy, creative exploration, prototyping mindset, and bias toward action."""
            },
        ]

    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0

    # Generate initial draft
    if verbose:
        print("\n" + "="*80)
        print("📄 GENERATING INITIAL DRAFT")
        print("="*80 + "\n")

    if verbose:
        # Stream the initial draft
        full_text = ""
        for chunk in llm_call_stream(task_input, provider=provider, model=model, temperature=temperature, **kwargs):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
                full_text += chunk
            else:
                draft_response = chunk
        print("\n")
    else:
        draft_response = llm_call(task_input, provider=provider, model=model, temperature=temperature, **kwargs)

    current_draft = draft_response.text

    total_tokens_in += draft_response.tokens_in or 0
    total_tokens_out += draft_response.tokens_out or 0
    total_cost += draft_response.cost_usd or 0.0

    versions = [{"version": 0, "content": current_draft}]
    all_critiques = []

    # Iterative critique and revision
    for iteration in range(n_iterations):
        if verbose:
            print("\n" + "="*80)
            print(f"🎨 ITERATION {iteration + 1}: Critique & Revision")
            print("="*80 + "\n")

        # Critique phase
        if verbose:
            print("─"*80)
            print("💬 CRITIQUE PHASE")
            print("─"*80 + "\n")

        critiques = []
        for critic in critique_panel:
            critique_prompt = f"""You are providing critique as: {critic['name']}

{critic['criteria']}

Draft to critique:
\"\"\"
{current_draft}
\"\"\"

Provide your critique focusing on {critic['focus']}:"""

            if verbose:
                print(f"\n{critic['name']}:")
                print("-" * 60 + "\n")
                full_text = ""
                for chunk in llm_call_stream(critique_prompt, provider=provider, model=model, temperature=0.7, **kwargs):
                    if isinstance(chunk, str):
                        print(chunk, end="", flush=True)
                        full_text += chunk
                    else:
                        critique_response = chunk
                print("\n")
            else:
                critique_response = llm_call(critique_prompt, provider=provider, model=model, temperature=0.7, **kwargs)

            critiques.append({
                "critic": critic['name'],
                "focus": critic['focus'],
                "feedback": critique_response.text
            })

            total_tokens_in += critique_response.tokens_in or 0
            total_tokens_out += critique_response.tokens_out or 0
            total_cost += critique_response.cost_usd or 0.0

        all_critiques.append(critiques)

        # Revision phase
        if verbose:
            print("\n" + "─"*80)
            print("✏️  REVISION PHASE")
            print("─"*80 + "\n")

        revision_prompt = f"""You are revising a draft based on structured critique.

Your job:
1. Carefully review all critiques
2. Identify the most important improvements
3. Revise the draft to address feedback
4. Balance different critique perspectives
5. Maintain the core message while improving quality

Current Draft:
\"\"\"
{current_draft}
\"\"\"

Critiques:

"""

        for critique in critiques:
            revision_prompt += f"{critique['critic']} ({critique['focus']}):\n{critique['feedback']}\n\n"

        revision_prompt += "Provide an improved version that addresses the feedback:\n"

        if verbose:
            full_text = ""
            for chunk in llm_call_stream(revision_prompt, provider=provider, model=model, temperature=0.7, **kwargs):
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                    full_text += chunk
                else:
                    revision_response = chunk
            print("\n")
        else:
            revision_response = llm_call(revision_prompt, provider=provider, model=model, temperature=0.7, **kwargs)

        current_draft = revision_response.text

        total_tokens_in += revision_response.tokens_in or 0
        total_tokens_out += revision_response.tokens_out or 0
        total_cost += revision_response.cost_usd or 0.0

        versions.append({"version": iteration + 1, "content": current_draft})

    latency = time.time() - start

    return StrategyResult(
        output=current_draft,
        strategy_name="design_critique",
        latency_s=latency,
        tokens_in=total_tokens_in,
        tokens_out=total_tokens_out,
        cost_usd=total_cost,
        metadata={
            "n_iterations": n_iterations,
            "n_critics": len(critique_panel),
            "all_versions": versions,
            "all_critiques": all_critiques,
            "provider": provider,
            "model": model
        }
    )


def interdisciplinary_team_strategy(
    task_input: str,
    expert_team: List[Dict[str, str]] = None,
    refinement_rounds: int = 1,
    provider: str = None,
    model: str = None,
    temperature: float = 0.7,
    verbose: bool = False,
    **kwargs
) -> StrategyResult:
    """
    Interdisciplinary team strategy: Domain experts collaborate on complex problems.

    Args:
        task_input: The problem/question
        expert_team: List of experts with 'name', 'role', 'perspective', 'system_prompt' (uses defaults if None)
        refinement_rounds: Number of refinement iterations after initial synthesis
        provider: LLM provider
        model: Model name
        temperature: Sampling temperature
        verbose: Print intermediate steps
    """
    start = time.time()
    provider = provider or DEFAULT_PROVIDER
    model = model or DEFAULT_MODEL

    # Default expert team if not provided - Classic tech team
    if expert_team is None:
        expert_team = [
            {
                "name": "Product Manager",
                "role": "Product Management",
                "perspective": "User needs, business value, roadmap, and strategic priorities",
                "system_prompt": """You are a Product Manager responsible for defining what to build and why.

Your mission: Ensure the solution creates real user value while achieving business objectives.

Focus on:
- User needs, pain points, and jobs-to-be-done
- Business impact, ROI, and strategic alignment
- Feature prioritization and tradeoffs
- Market fit and competitive positioning
- Success metrics and measurable outcomes
- Feasibility vs. value vs. risk assessment

Analyze problems by asking:
- What user problem does this solve?
- What is the business value?
- How do we measure success?
- What are the must-haves vs. nice-to-haves?
- What are the risks and mitigations?

Bring a balanced perspective that bridges user needs, business goals, and technical reality."""
            },
            {
                "name": "Software Engineer",
                "role": "Engineering",
                "perspective": "Technical implementation, architecture, scalability, and feasibility",
                "system_prompt": """You are a Software Engineer responsible for building and shipping reliable systems.

Your mission: Deliver technically sound solutions that are maintainable, scalable, and feasible within constraints.

Focus on:
- Technical feasibility and implementation complexity
- System architecture and design patterns
- Scalability, performance, and reliability
- Security, data integrity, and edge cases
- Technical debt and long-term maintainability
- Development velocity and engineering resources
- Integration with existing systems

Analyze problems by asking:
- Is this technically feasible?
- What is the implementation complexity?
- What are the technical risks?
- How does this scale?
- What are the dependencies and blockers?
- What technical debt are we taking on?

Bring a pragmatic engineering perspective focused on what we can actually build and ship."""
            },
            {
                "name": "Product Designer",
                "role": "Design",
                "perspective": "User experience, interaction design, usability, and design quality",
                "system_prompt": """You are a Product Designer responsible for crafting intuitive, delightful user experiences.

Your mission: Ensure the solution is usable, accessible, and provides a great user experience.

Focus on:
- User experience and interaction design
- Usability, learnability, and accessibility
- User flows and mental models
- Information architecture and navigation
- Visual design and brand consistency
- Edge cases and error states
- User research insights and validation

Analyze problems by asking:
- Is this intuitive for users?
- What is the user flow?
- Are there usability issues or friction points?
- Is it accessible to all users?
- How do we handle edge cases and errors?
- Does this match user mental models?

Bring a user-centered design perspective that ensures solutions are not just functional but delightful to use."""
            },
        ]

    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0

    # Expert analysis phase
    if verbose:
        print("\n" + "="*80)
        print("👥 EXPERT ANALYSIS PHASE")
        print("="*80 + "\n")

    expert_analyses = []
    for expert in expert_team:
        expert_prompt = f"""{expert['system_prompt']}

Problem:
{task_input}

Provide your analysis from the {expert['role']} perspective.
Focus on: {expert['perspective']}

Your analysis:"""

        if verbose:
            print(f"\n{expert['name']} ({expert['role']}):")
            print("─" * 60 + "\n")
            full_text = ""
            for chunk in llm_call_stream(expert_prompt, provider=provider, model=model, temperature=temperature, **kwargs):
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                    full_text += chunk
                else:
                    analysis_response = chunk
            print("\n")
        else:
            analysis_response = llm_call(expert_prompt, provider=provider, model=model, temperature=temperature, **kwargs)

        expert_analyses.append({
            "expert": expert['name'],
            "role": expert['role'],
            "analysis": analysis_response.text
        })

        total_tokens_in += analysis_response.tokens_in or 0
        total_tokens_out += analysis_response.tokens_out or 0
        total_cost += analysis_response.cost_usd or 0.0

    # Synthesis phase
    if verbose:
        print("\n" + "="*80)
        print("🔄 SYNTHESIS PHASE")
        print("="*80 + "\n")

    synthesis_prompt = f"""You are a project lead synthesizing insights from an interdisciplinary team.

Your job:
1. Review each expert's analysis
2. Identify key insights and potential conflicts
3. Synthesize into a coherent, actionable solution
4. Balance competing priorities (technical, user, business, etc.)
5. Propose concrete next steps

Original Problem:
{task_input}

Expert Analyses:

"""

    for analysis in expert_analyses:
        synthesis_prompt += f"{analysis['expert']} ({analysis['role']}):\n{analysis['analysis']}\n\n"
        synthesis_prompt += "-" * 60 + "\n\n"

    synthesis_prompt += """Based on all expert analyses:

1. Synthesize key insights
2. Identify any conflicts or tradeoffs
3. Propose an integrated solution
4. Provide concrete next steps

Integrated solution:"""

    if verbose:
        full_text = ""
        for chunk in llm_call_stream(synthesis_prompt, provider=provider, model=model, temperature=0.3, **kwargs):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
                full_text += chunk
            else:
                synthesis_response = chunk
        print("\n")
    else:
        synthesis_response = llm_call(synthesis_prompt, provider=provider, model=model, temperature=0.3, **kwargs)

    current_solution = synthesis_response.text

    total_tokens_in += synthesis_response.tokens_in or 0
    total_tokens_out += synthesis_response.tokens_out or 0
    total_cost += synthesis_response.cost_usd or 0.0

    solutions = [{"round": 0, "content": current_solution}]
    all_refinements = []

    # Refinement rounds
    for round_num in range(refinement_rounds):
        if verbose:
            print("\n" + "="*80)
            print(f"🔄 REFINEMENT ROUND {round_num + 1}")
            print("="*80 + "\n")

        refinements = []
        for expert in expert_team:
            refinement_prompt = f"""{expert['system_prompt']}

Original Problem:
{task_input}

Proposed Solution:
{current_solution}

Review this solution from your {expert['role']} perspective.
Provide specific suggestions for improvement or concerns.

Your feedback:"""

            if verbose:
                print(f"\n{expert['name']} - Refinement:")
                print("─" * 60 + "\n")
                full_text = ""
                for chunk in llm_call_stream(refinement_prompt, provider=provider, model=model, temperature=temperature, **kwargs):
                    if isinstance(chunk, str):
                        print(chunk, end="", flush=True)
                        full_text += chunk
                    else:
                        refinement_response = chunk
                print("\n")
            else:
                refinement_response = llm_call(refinement_prompt, provider=provider, model=model, temperature=temperature, **kwargs)

            refinements.append({
                "expert": expert['name'],
                "feedback": refinement_response.text
            })

            total_tokens_in += refinement_response.tokens_in or 0
            total_tokens_out += refinement_response.tokens_out or 0
            total_cost += refinement_response.cost_usd or 0.0

        all_refinements.append(refinements)

        # Integrate refinements
        if verbose:
            print("\n" + "─"*80)
            print("🔄 Integrating Refinements")
            print("─"*80 + "\n")

        integration_prompt = f"""You are a project lead integrating expert feedback.

Current Solution:
{current_solution}

Expert Refinements:

"""

        for refinement in refinements:
            integration_prompt += f"{refinement['expert']}:\n{refinement['feedback']}\n\n"

        integration_prompt += "Incorporate the expert feedback to improve the solution.\n\nRefined solution:"

        if verbose:
            full_text = ""
            for chunk in llm_call_stream(integration_prompt, provider=provider, model=model, temperature=0.3, **kwargs):
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                    full_text += chunk
                else:
                    integration_response = chunk
            print("\n")
        else:
            integration_response = llm_call(integration_prompt, provider=provider, model=model, temperature=0.3, **kwargs)

        current_solution = integration_response.text

        total_tokens_in += integration_response.tokens_in or 0
        total_tokens_out += integration_response.tokens_out or 0
        total_cost += integration_response.cost_usd or 0.0

        solutions.append({"round": round_num + 1, "content": current_solution})

    latency = time.time() - start

    return StrategyResult(
        output=current_solution,
        strategy_name="interdisciplinary_team",
        latency_s=latency,
        tokens_in=total_tokens_in,
        tokens_out=total_tokens_out,
        cost_usd=total_cost,
        metadata={
            "n_experts": len(expert_team),
            "refinement_rounds": refinement_rounds,
            "expert_analyses": expert_analyses,
            "all_solutions": solutions,
            "all_refinements": all_refinements,
            "provider": provider,
            "model": model
        }
    )


def adaptive_team_strategy(
    task_input: str,
    n_experts: int = 3,
    refinement_rounds: int = 1,
    provider: str = None,
    model: str = None,
    temperature: float = 0.7,
    verbose: bool = False,
    **kwargs
) -> StrategyResult:
    """
    Adaptive team strategy: Dynamically generate expert team tailored to each specific problem.

    This meta-strategy:
    1. Analyzes the problem to understand what expertise is needed
    2. Generates custom expert personas specifically for this problem
    3. Runs interdisciplinary team collaboration with the custom experts

    Args:
        task_input: The problem/question
        n_experts: Number of experts to generate (default: 3)
        refinement_rounds: Number of refinement iterations after initial synthesis
        provider: LLM provider
        model: Model name
        temperature: Sampling temperature
        verbose: Print intermediate steps
    """
    start = time.time()
    provider = provider or DEFAULT_PROVIDER
    model = model or DEFAULT_MODEL

    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0

    # Step 1: Analyze the problem and design expert team
    if verbose:
        print("\n" + "="*80)
        print("🧠 ANALYZING PROBLEM & DESIGNING EXPERT TEAM")
        print("="*80 + "\n")

    team_design_prompt = f"""You are designing an expert team to solve a specific problem.

Problem to solve:
\"\"\"
{task_input}
\"\"\"

Your task:
1. Analyze what kind of problem this is
2. Identify what types of expertise would be most valuable
3. Design {n_experts} expert personas specifically tailored to solve this problem

For each expert, provide:
- name: A descriptive name (e.g., "Number Theory Specialist", "Clinical Psychologist")
- role: Their domain/discipline
- perspective: What they focus on
- system_prompt: Detailed instructions for how they should analyze problems

Format your response as a JSON array:
[
  {{
    "name": "Expert Name",
    "role": "Domain",
    "perspective": "What they focus on",
    "system_prompt": "You are a [role]... Focus on: [specifics]... Analyze by: [approach]..."
  }},
  ...
]

Make the experts highly relevant and specific to THIS problem. Don't use generic experts - tailor them!

JSON array of {n_experts} experts:"""

    if verbose:
        print("Designing expert team for this specific problem...")
        print("─" * 60 + "\n")

    team_design_response = llm_call(team_design_prompt, provider=provider, model=model, temperature=0.7, **kwargs)

    total_tokens_in += team_design_response.tokens_in or 0
    total_tokens_out += team_design_response.tokens_out or 0
    total_cost += team_design_response.cost_usd or 0.0

    # Parse the expert team from JSON
    import json
    try:
        # Extract JSON from response (might have markdown code blocks)
        response_text = team_design_response.text
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        expert_team = json.loads(response_text.strip())

        if verbose:
            print(f"✅ Generated {len(expert_team)} custom experts:")
            for expert in expert_team:
                print(f"   • {expert['name']} ({expert['role']})")
            print("\n")
    except Exception as e:
        if verbose:
            print(f"⚠️  Failed to parse expert team JSON: {e}")
            print("Using fallback generic team...\n")

        # Fallback to generic team
        expert_team = [
            {
                "name": "Domain Expert",
                "role": "Subject Matter Expert",
                "perspective": "Deep domain knowledge",
                "system_prompt": "You are a domain expert. Analyze this problem with deep subject matter expertise."
            },
            {
                "name": "Analytical Thinker",
                "role": "Analytical Reasoning",
                "perspective": "Logical analysis and reasoning",
                "system_prompt": "You are an analytical thinker. Break down problems logically and reason through them systematically."
            },
            {
                "name": "Practical Advisor",
                "role": "Practical Application",
                "perspective": "Real-world feasibility",
                "system_prompt": "You are a practical advisor. Focus on what works in practice and real-world constraints."
            },
        ][:n_experts]

    # Step 2: Run expert analysis phase (same as interdisciplinary_team)
    if verbose:
        print("\n" + "="*80)
        print("👥 EXPERT ANALYSIS PHASE")
        print("="*80 + "\n")

    expert_analyses = []
    for expert in expert_team:
        expert_prompt = f"""{expert['system_prompt']}

Problem:
{task_input}

Provide your analysis from the {expert['role']} perspective.
Focus on: {expert['perspective']}

Your analysis:"""

        if verbose:
            print(f"\n{expert['name']} ({expert['role']}):")
            print("─" * 60 + "\n")
            full_text = ""
            for chunk in llm_call_stream(expert_prompt, provider=provider, model=model, temperature=temperature, **kwargs):
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                    full_text += chunk
                else:
                    analysis_response = chunk
            print("\n")
        else:
            analysis_response = llm_call(expert_prompt, provider=provider, model=model, temperature=temperature, **kwargs)

        expert_analyses.append({
            "expert": expert['name'],
            "role": expert['role'],
            "analysis": analysis_response.text
        })

        total_tokens_in += analysis_response.tokens_in or 0
        total_tokens_out += analysis_response.tokens_out or 0
        total_cost += analysis_response.cost_usd or 0.0

    # Step 3: Synthesis phase
    if verbose:
        print("\n" + "="*80)
        print("🔄 SYNTHESIS PHASE")
        print("="*80 + "\n")

    synthesis_prompt = f"""You are a project lead synthesizing insights from a custom expert team.

Your job:
1. Review each expert's analysis
2. Identify key insights and potential conflicts
3. Synthesize into a coherent, actionable solution
4. Balance competing priorities
5. Propose concrete next steps

Original Problem:
{task_input}

Expert Analyses:

"""

    for analysis in expert_analyses:
        synthesis_prompt += f"{analysis['expert']} ({analysis['role']}):\n{analysis['analysis']}\n\n"
        synthesis_prompt += "-" * 60 + "\n\n"

    synthesis_prompt += """Based on all expert analyses:

1. Synthesize key insights
2. Identify any conflicts or tradeoffs
3. Propose an integrated solution
4. Provide concrete next steps

Integrated solution:"""

    if verbose:
        full_text = ""
        for chunk in llm_call_stream(synthesis_prompt, provider=provider, model=model, temperature=0.3, **kwargs):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
                full_text += chunk
            else:
                synthesis_response = chunk
        print("\n")
    else:
        synthesis_response = llm_call(synthesis_prompt, provider=provider, model=model, temperature=0.3, **kwargs)

    current_solution = synthesis_response.text

    total_tokens_in += synthesis_response.tokens_in or 0
    total_tokens_out += synthesis_response.tokens_out or 0
    total_cost += synthesis_response.cost_usd or 0.0

    solutions = [{"round": 0, "content": current_solution}]
    all_refinements = []

    # Step 4: Refinement rounds (optional)
    for round_num in range(refinement_rounds):
        if verbose:
            print("\n" + "="*80)
            print(f"🔄 REFINEMENT ROUND {round_num + 1}")
            print("="*80 + "\n")

        refinements = []
        for expert in expert_team:
            refinement_prompt = f"""{expert['system_prompt']}

Original Problem:
{task_input}

Proposed Solution:
{current_solution}

Review this solution from your {expert['role']} perspective.
Provide specific suggestions for improvement or concerns.

Your feedback:"""

            if verbose:
                print(f"\n{expert['name']} - Refinement:")
                print("─" * 60 + "\n")
                full_text = ""
                for chunk in llm_call_stream(refinement_prompt, provider=provider, model=model, temperature=temperature, **kwargs):
                    if isinstance(chunk, str):
                        print(chunk, end="", flush=True)
                        full_text += chunk
                    else:
                        refinement_response = chunk
                print("\n")
            else:
                refinement_response = llm_call(refinement_prompt, provider=provider, model=model, temperature=temperature, **kwargs)

            refinements.append({
                "expert": expert['name'],
                "feedback": refinement_response.text
            })

            total_tokens_in += refinement_response.tokens_in or 0
            total_tokens_out += refinement_response.tokens_out or 0
            total_cost += refinement_response.cost_usd or 0.0

        all_refinements.append(refinements)

        # Integrate refinements
        if verbose:
            print("\n" + "─"*80)
            print("🔄 Integrating Refinements")
            print("─"*80 + "\n")

        integration_prompt = f"""You are a project lead integrating expert feedback.

Current Solution:
{current_solution}

Expert Refinements:

"""

        for refinement in refinements:
            integration_prompt += f"{refinement['expert']}:\n{refinement['feedback']}\n\n"

        integration_prompt += "Incorporate the expert feedback to improve the solution.\n\nRefined solution:"

        if verbose:
            full_text = ""
            for chunk in llm_call_stream(integration_prompt, provider=provider, model=model, temperature=0.3, **kwargs):
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                    full_text += chunk
                else:
                    integration_response = chunk
            print("\n")
        else:
            integration_response = llm_call(integration_prompt, provider=provider, model=model, temperature=0.3, **kwargs)

        current_solution = integration_response.text

        total_tokens_in += integration_response.tokens_in or 0
        total_tokens_out += integration_response.tokens_out or 0
        total_cost += integration_response.cost_usd or 0.0

        solutions.append({"round": round_num + 1, "content": current_solution})

    latency = time.time() - start

    return StrategyResult(
        output=current_solution,
        strategy_name="adaptive_team",
        latency_s=latency,
        tokens_in=total_tokens_in,
        tokens_out=total_tokens_out,
        cost_usd=total_cost,
        metadata={
            "n_experts": len(expert_team),
            "generated_experts": expert_team,
            "refinement_rounds": refinement_rounds,
            "expert_analyses": expert_analyses,
            "all_solutions": solutions,
            "all_refinements": all_refinements,
            "provider": provider,
            "model": model
        }
    )


# Registry of strategies
STRATEGIES = {
    "single": single_model_strategy,
    "debate": debate_strategy,
    "self_consistency": self_consistency_strategy,
    "manager_worker": manager_worker_strategy,
    "consensus": consensus_strategy,
    "design_critique": design_critique_strategy,
    "interdisciplinary_team": interdisciplinary_team_strategy,
    "adaptive_team": adaptive_team_strategy,
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
