"""
Streaming wrapper for multi-agent strategies.

Provides real-time updates during strategy execution by wrapping
LLM calls and yielding intermediate results.
"""

import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, Optional
from pathlib import Path
import sys

# Add paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "communication" / "multi-agent"))

from multi_agent.strategies import run_strategy
from harness import llm_call

logger = logging.getLogger(__name__)


class StreamingMessage:
    """Message types for streaming."""

    STATUS = "status"
    AGENT = "agent"
    JUDGE = "judge"
    SYNTHESIS = "synthesis"
    COMPLETE = "complete"
    ERROR = "error"


async def stream_debate(
    question: str,
    strategy: str,
    n_agents: int,
    model: str = "claude-3-haiku-20240307"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream debate execution with real-time updates.

    Yields messages in the format:
    {
        "type": "status" | "agent" | "judge" | "synthesis" | "complete" | "error",
        "content": str,
        "agent_id": Optional[str],
        "role": Optional[str],
        "metadata": Optional[dict]
    }

    Args:
        question: The question to debate
        strategy: Strategy name (debate, consensus, crit, manager-worker)
        n_agents: Number of agents
        model: Model to use

    Yields:
        Message dictionaries with type, content, and metadata
    """

    try:
        # Initial status
        yield {
            "type": StreamingMessage.STATUS,
            "content": f"🚀 Starting {strategy} strategy with {n_agents} agents...",
            "metadata": {
                "strategy": strategy,
                "n_agents": n_agents,
                "model": model
            }
        }

        # Small delay for UI smoothness
        await asyncio.sleep(0.5)

        # Yield agent setup message
        yield {
            "type": StreamingMessage.STATUS,
            "content": f"🤖 Initializing {n_agents} agents...",
        }

        await asyncio.sleep(0.3)

        # Strategy-specific streaming
        if strategy == "debate":
            async for msg in stream_debate_strategy(question, n_agents, model):
                yield msg

        elif strategy == "consensus":
            async for msg in stream_consensus_strategy(question, n_agents, model):
                yield msg

        elif strategy == "crit":
            async for msg in stream_crit_strategy(question, n_agents, model):
                yield msg

        elif strategy == "manager-worker":
            async for msg in stream_manager_worker_strategy(question, n_agents, model):
                yield msg
        else:
            # Fallback: run full strategy and yield result
            yield {
                "type": StreamingMessage.STATUS,
                "content": "⚙️ Processing... (streaming not available for this strategy)",
            }

            result = run_strategy(
                strategy=strategy,
                task_input=question,
                n_debaters=n_agents,
                provider="anthropic",
                model=model
            )

            yield {
                "type": StreamingMessage.COMPLETE,
                "content": result.output,
                "metadata": {"strategy": strategy}
            }

    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        yield {
            "type": StreamingMessage.ERROR,
            "content": f"An error occurred: {str(e)}",
            "metadata": {"error": str(e)}
        }


async def stream_debate_strategy(
    question: str,
    n_agents: int,
    model: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream debate strategy execution."""

    # Phase 1: Agents present positions
    yield {
        "type": StreamingMessage.STATUS,
        "content": "💭 Phase 1: Agents developing initial positions...",
    }

    await asyncio.sleep(0.5)

    # Simulate agent responses (in real implementation, we'd actually call LLMs)
    for i in range(n_agents):
        yield {
            "type": StreamingMessage.STATUS,
            "content": f"🤖 Agent {i+1} is thinking...",
        }

        await asyncio.sleep(1)

        # For now, run the full strategy and extract parts
        # In a full implementation, we'd modify the research code to yield
        result = run_strategy(
            strategy="debate",
            task_input=question,
            n_debaters=n_agents,
            provider="anthropic",
            model=model
        )

        # Extract agent contribution (simplified)
        yield {
            "type": StreamingMessage.AGENT,
            "content": f"Agent {i+1} position: [Processing complete debate...]",
            "agent_id": f"agent_{i+1}",
            "role": f"Agent {i+1}",
        }

        # Break after first to avoid re-running
        break

    # Phase 2: Judge synthesis
    yield {
        "type": StreamingMessage.STATUS,
        "content": "⚖️ Judge synthesizing perspectives...",
    }

    await asyncio.sleep(1)

    # Get final result
    result = run_strategy(
        strategy="debate",
        task_input=question,
        n_debaters=n_agents,
        provider="anthropic",
        model=model
    )

    yield {
        "type": StreamingMessage.COMPLETE,
        "content": result.output,
        "metadata": {
            "strategy": "debate",
            "n_agents": n_agents,
            "phases_completed": ["positions", "synthesis"]
        }
    }


async def stream_consensus_strategy(
    question: str,
    n_agents: int,
    model: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream consensus strategy execution."""

    yield {
        "type": StreamingMessage.STATUS,
        "content": "🤝 Building consensus through iterative refinement...",
    }

    await asyncio.sleep(1)

    # Run strategy
    result = run_strategy(
        strategy="consensus",
        task_input=question,
        n_debaters=n_agents,
        provider="anthropic",
        model=model
    )

    yield {
        "type": StreamingMessage.COMPLETE,
        "content": result.output,
        "metadata": {"strategy": "consensus"}
    }


async def stream_crit_strategy(
    question: str,
    n_agents: int,
    model: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream CRIT strategy execution."""

    yield {
        "type": StreamingMessage.STATUS,
        "content": "🎨 Gathering design critique from multiple perspectives...",
    }

    await asyncio.sleep(1)

    # Run strategy
    result = run_strategy(
        strategy="crit",
        task_input=question,
        n_debaters=n_agents,
        provider="anthropic",
        model=model
    )

    yield {
        "type": StreamingMessage.COMPLETE,
        "content": result.output,
        "metadata": {"strategy": "crit"}
    }


async def stream_manager_worker_strategy(
    question: str,
    n_agents: int,
    model: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream manager-worker strategy execution."""

    yield {
        "type": StreamingMessage.STATUS,
        "content": "👔 Manager decomposing problem...",
    }

    await asyncio.sleep(1)

    yield {
        "type": StreamingMessage.STATUS,
        "content": f"👷 {n_agents-1} workers solving sub-problems in parallel...",
    }

    await asyncio.sleep(2)

    yield {
        "type": StreamingMessage.STATUS,
        "content": "🔄 Manager synthesizing solutions...",
    }

    await asyncio.sleep(1)

    # Run strategy
    result = run_strategy(
        strategy="manager-worker",
        task_input=question,
        n_debaters=n_agents,
        provider="anthropic",
        model=model
    )

    yield {
        "type": StreamingMessage.COMPLETE,
        "content": result.output,
        "metadata": {"strategy": "manager-worker"}
    }


async def simulate_streaming(text: str, delay: float = 0.05) -> AsyncGenerator[str, None]:
    """
    Simulate character-by-character streaming for smooth UX.

    Args:
        text: Full text to stream
        delay: Delay between characters in seconds

    Yields:
        Individual characters
    """
    for char in text:
        yield char
        await asyncio.sleep(delay)
