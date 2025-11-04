"""
Multi-Agent Arena Backend

Streams multi-agent interactions in real-time via WebSockets.
"""

import os
import sys
from pathlib import Path

# Add project root to path to import research code
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import logging

# Import shared utilities
sys.path.insert(0, str(project_root / "web-tools" / "shared" / "backend"))
from auth import RateLimiter, APIKeyValidator, create_rate_limited_endpoint
from middleware import setup_all_middleware

# Import research code
from projects.multi_agent.code.strategies import run_strategy
from harness import llm_call

logger = logging.getLogger(__name__)

# Create app
app = FastAPI(
    title="Multi-Agent Arena",
    description="Watch AI agents debate and solve problems in real-time",
    version="0.1.0"
)

# Setup middleware
setup_all_middleware(app)

# Rate limiting
limiter = RateLimiter(
    requests=int(os.getenv("RATE_LIMIT_REQUESTS", "3")),
    window=int(os.getenv("RATE_LIMIT_WINDOW", "3600"))
)
api_key_validator = APIKeyValidator()


# Models
class DebateRequest(BaseModel):
    question: str
    strategy: str = "debate"
    n_agents: int = 3
    max_rounds: int = 3
    model: Optional[str] = None


class StrategyInfo(BaseModel):
    name: str
    display_name: str
    description: str
    best_for: str
    typical_agents: int


# Available strategies
STRATEGIES = {
    "debate": StrategyInfo(
        name="debate",
        display_name="Debate",
        description="Agents argue different perspectives, judge synthesizes",
        best_for="Controversial topics, multiple viewpoints",
        typical_agents=3
    ),
    "consensus": StrategyInfo(
        name="consensus",
        display_name="Consensus",
        description="Agents work together to find agreement",
        best_for="Collaborative solutions, finding common ground",
        typical_agents=3
    ),
    "crit": StrategyInfo(
        name="crit",
        display_name="Design Critique",
        description="Multiple perspectives critique a design",
        best_for="Design feedback, structured critique",
        typical_agents=4
    ),
    "manager-worker": StrategyInfo(
        name="manager-worker",
        display_name="Manager-Worker",
        description="Manager decomposes problem, workers solve in parallel",
        best_for="Complex problems, decomposable tasks",
        typical_agents=4
    )
}


@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "service": "multi-agent-arena",
        "version": "0.1.0"
    }


@app.get("/api/strategies")
async def list_strategies():
    """List available strategies."""
    return {"strategies": list(STRATEGIES.values())}


@app.get("/api/usage")
async def get_usage(request: Request):
    """Get current rate limit usage."""
    usage = limiter.get_usage(request)
    return usage


@app.post("/api/debate")
@create_rate_limited_endpoint(limiter, api_key_validator)
async def run_debate(
    request: Request,
    debate_request: DebateRequest,
    user_api_key: Optional[str] = None
):
    """
    Run a multi-agent debate (non-streaming).

    This endpoint runs the full debate and returns the result.
    For streaming, use the WebSocket endpoint.
    """
    # Validate input
    if debate_request.n_agents < 2 or debate_request.n_agents > 5:
        raise HTTPException(400, "Number of agents must be between 2 and 5")

    if debate_request.strategy not in STRATEGIES:
        raise HTTPException(400, f"Invalid strategy: {debate_request.strategy}")

    if len(debate_request.question) > 500:
        raise HTTPException(400, "Question too long (max 500 characters)")

    # Determine which API key to use
    if user_api_key:
        # User provided their own key
        os.environ["ANTHROPIC_API_KEY"] = user_api_key
        logger.info("Using user-provided API key")
    else:
        # Use our key (rate limited)
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise HTTPException(500, "API key not configured")

    # Select model
    model = debate_request.model
    if not model or not user_api_key:
        # Free tier: Haiku only
        model = "claude-3-haiku-20240307"

    try:
        # Run the strategy using research code
        result = run_strategy(
            strategy=debate_request.strategy,
            task_input=debate_request.question,
            n_debaters=debate_request.n_agents,
            provider="anthropic",
            model=model
        )

        return {
            "question": debate_request.question,
            "strategy": debate_request.strategy,
            "result": result.output,
            "metadata": {
                "n_agents": debate_request.n_agents,
                "model": model,
                "strategy_info": STRATEGIES[debate_request.strategy]
            }
        }

    except Exception as e:
        logger.error(f"Error running debate: {e}", exc_info=True)
        raise HTTPException(500, f"Error running debate: {str(e)}")


class AgentMessage(BaseModel):
    """Message from an agent in the debate."""
    type: str  # "agent", "judge", "status", "error", "complete"
    agent_id: Optional[str] = None
    role: Optional[str] = None  # "agent_1", "agent_2", "judge", etc.
    content: str
    metadata: Optional[dict] = None


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_message(self, message: AgentMessage, websocket: WebSocket):
        await websocket.send_json(message.dict())


manager = ConnectionManager()


@app.websocket("/ws/debate")
async def debate_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming debates.

    Client sends:
    {
        "question": "...",
        "strategy": "debate",
        "n_agents": 3
    }

    Server streams back:
    {
        "type": "status",
        "content": "Starting debate..."
    }
    {
        "type": "agent",
        "agent_id": "agent_1",
        "role": "Supporter",
        "content": "I believe that..."
    }
    {
        "type": "complete",
        "content": "Final synthesis..."
    }
    """
    await manager.connect(websocket)

    try:
        # Receive initial request
        data = await websocket.receive_json()

        # Parse request
        try:
            debate_request = DebateRequest(**data)
        except Exception as e:
            await manager.send_message(
                AgentMessage(type="error", content=f"Invalid request: {str(e)}"),
                websocket
            )
            return

        # Send status
        await manager.send_message(
            AgentMessage(
                type="status",
                content=f"Starting {debate_request.strategy} with {debate_request.n_agents} agents..."
            ),
            websocket
        )

        # TODO: Implement streaming version of strategies
        # For now, run the full debate and send result
        # In future: modify research code to yield intermediate results

        model = "claude-3-haiku-20240307"

        # Run strategy
        result = run_strategy(
            strategy=debate_request.strategy,
            task_input=debate_request.question,
            n_debaters=debate_request.n_agents,
            provider="anthropic",
            model=model
        )

        # Send completion
        await manager.send_message(
            AgentMessage(
                type="complete",
                content=result.output,
                metadata={
                    "strategy": debate_request.strategy,
                    "n_agents": debate_request.n_agents
                }
            ),
            websocket
        )

    except WebSocketDisconnect:
        logger.info("Client disconnected")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await manager.send_message(
                AgentMessage(type="error", content=str(e)),
                websocket
            )
        except:
            pass
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
