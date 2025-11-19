"""
AgentMesh FastAPI server.

Provides REST API for workflow management and execution.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentmesh.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    await init_db()
    print("✅ Database initialized")
    print("✅ AgentMesh API server ready")
    print("📚 API docs: http://localhost:8000/docs")

    yield

    # Shutdown
    print("👋 Shutting down AgentMesh API")


# Create FastAPI app
app = FastAPI(
    title="AgentMesh API",
    description="Multi-agent workflow orchestration built on Hidden Layer research",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware (for web frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "AgentMesh API",
        "version": "0.1.0",
        "description": "Multi-agent workflow orchestration",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


# Import and register route modules
from agentmesh.api.routes import workflows, runs, async_runs, human_steps

app.include_router(workflows.router, prefix="/api")
app.include_router(runs.router, prefix="/api")
app.include_router(async_runs.router, prefix="/api")
app.include_router(human_steps.router, prefix="/api")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agentmesh.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
    )
