"""Main FastAPI application for Steerability Dashboard."""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from sqlmodel import SQLModel, create_engine

from .config import get_settings
from .api import steering_routes, experiments_routes, metrics_routes

# Add shared utilities to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "web-tools" / "shared" / "backend"))

from middleware import setup_all_middleware
from auth import RateLimiter, APIKeyValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    logger.info("Starting Steerability Dashboard...")
    settings = get_settings()

    # Init database
    engine = create_engine(settings.database_url)
    SQLModel.metadata.create_all(engine)

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Steerability Dashboard API",
    description="Live LLM steering with adherence metrics",
    version="0.1.0",
    lifespan=lifespan,
)

# Setup all shared middleware (CORS, error handling, security headers)
setup_all_middleware(app)

# Initialize rate limiter
limiter = RateLimiter(requests=5, window=3600)  # 5 requests per hour
api_key_validator = APIKeyValidator()

# Store in app state for access in routes
app.state.limiter = limiter
app.state.api_key_validator = api_key_validator

# Routes
app.include_router(steering_routes.router, prefix="/api")
app.include_router(experiments_routes.router, prefix="/api")
app.include_router(metrics_routes.router, prefix="/api")


@app.get("/")
async def root():
    return {"status": "healthy", "service": "Steerability Dashboard", "version": "0.1.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/usage")
async def get_usage(request: Request):
    """Get current rate limit usage."""
    usage = app.state.limiter.get_usage(request)
    return usage
