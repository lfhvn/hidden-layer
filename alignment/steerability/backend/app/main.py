"""Main FastAPI application for Steerability Dashboard."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, create_engine

from .config import get_settings
from .api import steering_routes, experiments_routes, metrics_routes

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

# CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
