"""
Common middleware for FastAPI web tools.
"""

import os
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


def setup_cors(app: FastAPI):
    """Configure CORS middleware."""
    origins = os.getenv("CORS_ORIGINS", "").split(",")

    # Default to localhost for development
    if not origins or origins == [""]:
        origins = [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:8000",
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.info(f"CORS enabled for origins: {origins}")


def setup_error_handling(app: FastAPI):
    """Configure global error handling."""

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)

        # Don't expose internal errors in production
        if os.getenv("ENV") == "production":
            detail = "An internal error occurred"
        else:
            detail = str(exc)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": detail
            }
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Invalid input",
                "detail": str(exc)
            }
        )


def setup_logging():
    """Configure logging."""
    level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def add_response_headers(app: FastAPI):
    """Add security and cache headers to all responses."""

    @app.middleware("http")
    async def add_headers(request: Request, call_next):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Cache control (can be overridden per endpoint)
        if "Cache-Control" not in response.headers:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"

        return response


def setup_all_middleware(app: FastAPI):
    """Setup all common middleware."""
    setup_logging()
    setup_cors(app)
    setup_error_handling(app)
    add_response_headers(app)

    logger.info("All middleware configured")
