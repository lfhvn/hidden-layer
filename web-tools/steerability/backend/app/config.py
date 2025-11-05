"""Configuration management using Pydantic settings."""

import os
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Security
    api_key: str = "dev-key-change-in-production"

    # Database
    database_url: str = "sqlite:///./steerability.db"

    # Model Configuration
    default_model_name: str = "gpt2"
    hf_cache_dir: str = os.path.expanduser("~/.cache/huggingface/hub")  # Centralized cache
    device: str = "cpu"

    # Server
    backend_port: int = 8001
    frontend_port: int = 3001
    log_level: str = "INFO"

    # CORS
    allowed_origins: str = "http://localhost:3001,http://127.0.0.1:3001"

    # Steering Configuration
    max_steering_strength: float = 5.0
    default_steering_strength: float = 1.0
    max_steering_vectors: int = 10

    # Metrics Configuration
    adherence_window_size: int = 100
    metrics_retention_days: int = 30
    sampling_rate: float = 0.1

    # Experiment Configuration
    max_parallel_experiments: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
