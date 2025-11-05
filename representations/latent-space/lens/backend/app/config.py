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
    database_url: str = "sqlite:///./latent_lens.db"

    # Model Configuration
    default_model_name: str = "gpt2"
    hf_cache_dir: str = os.path.expanduser("~/.cache/huggingface/hub")  # Centralized cache
    device: str = "cpu"

    # Server
    backend_port: int = 8000
    frontend_port: int = 3000
    log_level: str = "INFO"

    # CORS
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Feature Extraction
    max_features: int = 1024
    sae_hidden_dim: int = 4096
    sparsity_coefficient: float = 0.01

    # Dataset
    dataset_name: str = "wikitext"
    dataset_split: str = "train"
    max_samples: int = 1000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
