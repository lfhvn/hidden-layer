"""
Model management for MLX Lab

Handles downloading, listing, removing, and managing MLX models.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from mlx_lab.utils import (
    get_huggingface_cache_dir,
    format_bytes,
    format_model_name,
    check_mlx_lm_installed,
)


@dataclass
class ModelInfo:
    """Information about an MLX model"""

    name: str
    path: Path
    size_bytes: int
    repo_id: str  # Full HuggingFace repo ID (e.g., "mlx-community/Qwen3-8B-4bit")


class ModelManager:
    """Manages MLX models in the HuggingFace cache"""

    # Recommended models for different use cases
    RECOMMENDED_MODELS = {
        "qwen3-8b-4bit": {
            "repo_id": "mlx-community/Qwen3-8B-4bit",
            "description": "Fast, capable general-purpose model",
            "ram_estimate": "~5GB",
            "use_case": "Interactive experiments, quick iterations",
        },
        "gpt-oss-20b-4bit": {
            "repo_id": "mlx-community/gpt-oss-20b-reasoning-4bit",
            "description": "Powerful reasoning model",
            "ram_estimate": "~12GB",
            "use_case": "Complex reasoning, research tasks",
        },
        "llama3.2-3b-4bit": {
            "repo_id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "description": "Lightweight, fast model",
            "ram_estimate": "~2GB",
            "use_case": "Quick testing, low memory",
        },
    }

    def __init__(self):
        self.cache_dir = get_huggingface_cache_dir()

    def list_models(self) -> List[ModelInfo]:
        """List all downloaded MLX models"""
        models = []

        # HuggingFace cache structure: ~/.cache/huggingface/hub/models--{org}--{model}
        hub_dir = self.cache_dir / "hub"
        if not hub_dir.exists():
            return models

        for model_dir in hub_dir.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith("models--"):
                # Parse model name from directory
                # e.g., "models--mlx-community--Qwen3-8B-4bit" -> "mlx-community/Qwen3-8B-4bit"
                parts = model_dir.name.replace("models--", "").split("--")
                if len(parts) >= 2:
                    repo_id = "/".join(parts)
                    model_name = parts[-1]  # Last part is the model name

                    # Calculate size
                    size = self._get_dir_size(model_dir)

                    models.append(
                        ModelInfo(name=model_name, path=model_dir, size_bytes=size, repo_id=repo_id)
                    )

        return sorted(models, key=lambda m: m.name)

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get info about a specific model"""
        models = self.list_models()

        # Try exact match first
        for model in models:
            if model.name == model_name or model.repo_id == model_name:
                return model

        # Try partial match
        for model in models:
            if model_name.lower() in model.name.lower():
                return model

        return None

    def download_model(self, repo_id: str, progress_callback=None) -> bool:
        """
        Download a model from HuggingFace

        Args:
            repo_id: Full repo ID (e.g., "mlx-community/Qwen3-8B-4bit")
            progress_callback: Optional callback for progress updates

        Returns:
            True if successful, False otherwise
        """
        if not check_mlx_lm_installed():
            raise ImportError("mlx-lm not installed. Run: pip install mlx mlx-lm")

        try:
            from mlx_lm import load

            # mlx_lm.load() will download the model if not cached
            if progress_callback:
                progress_callback(f"Downloading {repo_id}...")

            model, tokenizer = load(repo_id)

            if progress_callback:
                progress_callback(f"✅ Successfully downloaded {repo_id}")

            return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Error: {str(e)}")
            return False

    def remove_model(self, model_name: str) -> bool:
        """
        Remove a model from the cache

        Args:
            model_name: Model name or repo_id

        Returns:
            True if successful, False otherwise
        """
        model_info = self.get_model_info(model_name)
        if not model_info:
            return False

        try:
            shutil.rmtree(model_info.path)
            return True
        except Exception:
            return False

    def get_recommended_models(self) -> Dict[str, Dict[str, str]]:
        """Get list of recommended models"""
        return self.RECOMMENDED_MODELS

    def _get_dir_size(self, path: Path) -> int:
        """Calculate total size of directory"""
        total = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception:
            pass
        return total

    def format_model_list(self, models: List[ModelInfo]) -> str:
        """Format model list for display"""
        if not models:
            return "No models found in cache"

        lines = []
        lines.append("Downloaded MLX Models:")
        lines.append("=" * 70)

        for model in models:
            size_str = format_bytes(model.size_bytes)
            lines.append(f"  • {model.name}")
            lines.append(f"    Repo: {model.repo_id}")
            lines.append(f"    Size: {size_str}")
            lines.append("")

        total_size = sum(m.size_bytes for m in models)
        lines.append("=" * 70)
        lines.append(f"Total: {len(models)} models, {format_bytes(total_size)}")

        return "\n".join(lines)

    def format_model_info(self, model: ModelInfo) -> str:
        """Format detailed model info for display"""
        lines = []
        lines.append(f"Model: {model.name}")
        lines.append("=" * 70)
        lines.append(f"Repository: {model.repo_id}")
        lines.append(f"Size: {format_bytes(model.size_bytes)}")
        lines.append(f"Location: {model.path}")

        # Add recommended info if available
        for key, info in self.RECOMMENDED_MODELS.items():
            if info["repo_id"] == model.repo_id:
                lines.append("")
                lines.append("Recommended For:")
                lines.append(f"  Description: {info['description']}")
                lines.append(f"  RAM Estimate: {info['ram_estimate']}")
                lines.append(f"  Use Case: {info['use_case']}")
                break

        return "\n".join(lines)
