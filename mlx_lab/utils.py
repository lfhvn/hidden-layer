"""
Utility functions for MLX Lab
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any


def get_mlx_lab_dir() -> Path:
    """Get the MLX Lab configuration directory"""
    mlx_lab_dir = Path.home() / ".mlx-lab"
    mlx_lab_dir.mkdir(exist_ok=True)
    return mlx_lab_dir


def get_benchmark_cache_path() -> Path:
    """Get the benchmark cache file path"""
    return get_mlx_lab_dir() / "benchmarks.json"


def load_benchmark_cache() -> Dict[str, Any]:
    """Load cached benchmark results"""
    cache_path = get_benchmark_cache_path()
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


def save_benchmark_cache(cache: Dict[str, Any]) -> None:
    """Save benchmark results to cache"""
    cache_path = get_benchmark_cache_path()
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def get_huggingface_cache_dir() -> Path:
    """Get the HuggingFace cache directory"""
    # Check environment variable first
    cache_dir = os.environ.get("HF_HOME")
    if cache_dir:
        return Path(cache_dir)

    # Default location
    return Path.home() / ".cache" / "huggingface"


def get_repo_root() -> Path:
    """Get the hidden-layer repository root"""
    # This file is in hidden-layer/mlx_lab/utils.py
    return Path(__file__).parent.parent


def format_bytes(bytes: int) -> str:
    """Format bytes as human-readable string"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} PB"


def format_model_name(model_name: str) -> str:
    """Format model name for display (e.g., 'mlx-community/Qwen3-8B-4bit' -> 'Qwen3-8B-4bit')"""
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name


def check_mlx_installed() -> bool:
    """Check if MLX is installed"""
    try:
        import mlx.core
        return True
    except ImportError:
        return False


def check_mlx_lm_installed() -> bool:
    """Check if mlx-lm is installed"""
    try:
        import mlx_lm
        return True
    except ImportError:
        return False


def check_harness_installed() -> bool:
    """Check if harness is installed"""
    try:
        import harness
        return True
    except ImportError:
        return False
