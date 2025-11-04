"""
AI-to-AI Communication via Cache-to-Cache (C2C)

This package implements direct semantic communication between LLMs through
KV-Cache projection, bypassing text generation for more efficient and expressive
inter-model communication.

Based on the paper:
"Cache-to-Cache: Direct Semantic Communication Between Large Language Models"
https://arxiv.org/abs/2510.03215

Main components:
- C2CProjector: Neural network for transforming KV-Caches between model architectures
- RosettaModel: Wrapper for orchestrating multi-model cache-to-cache communication
- KV-Cache utilities: Helper functions for cache manipulation

Example usage:
    >>> from code import RosettaModel, create_c2c_projector, generate_kv_cache_index
    >>> from transformers import AutoModelForCausalLM
    >>>
    >>> # Load models
    >>> base_model = AutoModelForCausalLM.from_pretrained("model-a")
    >>> source_model = AutoModelForCausalLM.from_pretrained("model-b")
    >>>
    >>> # Create projectors
    >>> projectors = [create_c2c_projector(source_model.config, base_model.config)]
    >>>
    >>> # Create RosettaModel
    >>> rosetta = RosettaModel([base_model, source_model], projector_list=projectors)
    >>>
    >>> # Configure and use
    >>> rosetta.set_projector_config(1, 0, 0, 0, 0)  # Map layers
    >>> cache_idx = generate_kv_cache_index(10, 1)  # Create cache index
    >>> output = rosetta.generate(cache_idx, input_ids, max_new_tokens=50)
"""

from .c2c_projector import (
    C2CProjector,
    create_c2c_projector,
    RegularMLP,
    StandardFFNLayer,
)
from .rosetta_model import RosettaModel
from .kv_cache_utils import (
    generate_kv_cache_index,
    clone_kv_cache,
    extract_layer_cache,
    update_layer_cache,
    print_cache_stats,
    visualize_cache_structure,
    merge_kv_caches,
)

__all__ = [
    # Core components
    "C2CProjector",
    "RosettaModel",
    # Factory functions
    "create_c2c_projector",
    # Utility functions
    "generate_kv_cache_index",
    "clone_kv_cache",
    "extract_layer_cache",
    "update_layer_cache",
    "print_cache_stats",
    "visualize_cache_structure",
    "merge_kv_caches",
    # Building blocks
    "RegularMLP",
    "StandardFFNLayer",
]

__version__ = "0.1.0"

