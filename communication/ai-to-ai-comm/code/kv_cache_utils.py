"""
KV-Cache utilities for Cache-to-Cache communication

Helper functions for managing and manipulating KV-Caches in multi-model setups.
"""

import torch
from torch import Tensor
from typing import List, Tuple
from transformers.cache_utils import DynamicCache


def generate_kv_cache_index(
    instruction_length: int,
    response_length: int = 1,
    source_model_idx: int = 1,
    target_model_idx: int = 0,
    device: torch.device = None,
) -> List[Tensor]:
    """
    Generate KV-Cache index for cache-to-cache communication.

    Args:
        instruction_length: Length of instruction/prompt section
        response_length: Length of response section (default: 1 for single token)
        source_model_idx: Index of source model (default: 1)
        target_model_idx: Index of target model (default: 0)
        device: Device for tensors (default: CPU)

    Returns:
        List of cache index tensors:
        - instruction_index: (1, instruction_length, 2) with [source_idx, target_idx]
        - response_index: (1, response_length, 2) with [-1, target_idx] (no projection)

    Example:
        >>> # For a prompt of 10 tokens using model 1 -> model 0 projection
        >>> cache_idx = generate_kv_cache_index(10, 1, source_model_idx=1, target_model_idx=0)
        >>> # cache_idx[0] = [[1, 0]] repeated 10 times (use projection)
        >>> # cache_idx[1] = [[-1, 0]] (no projection for response)
    """
    if device is None:
        device = torch.device("cpu")

    # Instruction section: use source model projection
    instruction_index = (
        torch.tensor([source_model_idx, target_model_idx], dtype=torch.long)
        .repeat(instruction_length, 1)
        .unsqueeze(0)
        .to(device)
    )

    # Response section: no projection (use base model only)
    response_index = (
        torch.tensor([-1, target_model_idx], dtype=torch.long)
        .repeat(response_length, 1)
        .unsqueeze(0)
        .to(device)
    )

    return [instruction_index, response_index]


def clone_kv_cache(kv_cache: DynamicCache) -> DynamicCache:
    """
    Clone a DynamicCache for independent manipulation.

    Args:
        kv_cache: DynamicCache to clone

    Returns:
        Cloned DynamicCache
    """
    new_cache = DynamicCache()
    for k, v in zip(kv_cache.key_cache, kv_cache.value_cache):
        new_cache.key_cache.append(k.clone().detach())
        new_cache.value_cache.append(v.clone().detach())
    return new_cache


def extract_layer_cache(
    kv_cache: DynamicCache, layer_idx: int, start: int, end: int
) -> Tuple[Tensor, Tensor]:
    """
    Extract a section of KV-Cache from a specific layer.

    Args:
        kv_cache: DynamicCache to extract from
        layer_idx: Layer index
        start: Start position in sequence
        end: End position in sequence

    Returns:
        Tuple of (key, value) tensors (B, H, seq_len, D)
    """
    key = kv_cache.key_cache[layer_idx][:, :, start:end, :]
    value = kv_cache.value_cache[layer_idx][:, :, start:end, :]
    return key, value


def update_layer_cache(
    kv_cache: DynamicCache,
    layer_idx: int,
    start: int,
    end: int,
    new_key: Tensor,
    new_value: Tensor,
):
    """
    Update a section of KV-Cache in a specific layer.

    Args:
        kv_cache: DynamicCache to update
        layer_idx: Layer index
        start: Start position in sequence
        end: End position in sequence
        new_key: New key tensor (B, H, seq_len, D)
        new_value: New value tensor (B, H, seq_len, D)
    """
    kv_cache.key_cache[layer_idx][:, :, start:end, :] = new_key
    kv_cache.value_cache[layer_idx][:, :, start:end, :] = new_value


def print_cache_stats(kv_cache: DynamicCache, name: str = "Cache"):
    """
    Print statistics about a KV-Cache.

    Args:
        kv_cache: DynamicCache to analyze
        name: Name for display
    """
    if not kv_cache.key_cache:
        print(f"{name}: Empty cache")
        return

    num_layers = len(kv_cache.key_cache)
    batch_size, num_heads, seq_len, head_dim = kv_cache.key_cache[0].shape

    total_elements = num_layers * batch_size * num_heads * seq_len * head_dim * 2  # key + value
    memory_mb = (total_elements * 4) / (1024 * 1024)  # Assuming float32

    print(f"{name} Statistics:")
    print(f"  Layers: {num_layers}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Num Heads: {num_heads}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Head Dimension: {head_dim}")
    print(f"  Total Memory: ~{memory_mb:.2f} MB")


def visualize_cache_structure(
    kv_cache_index: List[Tensor], names: List[str] = None
):
    """
    Visualize the structure of KV-Cache index.

    Args:
        kv_cache_index: List of cache index tensors
        names: Optional names for each section
    """
    if names is None:
        names = [f"Section {i}" for i in range(len(kv_cache_index))]

    print("KV-Cache Index Structure:")
    for i, (section, name) in enumerate(zip(kv_cache_index, names)):
        source_idx = section[0, 0, 0].item()
        target_idx = section[0, 0, 1].item()
        seq_len = section.shape[1]

        projection_str = "No Projection" if source_idx == -1 else f"Model {source_idx} -> Model {target_idx}"
        print(f"  {name}: Length {seq_len}, {projection_str}")


def merge_kv_caches(cache_list: List[DynamicCache]) -> DynamicCache:
    """
    Merge multiple KV-Caches along the sequence dimension.

    Args:
        cache_list: List of DynamicCache objects to merge

    Returns:
        Merged DynamicCache
    """
    if not cache_list:
        return DynamicCache()

    merged = DynamicCache()
    num_layers = len(cache_list[0].key_cache)

    for layer_idx in range(num_layers):
        # Concatenate keys and values from all caches for this layer
        keys = [cache.key_cache[layer_idx] for cache in cache_list]
        values = [cache.value_cache[layer_idx] for cache in cache_list]

        merged_key = torch.cat(keys, dim=2)  # Concatenate along sequence dimension
        merged_value = torch.cat(values, dim=2)

        merged.update(merged_key, merged_value, layer_idx)

    return merged
