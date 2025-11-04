"""Library for managing and creating steering vectors."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np

from .engine import SteeringVector, SteeringMethod

logger = logging.getLogger(__name__)


class VectorLibrary:
    """Manages a collection of steering vectors.

    Provides storage, retrieval, and manipulation of steering vectors.
    """

    def __init__(self, storage_dir: str = "./steering_vectors"):
        """Initialize vector library.

        Args:
            storage_dir: Directory to store vectors
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self.vectors: Dict[str, SteeringVector] = {}

        logger.info(f"VectorLibrary initialized with storage at {storage_dir}")

    def add(self, vector: SteeringVector) -> None:
        """Add vector to library.

        Args:
            vector: Steering vector to add
        """
        self.vectors[vector.name] = vector
        logger.info(f"Added vector '{vector.name}' to library")

    def get(self, name: str) -> Optional[SteeringVector]:
        """Get vector by name.

        Args:
            name: Vector name

        Returns:
            SteeringVector or None if not found
        """
        return self.vectors.get(name)

    def list(self) -> List[str]:
        """List all vector names.

        Returns:
            List of vector names
        """
        return list(self.vectors.keys())

    def remove(self, name: str) -> bool:
        """Remove vector from library.

        Args:
            name: Vector name

        Returns:
            True if removed, False if not found
        """
        if name in self.vectors:
            del self.vectors[name]
            logger.info(f"Removed vector '{name}' from library")
            return True
        return False

    def save(self, name: str) -> None:
        """Save vector to disk.

        Args:
            name: Vector name
        """
        if name not in self.vectors:
            raise ValueError(f"Vector '{name}' not found in library")

        vector = self.vectors[name]
        save_path = self.storage_dir / f"{name}.pt"

        torch.save(
            {
                "name": vector.name,
                "vector": vector.vector,
                "layer_index": vector.layer_index,
                "strength": vector.strength,
                "method": vector.method.value,
                "description": vector.description,
                "metadata": vector.metadata,
            },
            save_path,
        )

        logger.info(f"Saved vector '{name}' to {save_path}")

    def load(self, name: str) -> SteeringVector:
        """Load vector from disk.

        Args:
            name: Vector name

        Returns:
            Loaded SteeringVector
        """
        load_path = self.storage_dir / f"{name}.pt"

        if not load_path.exists():
            raise FileNotFoundError(f"Vector file not found: {load_path}")

        data = torch.load(load_path)

        vector = SteeringVector(
            name=data["name"],
            vector=data["vector"],
            layer_index=data["layer_index"],
            strength=data.get("strength", 1.0),
            method=SteeringMethod(data.get("method", "add")),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
        )

        self.vectors[name] = vector
        logger.info(f"Loaded vector '{name}' from {load_path}")

        return vector

    def load_all(self) -> int:
        """Load all vectors from storage directory.

        Returns:
            Number of vectors loaded
        """
        count = 0
        for file_path in self.storage_dir.glob("*.pt"):
            try:
                name = file_path.stem
                self.load(name)
                count += 1
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {count} vectors from storage")
        return count


def create_steering_vector(
    name: str,
    positive_examples: List[str],
    negative_examples: List[str],
    model,
    tokenizer,
    layer_index: int,
    strength: float = 1.0,
    device: str = "cpu",
) -> SteeringVector:
    """Create a steering vector from positive/negative examples.

    This uses mean difference of activations between positive and
    negative examples to find a steering direction.

    Args:
        name: Name for the steering vector
        positive_examples: Examples of desired behavior
        negative_examples: Examples of undesired behavior
        model: Language model
        tokenizer: Tokenizer
        layer_index: Which layer to extract from
        strength: Steering strength
        device: Device to run on

    Returns:
        SteeringVector
    """
    logger.info(
        f"Creating steering vector '{name}' from "
        f"{len(positive_examples)} pos / {len(negative_examples)} neg examples"
    )

    def get_mean_activation(examples: List[str]) -> torch.Tensor:
        """Get mean activation for examples."""
        activations = []

        # Hook to capture activations
        captured = []

        def hook_fn(module, input_tensor, output_tensor):
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]
            captured.append(output_tensor.detach().cpu())

        # Get the layer module
        if hasattr(model, "transformer"):
            layer = model.transformer.h[layer_index]
        elif hasattr(model, "encoder"):
            layer = model.encoder.layer[layer_index]
        else:
            raise ValueError("Unsupported model architecture")

        hook = layer.register_forward_hook(hook_fn)

        try:
            for example in examples:
                captured.clear()
                inputs = tokenizer(example, return_tensors="pt").to(device)

                with torch.no_grad():
                    _ = model(**inputs)

                if captured:
                    # Average over sequence length
                    act = captured[0].mean(dim=1)  # (batch, hidden_dim)
                    activations.append(act)

        finally:
            hook.remove()

        # Average across all examples
        mean_act = torch.stack(activations).mean(dim=0).squeeze()

        return mean_act

    # Get mean activations for positive and negative examples
    pos_mean = get_mean_activation(positive_examples)
    neg_mean = get_mean_activation(negative_examples)

    # Steering vector is the difference
    steering_vec = pos_mean - neg_mean

    # Normalize
    steering_vec = steering_vec / (steering_vec.norm() + 1e-8)

    logger.info(f"Created steering vector '{name}' with norm {steering_vec.norm():.4f}")

    return SteeringVector(
        name=name,
        vector=steering_vec,
        layer_index=layer_index,
        strength=strength,
        method=SteeringMethod.ADD,
        description=f"Learned from {len(positive_examples)} pos, {len(negative_examples)} neg examples",
        metadata={
            "num_positive": len(positive_examples),
            "num_negative": len(negative_examples),
            "vector_norm": float(steering_vec.norm().item()),
        },
    )
