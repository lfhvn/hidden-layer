"""Core steering engine for modifying model behavior."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class SteeringMethod(str, Enum):
    """Method for applying steering vectors."""

    ADD = "add"  # Add steering vector to activations
    SCALE = "scale"  # Scale activations by steering vector
    REPLACE = "replace"  # Replace activations with steering vector
    PROJECT = "project"  # Project activations onto steering direction


@dataclass
class SteeringVector:
    """Represents a steering vector for behavior modification.

    A steering vector is a direction in activation space that
    corresponds to a specific behavior or attribute.
    """

    name: str
    vector: torch.Tensor  # The actual steering direction
    layer_index: int  # Which layer to apply steering
    strength: float = 1.0  # Multiplier for steering effect
    method: SteeringMethod = SteeringMethod.ADD
    description: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def apply(
        self, activations: torch.Tensor, method: Optional[SteeringMethod] = None
    ) -> torch.Tensor:
        """Apply steering to activations.

        Args:
            activations: Input activations (batch, seq_len, hidden_dim)
            method: Steering method (uses self.method if None)

        Returns:
            Steered activations
        """
        method = method or self.method
        vector = self.vector.to(activations.device)

        # Ensure vector has compatible shape
        if len(vector.shape) == 1:
            vector = vector.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)

        steered = activations.clone()

        if method == SteeringMethod.ADD:
            # Add scaled steering vector
            steered = activations + self.strength * vector

        elif method == SteeringMethod.SCALE:
            # Element-wise scaling
            steered = activations * (1 + self.strength * vector)

        elif method == SteeringMethod.REPLACE:
            # Replace with scaled vector
            steered = self.strength * vector.expand_as(activations)

        elif method == SteeringMethod.PROJECT:
            # Project onto steering direction
            # projection = (act · vec / ||vec||^2) * vec
            vec_norm = vector.flatten()
            act_flat = activations.flatten(1)
            projection = (
                torch.matmul(act_flat, vec_norm) / (vec_norm.norm() ** 2 + 1e-8)
            )
            projection = projection.unsqueeze(-1) * vec_norm
            steered = activations + self.strength * projection.view_as(activations)

        return steered


@dataclass
class SteeringResult:
    """Result from steering operation."""

    output_text: str
    steered_activations: Dict[int, torch.Tensor]
    original_activations: Optional[Dict[int, torch.Tensor]] = None
    steering_vectors_applied: List[SteeringVector] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class SteeringEngine:
    """Engine for applying steering vectors to language models.

    This engine hooks into model layers and applies steering vectors
    to modify behavior in real-time.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cpu",
    ):
        """Initialize steering engine.

        Args:
            model: Language model to steer
            tokenizer: Tokenizer for model
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

        # Active steering vectors per layer
        self.active_vectors: Dict[int, List[SteeringVector]] = {}

        # Hooks for capturing and modifying activations
        self.hooks = []

        # Store activations for analysis
        self.captured_activations: Dict[int, torch.Tensor] = {}

        logger.info(f"SteeringEngine initialized for {model.config.model_type}")

    def add_steering_vector(self, vector: SteeringVector) -> None:
        """Add a steering vector to be applied.

        Args:
            vector: Steering vector to add
        """
        layer_idx = vector.layer_index

        if layer_idx not in self.active_vectors:
            self.active_vectors[layer_idx] = []

        self.active_vectors[layer_idx].append(vector)

        logger.info(
            f"Added steering vector '{vector.name}' to layer {layer_idx} "
            f"(strength={vector.strength}, method={vector.method})"
        )

    def remove_steering_vector(self, name: str, layer_index: int) -> bool:
        """Remove a steering vector.

        Args:
            name: Name of vector to remove
            layer_index: Layer index

        Returns:
            True if removed, False if not found
        """
        if layer_index in self.active_vectors:
            original_len = len(self.active_vectors[layer_index])
            self.active_vectors[layer_index] = [
                v for v in self.active_vectors[layer_index] if v.name != name
            ]

            if len(self.active_vectors[layer_index]) < original_len:
                logger.info(f"Removed steering vector '{name}' from layer {layer_index}")
                return True

        return False

    def clear_steering_vectors(self, layer_index: Optional[int] = None) -> None:
        """Clear steering vectors.

        Args:
            layer_index: Specific layer to clear (None = all layers)
        """
        if layer_index is not None:
            self.active_vectors.pop(layer_index, None)
            logger.info(f"Cleared steering vectors from layer {layer_index}")
        else:
            self.active_vectors.clear()
            logger.info("Cleared all steering vectors")

    def _make_steering_hook(self, layer_index: int):
        """Create forward hook for steering.

        Args:
            layer_index: Layer to hook

        Returns:
            Hook function
        """

        def hook_fn(module, input_tensor, output_tensor):
            """Forward hook that applies steering."""
            # Handle tuple outputs
            if isinstance(output_tensor, tuple):
                activations = output_tensor[0]
                rest = output_tensor[1:]
            else:
                activations = output_tensor
                rest = None

            # Store original activations
            self.captured_activations[layer_index] = activations.detach().cpu()

            # Apply steering vectors
            if layer_index in self.active_vectors:
                for vector in self.active_vectors[layer_index]:
                    activations = vector.apply(activations)

            # Return modified output
            if rest is not None:
                return (activations,) + rest
            return activations

        return hook_fn

    def _get_layer_module(self, layer_index: int) -> Optional[nn.Module]:
        """Get module for a specific layer.

        Args:
            layer_index: Layer index

        Returns:
            Module or None if not found
        """
        model_type = self.model.config.model_type

        if model_type in ["gpt2", "gpt_neo", "gpt_neox"]:
            layer_name = f"transformer.h.{layer_index}"
        elif model_type in ["bert", "roberta"]:
            layer_name = f"encoder.layer.{layer_index}"
        elif model_type == "llama":
            layer_name = f"model.layers.{layer_index}"
        else:
            layer_name = f"transformer.h.{layer_index}"

        # Navigate to module
        module = self.model
        for part in layer_name.split("."):
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                logger.warning(f"Layer {layer_name} not found")
                return None

        return module

    def register_hooks(self) -> None:
        """Register forward hooks for steering."""
        self.remove_hooks()  # Clear any existing hooks

        for layer_index in self.active_vectors.keys():
            module = self._get_layer_module(layer_index)

            if module is not None:
                hook = module.register_forward_hook(
                    self._make_steering_hook(layer_index)
                )
                self.hooks.append(hook)
                logger.debug(f"Registered steering hook on layer {layer_index}")

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 0.7,
        **kwargs,
    ) -> SteeringResult:
        """Generate text with active steering vectors.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            SteeringResult with generated text and metadata
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Clear captured activations
        self.captured_activations = {}

        # Register hooks
        self.register_hooks()

        try:
            # Generate with steering
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                    or self.tokenizer.eos_token_id,
                    **kwargs,
                )

            # Decode output
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Build result
            result = SteeringResult(
                output_text=output_text,
                steered_activations=self.captured_activations.copy(),
                steering_vectors_applied=[
                    v for vectors in self.active_vectors.values() for v in vectors
                ],
                metadata={
                    "prompt": prompt,
                    "max_length": max_length,
                    "temperature": temperature,
                    "num_steering_vectors": sum(
                        len(v) for v in self.active_vectors.values()
                    ),
                },
            )

            return result

        finally:
            # Always remove hooks
            self.remove_hooks()

    def compare_steered_unsteered(
        self, prompt: str, **generation_kwargs
    ) -> Tuple[str, str]:
        """Generate both steered and unsteered versions for comparison.

        Args:
            prompt: Input prompt
            **generation_kwargs: Generation parameters

        Returns:
            Tuple of (unsteered_text, steered_text)
        """
        # Generate unsteered
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs_unsteered = self.model.generate(**inputs, **generation_kwargs)

        unsteered_text = self.tokenizer.decode(
            outputs_unsteered[0], skip_special_tokens=True
        )

        # Generate steered
        result_steered = self.generate(prompt, **generation_kwargs)
        steered_text = result_steered.output_text

        return unsteered_text, steered_text
