"""Activation capture utilities for hooking into language model layers.

This module provides tools to extract activations from specific layers
of transformer models for SAE training and analysis.
"""

import logging
from typing import Dict, List, Optional, Callable, Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class ActivationCapture:
    """Captures activations from specified layers of a neural network.

    Usage:
        model = AutoModel.from_pretrained("gpt2")
        capture = ActivationCapture(model, layer_names=["transformer.h.6"])

        with capture:
            outputs = model(input_ids)
            activations = capture.get_activations()
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None,
        capture_input: bool = False,
        capture_output: bool = True,
    ):
        """Initialize activation capture.

        Args:
            model: The neural network model
            layer_names: List of layer names to hook (e.g., ["transformer.h.6.mlp"])
                        If None, will auto-detect transformer layers
            capture_input: Whether to capture layer inputs
            capture_output: Whether to capture layer outputs
        """
        self.model = model
        self.layer_names = layer_names or self._auto_detect_layers()
        self.capture_input = capture_input
        self.capture_output = capture_output

        self.activations: Dict[str, List[torch.Tensor]] = {}
        self.hooks: List[Any] = []

        logger.info(f"ActivationCapture initialized for layers: {self.layer_names}")

    def _auto_detect_layers(self) -> List[str]:
        """Auto-detect transformer layers in the model."""
        layers = []

        for name, module in self.model.named_modules():
            # Look for transformer blocks
            if any(
                pattern in name.lower()
                for pattern in ["transformer.h", "layers", "blocks"]
            ):
                if isinstance(module, nn.Module) and len(list(module.children())) > 0:
                    layers.append(name)

        logger.info(f"Auto-detected {len(layers)} layers")
        return layers[:12]  # Limit to first 12 layers

    def _get_module_by_name(self, name: str) -> Optional[nn.Module]:
        """Get a module by its qualified name."""
        parts = name.split(".")
        module = self.model

        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                logger.warning(f"Module {name} not found in model")
                return None

        return module

    def _make_hook(self, layer_name: str) -> Callable:
        """Create a forward hook for a specific layer.

        Args:
            layer_name: Name of the layer to hook

        Returns:
            Hook function
        """

        def hook_fn(module, input_tensor, output_tensor):
            """Forward hook that captures activations."""
            if layer_name not in self.activations:
                self.activations[layer_name] = []

            if self.capture_output:
                # Handle tuple outputs (e.g., attention layers)
                if isinstance(output_tensor, tuple):
                    output_tensor = output_tensor[0]

                # Detach and move to CPU to save memory
                self.activations[layer_name].append(output_tensor.detach().cpu())

            if self.capture_input:
                # Handle tuple inputs
                if isinstance(input_tensor, tuple):
                    input_tensor = input_tensor[0]

                key = f"{layer_name}_input"
                if key not in self.activations:
                    self.activations[key] = []
                self.activations[key].append(input_tensor.detach().cpu())

        return hook_fn

    def register_hooks(self) -> None:
        """Register forward hooks on specified layers."""
        for layer_name in self.layer_names:
            module = self._get_module_by_name(layer_name)

            if module is not None:
                hook = module.register_forward_hook(self._make_hook(layer_name))
                self.hooks.append(hook)
                logger.debug(f"Registered hook on {layer_name}")
            else:
                logger.warning(f"Could not register hook on {layer_name}")

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.debug("Removed all hooks")

    def clear_activations(self) -> None:
        """Clear stored activations."""
        self.activations = {}

    def get_activations(
        self, layer_name: Optional[str] = None, concatenate: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Get captured activations.

        Args:
            layer_name: Specific layer to get activations for (None = all)
            concatenate: Whether to concatenate activations along batch dim

        Returns:
            Dictionary mapping layer names to activation tensors
        """
        if layer_name is not None:
            acts = self.activations.get(layer_name, [])
            if concatenate and acts:
                return {layer_name: torch.cat(acts, dim=0)}
            return {layer_name: acts}

        # Return all activations
        result = {}
        for name, acts in self.activations.items():
            if concatenate and acts:
                result[name] = torch.cat(acts, dim=0)
            else:
                result[name] = acts

        return result

    def __enter__(self):
        """Context manager entry - register hooks."""
        self.clear_activations()
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - remove hooks."""
        self.remove_hooks()


def get_activation_hook(
    model: PreTrainedModel, layer_idx: int = -1, component: str = "mlp"
) -> ActivationCapture:
    """Convenience function to create an activation hook for a specific layer.

    Args:
        model: HuggingFace transformer model
        layer_idx: Layer index (-1 for last layer)
        component: Which component to hook ("mlp", "attn", or "block")

    Returns:
        ActivationCapture instance

    Example:
        >>> model = AutoModel.from_pretrained("gpt2")
        >>> hook = get_activation_hook(model, layer_idx=6, component="mlp")
        >>> with hook:
        >>>     outputs = model(**inputs)
        >>>     activations = hook.get_activations()
    """
    # Detect model architecture
    config = model.config
    num_layers = config.num_hidden_layers if hasattr(config, "num_hidden_layers") else 12

    # Handle negative indexing
    if layer_idx < 0:
        layer_idx = num_layers + layer_idx

    # Build layer name based on model architecture
    model_type = config.model_type if hasattr(config, "model_type") else "gpt2"

    if model_type in ["gpt2", "gpt_neo", "gpt_neox"]:
        base = f"transformer.h.{layer_idx}"
    elif model_type in ["bert", "roberta"]:
        base = f"encoder.layer.{layer_idx}"
    elif model_type == "llama":
        base = f"model.layers.{layer_idx}"
    else:
        # Default to GPT-2 style
        base = f"transformer.h.{layer_idx}"
        logger.warning(f"Unknown model type {model_type}, using GPT-2 naming")

    # Add component suffix
    if component == "mlp":
        layer_name = f"{base}.mlp"
    elif component == "attn":
        layer_name = f"{base}.attn"
    else:
        layer_name = base

    logger.info(f"Creating activation hook for {layer_name}")

    return ActivationCapture(model, layer_names=[layer_name])
