"""
Activation Steering Module

Enables injection of concept vectors into MLX model activations
to test introspective awareness following the methodology from:
https://transformer-circuits.pub/2025/introspection/index.html
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np


@dataclass
class SteeringConfig:
    """Configuration for activation steering"""

    layer_idx: int  # Which layer to inject into
    position: str = "last"  # "last", "mean", "first", or token index
    strategy: str = "add"  # "add", "replace", "scale"
    strength: float = 1.0  # Scaling factor for injection
    normalize: bool = False  # Normalize vectors before injection


class ActivationSteerer:
    """
    Hook into MLX model forward pass to extract and inject activations.

    Usage:
        steerer = ActivationSteerer(model)

        # Extract concept vector
        concept_vec = steerer.extract_activation(
            prompt="I feel very happy",
            layer=15,
            position="last"
        )

        # Inject concept and generate
        response = steerer.generate_with_steering(
            prompt="Tell me a story",
            concept_vector=concept_vec,
            config=SteeringConfig(layer_idx=15, strength=1.5)
        )
    """

    def __init__(self, model, tokenizer):
        """
        Args:
            model: MLX model instance
            tokenizer: Model tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.activations: Dict[str, mx.array] = {}
        self.hooks: List = []

    def _get_layer_module(self, layer_idx: int):
        """Get the module for a specific layer"""
        # MLX models typically have a 'layers' attribute
        if hasattr(self.model, "layers"):
            return self.model.layers[layer_idx]
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx]
        else:
            raise ValueError("Could not find layers in model structure")

    def _position_to_index(self, position: str, seq_len: int) -> int:
        """Convert position string to token index"""
        if position == "last":
            return seq_len - 1
        elif position == "first":
            return 0
        elif position.isdigit():
            return int(position)
        else:
            raise ValueError(f"Unknown position: {position}")

    def extract_activation(self, prompt: str, layer_idx: int, position: str = "last") -> np.ndarray:
        """
        Extract activation vector from a specific layer and position.

        Args:
            prompt: Input text
            layer_idx: Layer to extract from (0-indexed)
            position: Token position ("last", "first", "mean", or index)

        Returns:
            Activation vector as numpy array
        """
        self.activations.clear()

        # Register forward hook to capture activations
        layer = self._get_layer_module(layer_idx)

        def hook_fn(module, args, output):
            # Store the output activations
            self.activations[f"layer_{layer_idx}"] = output
            return output

        # Register hook
        layer.register_forward_hook(hook_fn)

        try:
            # Tokenize and run forward pass
            tokens = self.tokenizer.encode(prompt)
            tokens = mx.array([tokens])  # Add batch dimension

            # Forward pass (don't generate, just get activations)
            with mx.no_grad():
                _ = self.model(tokens)

            # Extract activations
            act = self.activations[f"layer_{layer_idx}"]

            # Select position
            if position == "mean":
                act_vec = mx.mean(act[0], axis=0)  # Mean over sequence
            else:
                pos_idx = self._position_to_index(position, act.shape[1])
                act_vec = act[0, pos_idx]  # Select specific token

            # Convert to numpy
            return np.array(act_vec)

        finally:
            # Clean up hook
            # Note: MLX hooks are automatically removed when out of scope
            self.activations.clear()

    def extract_contrastive_concept(
        self, positive_prompt: str, negative_prompt: str, layer_idx: int, position: str = "last"
    ) -> np.ndarray:
        """
        Extract concept vector using contrastive method.

        Args:
            positive_prompt: Prompt representing the concept
            negative_prompt: Neutral/opposite prompt
            layer_idx: Layer to extract from
            position: Token position

        Returns:
            Concept vector (positive - negative)
        """
        pos_act = self.extract_activation(positive_prompt, layer_idx, position)
        neg_act = self.extract_activation(negative_prompt, layer_idx, position)

        concept_vec = pos_act - neg_act

        # Optional: normalize
        norm = np.linalg.norm(concept_vec)
        if norm > 0:
            concept_vec = concept_vec / norm

        return concept_vec

    def generate_with_steering(
        self,
        prompt: str,
        concept_vector: np.ndarray,
        config: SteeringConfig,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> Tuple[str, Dict]:
        """
        Generate text with concept vector injected into activations.

        Args:
            prompt: Input prompt
            concept_vector: Vector to inject
            config: Steering configuration
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Tuple of (generated_text, metadata)
        """
        # Convert concept vector to MLX array
        concept_mx = mx.array(concept_vector)

        # Normalize if requested
        if config.normalize:
            concept_mx = concept_mx / mx.linalg.norm(concept_mx)

        # Apply strength scaling
        concept_mx = concept_mx * config.strength

        # Register intervention hook
        layer = self._get_layer_module(config.layer_idx)

        def steering_hook(module, args, output):
            """Modify activations during forward pass"""
            # output shape: [batch, seq_len, hidden_dim]

            if config.position == "mean":
                # Apply to all positions
                if config.strategy == "add":
                    output = output + concept_mx
                elif config.strategy == "scale":
                    output = output * (1 + config.strength)
                elif config.strategy == "replace":
                    output = mx.broadcast_to(concept_mx, output.shape)
            else:
                # Apply to specific position
                pos_idx = self._position_to_index(config.position, output.shape[1])

                if config.strategy == "add":
                    output = mx.concatenate(
                        [
                            output[:, :pos_idx],
                            (output[:, pos_idx : pos_idx + 1] + concept_mx).reshape(1, 1, -1),
                            output[:, pos_idx + 1 :],
                        ],
                        axis=1,
                    )
                elif config.strategy == "replace":
                    output = mx.concatenate(
                        [output[:, :pos_idx], concept_mx.reshape(1, 1, -1), output[:, pos_idx + 1 :]], axis=1
                    )
                # scale strategy modifies globally

            return output

        # Register hook
        layer.register_forward_hook(steering_hook)

        try:
            # Generate with steering
            from mlx_lm import generate as mlx_generate

            output = mlx_generate(
                self.model, self.tokenizer, prompt=prompt, temp=temperature, max_tokens=max_tokens, verbose=False
            )

            metadata = {
                "layer": config.layer_idx,
                "position": config.position,
                "strategy": config.strategy,
                "strength": config.strength,
                "normalized": config.normalize,
            }

            return output, metadata

        finally:
            # Clean up
            pass

    def compare_with_baseline(
        self,
        prompt: str,
        concept_vector: np.ndarray,
        config: SteeringConfig,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> Dict:
        """
        Generate both baseline and steered outputs for comparison.

        Returns:
            Dictionary with 'baseline', 'steered', and 'config' keys
        """
        # Baseline generation
        from mlx_lm import generate as mlx_generate

        baseline = mlx_generate(
            self.model, self.tokenizer, prompt=prompt, temp=temperature, max_tokens=max_tokens, verbose=False
        )

        # Steered generation
        steered, metadata = self.generate_with_steering(
            prompt=prompt, concept_vector=concept_vector, config=config, max_tokens=max_tokens, temperature=temperature
        )

        return {"baseline": baseline, "steered": steered, "config": metadata, "prompt": prompt}


class ActivationCache:
    """
    Cache activations across multiple layers for analysis.
    """

    def __init__(self, model, tokenizer, layer_indices: List[int]):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_indices = layer_indices
        self.cache: Dict[int, np.ndarray] = {}

    def capture_activations(self, prompt: str, position: str = "last") -> Dict[int, np.ndarray]:
        """
        Capture activations from multiple layers in a single forward pass.

        Args:
            prompt: Input text
            position: Token position to extract

        Returns:
            Dictionary mapping layer_idx -> activation vector
        """
        steerer = ActivationSteerer(self.model, self.tokenizer)

        for layer_idx in self.layer_indices:
            act = steerer.extract_activation(prompt, layer_idx, position)
            self.cache[layer_idx] = act

        return self.cache

    def get_layer_activation(self, layer_idx: int) -> Optional[np.ndarray]:
        """Get cached activation for a specific layer"""
        return self.cache.get(layer_idx)

    def clear(self):
        """Clear the cache"""
        self.cache.clear()


def test_steering_basic():
    """Basic test of activation steering"""
    print("Testing activation steering...")

    # This is a placeholder - would need actual model
    from mlx_lm import load

    model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
    steerer = ActivationSteerer(model, tokenizer)

    # Extract happiness concept
    happiness = steerer.extract_contrastive_concept(
        positive_prompt="I feel very happy and joyful!",
        negative_prompt="I feel neutral, neither happy nor sad.",
        layer_idx=15,
        position="last",
    )

    print(f"Extracted happiness vector: shape {happiness.shape}")

    # Test steering
    result = steerer.compare_with_baseline(
        prompt="Tell me about your day.",
        concept_vector=happiness,
        config=SteeringConfig(layer_idx=15, strength=1.0, strategy="add"),
        max_tokens=50,
    )

    print(f"\nBaseline: {result['baseline']}")
    print(f"\nSteered: {result['steered']}")


if __name__ == "__main__":
    test_steering_basic()
