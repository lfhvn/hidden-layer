"""
Example: Cache-to-Cache Communication between two LLMs

This example demonstrates how to use C2C for direct semantic communication
between two different LLM architectures without text generation.

Usage:
    python example_c2c.py

Requirements:
    - transformers
    - torch
    - Two compatible models (e.g., Qwen2-0.5B and Qwen2-1.5B)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from c2c_projector import create_c2c_projector
from rosetta_model import RosettaModel
from kv_cache_utils import generate_kv_cache_index, print_cache_stats


def load_models(
    base_model_name: str = "Qwen/Qwen2-0.5B-Instruct",
    source_model_name: str = "Qwen/Qwen2-1.5B-Instruct",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Load base and source models.

    Args:
        base_model_name: Name or path of base model
        source_model_name: Name or path of source model
        device: Device to load models on

    Returns:
        Tuple of (base_model, source_model, tokenizer)
    """
    print(f"Loading models on {device}...")

    # Load tokenizer (use base model's tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load models
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    source_model = AutoModelForCausalLM.from_pretrained(
        source_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )

    base_model.eval()
    source_model.eval()

    print(f"Base model: {base_model_name}")
    print(f"  Layers: {base_model.config.num_hidden_layers}")
    print(f"  Hidden size: {base_model.config.hidden_size}")
    print(f"  Attention heads: {base_model.config.num_attention_heads}")

    print(f"\nSource model: {source_model_name}")
    print(f"  Layers: {source_model.config.num_hidden_layers}")
    print(f"  Hidden size: {source_model.config.hidden_size}")
    print(f"  Attention heads: {source_model.config.num_attention_heads}")

    return base_model, source_model, tokenizer


def setup_rosetta_model(base_model, source_model, device):
    """
    Set up RosettaModel with C2C projectors.

    Args:
        base_model: Base model
        source_model: Source model
        device: Device

    Returns:
        Configured RosettaModel
    """
    print("\nSetting up RosettaModel...")

    # Create projectors for each layer
    num_layers = base_model.config.num_hidden_layers
    projector_list = []

    for layer_idx in range(num_layers):
        projector = create_c2c_projector(
            source_model.config,
            base_model.config,
            hidden_dim=1024,
            intermediate_dim=2048,
            num_layers=3,
            dropout=0.1,
        )
        projector.to(device)
        projector_list.append(projector)

    print(f"Created {len(projector_list)} projectors")

    # Create RosettaModel
    rosetta = RosettaModel(
        model_list=[base_model, source_model],
        base_model_idx=0,
        projector_list=projector_list,
    )

    # Configure layer mappings
    for layer_idx in range(num_layers):
        rosetta.set_projector_config(
            source_model_idx=1,
            source_model_layer_idx=layer_idx,
            target_model_idx=0,
            target_model_layer_idx=layer_idx,
            projector_idx=layer_idx,
        )

    print("Configured layer-wise projections")
    return rosetta


def run_inference(
    rosetta_model,
    tokenizer,
    prompt: str = "Explain quantum computing in simple terms:",
    max_new_tokens: int = 50,
):
    """
    Run inference with cache-to-cache communication.

    Args:
        rosetta_model: RosettaModel instance
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    print(f"\n{'='*60}")
    print("Running C2C Inference")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}")

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(rosetta_model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    seq_len = input_ids.shape[1]
    print(f"Input length: {seq_len} tokens")

    # Generate KV-Cache index
    # - Use source model (model 1) for instruction processing
    # - Use base model (model 0) for response generation
    kv_cache_index = generate_kv_cache_index(
        instruction_length=seq_len - 1,  # All tokens except last
        response_length=1,  # Start generation from last token
        source_model_idx=1,
        target_model_idx=0,
        device=rosetta_model.device,
    )

    print("\nGenerating with C2C...")
    with torch.no_grad():
        output_ids = rosetta_model.generate(
            kv_cache_index=kv_cache_index,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{generated_text}")
    print(f"{'='*60}\n")

    return generated_text


def compare_with_baseline(
    base_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
):
    """
    Run baseline inference with just the base model.

    Args:
        base_model: Base model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    print(f"\n{'='*60}")
    print("Running Baseline Inference (Base Model Only)")
    print(f"{'='*60}")

    inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

    print("Generating with base model...")
    with torch.no_grad():
        output_ids = base_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{generated_text}")
    print(f"{'='*60}\n")

    return generated_text


def main():
    """Main example function."""
    print("Cache-to-Cache (C2C) Communication Example")
    print("=" * 60)

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_name = "Qwen/Qwen2-0.5B-Instruct"
    source_model_name = "Qwen/Qwen2-1.5B-Instruct"

    # Load models
    base_model, source_model, tokenizer = load_models(
        base_model_name, source_model_name, device
    )

    # Setup RosettaModel
    rosetta_model = setup_rosetta_model(base_model, source_model, device)

    # Test prompt
    prompt = "Explain the concept of artificial intelligence in one sentence:"

    # Run C2C inference
    c2c_output = run_inference(
        rosetta_model, tokenizer, prompt, max_new_tokens=30
    )

    # Run baseline for comparison
    baseline_output = compare_with_baseline(
        base_model, tokenizer, prompt, max_new_tokens=30
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"C2C leverages {source_model_name} for understanding")
    print(f"and {base_model_name} for generation, enabling:")
    print("  - Better instruction understanding from larger model")
    print("  - Efficient generation from smaller model")
    print("  - Direct semantic transfer without text encoding")
    print("=" * 60)


if __name__ == "__main__":
    main()
