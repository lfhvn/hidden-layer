"""
LLM Provider Layer - supports local MLX models, Ollama, and API providers
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

from .defaults import DEFAULT_MODEL, DEFAULT_PROVIDER
from .system_prompts import resolve_system_prompt


@dataclass
class LLMResponse:
    """Standardized response format"""

    text: str
    model: str
    provider: str
    latency_s: float
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_usd: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMProvider:
    """
    Unified interface for calling different LLM providers.
    Supports: MLX (local), Ollama (local), Anthropic, OpenAI
    """

    def __init__(self):
        self.mlx_model = None
        self.mlx_tokenizer = None

    def call(
        self,
        prompt: str,
        provider: str = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Route to appropriate provider.

        Args:
            prompt: The input prompt
            provider: "mlx", "ollama", "anthropic", "openai" (defaults to DEFAULT_PROVIDER)
            model: Model identifier (defaults to DEFAULT_MODEL)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            system_prompt: System prompt (role/persona). Can be:
                - A prompt name (e.g., "researcher") that loads from config/system_prompts/
                - Inline prompt text
                - None (no system prompt)
            **kwargs: Provider-specific args
        """
        # Use defaults if not specified
        if provider is None:
            provider = DEFAULT_PROVIDER
        if model is None:
            model = DEFAULT_MODEL

        # Resolve system prompt (load from file if named, or use as-is)
        resolved_system_prompt = resolve_system_prompt(system_prompt)

        start = time.time()

        if provider == "mlx":
            response = self._call_mlx(prompt, model, temperature, max_tokens, resolved_system_prompt, **kwargs)
        elif provider == "ollama":
            response = self._call_ollama(prompt, model, temperature, max_tokens, resolved_system_prompt, **kwargs)
        elif provider == "anthropic":
            response = self._call_anthropic(prompt, model, temperature, max_tokens, resolved_system_prompt, **kwargs)
        elif provider == "openai":
            response = self._call_openai(prompt, model, temperature, max_tokens, resolved_system_prompt, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        response.latency_s = time.time() - start
        return response

    def call_stream(
        self,
        prompt: str,
        provider: str = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Generator[str, None, LLMResponse]:
        """
        Stream response from LLM provider.

        Yields text chunks as they arrive, then returns final LLMResponse.

        Usage:
            full_text = ""
            for chunk in provider.call_stream(prompt, ...):
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                    full_text += chunk
                else:
                    # Last item is the LLMResponse
                    response = chunk
        """
        # Use defaults if not specified
        if provider is None:
            provider = DEFAULT_PROVIDER
        if model is None:
            model = DEFAULT_MODEL

        # Resolve system prompt
        resolved_system_prompt = resolve_system_prompt(system_prompt)

        _start = time.time()  # noqa: F841

        if provider == "ollama":
            yield from self._call_ollama_stream(
                prompt, model, temperature, max_tokens, resolved_system_prompt, **kwargs
            )
        elif provider == "anthropic":
            yield from self._call_anthropic_stream(
                prompt, model, temperature, max_tokens, resolved_system_prompt, **kwargs
            )
        elif provider == "openai":
            yield from self._call_openai_stream(
                prompt, model, temperature, max_tokens, resolved_system_prompt, **kwargs
            )
        else:
            # Fallback: non-streaming providers just yield full response
            response = self.call(prompt, provider, model, temperature, max_tokens, system_prompt, **kwargs)
            yield response.text
            yield response

    def _call_mlx(
        self, prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: Optional[str], **kwargs
    ) -> LLMResponse:
        """Call MLX local model"""
        try:
            from mlx_lm import generate, load
            from pathlib import Path
            import os

            # Lazy load model
            if self.mlx_model is None or kwargs.get("reload_model"):
                # Check if model is cached
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
                model_cache = cache_dir / f"models--{model.replace('/', '--')}"

                if model_cache.exists():
                    print(f"📦 Loading cached model: {model}")
                else:
                    print(f"⬇️  Downloading model: {model}")
                    print("   (This may take a few minutes - progress bars will appear below)")

                self.mlx_model, self.mlx_tokenizer = load(model)
                print(f"✓ Model loaded successfully")

            # Combine system prompt with user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Generate
            # Note: mlx-lm 0.28.3+ doesn't accept temperature parameter directly
            # Temperature control requires using sampler parameter (not implemented yet)
            output = generate(
                self.mlx_model,
                self.mlx_tokenizer,
                prompt=full_prompt,
                max_tokens=max_tokens,
                verbose=False,
            )

            return LLMResponse(
                text=output,
                model=model,
                provider="mlx",
                latency_s=0.0,  # Will be set by caller
                tokens_in=len(self.mlx_tokenizer.encode(full_prompt)),
                tokens_out=len(self.mlx_tokenizer.encode(output)),
            )

        except ImportError:
            return LLMResponse(
                text="[MLX not installed. Run: pip install mlx mlx-lm]", model=model, provider="mlx", latency_s=0.0
            )

    def _call_ollama(
        self, prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: Optional[str], **kwargs
    ) -> LLMResponse:
        """Call Ollama local model"""
        try:
            import ollama

            # Build options dict with reasoning/thinking parameters
            options = {
                "temperature": temperature,
                "num_predict": max_tokens,
            }

            # Add thinking budget and other reasoning parameters if provided
            reasoning_params = [
                "thinking_budget",  # Budget for reasoning tokens
                "num_ctx",  # Context window size
                "top_k",  # Top-K sampling
                "top_p",  # Top-P sampling
                "repeat_penalty",  # Repetition penalty
                "num_thread",  # Number of threads
                "num_gpu",  # Number of GPUs
                "seed",  # Random seed for reproducibility
            ]
            for param in reasoning_params:
                if param in kwargs:
                    options[param] = kwargs[param]

            # Combine system prompt with user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            response = ollama.generate(model=model, prompt=full_prompt, options=options)

            return LLMResponse(
                text=response["response"],
                model=model,
                provider="ollama",
                latency_s=0.0,  # Will be set by caller
                tokens_in=response.get("prompt_eval_count"),
                tokens_out=response.get("eval_count"),
                metadata={
                    "total_duration": response.get("total_duration"),
                    "thinking_budget": options.get("thinking_budget"),
                },
            )

        except ImportError:
            return LLMResponse(
                text="[Ollama not installed. Run: pip install ollama]", model=model, provider="ollama", latency_s=0.0
            )

    def _call_ollama_stream(
        self, prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: Optional[str], **kwargs
    ) -> Generator[str, None, LLMResponse]:
        """Stream from Ollama local model"""
        try:
            import ollama

            # Build options dict
            options = {
                "temperature": temperature,
                "num_predict": max_tokens,
            }

            # Add reasoning parameters
            reasoning_params = [
                "thinking_budget",
                "num_ctx",
                "top_k",
                "top_p",
                "repeat_penalty",
                "num_thread",
                "num_gpu",
                "seed",
            ]
            for param in reasoning_params:
                if param in kwargs:
                    options[param] = kwargs[param]

            # Combine system prompt with user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            start = time.time()
            full_text = ""
            total_tokens_in = 0
            total_tokens_out = 0

            # Stream the response
            for chunk in ollama.generate(model=model, prompt=full_prompt, options=options, stream=True):
                text_chunk = chunk.get("response", "")
                full_text += text_chunk
                yield text_chunk

                # Update token counts if available
                if "prompt_eval_count" in chunk:
                    total_tokens_in = chunk["prompt_eval_count"]
                if "eval_count" in chunk:
                    total_tokens_out = chunk["eval_count"]

            # Return final response
            latency = time.time() - start
            yield LLMResponse(
                text=full_text,
                model=model,
                provider="ollama",
                latency_s=latency,
                tokens_in=total_tokens_in,
                tokens_out=total_tokens_out,
                metadata={"thinking_budget": options.get("thinking_budget")},
            )

        except ImportError:
            yield "[Ollama not installed. Run: pip install ollama]"
            yield LLMResponse(text="", model=model, provider="ollama", latency_s=0.0)

    def _call_anthropic(
        self, prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: Optional[str], **kwargs
    ) -> LLMResponse:
        """Call Anthropic API"""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

            # Build message create args
            create_args = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

            # Add system prompt if provided (Anthropic native support)
            if system_prompt:
                create_args["system"] = system_prompt

            message = client.messages.create(**create_args)

            # Calculate cost (approximate)
            cost = self._estimate_cost_anthropic(model, message.usage.input_tokens, message.usage.output_tokens)

            return LLMResponse(
                text=message.content[0].text,
                model=model,
                provider="anthropic",
                latency_s=0.0,
                tokens_in=message.usage.input_tokens,
                tokens_out=message.usage.output_tokens,
                cost_usd=cost,
            )

        except ImportError:
            return LLMResponse(
                text="[Anthropic not installed. Run: pip install anthropic]",
                model=model,
                provider="anthropic",
                latency_s=0.0,
            )

    def _call_openai(
        self, prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: Optional[str], **kwargs
    ) -> LLMResponse:
        """Call OpenAI API"""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            # Build messages array
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )

            # Calculate cost (approximate)
            cost = self._estimate_cost_openai(model, response.usage.prompt_tokens, response.usage.completion_tokens)

            return LLMResponse(
                text=response.choices[0].message.content,
                model=model,
                provider="openai",
                latency_s=0.0,
                tokens_in=response.usage.prompt_tokens,
                tokens_out=response.usage.completion_tokens,
                cost_usd=cost,
            )

        except ImportError:
            return LLMResponse(
                text="[OpenAI not installed. Run: pip install openai]", model=model, provider="openai", latency_s=0.0
            )

    def _call_anthropic_stream(
        self, prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: Optional[str], **kwargs
    ) -> Generator[str, None, LLMResponse]:
        """Stream from Anthropic API"""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

            start = time.time()
            full_text = ""
            tokens_in = 0
            tokens_out = 0

            # Build stream args
            stream_args = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

            # Add system prompt if provided
            if system_prompt:
                stream_args["system"] = system_prompt

            # Stream the response
            with client.messages.stream(**stream_args) as stream:
                for text_chunk in stream.text_stream:
                    full_text += text_chunk
                    yield text_chunk

                # Get final message for token counts
                message = stream.get_final_message()
                tokens_in = message.usage.input_tokens
                tokens_out = message.usage.output_tokens

            latency = time.time() - start
            cost = self._estimate_cost_anthropic(model, tokens_in, tokens_out)

            yield LLMResponse(
                text=full_text,
                model=model,
                provider="anthropic",
                latency_s=latency,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
            )

        except ImportError:
            yield "[Anthropic not installed. Run: pip install anthropic]"
            yield LLMResponse(text="", model=model, provider="anthropic", latency_s=0.0)

    def _call_openai_stream(
        self, prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: Optional[str], **kwargs
    ) -> Generator[str, None, LLMResponse]:
        """Stream from OpenAI API"""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            start = time.time()
            full_text = ""
            tokens_in = 0
            tokens_out = 0

            # Build messages array
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Stream the response
            stream = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text_chunk = chunk.choices[0].delta.content
                    full_text += text_chunk
                    yield text_chunk

            latency = time.time() - start

            # Note: Token counts not available in streaming mode for OpenAI
            # Estimate based on text length
            tokens_out = len(full_text.split()) * 1.3  # Rough estimate

            cost = self._estimate_cost_openai(model, tokens_in, tokens_out)

            yield LLMResponse(
                text=full_text,
                model=model,
                provider="openai",
                latency_s=latency,
                tokens_in=None,
                tokens_out=int(tokens_out),
                cost_usd=cost,
            )

        except ImportError:
            yield "[OpenAI not installed. Run: pip install openai]"
            yield LLMResponse(text="", model=model, provider="openai", latency_s=0.0)

    def _estimate_cost_anthropic(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Rough cost estimation for Anthropic"""
        # Prices as of Oct 2025 (check https://www.anthropic.com/pricing for updates)
        prices = {
            "claude-3-5-sonnet-20241022": (3.0, 15.0),  # per 1M tokens (input, output)
            "claude-3-5-haiku-20241022": (1.0, 5.0),
            "claude-3-opus-20240229": (15.0, 75.0),
        }

        if model in prices:
            input_price, output_price = prices[model]
            return (tokens_in * input_price + tokens_out * output_price) / 1_000_000
        return 0.0

    def _estimate_cost_openai(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Rough cost estimation for OpenAI"""
        # Prices as of Oct 2025 (check https://openai.com/pricing for updates)
        prices = {
            "gpt-4": (30.0, 60.0),
            "gpt-4-turbo": (10.0, 30.0),
            "gpt-4o": (2.5, 10.0),
            "gpt-4o-mini": (0.15, 0.6),
            "gpt-3.5-turbo": (0.5, 1.5),
        }

        for key, (input_price, output_price) in prices.items():
            if key in model:
                return (tokens_in * input_price + tokens_out * output_price) / 1_000_000
        return 0.0


# Singleton instance
_provider = None


def get_provider() -> LLMProvider:
    """Get or create singleton provider"""
    global _provider
    if _provider is None:
        _provider = LLMProvider()
    return _provider


def llm_call(
    prompt: str, provider: str = None, model: str = None, system_prompt: Optional[str] = None, **kwargs
) -> LLMResponse:
    """
    Convenience function for calling LLMs.

    Defaults to DEFAULT_PROVIDER and DEFAULT_MODEL if not specified.

    Examples:
        # Use defaults (from defaults.py)
        response = llm_call("What is 2+2?")

        # Local Ollama
        response = llm_call("What is 2+2?", provider="ollama", model="gpt-oss:20b")

        # Local MLX
        response = llm_call("What is 2+2?", provider="mlx", model="mlx-community/Llama-3.2-3B-Instruct-4bit")

        # Anthropic API
        response = llm_call("What is 2+2?", provider="anthropic", model="claude-3-5-sonnet-20241022")

        # With thinking budget
        response = llm_call("Complex problem...", thinking_budget=2000)

        # With system prompt (name or inline)
        response = llm_call("Design a new architecture", system_prompt="researcher")
        response = llm_call("Question", system_prompt="You are an expert in...")
    """
    return get_provider().call(prompt, provider=provider, model=model, system_prompt=system_prompt, **kwargs)


def llm_call_stream(
    prompt: str, provider: str = None, model: str = None, system_prompt: Optional[str] = None, **kwargs
) -> Generator[str, None, LLMResponse]:
    """
    Convenience function for streaming LLM calls.

    Defaults to DEFAULT_PROVIDER and DEFAULT_MODEL if not specified.

    Yields text chunks as they arrive, with final LLMResponse at the end.

    Examples:
        # Use defaults (from defaults.py)
        for chunk in llm_call_stream("Tell me a story"):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
            else:
                response = chunk  # Final LLMResponse

        # With thinking budget
        for chunk in llm_call_stream("Solve this...", thinking_budget=2000):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)

        # With system prompt
        for chunk in llm_call_stream("Question", system_prompt="researcher"):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
    """
    return get_provider().call_stream(prompt, provider=provider, model=model, system_prompt=system_prompt, **kwargs)
