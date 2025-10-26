"""
LLM Provider Layer - supports local MLX models, Ollama, and API providers
"""
import os
from typing import Optional, Dict, Any, Generator
from dataclasses import dataclass
import time
import sys

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
        provider: str = "ollama",
        model: str = "llama3.2:latest",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        """
        Route to appropriate provider.
        
        Args:
            prompt: The input prompt
            provider: "mlx", "ollama", "anthropic", "openai"
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            **kwargs: Provider-specific args
        """
        start = time.time()
        
        if provider == "mlx":
            response = self._call_mlx(prompt, model, temperature, max_tokens, **kwargs)
        elif provider == "ollama":
            response = self._call_ollama(prompt, model, temperature, max_tokens, **kwargs)
        elif provider == "anthropic":
            response = self._call_anthropic(prompt, model, temperature, max_tokens, **kwargs)
        elif provider == "openai":
            response = self._call_openai(prompt, model, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        response.latency_s = time.time() - start
        return response

    def call_stream(
        self,
        prompt: str,
        provider: str = "ollama",
        model: str = "llama3.2:latest",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
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
        start = time.time()

        if provider == "ollama":
            yield from self._call_ollama_stream(prompt, model, temperature, max_tokens, **kwargs)
        elif provider == "anthropic":
            yield from self._call_anthropic_stream(prompt, model, temperature, max_tokens, **kwargs)
        elif provider == "openai":
            yield from self._call_openai_stream(prompt, model, temperature, max_tokens, **kwargs)
        else:
            # Fallback: non-streaming providers just yield full response
            response = self.call(prompt, provider, model, temperature, max_tokens, **kwargs)
            yield response.text
            yield response

    def _call_mlx(self, prompt: str, model: str, temperature: float, max_tokens: int, **kwargs) -> LLMResponse:
        """Call MLX local model"""
        try:
            import mlx.core as mx
            from mlx_lm import load, generate
            
            # Lazy load model
            if self.mlx_model is None or kwargs.get('reload_model'):
                print(f"Loading MLX model: {model}")
                self.mlx_model, self.mlx_tokenizer = load(model)
            
            # Generate
            output = generate(
                self.mlx_model,
                self.mlx_tokenizer,
                prompt=prompt,
                temp=temperature,
                max_tokens=max_tokens,
                verbose=False
            )
            
            return LLMResponse(
                text=output,
                model=model,
                provider="mlx",
                latency_s=0.0,  # Will be set by caller
                tokens_in=len(self.mlx_tokenizer.encode(prompt)),
                tokens_out=len(self.mlx_tokenizer.encode(output))
            )
            
        except ImportError:
            return LLMResponse(
                text="[MLX not installed. Run: pip install mlx mlx-lm]",
                model=model,
                provider="mlx",
                latency_s=0.0
            )
    
    def _call_ollama(self, prompt: str, model: str, temperature: float, max_tokens: int, **kwargs) -> LLMResponse:
        """Call Ollama local model"""
        try:
            import ollama

            # Build options dict with reasoning/thinking parameters
            options = {
                'temperature': temperature,
                'num_predict': max_tokens,
            }

            # Add thinking budget and other reasoning parameters if provided
            reasoning_params = [
                'thinking_budget',  # Budget for reasoning tokens
                'num_ctx',          # Context window size
                'top_k',            # Top-K sampling
                'top_p',            # Top-P sampling
                'repeat_penalty',   # Repetition penalty
                'num_thread',       # Number of threads
                'num_gpu',          # Number of GPUs
                'seed',             # Random seed for reproducibility
            ]
            for param in reasoning_params:
                if param in kwargs:
                    options[param] = kwargs[param]

            response = ollama.generate(
                model=model,
                prompt=prompt,
                options=options
            )

            return LLMResponse(
                text=response['response'],
                model=model,
                provider="ollama",
                latency_s=0.0,  # Will be set by caller
                tokens_in=response.get('prompt_eval_count'),
                tokens_out=response.get('eval_count'),
                metadata={
                    'total_duration': response.get('total_duration'),
                    'thinking_budget': options.get('thinking_budget')
                }
            )
            
        except ImportError:
            return LLMResponse(
                text="[Ollama not installed. Run: pip install ollama]",
                model=model,
                provider="ollama",
                latency_s=0.0
            )

    def _call_ollama_stream(self, prompt: str, model: str, temperature: float, max_tokens: int, **kwargs) -> Generator[str, None, LLMResponse]:
        """Stream from Ollama local model"""
        try:
            import ollama

            # Build options dict
            options = {
                'temperature': temperature,
                'num_predict': max_tokens,
            }

            # Add reasoning parameters
            reasoning_params = [
                'thinking_budget', 'num_ctx', 'top_k', 'top_p',
                'repeat_penalty', 'num_thread', 'num_gpu', 'seed'
            ]
            for param in reasoning_params:
                if param in kwargs:
                    options[param] = kwargs[param]

            start = time.time()
            full_text = ""
            total_tokens_in = 0
            total_tokens_out = 0

            # Stream the response
            for chunk in ollama.generate(
                model=model,
                prompt=prompt,
                options=options,
                stream=True
            ):
                text_chunk = chunk.get('response', '')
                full_text += text_chunk
                yield text_chunk

                # Update token counts if available
                if 'prompt_eval_count' in chunk:
                    total_tokens_in = chunk['prompt_eval_count']
                if 'eval_count' in chunk:
                    total_tokens_out = chunk['eval_count']

            # Return final response
            latency = time.time() - start
            yield LLMResponse(
                text=full_text,
                model=model,
                provider="ollama",
                latency_s=latency,
                tokens_in=total_tokens_in,
                tokens_out=total_tokens_out,
                metadata={'thinking_budget': options.get('thinking_budget')}
            )

        except ImportError:
            yield "[Ollama not installed. Run: pip install ollama]"
            yield LLMResponse(
                text="",
                model=model,
                provider="ollama",
                latency_s=0.0
            )

    def _call_anthropic(self, prompt: str, model: str, temperature: float, max_tokens: int, **kwargs) -> LLMResponse:
        """Call Anthropic API"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Calculate cost (approximate)
            cost = self._estimate_cost_anthropic(model, message.usage.input_tokens, message.usage.output_tokens)
            
            return LLMResponse(
                text=message.content[0].text,
                model=model,
                provider="anthropic",
                latency_s=0.0,
                tokens_in=message.usage.input_tokens,
                tokens_out=message.usage.output_tokens,
                cost_usd=cost
            )
            
        except ImportError:
            return LLMResponse(
                text="[Anthropic not installed. Run: pip install anthropic]",
                model=model,
                provider="anthropic",
                latency_s=0.0
            )
    
    def _call_openai(self, prompt: str, model: str, temperature: float, max_tokens: int, **kwargs) -> LLMResponse:
        """Call OpenAI API"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
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
                cost_usd=cost
            )
            
        except ImportError:
            return LLMResponse(
                text="[OpenAI not installed. Run: pip install openai]",
                model=model,
                provider="openai",
                latency_s=0.0
            )

    def _call_anthropic_stream(self, prompt: str, model: str, temperature: float, max_tokens: int, **kwargs) -> Generator[str, None, LLMResponse]:
        """Stream from Anthropic API"""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

            start = time.time()
            full_text = ""
            tokens_in = 0
            tokens_out = 0

            # Stream the response
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
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
                cost_usd=cost
            )

        except ImportError:
            yield "[Anthropic not installed. Run: pip install anthropic]"
            yield LLMResponse(text="", model=model, provider="anthropic", latency_s=0.0)

    def _call_openai_stream(self, prompt: str, model: str, temperature: float, max_tokens: int, **kwargs) -> Generator[str, None, LLMResponse]:
        """Stream from OpenAI API"""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            start = time.time()
            full_text = ""
            tokens_in = 0
            tokens_out = 0

            # Stream the response
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
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
                cost_usd=cost
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


def llm_call(prompt: str, provider: str = "ollama", model: str = "llama3.2:latest", **kwargs) -> LLMResponse:
    """
    Convenience function for calling LLMs.

    Examples:
        # Local Ollama
        response = llm_call("What is 2+2?", provider="ollama", model="llama3.2:latest")

        # Local MLX
        response = llm_call("What is 2+2?", provider="mlx", model="mlx-community/Llama-3.2-3B-Instruct-4bit")

        # Anthropic API
        response = llm_call("What is 2+2?", provider="anthropic", model="claude-3-5-sonnet-20241022")

        # With thinking budget
        response = llm_call("Complex problem...", provider="ollama",
                           model="gpt-oss:20b", thinking_budget=2000)
    """
    return get_provider().call(prompt, provider=provider, model=model, **kwargs)


def llm_call_stream(prompt: str, provider: str = "ollama", model: str = "llama3.2:latest", **kwargs) -> Generator[str, None, LLMResponse]:
    """
    Convenience function for streaming LLM calls.

    Yields text chunks as they arrive, with final LLMResponse at the end.

    Examples:
        # Stream and print in real-time
        for chunk in llm_call_stream("Tell me a story", provider="ollama"):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
            else:
                response = chunk  # Final LLMResponse

        # With thinking budget
        for chunk in llm_call_stream("Solve this...", provider="ollama",
                                     model="gpt-oss:20b", thinking_budget=2000):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
    """
    return get_provider().call_stream(prompt, provider=provider, model=model, **kwargs)
