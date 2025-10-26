"""
LLM Provider Layer - supports local MLX models, Ollama, and API providers
"""
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
import time

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
            
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                }
            )
            
            return LLMResponse(
                text=response['response'],
                model=model,
                provider="ollama",
                latency_s=0.0,  # Will be set by caller
                tokens_in=response.get('prompt_eval_count'),
                tokens_out=response.get('eval_count'),
                metadata={'total_duration': response.get('total_duration')}
            )
            
        except ImportError:
            return LLMResponse(
                text="[Ollama not installed. Run: pip install ollama]",
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
    
    def _estimate_cost_anthropic(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Rough cost estimation for Anthropic"""
        # Prices as of Oct 2024 (update as needed)
        prices = {
            "claude-3-5-sonnet-20241022": (3.0, 15.0),  # per 1M tokens (input, output)
            "claude-3-5-haiku-20241022": (1.0, 5.0),
        }
        
        if model in prices:
            input_price, output_price = prices[model]
            return (tokens_in * input_price + tokens_out * output_price) / 1_000_000
        return 0.0
    
    def _estimate_cost_openai(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Rough cost estimation for OpenAI"""
        # Prices as of Oct 2024 (update as needed)
        prices = {
            "gpt-4": (30.0, 60.0),
            "gpt-4-turbo": (10.0, 30.0),
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
    """
    return get_provider().call(prompt, provider=provider, model=model, **kwargs)
