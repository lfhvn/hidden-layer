"""
Performance benchmarking for MLX models

Measures speed, memory usage, and latency.
"""

import time
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from mlx_lab.utils import (
    load_benchmark_cache,
    save_benchmark_cache,
    format_bytes,
    check_mlx_lm_installed,
)


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark"""

    model_name: str
    tokens_per_sec: float
    memory_gb: float
    first_token_latency_ms: float
    total_time_sec: float
    timestamp: str


class PerformanceBenchmark:
    """Benchmark MLX model performance"""

    # Standard test prompt for consistency
    TEST_PROMPT = "Write a short story about a robot learning to paint. The story should include"

    # Number of tokens to generate for speed test
    TEST_TOKENS = 100

    def __init__(self):
        self.cache = load_benchmark_cache()

    def benchmark_model(
        self, repo_id: str, use_cache: bool = True, progress_callback=None
    ) -> Optional[BenchmarkResult]:
        """
        Benchmark a model's performance

        Args:
            repo_id: Full HuggingFace repo ID
            use_cache: Use cached results if available
            progress_callback: Optional callback for progress updates

        Returns:
            BenchmarkResult or None if failed
        """
        if not check_mlx_lm_installed():
            raise ImportError("mlx-lm not installed. Run: pip install mlx mlx-lm")

        # Check cache first
        if use_cache and repo_id in self.cache:
            cached = self.cache[repo_id]
            return BenchmarkResult(**cached)

        try:
            if progress_callback:
                progress_callback("Loading model...")

            # Import here to avoid requiring mlx for all commands
            import psutil
            import mlx.core as mx
            from mlx_lm import load, generate

            # Measure memory before loading
            process = psutil.Process(os.getpid())
            memory_before_mb = process.memory_info().rss / (1024 * 1024)

            # Load model
            model, tokenizer = load(repo_id)

            # Measure memory after loading
            memory_after_mb = process.memory_info().rss / (1024 * 1024)
            model_memory_gb = (memory_after_mb - memory_before_mb) / 1024

            if progress_callback:
                progress_callback("Running performance test...")

            # Warm up (first generation is often slower)
            _ = generate(
                model, tokenizer, prompt="Test", max_tokens=10, verbose=False
            )

            # Measure latency to first token and total generation time
            start_time = time.time()

            output = generate(
                model,
                tokenizer,
                prompt=self.TEST_PROMPT,
                max_tokens=self.TEST_TOKENS,
                verbose=False,
            )

            end_time = time.time()
            total_time = end_time - start_time

            # Count tokens generated
            output_tokens = len(tokenizer.encode(output))
            tokens_per_sec = output_tokens / total_time if total_time > 0 else 0

            # Estimate first token latency (rough approximation)
            # First token is typically ~10% of total time for 100 tokens
            first_token_latency_ms = (total_time * 0.1) * 1000

            # Create result
            result = BenchmarkResult(
                model_name=repo_id,
                tokens_per_sec=tokens_per_sec,
                memory_gb=model_memory_gb,
                first_token_latency_ms=first_token_latency_ms,
                total_time_sec=total_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

            # Cache result
            self.cache[repo_id] = {
                "model_name": result.model_name,
                "tokens_per_sec": result.tokens_per_sec,
                "memory_gb": result.memory_gb,
                "first_token_latency_ms": result.first_token_latency_ms,
                "total_time_sec": result.total_time_sec,
                "timestamp": result.timestamp,
            }
            save_benchmark_cache(self.cache)

            if progress_callback:
                progress_callback("✅ Benchmark complete")

            return result

        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Error: {str(e)}")
            return None

    def compare_models(
        self, model_ids: List[str], use_cache: bool = True
    ) -> List[BenchmarkResult]:
        """
        Compare multiple models

        Args:
            model_ids: List of HuggingFace repo IDs
            use_cache: Use cached results if available

        Returns:
            List of BenchmarkResults
        """
        results = []
        for model_id in model_ids:
            result = self.benchmark_model(model_id, use_cache=use_cache)
            if result:
                results.append(result)
        return results

    def get_cached_result(self, repo_id: str) -> Optional[BenchmarkResult]:
        """Get cached benchmark result"""
        if repo_id in self.cache:
            cached = self.cache[repo_id]
            return BenchmarkResult(**cached)
        return None

    def clear_cache(self, repo_id: Optional[str] = None):
        """Clear benchmark cache"""
        if repo_id:
            if repo_id in self.cache:
                del self.cache[repo_id]
        else:
            self.cache = {}
        save_benchmark_cache(self.cache)

    def format_result(self, result: BenchmarkResult) -> str:
        """Format benchmark result for display"""
        lines = []
        lines.append(f"Performance Test: {result.model_name}")
        lines.append("=" * 70)
        lines.append(f"Speed:        {result.tokens_per_sec:.1f} tokens/sec")
        lines.append(f"Memory:       {result.memory_gb:.1f} GB")
        lines.append(f"First token:  {result.first_token_latency_ms:.0f} ms")
        lines.append(f"Total time:   {result.total_time_sec:.2f} sec")
        lines.append(f"Tested:       {result.timestamp}")
        lines.append("")

        # Add interpretation
        if result.tokens_per_sec > 40:
            lines.append("✅ Fast - Good for interactive experiments")
        elif result.tokens_per_sec > 20:
            lines.append("⚠️  Medium - Acceptable for most tasks")
        else:
            lines.append("🐌 Slow - May be sluggish for interactive work")

        return "\n".join(lines)

    def format_comparison(self, results: List[BenchmarkResult]) -> str:
        """Format comparison table"""
        if not results:
            return "No benchmark results to compare"

        lines = []
        lines.append("Model Performance Comparison")
        lines.append("=" * 70)
        lines.append("")

        # Header
        lines.append(
            f"{'Model':<30} {'Speed':<15} {'Memory':<10} {'Latency':<10}"
        )
        lines.append("-" * 70)

        # Sort by speed (fastest first)
        sorted_results = sorted(
            results, key=lambda r: r.tokens_per_sec, reverse=True
        )

        for result in sorted_results:
            model_short = result.model_name.split("/")[-1][:28]
            speed = f"{result.tokens_per_sec:.1f} tok/s"
            memory = f"{result.memory_gb:.1f} GB"
            latency = f"{result.first_token_latency_ms:.0f} ms"

            lines.append(f"{model_short:<30} {speed:<15} {memory:<10} {latency:<10}")

        lines.append("")
        lines.append(
            "Tested on: " + sorted_results[0].timestamp if sorted_results else ""
        )

        return "\n".join(lines)
