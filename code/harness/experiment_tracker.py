"""
Experiment tracking and logging system.
Designed for notebook-based research with easy integration.

Optional W&B integration for professional dashboards and visualization.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
import hashlib
import os

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run"""
    experiment_name: str
    task_type: str  # e.g., "reasoning", "planning", "creative"
    strategy: str  # e.g., "single", "debate", "manager-worker"
    
    # Model config
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    
    # Strategy-specific params
    n_agents: Optional[int] = None
    debate_rounds: Optional[int] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    git_hash: Optional[str] = None
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def hash(self) -> str:
        """Generate unique hash for this configuration"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class ExperimentResult:
    """Single run result"""
    config: ExperimentConfig
    
    # Input/Output
    task_input: str
    output: str
    
    # Metrics
    latency_s: float
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_usd: Optional[float] = None
    
    # Evaluation (to be filled by eval functions)
    eval_scores: Dict[str, float] = field(default_factory=dict)
    eval_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Success/failure
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'config': self.config.to_dict(),
            'task_input': self.task_input,
            'output': self.output,
            'latency_s': self.latency_s,
            'tokens_in': self.tokens_in,
            'tokens_out': self.tokens_out,
            'cost_usd': self.cost_usd,
            'eval_scores': self.eval_scores,
            'eval_metadata': self.eval_metadata,
            'success': self.success,
            'error': self.error
        }


class ExperimentTracker:
    """
    Tracks experiments with automatic logging to disk.
    Designed for notebook-based workflows.

    Optional W&B integration for professional tracking and visualization.
    """

    def __init__(self, base_dir: str = "./experiments", use_wandb: bool = False):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)

        self.current_experiment: Optional[str] = None
        self.current_run_dir: Optional[Path] = None
        self.results: List[ExperimentResult] = []

        # W&B integration (optional)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if use_wandb and not WANDB_AVAILABLE:
            print("⚠️  W&B requested but not installed. Run: pip install wandb")
            print("   Continuing with local tracking only.")
    
    def start_experiment(self, config: ExperimentConfig) -> Path:
        """
        Start a new experiment run.
        Creates a directory: experiments/{experiment_name}_{timestamp}_{hash}/

        If W&B enabled, also initializes W&B run.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = config.hash()

        run_name = f"{config.experiment_name}_{timestamp}_{config_hash}"
        self.current_run_dir = self.base_dir / run_name
        self.current_run_dir.mkdir(exist_ok=True, parents=True)

        self.current_experiment = run_name
        self.results = []

        # Save config
        with open(self.current_run_dir / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        print(f"📊 Started experiment: {run_name}")
        print(f"📁 Logging to: {self.current_run_dir}")

        # Initialize W&B if enabled
        if self.use_wandb:
            wandb.init(
                project="hidden-layer",
                name=config.experiment_name,
                config={
                    "strategy": config.strategy,
                    "provider": config.provider,
                    "model": config.model,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "n_agents": config.n_agents,
                    "task_type": config.task_type,
                },
                tags=[config.strategy, config.provider, config.task_type],
            )
            print(f"🚀 W&B tracking: {wandb.run.get_url()}")

        return self.current_run_dir
    
    def log_result(self, result: ExperimentResult):
        """Log a single result"""
        self.results.append(result)

        # Append to results.jsonl (one JSON per line for easy streaming)
        with open(self.current_run_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

        # Log to W&B if enabled
        if self.use_wandb:
            log_dict = {
                "latency_s": result.latency_s,
                "success": result.success,
            }

            # Add optional metrics if available
            if result.tokens_in is not None:
                log_dict["tokens_in"] = result.tokens_in
            if result.tokens_out is not None:
                log_dict["tokens_out"] = result.tokens_out
            if result.cost_usd is not None:
                log_dict["cost_usd"] = result.cost_usd

            # Log eval scores
            for key, value in result.eval_scores.items():
                log_dict[f"eval/{key}"] = value

            wandb.log(log_dict)
    
    def finish_experiment(self):
        """
        Finish experiment and generate summary.
        """
        if not self.results:
            print("⚠️  No results to summarize")
            if self.use_wandb:
                wandb.finish()
            return

        summary = self._generate_summary()

        # Save summary
        with open(self.current_run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Experiment complete: {self.current_experiment}")
        print(f"📊 Results: {summary['total_runs']} runs")
        print(f"⏱️  Avg latency: {summary['avg_latency_s']:.2f}s")
        if summary['total_cost_usd'] > 0:
            print(f"💰 Total cost: ${summary['total_cost_usd']:.4f}")

        # Log summary to W&B and finish run
        if self.use_wandb:
            # Log aggregate metrics
            wandb.run.summary.update({
                "total_runs": summary['total_runs'],
                "successful_runs": summary['successful_runs'],
                "success_rate": summary['success_rate'],
                "avg_latency_s": summary['avg_latency_s'],
                "total_cost_usd": summary['total_cost_usd'],
                "avg_tokens_in": summary['avg_tokens_in'],
                "avg_tokens_out": summary['avg_tokens_out'],
                "total_tokens": summary['total_tokens'],
            })

            # Log eval score aggregates
            for eval_name, stats in summary['eval_scores'].items():
                wandb.run.summary[f"eval_summary/{eval_name}/mean"] = stats['mean']
                wandb.run.summary[f"eval_summary/{eval_name}/min"] = stats['min']
                wandb.run.summary[f"eval_summary/{eval_name}/max"] = stats['max']

            wandb.finish()
            print("📊 W&B run finished")

        return summary
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_runs = len(self.results)
        successful_runs = sum(1 for r in self.results if r.success)
        
        latencies = [r.latency_s for r in self.results if r.latency_s]
        costs = [r.cost_usd for r in self.results if r.cost_usd]
        tokens_in = [r.tokens_in for r in self.results if r.tokens_in]
        tokens_out = [r.tokens_out for r in self.results if r.tokens_out]
        
        # Aggregate eval scores
        eval_aggregates = {}
        if self.results[0].eval_scores:
            for key in self.results[0].eval_scores.keys():
                scores = [r.eval_scores[key] for r in self.results if key in r.eval_scores]
                if scores:
                    eval_aggregates[key] = {
                        'mean': sum(scores) / len(scores),
                        'min': min(scores),
                        'max': max(scores),
                        'count': len(scores)
                    }
        
        return {
            'experiment': self.current_experiment,
            'total_runs': total_runs,
            'successful_runs': successful_runs,
            'success_rate': successful_runs / total_runs if total_runs > 0 else 0,
            
            'avg_latency_s': sum(latencies) / len(latencies) if latencies else 0,
            'total_cost_usd': sum(costs) if costs else 0,
            
            'avg_tokens_in': sum(tokens_in) / len(tokens_in) if tokens_in else 0,
            'avg_tokens_out': sum(tokens_out) / len(tokens_out) if tokens_out else 0,
            'total_tokens': sum(tokens_in or []) + sum(tokens_out or []),
            
            'eval_scores': eval_aggregates,
            
            'timestamp': datetime.now().isoformat()
        }
    
    def load_experiment(self, run_dir: str) -> List[ExperimentResult]:
        """Load results from a previous experiment"""
        run_path = Path(run_dir)
        
        results = []
        with open(run_path / "results.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                # Reconstruct objects (simplified, doesn't fully restore)
                results.append(data)
        
        return results


# Singleton tracker
_tracker = None

def get_tracker(use_wandb: bool = False) -> ExperimentTracker:
    """Get or create singleton tracker"""
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker(use_wandb=use_wandb)
    return _tracker


def compare_experiments(run_dirs: List[str], metric: str = "latency_s") -> Dict[str, Any]:
    """
    Compare multiple experiment runs.
    
    Args:
        run_dirs: List of experiment directory paths
        metric: Metric to compare (e.g., "latency_s", "cost_usd", or eval score name)
    
    Returns:
        Comparison dict with stats for each run
    """
    tracker = get_tracker()
    comparison = {}
    
    for run_dir in run_dirs:
        results = tracker.load_experiment(run_dir)
        run_name = Path(run_dir).name
        
        if metric in ["latency_s", "cost_usd", "tokens_in", "tokens_out"]:
            values = [r.get(metric) for r in results if r.get(metric) is not None]
        else:
            # Assume it's an eval score
            values = [r.get('eval_scores', {}).get(metric) for r in results 
                     if r.get('eval_scores', {}).get(metric) is not None]
        
        if values:
            comparison[run_name] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
    
    return comparison
