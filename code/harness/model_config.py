"""
Model configuration management system.

Allows defining and loading model-specific configurations including:
- System prompts
- Temperature and sampling parameters
- Thinking budgets and reasoning controls
- Provider-specific options
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """Configuration for a specific model setup"""

    # Identity
    name: str
    provider: str
    model: str

    # System prompts
    system_prompt: Optional[str] = None

    # Sampling parameters
    temperature: float = 0.7
    max_tokens: int = 2048
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    # Reasoning parameters
    thinking_budget: Optional[int] = None
    num_ctx: Optional[int] = None
    repeat_penalty: Optional[float] = None

    # Other options
    seed: Optional[int] = None

    # Metadata
    description: Optional[str] = None
    tags: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None and k not in ['name', 'description', 'tags']}

    def to_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for llm_call/run_strategy"""
        kwargs = {
            'provider': self.provider,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }

        # Add optional parameters
        if self.system_prompt:
            kwargs['system_prompt'] = self.system_prompt
        if self.top_k:
            kwargs['top_k'] = self.top_k
        if self.top_p:
            kwargs['top_p'] = self.top_p
        if self.thinking_budget:
            kwargs['thinking_budget'] = self.thinking_budget
        if self.num_ctx:
            kwargs['num_ctx'] = self.num_ctx
        if self.repeat_penalty:
            kwargs['repeat_penalty'] = self.repeat_penalty
        if self.seed:
            kwargs['seed'] = self.seed

        return kwargs


class ModelConfigManager:
    """Manages loading and saving model configurations"""

    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True, parents=True)
        self.config_file = self.config_dir / "models.yaml"
        self._configs: Dict[str, ModelConfig] = {}
        self._load_configs()

    def _load_configs(self):
        """Load configurations from YAML file"""
        if not self.config_file.exists():
            return

        try:
            with open(self.config_file, 'r') as f:
                data = yaml.safe_load(f) or {}

            for name, config_data in data.items():
                config_data['name'] = name
                self._configs[name] = ModelConfig(**config_data)
        except Exception as e:
            print(f"Warning: Could not load model configs: {e}")

    def get(self, name: str) -> Optional[ModelConfig]:
        """Get a configuration by name"""
        return self._configs.get(name)

    def list(self) -> Dict[str, ModelConfig]:
        """List all available configurations"""
        return self._configs.copy()

    def add(self, config: ModelConfig, save: bool = True):
        """Add a new configuration"""
        self._configs[config.name] = config
        if save:
            self.save()

    def save(self):
        """Save configurations to YAML file"""
        data = {}
        for name, config in self._configs.items():
            config_dict = asdict(config)
            # Remove name from the dict as it's the key
            config_dict.pop('name')
            # Remove None values
            data[name] = {k: v for k, v in config_dict.items() if v is not None}

        with open(self.config_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def create_default_configs(self):
        """Create default configurations"""
        defaults = [
            ModelConfig(
                name="gpt-oss-20b-reasoning",
                provider="ollama",
                model="gpt-oss:20b",
                temperature=0.7,
                thinking_budget=2000,
                num_ctx=4096,
                description="GPT-OSS 20B with extended reasoning budget",
                tags=["reasoning", "large"]
            ),
            ModelConfig(
                name="gpt-oss-20b-creative",
                provider="ollama",
                model="gpt-oss:20b",
                temperature=0.9,
                top_p=0.95,
                num_ctx=4096,
                description="GPT-OSS 20B optimized for creative tasks",
                tags=["creative", "large"]
            ),
            ModelConfig(
                name="gpt-oss-20b-precise",
                provider="ollama",
                model="gpt-oss:20b",
                temperature=0.3,
                top_k=40,
                num_ctx=4096,
                seed=42,
                description="GPT-OSS 20B for precise, deterministic outputs",
                tags=["precise", "deterministic"]
            ),
            ModelConfig(
                name="llama3.2-fast",
                provider="ollama",
                model="llama3.2:latest",
                temperature=0.7,
                max_tokens=1024,
                description="Llama 3.2 3B for fast iteration",
                tags=["fast", "small"]
            ),
            ModelConfig(
                name="claude-sonnet",
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                temperature=0.7,
                max_tokens=4096,
                description="Claude 3.5 Sonnet for high-quality outputs",
                tags=["api", "premium"]
            ),
            ModelConfig(
                name="claude-haiku",
                provider="anthropic",
                model="claude-3-5-haiku-20241022",
                temperature=0.7,
                max_tokens=2048,
                description="Claude 3.5 Haiku for fast, cost-effective API usage",
                tags=["api", "fast", "cheap"]
            ),
        ]

        for config in defaults:
            self.add(config, save=False)

        self.save()
        return defaults


# Singleton instance
_manager = None

def get_config_manager() -> ModelConfigManager:
    """Get or create singleton config manager"""
    global _manager
    if _manager is None:
        _manager = ModelConfigManager()
    return _manager


def get_model_config(name: str) -> Optional[ModelConfig]:
    """Convenience function to get a model config by name"""
    return get_config_manager().get(name)


def list_model_configs() -> Dict[str, ModelConfig]:
    """Convenience function to list all model configs"""
    return get_config_manager().list()
