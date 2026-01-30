"""
Configuration for the AI Research Aggregator.

Defines user interests, source settings, and LLM preferences.
Loads from a YAML config file with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "default_config.yaml"
)

USER_CONFIG_PATH = os.path.expanduser("~/.config/ai-research-aggregator/config.yaml")


@dataclass
class UserInterests:
    """Topics and themes the user is most interested in."""

    # Core research topics (used for ranking relevance)
    topics: List[str] = field(default_factory=lambda: [
        "agent communication and coordination",
        "theory of mind in AI",
        "model interpretability and mechanistic interpretability",
        "AI alignment and safety",
        "AI steerability and control",
        "sparse autoencoders (SAEs)",
        "activation steering and representation engineering",
        "multi-agent systems",
        "LLM self-knowledge and introspection",
        "latent space representations",
        "AI deception detection",
        "emergent capabilities in language models",
        "chain-of-thought reasoning",
        "RLHF and preference learning",
        "long-context and memory in LLMs",
    ])

    # Extra search terms for paper sources
    search_terms: List[str] = field(default_factory=lambda: [
        "interpretability",
        "alignment",
        "multi-agent",
        "theory of mind",
        "sparse autoencoder",
        "steering vectors",
        "activation engineering",
        "chain of thought",
        "model introspection",
        "AI safety",
    ])

    # Key figures to track
    key_figures: List[str] = field(default_factory=lambda: [
        "Dario Amodei",
        "Daniela Amodei",
        "Ilya Sutskever",
        "Yann LeCun",
        "Andrej Karpathy",
        "Sam Altman",
        "Demis Hassabis",
        "Geoffrey Hinton",
        "Yoshua Bengio",
        "Jan Leike",
        "Chris Olah",
        "Sasha Rush",
        "Percy Liang",
        "Jacob Steinhardt",
        "Paul Christiano",
        "Eliezer Yudkowsky",
    ])

    # arXiv categories to monitor
    arxiv_categories: List[str] = field(default_factory=lambda: [
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "cs.CV",
        "cs.MA",
        "cs.NE",
        "stat.ML",
    ])


@dataclass
class SourceSettings:
    """Settings for content sources."""

    # Maximum items to fetch per source
    max_papers: int = 50
    max_blog_posts: int = 30
    max_community_posts: int = 30
    max_events: int = 20

    # Days to look back
    papers_days_back: int = 3
    community_days_back: int = 3

    # Reddit subreddits
    subreddits: List[str] = field(default_factory=lambda: [
        "MachineLearning",
        "artificial",
        "LocalLLaMA",
    ])

    # Enable/disable individual sources
    enable_arxiv: bool = True
    enable_blogs: bool = True
    enable_communities: bool = True
    enable_events: bool = True


@dataclass
class LLMSettings:
    """Settings for LLM-based ranking and summarization."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3
    max_tokens: int = 4096

    # How many items to send to LLM for ranking at once (batch size)
    ranking_batch_size: int = 15

    # Maximum items to include in final digest per section
    top_papers: int = 10
    top_blog_posts: int = 8
    top_community: int = 8
    top_events: int = 10


@dataclass
class OutputSettings:
    """Settings for digest output."""

    # Output directory for generated digests
    output_dir: str = "digests"

    # Formats to generate
    markdown: bool = True
    terminal: bool = True

    # Include these sections
    show_papers: bool = True
    show_blogs: bool = True
    show_community: bool = True
    show_events: bool = True


@dataclass
class AggregatorConfig:
    """Full aggregator configuration."""

    interests: UserInterests = field(default_factory=UserInterests)
    sources: SourceSettings = field(default_factory=SourceSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "AggregatorConfig":
        """Load config from YAML file, falling back to defaults."""
        config = cls()

        # Try user config first, then explicit path
        config_path = path or USER_CONFIG_PATH

        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    data = yaml.safe_load(f) or {}
                config = cls._from_dict(data)
            except Exception as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")
                print("Using default configuration.")

        return config

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "AggregatorConfig":
        """Create config from a dictionary (parsed YAML)."""
        config = cls()

        if "interests" in data:
            interests = data["interests"]
            if "topics" in interests:
                config.interests.topics = interests["topics"]
            if "search_terms" in interests:
                config.interests.search_terms = interests["search_terms"]
            if "key_figures" in interests:
                config.interests.key_figures = interests["key_figures"]
            if "arxiv_categories" in interests:
                config.interests.arxiv_categories = interests["arxiv_categories"]

        if "sources" in data:
            src = data["sources"]
            for attr in (
                "max_papers", "max_blog_posts", "max_community_posts",
                "max_events", "papers_days_back", "community_days_back",
                "subreddits", "enable_arxiv", "enable_blogs",
                "enable_communities", "enable_events",
            ):
                if attr in src:
                    setattr(config.sources, attr, src[attr])

        if "llm" in data:
            llm = data["llm"]
            for attr in (
                "provider", "model", "temperature", "max_tokens",
                "ranking_batch_size", "top_papers", "top_blog_posts",
                "top_community", "top_events",
            ):
                if attr in llm:
                    setattr(config.llm, attr, llm[attr])

        if "output" in data:
            out = data["output"]
            for attr in (
                "output_dir", "markdown", "terminal",
                "show_papers", "show_blogs", "show_community", "show_events",
            ):
                if attr in out:
                    setattr(config.output, attr, out[attr])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a dictionary for YAML output."""
        return {
            "interests": {
                "topics": self.interests.topics,
                "search_terms": self.interests.search_terms,
                "key_figures": self.interests.key_figures,
                "arxiv_categories": self.interests.arxiv_categories,
            },
            "sources": {
                "max_papers": self.sources.max_papers,
                "max_blog_posts": self.sources.max_blog_posts,
                "max_community_posts": self.sources.max_community_posts,
                "max_events": self.sources.max_events,
                "papers_days_back": self.sources.papers_days_back,
                "subreddits": self.sources.subreddits,
                "enable_arxiv": self.sources.enable_arxiv,
                "enable_blogs": self.sources.enable_blogs,
                "enable_communities": self.sources.enable_communities,
                "enable_events": self.sources.enable_events,
            },
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "ranking_batch_size": self.llm.ranking_batch_size,
                "top_papers": self.llm.top_papers,
                "top_blog_posts": self.llm.top_blog_posts,
                "top_community": self.llm.top_community,
                "top_events": self.llm.top_events,
            },
            "output": {
                "output_dir": self.output.output_dir,
                "markdown": self.output.markdown,
                "terminal": self.output.terminal,
            },
        }

    def save(self, path: Optional[str] = None):
        """Save config to YAML file."""
        save_path = path or USER_CONFIG_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
