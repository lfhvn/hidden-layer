"""Offline smoke tests for the AI research aggregator (no network)."""

from ai_research_aggregator.config import AggregatorConfig
from ai_research_aggregator.ranking import RANKING_PROMPT, ContentType


def test_default_config_has_expected_sections():
    cfg = AggregatorConfig()
    for section in ("interests", "sources", "llm", "output"):
        assert hasattr(cfg, section)
    assert isinstance(cfg.to_dict(), dict)


def test_content_type_taxonomy():
    for member in ("PAPER", "BLOG_POST", "SOCIAL_POST"):
        assert hasattr(ContentType, member)


def test_ranking_prompt_is_nonempty_text():
    assert isinstance(RANKING_PROMPT, str) and RANKING_PROMPT.strip()
