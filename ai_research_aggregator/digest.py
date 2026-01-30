"""
Daily digest generator.

Orchestrates fetching, ranking, and output generation.
"""

import logging
import os
import time
from datetime import datetime
from typing import List, Optional

from ai_research_aggregator.config import AggregatorConfig
from ai_research_aggregator.models import (
    ContentItem,
    ContentType,
    DailyDigest,
    DigestSection,
    EventItem,
)
from ai_research_aggregator.ranking import rank_with_keywords, rank_with_llm
from ai_research_aggregator.sources.arxiv import ArxivSource
from ai_research_aggregator.sources.blogs import BlogAggregatorSource
from ai_research_aggregator.sources.communities import CommunitySource
from ai_research_aggregator.sources.events import SFEventsSource

logger = logging.getLogger(__name__)


def generate_digest(
    config: Optional[AggregatorConfig] = None,
    use_llm: bool = True,
    skip_events: bool = False,
) -> DailyDigest:
    """
    Generate a complete daily digest.

    Args:
        config: Aggregator configuration. Uses defaults if None.
        use_llm: Whether to use LLM for ranking/summarization.
        skip_events: Skip fetching events (faster).

    Returns:
        DailyDigest with ranked and summarized content.
    """
    if config is None:
        config = AggregatorConfig()

    start_time = time.time()
    total_items = 0

    sections = []

    # --- Fetch from all sources ---
    print("Fetching AI research papers...")
    papers = _fetch_papers(config)
    total_items += len(papers)
    print(f"  Found {len(papers)} papers")

    print("Fetching blog posts...")
    blogs = _fetch_blogs(config)
    total_items += len(blogs)
    print(f"  Found {len(blogs)} blog posts")

    print("Fetching community posts...")
    community = _fetch_community(config)
    total_items += len(community)
    print(f"  Found {len(community)} community posts")

    events = []
    if not skip_events and config.sources.enable_events:
        print("Fetching SF AI events...")
        events = _fetch_events(config)
        total_items += len(events)
        print(f"  Found {len(events)} events")

    # --- Rank and summarize ---
    rank_fn = rank_with_llm if use_llm else rank_with_keywords

    if papers:
        print("Ranking papers...")
        papers = rank_fn(papers, config)
        top_papers = papers[:config.llm.top_papers]
        sections.append(DigestSection(
            title="Top Research Papers",
            description="Most relevant new papers from arXiv",
            items=top_papers,
        ))

    if blogs:
        print("Ranking blog posts...")
        blogs = rank_fn(blogs, config)
        top_blogs = blogs[:config.llm.top_blog_posts]
        sections.append(DigestSection(
            title="AI Lab & Industry Updates",
            description="From OpenAI, Anthropic, DeepMind, Meta AI, and more",
            items=top_blogs,
        ))

    if community:
        print("Ranking community posts...")
        community = rank_fn(community, config)
        top_community = community[:config.llm.top_community]
        sections.append(DigestSection(
            title="Community Highlights",
            description="Trending from Hacker News and Reddit ML communities",
            items=top_community,
        ))

    if events:
        print("Processing events...")
        # Events don't need LLM ranking - sort by date
        sections.append(DigestSection(
            title="Upcoming SF Bay Area AI Events",
            description="AI/ML events happening in and around San Francisco",
            items=events[:config.llm.top_events],
        ))

    elapsed = time.time() - start_time
    print(f"\nDigest generated in {elapsed:.1f}s ({total_items} items scanned)")

    return DailyDigest(
        date=datetime.now(),
        sections=sections,
        total_items_scanned=total_items,
        generation_time_s=elapsed,
    )


def _fetch_papers(config: AggregatorConfig) -> List[ContentItem]:
    """Fetch papers from configured sources."""
    if not config.sources.enable_arxiv:
        return []

    source = ArxivSource(
        categories=config.interests.arxiv_categories,
        search_terms=config.interests.search_terms,
        days_back=config.sources.papers_days_back,
    )
    return source.fetch_safe(max_items=config.sources.max_papers)


def _fetch_blogs(config: AggregatorConfig) -> List[ContentItem]:
    """Fetch blog posts from configured sources."""
    if not config.sources.enable_blogs:
        return []

    source = BlogAggregatorSource()
    return source.fetch_safe(max_items=config.sources.max_blog_posts)


def _fetch_community(config: AggregatorConfig) -> List[ContentItem]:
    """Fetch community posts from configured sources."""
    if not config.sources.enable_communities:
        return []

    source = CommunitySource()
    return source.fetch_safe(max_items=config.sources.max_community_posts)


def _fetch_events(config: AggregatorConfig) -> List[ContentItem]:
    """Fetch events from configured sources."""
    if not config.sources.enable_events:
        return []

    source = SFEventsSource()
    return source.fetch_safe(max_items=config.sources.max_events)


def save_digest(digest: DailyDigest, config: Optional[AggregatorConfig] = None) -> str:
    """
    Save digest to a markdown file.

    Returns:
        Path to the saved file.
    """
    if config is None:
        config = AggregatorConfig()

    output_dir = config.output.output_dir
    os.makedirs(output_dir, exist_ok=True)

    date_str = digest.date.strftime("%Y-%m-%d")
    filename = f"digest-{date_str}.md"
    filepath = os.path.join(output_dir, filename)

    markdown = digest.to_markdown()

    with open(filepath, "w") as f:
        f.write(markdown)

    return filepath


def print_digest_terminal(digest: DailyDigest):
    """Print a condensed digest to the terminal."""
    date_str = digest.date.strftime("%A, %B %d, %Y")

    print("")
    print("=" * 70)
    print(f"  AI RESEARCH DIGEST - {date_str}")
    print(f"  Scanned {digest.total_items_scanned} items | Generated in {digest.generation_time_s:.1f}s")
    print("=" * 70)

    for section in digest.sections:
        print("")
        print(f"  {section.title.upper()}")
        print(f"  {section.description}")
        print("-" * 70)

        if not section.items:
            print("  No items found for this section today.")
            continue

        for i, item in enumerate(section.items, 1):
            score_str = ""
            if item.relevance_score > 0:
                score_str = f" [{item.relevance_score:.0f}/100]"

            print(f"  {i}. {item.title}{score_str}")

            if item.authors:
                authors_str = ", ".join(item.authors[:3])
                if len(item.authors) > 3:
                    authors_str += f" +{len(item.authors) - 3}"
                print(f"     By: {authors_str}")

            if item.summary:
                # Wrap summary to terminal width
                summary = item.summary[:200]
                print(f"     {summary}")
            elif item.abstract:
                abstract = item.abstract[:150]
                if len(item.abstract) > 150:
                    abstract += "..."
                print(f"     {abstract}")

            if isinstance(item, EventItem):
                if item.event_date:
                    print(f"     When: {item.event_date.strftime('%B %d, %Y %I:%M %p')}")
                if item.location:
                    print(f"     Where: {item.location}")

            print(f"     {item.url}")
            print("")

    print("=" * 70)
    print("  Generated by Hidden Layer AI Research Aggregator")
    print("=" * 70)
