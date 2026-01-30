"""
Data models for the AI Research Aggregator.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class ContentType(Enum):
    PAPER = "paper"
    BLOG_POST = "blog_post"
    SOCIAL_POST = "social_post"
    EVENT = "event"
    COMMUNITY_POST = "community_post"


class SourceName(Enum):
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    OPENAI_BLOG = "openai_blog"
    ANTHROPIC_BLOG = "anthropic_blog"
    DEEPMIND_BLOG = "deepmind_blog"
    META_AI_BLOG = "meta_ai_blog"
    HACKER_NEWS = "hacker_news"
    REDDIT_ML = "reddit_ml"
    TWITTER = "twitter"
    LUMA = "luma"
    MEETUP = "meetup"
    EVENTBRITE = "eventbrite"


@dataclass
class ContentItem:
    """A single piece of content from any source."""

    title: str
    url: str
    source: SourceName
    content_type: ContentType
    published_date: Optional[datetime] = None
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    # Populated by the ranking engine
    relevance_score: float = 0.0
    summary: str = ""
    relevance_reason: str = ""


@dataclass
class EventItem(ContentItem):
    """An event with location and time details."""

    event_date: Optional[datetime] = None
    event_end_date: Optional[datetime] = None
    location: str = ""
    is_virtual: bool = False
    price: str = ""
    rsvp_count: int = 0


@dataclass
class DigestSection:
    """A section of the daily digest."""

    title: str
    description: str
    items: List[ContentItem] = field(default_factory=list)


@dataclass
class SourceHealth:
    """Health report for a source fetch."""

    source_name: str
    items_count: int = 0
    error: Optional[str] = None
    latency_s: float = 0.0


@dataclass
class DailyDigest:
    """The full daily digest output."""

    date: datetime
    sections: List[DigestSection] = field(default_factory=list)
    total_items_scanned: int = 0
    generation_time_s: float = 0.0

    # LLM-generated synthesis identifying a business or research opportunity
    # that emerges from the day's highlights
    opportunity_analysis: str = ""

    # Per-source health data
    source_health: List["SourceHealth"] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the digest as markdown."""
        lines = []
        date_str = self.date.strftime("%A, %B %d, %Y")
        lines.append(f"# AI Research Digest - {date_str}")
        lines.append("")
        lines.append(f"*Scanned {self.total_items_scanned} items | Generated in {self.generation_time_s:.1f}s*")
        lines.append("")
        lines.append("---")
        lines.append("")

        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            if section.description:
                lines.append(f"*{section.description}*")
                lines.append("")

            if not section.items:
                lines.append("*No items found for this section today.*")
                lines.append("")
                continue

            for i, item in enumerate(section.items, 1):
                lines.append(f"### {i}. [{item.title}]({item.url})")
                lines.append("")

                # Metadata line
                meta_parts = []
                if item.authors:
                    authors_str = ", ".join(item.authors[:3])
                    if len(item.authors) > 3:
                        authors_str += f" +{len(item.authors) - 3} more"
                    meta_parts.append(f"**Authors:** {authors_str}")
                if item.published_date:
                    meta_parts.append(f"**Published:** {item.published_date.strftime('%Y-%m-%d')}")
                meta_parts.append(f"**Source:** {item.source.value}")
                if item.relevance_score > 0:
                    score_bar = _score_bar(item.relevance_score)
                    meta_parts.append(f"**Relevance:** {score_bar} ({item.relevance_score:.0f}/100)")

                lines.append(" | ".join(meta_parts))
                lines.append("")

                if item.summary:
                    lines.append(f"> {item.summary}")
                    lines.append("")
                elif item.abstract:
                    abstract = item.abstract[:300]
                    if len(item.abstract) > 300:
                        abstract += "..."
                    lines.append(f"> {abstract}")
                    lines.append("")

                if item.relevance_reason:
                    lines.append(f"**Why this matters:** {item.relevance_reason}")
                    lines.append("")

                if item.tags:
                    lines.append(f"**Tags:** {', '.join(item.tags)}")
                    lines.append("")

                # Event-specific fields
                if isinstance(item, EventItem):
                    if item.event_date:
                        lines.append(f"**When:** {item.event_date.strftime('%B %d, %Y %I:%M %p')}")
                    if item.location:
                        lines.append(f"**Where:** {item.location}")
                    if item.price:
                        lines.append(f"**Price:** {item.price}")
                    if item.rsvp_count:
                        lines.append(f"**RSVPs:** {item.rsvp_count}")
                    lines.append("")

                lines.append("---")
                lines.append("")

        if self.opportunity_analysis:
            lines.append("")
            lines.append("## Opportunity Spotlight")
            lines.append("")
            lines.append(self.opportunity_analysis)
            lines.append("")
            lines.append("---")
            lines.append("")

        lines.append("")
        lines.append("*Generated by Hidden Layer AI Research Aggregator*")
        return "\n".join(lines)


def _score_bar(score: float, length: int = 10) -> str:
    """Create a visual score bar."""
    filled = int(score / 100 * length)
    return "[" + "#" * filled + "-" * (length - filled) + "]"
