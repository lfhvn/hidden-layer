"""
Blog and news sources.

Scrapes AI research blogs and key figure feeds via RSS and web scraping.
"""

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple

import requests

from ai_research_aggregator.models import ContentItem, ContentType, SourceName

from .base import BaseSource

logger = logging.getLogger(__name__)

# Blog RSS/Atom feed URLs
BLOG_FEEDS: List[Dict] = [
    {
        "name": "OpenAI Blog",
        "url": "https://openai.com/blog/rss.xml",
        "source": SourceName.OPENAI_BLOG,
        "format": "rss",
    },
    {
        "name": "Anthropic Research",
        "url": "https://www.anthropic.com/research/rss.xml",
        "source": SourceName.ANTHROPIC_BLOG,
        "format": "rss",
    },
    {
        "name": "Google DeepMind Blog",
        "url": "https://deepmind.google/blog/rss.xml",
        "source": SourceName.DEEPMIND_BLOG,
        "format": "rss",
    },
    {
        "name": "Meta AI Blog",
        "url": "https://ai.meta.com/blog/rss/",
        "source": SourceName.META_AI_BLOG,
        "format": "rss",
    },
]

# Key AI figures to track (name -> relevant URLs/identifiers)
KEY_FIGURES = [
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
]

HEADERS = {
    "User-Agent": "HiddenLayerResearchAggregator/0.1 (research tool)"
}


class BlogAggregatorSource(BaseSource):
    """Aggregates content from AI research blogs."""

    def __init__(self, extra_feeds: Optional[List[Dict]] = None):
        self.feeds = BLOG_FEEDS.copy()
        if extra_feeds:
            self.feeds.extend(extra_feeds)

    @property
    def name(self) -> str:
        return "AI Blogs"

    def fetch(self, max_items: int = 50) -> List[ContentItem]:
        """Fetch recent posts from all configured blog feeds."""
        all_items = []

        for feed_config in self.feeds:
            try:
                items = self._fetch_feed(feed_config, max_per_feed=max_items // len(self.feeds) + 5)
                all_items.extend(items)
            except Exception as e:
                logger.warning(f"Failed to fetch {feed_config['name']}: {e}")

        # Sort by date, newest first
        all_items.sort(key=lambda x: x.published_date or datetime.min, reverse=True)
        return all_items[:max_items]

    def _fetch_feed(self, config: Dict, max_per_feed: int = 20) -> List[ContentItem]:
        """Fetch and parse a single RSS/Atom feed."""
        response = requests.get(config["url"], headers=HEADERS, timeout=15)
        response.raise_for_status()

        items = []
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            logger.warning(f"Failed to parse XML from {config['name']}")
            return []

        # Try RSS format first
        channel = root.find("channel")
        if channel is not None:
            items = self._parse_rss(channel, config, max_per_feed)
        else:
            # Try Atom format
            items = self._parse_atom(root, config, max_per_feed)

        return items

    def _parse_rss(self, channel: ET.Element, config: Dict, max_items: int) -> List[ContentItem]:
        """Parse RSS 2.0 feed items."""
        items = []
        for item_elem in channel.findall("item")[:max_items]:
            title = self._get_text(item_elem, "title")
            link = self._get_text(item_elem, "link")
            description = self._get_text(item_elem, "description")
            pub_date_str = self._get_text(item_elem, "pubDate")

            if not title or not link:
                continue

            # Clean HTML from description
            description = self._strip_html(description)

            pub_date = None
            if pub_date_str:
                pub_date = self._parse_date(pub_date_str)

            # Extract author if present
            author = self._get_text(item_elem, "author") or self._get_text(item_elem, "dc:creator")
            authors = [author] if author else []

            # Extract categories/tags
            tags = [cat.text for cat in item_elem.findall("category") if cat.text]

            # Check if any key figures are mentioned
            full_text = f"{title} {description}".lower()
            mentioned_figures = [f for f in KEY_FIGURES if f.lower() in full_text]
            if mentioned_figures:
                tags.extend([f"mentions:{f}" for f in mentioned_figures])

            items.append(ContentItem(
                title=title.strip(),
                url=link.strip(),
                source=config["source"],
                content_type=ContentType.BLOG_POST,
                published_date=pub_date,
                authors=authors,
                abstract=description[:500] if description else "",
                tags=tags,
                metadata={"feed_name": config["name"]},
            ))

        return items

    def _parse_atom(self, root: ET.Element, config: Dict, max_items: int) -> List[ContentItem]:
        """Parse Atom feed entries."""
        ns = "{http://www.w3.org/2005/Atom}"
        items = []

        for entry in root.findall(f"{ns}entry")[:max_items]:
            title_elem = entry.find(f"{ns}title")
            title = title_elem.text if title_elem is not None else ""

            link = ""
            for l in entry.findall(f"{ns}link"):
                href = l.get("href")
                if href and (l.get("rel", "alternate") == "alternate" or l.get("type", "") == "text/html"):
                    link = href
                    break
            if not link:
                for l in entry.findall(f"{ns}link"):
                    href = l.get("href")
                    if href:
                        link = href
                        break

            summary_elem = entry.find(f"{ns}summary") or entry.find(f"{ns}content")
            summary = summary_elem.text if summary_elem is not None else ""
            summary = self._strip_html(summary)

            pub_elem = entry.find(f"{ns}published") or entry.find(f"{ns}updated")
            pub_date = None
            if pub_elem is not None and pub_elem.text:
                pub_date = self._parse_date(pub_elem.text)

            authors = []
            for author in entry.findall(f"{ns}author"):
                name = author.find(f"{ns}name")
                if name is not None and name.text:
                    authors.append(name.text)

            tags = [cat.get("term", "") for cat in entry.findall(f"{ns}category") if cat.get("term")]

            if not title or not link:
                continue

            items.append(ContentItem(
                title=title.strip(),
                url=link.strip(),
                source=config["source"],
                content_type=ContentType.BLOG_POST,
                published_date=pub_date,
                authors=authors,
                abstract=summary[:500] if summary else "",
                tags=tags,
                metadata={"feed_name": config["name"]},
            ))

        return items

    @staticmethod
    def _get_text(elem: ET.Element, tag: str) -> str:
        """Get text content of a child element."""
        child = elem.find(tag)
        if child is not None and child.text:
            return child.text
        return ""

    @staticmethod
    def _strip_html(text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return ""
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Parse various date formats found in feeds."""
        if not date_str:
            return None

        # Try RFC 2822 (RSS standard)
        try:
            return parsedate_to_datetime(date_str)
        except (ValueError, TypeError):
            pass

        # Try ISO 8601 (Atom standard)
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

        # Try common formats
        for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%B %d, %Y"]:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        return None
