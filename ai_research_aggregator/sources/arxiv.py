"""
arXiv paper source.

Uses the arXiv API to fetch recent papers from AI-relevant categories.
"""

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Optional

from ai_research_aggregator.models import ContentItem, ContentType, SourceName

from .base import BaseSource

logger = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"

# AI-relevant arXiv categories
DEFAULT_CATEGORIES = [
    "cs.AI",   # Artificial Intelligence
    "cs.CL",   # Computation and Language (NLP)
    "cs.LG",   # Machine Learning
    "cs.CV",   # Computer Vision
    "cs.MA",   # Multi-Agent Systems
    "cs.NE",   # Neural and Evolutionary Computing
    "stat.ML", # Machine Learning (Stats)
]

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

HEADERS = {
    "User-Agent": "HiddenLayerResearchAggregator/0.1 (research tool; mailto:research@hiddenlayer.ai)"
}


class ArxivSource(BaseSource):
    """Fetches recent AI papers from arXiv."""

    cache_ttl_s = 14400  # arXiv updates slowly; cache 4 hours
    request_delay_s = 1.0  # arXiv rate-limits aggressively

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        search_terms: Optional[List[str]] = None,
        days_back: int = 2,
    ):
        super().__init__()
        self.categories = categories or DEFAULT_CATEGORIES
        self.search_terms = search_terms or []
        self.days_back = days_back

    @property
    def name(self) -> str:
        return "arXiv"

    def fetch(self, max_items: int = 50) -> List[ContentItem]:
        """Fetch recent papers from arXiv API."""
        items = []

        # Build category query
        cat_query = " OR ".join(f"cat:{cat}" for cat in self.categories)
        query = f"({cat_query})"

        # Add search terms if provided
        if self.search_terms:
            terms_query = " OR ".join(
                f'all:"{term}"' for term in self.search_terms
            )
            query = f"({cat_query}) AND ({terms_query})"

        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_items,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        response = self._cached_get(ARXIV_API_URL, params=params, headers=HEADERS, timeout=30)

        root = ET.fromstring(response.text)

        for entry in root.findall(f"{ATOM_NS}entry"):
            item = self._parse_entry(entry)
            if item:
                items.append(item)

        return items

    def _parse_entry(self, entry: ET.Element) -> Optional[ContentItem]:
        """Parse a single arXiv Atom entry into a ContentItem."""
        try:
            title = entry.find(f"{ATOM_NS}title")
            if title is None or title.text is None:
                return None
            title_text = " ".join(title.text.strip().split())

            # Get the abstract link
            link = None
            for l in entry.findall(f"{ATOM_NS}link"):
                if l.get("type") == "text/html" or l.get("rel") == "alternate":
                    link = l.get("href")
                    break
            if not link:
                id_elem = entry.find(f"{ATOM_NS}id")
                link = id_elem.text if id_elem is not None else ""

            # Authors
            authors = []
            for author in entry.findall(f"{ATOM_NS}author"):
                name = author.find(f"{ATOM_NS}name")
                if name is not None and name.text:
                    authors.append(name.text.strip())

            # Abstract / summary
            summary = entry.find(f"{ATOM_NS}summary")
            abstract = ""
            if summary is not None and summary.text:
                abstract = " ".join(summary.text.strip().split())

            # Published date
            published = entry.find(f"{ATOM_NS}published")
            pub_date = None
            if published is not None and published.text:
                try:
                    pub_date = datetime.fromisoformat(published.text.replace("Z", "+00:00"))
                except ValueError:
                    pass

            # Categories as tags
            tags = []
            for category in entry.findall(f"{ARXIV_NS}primary_category"):
                term = category.get("term")
                if term:
                    tags.append(term)
            for category in entry.findall(f"{ATOM_NS}category"):
                term = category.get("term")
                if term and term not in tags:
                    tags.append(term)

            # arXiv ID for metadata
            arxiv_id = ""
            id_elem = entry.find(f"{ATOM_NS}id")
            if id_elem is not None and id_elem.text:
                arxiv_id = id_elem.text.split("/abs/")[-1]

            return ContentItem(
                title=title_text,
                url=link,
                source=SourceName.ARXIV,
                content_type=ContentType.PAPER,
                published_date=pub_date,
                authors=authors,
                abstract=abstract,
                tags=tags,
                metadata={"arxiv_id": arxiv_id},
            )
        except Exception as e:
            logger.debug(f"Failed to parse arXiv entry: {e}")
            return None
