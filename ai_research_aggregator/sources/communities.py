"""
AI community sources.

Fetches trending AI content from Hacker News and Reddit.
"""

import logging
from datetime import datetime
from typing import List, Optional

from ai_research_aggregator.models import ContentItem, ContentType, SourceName

from .base import BaseSource

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "HiddenLayerResearchAggregator/0.1 (research tool)"
}

# Hacker News Algolia API
HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"

# Reddit JSON API (no auth needed for public subreddits)
REDDIT_URL = "https://www.reddit.com/r/{subreddit}/hot.json"


class HackerNewsSource(BaseSource):
    """Fetches AI-related posts from Hacker News."""

    cache_ttl_s = 3600  # 1 hour
    request_delay_s = 0.5

    def __init__(self, search_terms: Optional[List[str]] = None):
        super().__init__()
        self.search_terms = search_terms or [
            "AI", "LLM", "GPT", "Claude", "machine learning",
            "neural network", "transformer", "deep learning",
            "Anthropic", "OpenAI", "DeepMind",
            "language model", "artificial intelligence",
        ]

    @property
    def name(self) -> str:
        return "Hacker News"

    def fetch(self, max_items: int = 30) -> List[ContentItem]:
        """Fetch AI-related stories from HN."""
        items = []
        seen_ids = set()

        for term in self.search_terms[:8]:  # Limit API calls
            try:
                params = {
                    "query": term,
                    "tags": "story",
                    "hitsPerPage": min(10, max_items),
                    "numericFilters": f"created_at_i>{int(datetime.now().timestamp()) - 86400 * 3}",
                }
                response = self._cached_get(HN_SEARCH_URL, params=params, headers=HEADERS, timeout=10)
                data = response.json()

                for hit in data.get("hits", []):
                    object_id = hit.get("objectID", "")
                    if object_id in seen_ids:
                        continue
                    seen_ids.add(object_id)

                    title = hit.get("title", "")
                    url = hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}"
                    points = hit.get("points", 0)
                    num_comments = hit.get("num_comments", 0)
                    author = hit.get("author", "")
                    created_at = hit.get("created_at_i")

                    pub_date = None
                    if created_at:
                        pub_date = datetime.fromtimestamp(created_at)

                    if not title:
                        continue

                    items.append(ContentItem(
                        title=title,
                        url=url,
                        source=SourceName.HACKER_NEWS,
                        content_type=ContentType.COMMUNITY_POST,
                        published_date=pub_date,
                        authors=[author] if author else [],
                        tags=["hacker_news"],
                        metadata={
                            "points": points,
                            "num_comments": num_comments,
                            "hn_id": object_id,
                            "hn_url": f"https://news.ycombinator.com/item?id={object_id}",
                        },
                    ))
            except Exception as e:
                logger.debug(f"HN search failed for '{term}': {e}")

        # Deduplicate and sort by points
        items.sort(key=lambda x: x.metadata.get("points", 0), reverse=True)
        return items[:max_items]


class RedditMLSource(BaseSource):
    """Fetches posts from ML-related subreddits."""

    cache_ttl_s = 3600  # 1 hour
    request_delay_s = 1.0  # Reddit is strict about rate limits

    def __init__(self, subreddits: Optional[List[str]] = None):
        super().__init__()
        self.subreddits = subreddits or [
            "MachineLearning",
            "artificial",
            "LocalLLaMA",
        ]

    @property
    def name(self) -> str:
        return "Reddit ML"

    def fetch(self, max_items: int = 30) -> List[ContentItem]:
        """Fetch hot posts from ML subreddits."""
        items = []

        per_sub = max_items // len(self.subreddits) + 5

        for subreddit in self.subreddits:
            try:
                url = REDDIT_URL.format(subreddit=subreddit)
                response = self._cached_get(
                    url,
                    headers={**HEADERS, "Accept": "application/json"},
                    params={"limit": per_sub, "raw_json": 1},
                    timeout=10,
                )
                data = response.json()

                for post_data in data.get("data", {}).get("children", []):
                    post = post_data.get("data", {})

                    title = post.get("title", "")
                    post_url = post.get("url", "")
                    permalink = post.get("permalink", "")
                    author = post.get("author", "")
                    score = post.get("score", 0)
                    num_comments = post.get("num_comments", 0)
                    selftext = post.get("selftext", "")
                    created_utc = post.get("created_utc")

                    if not title or post.get("stickied"):
                        continue

                    pub_date = None
                    if created_utc:
                        pub_date = datetime.fromtimestamp(created_utc)

                    full_url = f"https://reddit.com{permalink}" if permalink else post_url

                    items.append(ContentItem(
                        title=title,
                        url=full_url,
                        source=SourceName.REDDIT_ML,
                        content_type=ContentType.COMMUNITY_POST,
                        published_date=pub_date,
                        authors=[author] if author else [],
                        abstract=selftext[:500] if selftext else "",
                        tags=[f"r/{subreddit}"],
                        metadata={
                            "score": score,
                            "num_comments": num_comments,
                            "subreddit": subreddit,
                            "external_url": post_url,
                        },
                    ))
            except Exception as e:
                logger.warning(f"Reddit fetch failed for r/{subreddit}: {e}")

        items.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)
        return items[:max_items]


class CommunitySource(BaseSource):
    """Meta-source that aggregates all community sources."""

    def __init__(self):
        super().__init__()
        self.sources = [
            HackerNewsSource(),
            RedditMLSource(),
        ]

    @property
    def name(self) -> str:
        return "AI Communities"

    def fetch(self, max_items: int = 50) -> List[ContentItem]:
        items = []
        per_source = max_items // len(self.sources) + 5

        for source in self.sources:
            items.extend(source.fetch_safe(max_items=per_source))

        # Sort by engagement (points/score)
        def engagement(item: ContentItem) -> int:
            return item.metadata.get("points", 0) + item.metadata.get("score", 0)

        items.sort(key=engagement, reverse=True)
        return items[:max_items]
