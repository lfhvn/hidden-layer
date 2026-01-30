"""
Persistent storage for cross-day deduplication and run history.

Uses SQLite at ~/.config/ai-research-aggregator/history.db.
"""

import hashlib
import logging
import os
import re
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Set
from urllib.parse import urlparse, urlunparse

from ai_research_aggregator.models import ContentItem

logger = logging.getLogger(__name__)

DB_DIR = os.path.join(
    os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
    "ai-research-aggregator",
)
DB_PATH = os.path.join(DB_DIR, "history.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_items (
    url_hash TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    published_in_digest INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_seen_last ON seen_items(last_seen);
CREATE INDEX IF NOT EXISTS idx_seen_published ON seen_items(published_in_digest);
"""


class HistoryDB:
    """SQLite-backed history for deduplication."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def get_recently_published_hashes(self, days: int = 7) -> Set[str]:
        """Return URL hashes of items published in digest within the last N days."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT url_hash FROM seen_items "
                "WHERE published_in_digest = 1 AND last_seen >= ?",
                (cutoff,),
            ).fetchall()
        return {row[0] for row in rows}

    def mark_published(self, items: List[ContentItem]):
        """Mark items as published in digest."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            for item in items:
                url_hash = normalize_url_hash(item.url)
                conn.execute(
                    "INSERT INTO seen_items (url_hash, title, source, first_seen, last_seen, published_in_digest) "
                    "VALUES (?, ?, ?, ?, ?, 1) "
                    "ON CONFLICT(url_hash) DO UPDATE SET "
                    "last_seen = ?, published_in_digest = 1",
                    (url_hash, item.title, item.source.value, now, now, now),
                )

    def record_seen(self, items: List[ContentItem]):
        """Record items as seen (but not necessarily published)."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            for item in items:
                url_hash = normalize_url_hash(item.url)
                conn.execute(
                    "INSERT INTO seen_items (url_hash, title, source, first_seen, last_seen, published_in_digest) "
                    "VALUES (?, ?, ?, ?, ?, 0) "
                    "ON CONFLICT(url_hash) DO UPDATE SET last_seen = ?",
                    (url_hash, item.title, item.source.value, now, now, now),
                )

    def clear(self):
        """Delete all history."""
        with self._connect() as conn:
            conn.execute("DELETE FROM seen_items")
        logger.info("History cleared")

    def stats(self) -> dict:
        """Return database statistics."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM seen_items").fetchone()[0]
            published = conn.execute(
                "SELECT COUNT(*) FROM seen_items WHERE published_in_digest = 1"
            ).fetchone()[0]
            recent = conn.execute(
                "SELECT COUNT(*) FROM seen_items WHERE last_seen >= ?",
                ((datetime.now() - timedelta(days=7)).isoformat(),),
            ).fetchone()[0]
        return {
            "total_items": total,
            "published_items": published,
            "seen_last_7_days": recent,
            "db_path": self.db_path,
        }


# ----- URL normalization -----

def normalize_url(url: str) -> str:
    """Normalize a URL for deduplication: strip tracking params, trailing slashes, etc."""
    if not url:
        return ""

    parsed = urlparse(url)

    # Strip common tracking params
    tracking_params = {"utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term", "ref", "source"}
    if parsed.query:
        params = parsed.query.split("&")
        filtered = [p for p in params if p.split("=")[0] not in tracking_params]
        query = "&".join(filtered)
    else:
        query = ""

    # Strip trailing slashes from path
    path = parsed.path.rstrip("/")

    # Normalize arXiv URLs: strip version suffix
    if "arxiv.org" in parsed.netloc:
        path = re.sub(r'v\d+$', '', path)

    return urlunparse((parsed.scheme, parsed.netloc, path, "", query, ""))


def normalize_url_hash(url: str) -> str:
    """Return a hash of the normalized URL."""
    return hashlib.sha256(normalize_url(url).encode()).hexdigest()[:32]


# ----- Within-run deduplication -----

def _title_words(title: str) -> set:
    """Extract lowercase word set for Jaccard similarity."""
    return set(re.findall(r'\w+', title.lower()))


def _jaccard_similarity(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate_items(
    items: List[ContentItem],
    title_threshold: float = 0.7,
    recently_published: Optional[Set[str]] = None,
) -> List[ContentItem]:
    """
    Deduplicate items within a single run.

    - Exact URL match (after normalization)
    - Title Jaccard similarity above threshold
    - Skip items recently published (cross-day dedup)

    When merging duplicates, keeps the item with the most metadata.
    """
    recently_published = recently_published or set()

    seen_urls = {}  # normalized_url -> index in result
    seen_titles = []  # (word_set, index_in_result)
    result = []

    for item in items:
        norm_url = normalize_url(item.url)
        url_hash = hashlib.sha256(norm_url.encode()).hexdigest()[:32]

        # Cross-day dedup: skip if recently published
        if url_hash in recently_published:
            logger.debug(f"Skipping recently published: {item.title[:60]}")
            continue

        # URL-based dedup
        if norm_url in seen_urls:
            existing_idx = seen_urls[norm_url]
            result[existing_idx] = _pick_best(result[existing_idx], item)
            continue

        # Title-based dedup
        title_words = _title_words(item.title)
        duplicate_found = False
        for existing_words, existing_idx in seen_titles:
            if _jaccard_similarity(title_words, existing_words) >= title_threshold:
                result[existing_idx] = _pick_best(result[existing_idx], item)
                duplicate_found = True
                logger.debug(
                    f"Title dedup: '{item.title[:50]}' matches '{result[existing_idx].title[:50]}'"
                )
                break

        if not duplicate_found:
            idx = len(result)
            seen_urls[norm_url] = idx
            seen_titles.append((title_words, idx))
            result.append(item)

    removed = len(items) - len(result)
    if removed > 0:
        logger.info(f"Deduplication removed {removed} items ({len(items)} -> {len(result)})")

    return result


def _pick_best(a: ContentItem, b: ContentItem) -> ContentItem:
    """Return whichever item has more metadata."""
    def richness(item: ContentItem) -> int:
        score = 0
        if item.abstract:
            score += len(item.abstract)
        if item.authors:
            score += len(item.authors) * 10
        if item.summary:
            score += 50
        if item.tags:
            score += len(item.tags) * 5
        return score

    return a if richness(a) >= richness(b) else b
