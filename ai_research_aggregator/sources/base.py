"""
Base class for all content sources.

Provides retry with exponential backoff, per-source rate limiting,
file-based response caching, and health reporting.
"""

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

from ai_research_aggregator.models import ContentItem

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "ai-research-aggregator",
)

# HTTP status codes that should be retried
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass
class SourceResult:
    """Health report for a single source fetch."""

    source_name: str
    items_count: int = 0
    error: Optional[str] = None
    latency_s: float = 0.0
    cache_hit: bool = False


class BaseSource(ABC):
    """Base class all content sources must implement."""

    # Subclasses can override for longer/shorter delays
    request_delay_s: float = 0.5

    # Cache TTL in seconds (subclasses can override)
    cache_ttl_s: int = 3600  # 1 hour default

    # Retry settings
    max_retries: int = 3
    backoff_base: float = 2.0

    def __init__(self):
        self._last_request_time: float = 0.0

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable source name."""
        ...

    @abstractmethod
    def fetch(self, max_items: int = 50) -> List[ContentItem]:
        """
        Fetch latest content items from this source.

        Args:
            max_items: Maximum number of items to return.

        Returns:
            List of ContentItem objects.
        """
        ...

    def fetch_safe(self, max_items: int = 50) -> List[ContentItem]:
        """Fetch with error handling - never raises."""
        try:
            items = self.fetch(max_items=max_items)
            logger.info(f"[{self.name}] Fetched {len(items)} items")
            return items
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to fetch: {e}")
            return []

    def fetch_with_health(self, max_items: int = 50) -> Tuple[List[ContentItem], SourceResult]:
        """Fetch and return (items, SourceResult) for health reporting."""
        start = time.time()
        try:
            items = self.fetch(max_items=max_items)
            elapsed = time.time() - start
            logger.info(f"[{self.name}] Fetched {len(items)} items in {elapsed:.1f}s")
            result = SourceResult(
                source_name=self.name,
                items_count=len(items),
                latency_s=elapsed,
            )
            return items, result
        except Exception as e:
            elapsed = time.time() - start
            logger.warning(f"[{self.name}] Failed to fetch: {e}")
            result = SourceResult(
                source_name=self.name,
                error=str(e),
                latency_s=elapsed,
            )
            return [], result

    # ----- Retry with exponential backoff -----

    def _request_with_retry(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: int = 15,
        method: str = "GET",
    ) -> requests.Response:
        """
        Make an HTTP request with automatic retry and exponential backoff.

        Retries on 429, 5xx status codes, and connection/timeout errors.
        Respects Retry-After header when present.
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                self._enforce_rate_limit()

                response = requests.request(
                    method, url, params=params, headers=headers, timeout=timeout,
                )

                if response.status_code not in RETRYABLE_STATUS_CODES:
                    response.raise_for_status()
                    return response

                # Retryable status code — sleep and retry
                if attempt < self.max_retries:
                    wait = self._get_retry_wait(response, attempt)
                    logger.debug(
                        f"[{self.name}] HTTP {response.status_code} for {url}, "
                        f"retrying in {wait:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait)
                else:
                    response.raise_for_status()

            except (requests.ConnectionError, requests.Timeout) as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait = self.backoff_base ** (attempt + 1)
                    logger.debug(
                        f"[{self.name}] {type(e).__name__} for {url}, "
                        f"retrying in {wait:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait)
                else:
                    raise

        # Should not reach here, but just in case
        raise last_exception or requests.RequestException(
            f"Request failed after {self.max_retries} retries"
        )

    def _get_retry_wait(self, response: requests.Response, attempt: int) -> float:
        """Calculate retry wait time, respecting Retry-After header."""
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return self.backoff_base ** (attempt + 1)

    # ----- Rate limiting -----

    def _enforce_rate_limit(self):
        """Sleep if needed to respect per-source rate limiting."""
        if self.request_delay_s <= 0:
            return
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.request_delay_s:
            time.sleep(self.request_delay_s - elapsed)
        self._last_request_time = time.time()

    # ----- File-based caching -----

    def _cached_get(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: int = 15,
        cache_ttl_s: Optional[int] = None,
    ) -> requests.Response:
        """
        HTTP GET with file-based caching and retry.

        Checks cache first; on miss, fetches with retry and stores result.
        """
        ttl = cache_ttl_s if cache_ttl_s is not None else self.cache_ttl_s
        cache_key = self._cache_key(url, params)
        cache_path = os.path.join(CACHE_DIR, cache_key)

        # Check cache
        if ttl > 0 and os.path.exists(cache_path):
            try:
                mtime = os.path.getmtime(cache_path)
                age = time.time() - mtime
                if age < ttl:
                    with open(cache_path, "r") as f:
                        cached = json.load(f)
                    logger.debug(
                        f"[{self.name}] Cache hit for {url} (age={age:.0f}s, ttl={ttl}s)"
                    )
                    return _CachedResponse(
                        status_code=cached["status_code"],
                        text=cached["text"],
                        headers=cached.get("headers", {}),
                    )
                else:
                    logger.debug(
                        f"[{self.name}] Cache expired for {url} (age={age:.0f}s > ttl={ttl}s)"
                    )
            except (json.JSONDecodeError, OSError, KeyError) as e:
                logger.debug(f"[{self.name}] Cache read error: {e}")

        # Cache miss — fetch with retry
        response = self._request_with_retry(
            url, params=params, headers=headers, timeout=timeout
        )

        # Write to cache
        if ttl > 0:
            try:
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(
                        {
                            "status_code": response.status_code,
                            "text": response.text,
                            "headers": dict(response.headers),
                            "url": url,
                        },
                        f,
                    )
                logger.debug(f"[{self.name}] Cached response for {url}")
            except OSError as e:
                logger.debug(f"[{self.name}] Cache write error: {e}")

        return response

    @staticmethod
    def _cache_key(url: str, params: Optional[Dict] = None) -> str:
        """Generate a deterministic cache key from URL and params."""
        key_data = url
        if params:
            sorted_params = sorted(params.items())
            key_data += "?" + "&".join(f"{k}={v}" for k, v in sorted_params)
        return hashlib.sha256(key_data.encode()).hexdigest()[:32] + ".json"

    @staticmethod
    def clear_cache():
        """Remove all cached responses."""
        import shutil

        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            logger.info(f"Cleared cache at {CACHE_DIR}")


class _CachedResponse:
    """Minimal response-like object for cache hits."""

    def __init__(self, status_code: int, text: str, headers: Optional[Dict] = None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"Cached response with status {self.status_code}")

    def json(self):
        return json.loads(self.text)
