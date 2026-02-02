"""
Base class for all content sources.
"""

import logging
from abc import ABC, abstractmethod
from typing import List

from ai_research_aggregator.models import ContentItem

logger = logging.getLogger(__name__)


class BaseSource(ABC):
    """Base class all content sources must implement."""

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
