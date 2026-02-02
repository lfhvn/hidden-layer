"""
Content sources for the AI Research Aggregator.

Each source implements the BaseSource interface and returns ContentItems.
"""

from .base import BaseSource
from .arxiv import ArxivSource
from .blogs import BlogAggregatorSource
from .communities import CommunitySource
from .events import SFEventsSource

__all__ = [
    "BaseSource",
    "ArxivSource",
    "BlogAggregatorSource",
    "CommunitySource",
    "SFEventsSource",
]
