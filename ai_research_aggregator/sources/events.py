"""
SF Bay Area AI events source.

Scrapes event platforms for AI/ML events in San Francisco and the Bay Area.
Uses Lu.ma API, and web scraping for Eventbrite and Meetup.
"""

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional

import requests

from ai_research_aggregator.models import ContentType, EventItem, SourceName

from .base import BaseSource

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "HiddenLayerResearchAggregator/0.1 (research tool)"
}

# Lu.ma discover API for SF AI events
LUMA_DISCOVER_URL = "https://api.lu.ma/public/v2/event/search"

# Eventbrite API
EVENTBRITE_SEARCH_URL = "https://www.eventbriteapi.com/v3/events/search/"

AI_EVENT_KEYWORDS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "LLM",
    "AI",
    "neural network",
    "NLP",
    "computer vision",
    "generative AI",
    "foundation model",
    "transformer",
    "AI safety",
    "AI alignment",
    "MLOps",
    "data science",
]

SF_LOCATIONS = [
    "San Francisco",
    "SF",
    "Bay Area",
    "Palo Alto",
    "Mountain View",
    "Berkeley",
    "Oakland",
    "San Jose",
    "Menlo Park",
    "Sunnyvale",
    "South Bay",
]


class LumaEventsSource(BaseSource):
    """Fetches AI events from Lu.ma."""

    @property
    def name(self) -> str:
        return "Lu.ma Events"

    def fetch(self, max_items: int = 20) -> List[EventItem]:
        """Fetch upcoming AI events from Lu.ma."""
        items = []

        # Lu.ma has a public discover page we can query
        search_queries = ["AI San Francisco", "machine learning SF", "AI meetup Bay Area"]

        for query in search_queries:
            try:
                # Lu.ma public search endpoint
                response = requests.get(
                    "https://api.lu.ma/public/v2/event/search",
                    params={"query": query, "limit": 10},
                    headers=HEADERS,
                    timeout=10,
                )

                if response.status_code == 200:
                    data = response.json()
                    for event_data in data.get("data", data.get("entries", [])):
                        item = self._parse_luma_event(event_data)
                        if item:
                            items.append(item)
            except Exception as e:
                logger.debug(f"Lu.ma search failed for '{query}': {e}")

        # Fallback: try scraping lu.ma/sf page for curated events
        if not items:
            items = self._fetch_luma_discover()

        return items[:max_items]

    def _fetch_luma_discover(self) -> List[EventItem]:
        """Fallback: try Lu.ma discover for SF area."""
        items = []
        try:
            response = requests.get(
                "https://lu.ma/sf",
                headers=HEADERS,
                timeout=10,
            )
            if response.status_code == 200:
                # Extract event URLs from the page using regex
                event_urls = re.findall(r'href="/([\w-]+)"', response.text)
                for slug in event_urls[:15]:
                    if len(slug) > 5 and slug not in ("about", "signin", "signup", "create", "explore"):
                        items.append(EventItem(
                            title=slug.replace("-", " ").title(),
                            url=f"https://lu.ma/{slug}",
                            source=SourceName.LUMA,
                            content_type=ContentType.EVENT,
                            location="San Francisco, CA",
                            tags=["luma", "sf"],
                        ))
        except Exception as e:
            logger.debug(f"Lu.ma discover fallback failed: {e}")

        return items

    def _parse_luma_event(self, data: dict) -> Optional[EventItem]:
        """Parse a Lu.ma event API response."""
        try:
            event = data.get("event", data)
            name = event.get("name", "")
            url = event.get("url", "")
            if not url and event.get("slug"):
                url = f"https://lu.ma/{event['slug']}"

            start_at = event.get("start_at")
            end_at = event.get("end_at")
            location_str = ""

            geo = event.get("geo_address_info") or event.get("location", {})
            if isinstance(geo, dict):
                location_str = geo.get("full_address", geo.get("city", ""))
            elif isinstance(geo, str):
                location_str = geo

            event_date = None
            if start_at:
                try:
                    event_date = datetime.fromisoformat(start_at.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            end_date = None
            if end_at:
                try:
                    end_date = datetime.fromisoformat(end_at.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            if not name:
                return None

            return EventItem(
                title=name,
                url=url,
                source=SourceName.LUMA,
                content_type=ContentType.EVENT,
                abstract=event.get("description", "")[:500],
                event_date=event_date,
                event_end_date=end_date,
                location=location_str,
                is_virtual=event.get("is_online", False),
                tags=["luma"],
                metadata={"cover_url": event.get("cover_url", "")},
            )
        except Exception as e:
            logger.debug(f"Failed to parse Lu.ma event: {e}")
            return None


class EventbriteSource(BaseSource):
    """Fetches AI events from Eventbrite (public search, no API key required)."""

    @property
    def name(self) -> str:
        return "Eventbrite"

    def fetch(self, max_items: int = 15) -> List[EventItem]:
        """Search Eventbrite for AI events in SF."""
        items = []

        search_url = "https://www.eventbrite.com/d/ca--san-francisco/ai-artificial-intelligence/"

        try:
            response = requests.get(search_url, headers=HEADERS, timeout=15)
            if response.status_code == 200:
                # Extract structured data from the page
                items = self._parse_eventbrite_html(response.text)
        except Exception as e:
            logger.debug(f"Eventbrite fetch failed: {e}")

        return items[:max_items]

    def _parse_eventbrite_html(self, html: str) -> List[EventItem]:
        """Extract event data from Eventbrite search results page."""
        items = []

        # Look for JSON-LD structured data
        import json
        ld_pattern = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.DOTALL)
        for match in ld_pattern.finditer(html):
            try:
                data = json.loads(match.group(1))
                if isinstance(data, list):
                    for item_data in data:
                        event = self._parse_ld_event(item_data)
                        if event:
                            items.append(event)
                elif isinstance(data, dict):
                    if data.get("@type") == "Event":
                        event = self._parse_ld_event(data)
                        if event:
                            items.append(event)
                    elif "itemListElement" in data:
                        for elem in data["itemListElement"]:
                            event_data = elem.get("item", elem)
                            event = self._parse_ld_event(event_data)
                            if event:
                                items.append(event)
            except (json.JSONDecodeError, KeyError):
                continue

        return items

    def _parse_ld_event(self, data: dict) -> Optional[EventItem]:
        """Parse a JSON-LD event object."""
        try:
            if data.get("@type") not in ("Event", "SocialEvent", None):
                if "name" not in data:
                    return None

            name = data.get("name", "")
            url = data.get("url", "")
            description = data.get("description", "")[:500]

            start_date = data.get("startDate")
            end_date = data.get("endDate")

            event_date = None
            if start_date:
                try:
                    event_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            event_end = None
            if end_date:
                try:
                    event_end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            location = ""
            loc_data = data.get("location", {})
            if isinstance(loc_data, dict):
                location = loc_data.get("name", "")
                address = loc_data.get("address", {})
                if isinstance(address, dict):
                    parts = [address.get("streetAddress", ""), address.get("addressLocality", "")]
                    addr_str = ", ".join(p for p in parts if p)
                    if addr_str:
                        location = f"{location} - {addr_str}" if location else addr_str
                elif isinstance(address, str):
                    location = f"{location} - {address}" if location else address

            is_virtual = False
            if isinstance(loc_data, dict):
                is_virtual = loc_data.get("@type") == "VirtualLocation"

            price = ""
            offers = data.get("offers", {})
            if isinstance(offers, dict):
                price_val = offers.get("price", "")
                currency = offers.get("priceCurrency", "USD")
                if price_val:
                    price = f"{currency} {price_val}" if price_val != "0" else "Free"

            if not name:
                return None

            return EventItem(
                title=name,
                url=url,
                source=SourceName.EVENTBRITE,
                content_type=ContentType.EVENT,
                abstract=description,
                event_date=event_date,
                event_end_date=event_end,
                location=location,
                is_virtual=is_virtual,
                price=price,
                tags=["eventbrite", "sf"],
            )
        except Exception as e:
            logger.debug(f"Failed to parse Eventbrite event: {e}")
            return None


class SFEventsSource(BaseSource):
    """Meta-source that aggregates all SF AI event sources."""

    def __init__(self):
        self.sources = [
            LumaEventsSource(),
            EventbriteSource(),
        ]

    @property
    def name(self) -> str:
        return "SF AI Events"

    def fetch(self, max_items: int = 20) -> List[EventItem]:
        items = []
        per_source = max_items // len(self.sources) + 5

        for source in self.sources:
            items.extend(source.fetch_safe(max_items=per_source))

        # Sort by event date (upcoming first)
        now = datetime.now()
        items.sort(key=lambda x: x.event_date or datetime.max if isinstance(x, EventItem) and hasattr(x, "event_date") else datetime.max)
        return items[:max_items]
