"""
Ranking and summarization engine.

Uses LLM to score content relevance and generate concise summaries.
Also provides keyword-based fallback ranking when LLM is unavailable.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from ai_research_aggregator.config import AggregatorConfig
from ai_research_aggregator.models import ContentItem, ContentType

logger = logging.getLogger(__name__)

# Prompt for LLM-based ranking
RANKING_PROMPT = """You are an AI research analyst. Given the user's research interests and a list of content items, score each item's relevance and provide a brief summary.

## User's Research Interests
{interests}

## Key Figures of Interest
{key_figures}

## Content Items to Rank
{items_json}

## Instructions
For each item, provide:
1. **relevance_score**: 0-100 integer (100 = extremely relevant to the user's interests)
2. **summary**: 1-2 sentence summary of what this item is about
3. **relevance_reason**: Brief explanation of why this is (or isn't) relevant

Consider:
- Direct topic match with user interests (highest weight)
- Authored by or mentioning key figures of interest (bonus)
- Novelty and potential impact of the work
- Recency (prefer newer content)
- Community engagement signals (points, comments) for community posts

Return a JSON array with objects containing: "index", "relevance_score", "summary", "relevance_reason"

Example:
```json
[
  {{"index": 0, "relevance_score": 85, "summary": "Proposes a new method for steering LLM behavior using activation vectors.", "relevance_reason": "Directly relates to steerability and activation engineering interests."}},
  {{"index": 1, "relevance_score": 40, "summary": "Benchmark for image classification on medical datasets.", "relevance_reason": "Computer vision paper but not in core interest areas."}}
]
```

Return ONLY the JSON array, no other text."""


def rank_with_llm(
    items: List[ContentItem],
    config: AggregatorConfig,
) -> List[ContentItem]:
    """
    Use LLM to rank and summarize content items.

    Items are processed in batches and scored 0-100 for relevance.
    """
    try:
        from harness import llm_call
    except ImportError:
        logger.warning("Harness not available, falling back to keyword ranking")
        return rank_with_keywords(items, config)

    if not items:
        return []

    batch_size = config.llm.ranking_batch_size
    all_scored = []

    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start:batch_start + batch_size]

        # Prepare items for the prompt
        items_for_prompt = []
        for i, item in enumerate(batch):
            entry = {
                "index": i,
                "title": item.title,
                "source": item.source.value,
                "type": item.content_type.value,
                "abstract": item.abstract[:300] if item.abstract else "",
                "authors": item.authors[:5],
                "tags": item.tags[:10],
            }
            # Add engagement metrics for community posts
            if item.content_type == ContentType.COMMUNITY_POST:
                entry["points"] = item.metadata.get("points", 0) + item.metadata.get("score", 0)
                entry["comments"] = item.metadata.get("num_comments", 0)
            items_for_prompt.append(entry)

        prompt = RANKING_PROMPT.format(
            interests="\n".join(f"- {t}" for t in config.interests.topics),
            key_figures=", ".join(config.interests.key_figures),
            items_json=json.dumps(items_for_prompt, indent=2),
        )

        try:
            response = llm_call(
                prompt=prompt,
                provider=config.llm.provider,
                model=config.llm.model,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
            )

            rankings = _parse_ranking_response(response.text)

            for ranking in rankings:
                idx = ranking.get("index", -1)
                if 0 <= idx < len(batch):
                    batch[idx].relevance_score = ranking.get("relevance_score", 0)
                    batch[idx].summary = ranking.get("summary", "")
                    batch[idx].relevance_reason = ranking.get("relevance_reason", "")

        except Exception as e:
            logger.warning(f"LLM ranking failed for batch starting at {batch_start}: {e}")
            # Fall back to keyword scoring for this batch
            batch = _keyword_score_batch(batch, config)

        all_scored.extend(batch)

    # Sort by relevance score descending
    all_scored.sort(key=lambda x: x.relevance_score, reverse=True)
    return all_scored


def rank_with_keywords(
    items: List[ContentItem],
    config: AggregatorConfig,
) -> List[ContentItem]:
    """
    Keyword-based ranking fallback (no LLM needed).

    Scores items based on keyword overlap with user interests.
    """
    return _keyword_score_batch(items, config)


def _keyword_score_batch(
    items: List[ContentItem],
    config: AggregatorConfig,
) -> List[ContentItem]:
    """Score a batch of items using keyword matching."""
    # Build keyword set from interests
    keywords = set()
    for topic in config.interests.topics:
        keywords.update(topic.lower().split())
    for term in config.interests.search_terms:
        keywords.update(term.lower().split())

    # Remove common words
    stopwords = {"and", "or", "the", "in", "of", "for", "a", "an", "is", "to", "with", "on", "at"}
    keywords -= stopwords

    figure_names = {name.lower() for name in config.interests.key_figures}

    for item in items:
        score = 0.0
        text = f"{item.title} {item.abstract} {' '.join(item.tags)}".lower()

        # Keyword matching (up to 60 points)
        matches = sum(1 for kw in keywords if kw in text)
        score += min(60, matches * 5)

        # Key figure mention (up to 20 points)
        figure_matches = sum(1 for name in figure_names if name in text)
        score += min(20, figure_matches * 10)

        # Recency bonus (up to 10 points)
        if item.published_date:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc) if item.published_date.tzinfo else datetime.now()
            days_old = (now - item.published_date).days
            if days_old <= 1:
                score += 10
            elif days_old <= 3:
                score += 7
            elif days_old <= 7:
                score += 4

        # Engagement bonus for community posts (up to 10 points)
        if item.content_type == ContentType.COMMUNITY_POST:
            engagement = item.metadata.get("points", 0) + item.metadata.get("score", 0)
            if engagement > 500:
                score += 10
            elif engagement > 200:
                score += 7
            elif engagement > 50:
                score += 4
            elif engagement > 10:
                score += 2

        item.relevance_score = min(100, score)

    items.sort(key=lambda x: x.relevance_score, reverse=True)
    return items


OPPORTUNITY_PROMPT = """You are a strategic AI research analyst for an independent research lab called Hidden Layer. Given today's top-ranked research highlights, identify ONE compelling opportunity — either a business opportunity or a promising direction for further research — that emerges from the convergence of these items.

## User's Research Interests
{interests}

## Today's Top Highlights
{highlights}

## Instructions
Write a 2-3 paragraph analysis that:
1. Identifies a specific opportunity (business venture, research direction, tool/product, or collaboration) that connects two or more of today's highlights
2. Explains WHY this opportunity exists now — what convergence of advances makes it timely
3. Outlines a concrete first step someone could take to pursue it

Be specific and sophisticated. Avoid generic advice like "stay informed" or "consider AI safety." Instead, identify non-obvious connections between the day's highlights and articulate an actionable opportunity that a technically sophisticated reader would find genuinely insightful.

Write in a direct, analytical tone. No bullet points — use flowing prose. Do not include a title or heading; just the analysis text."""


def generate_opportunity_analysis(
    digest_sections: list,
    config: "AggregatorConfig",
) -> str:
    """
    Use LLM to synthesize an opportunity from the day's top highlights.

    Returns the analysis text, or empty string if LLM is unavailable.
    """
    try:
        from harness import llm_call
    except ImportError:
        logger.warning("Harness not available, skipping opportunity analysis")
        return ""

    # Collect top items from each section for the prompt
    highlights = []
    for section in digest_sections:
        for item in section.items[:5]:
            entry = {
                "title": item.title,
                "source": item.source.value,
                "summary": item.summary or item.abstract[:200],
            }
            if item.relevance_reason:
                entry["why_it_matters"] = item.relevance_reason
            highlights.append(entry)

    if not highlights:
        return ""

    prompt = OPPORTUNITY_PROMPT.format(
        interests="\n".join(f"- {t}" for t in config.interests.topics),
        highlights=json.dumps(highlights, indent=2),
    )

    try:
        response = llm_call(
            prompt=prompt,
            provider=config.llm.provider,
            model=config.llm.model,
            temperature=0.5,
            max_tokens=1024,
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"Opportunity analysis generation failed: {e}")
        return ""


def _parse_ranking_response(text: str) -> List[Dict]:
    """Parse LLM ranking response, extracting JSON array."""
    # Try to find JSON array in the response
    # First try direct parse
    text = text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from text
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse LLM ranking response")
    return []
