"""
Ranking and summarization engine.

Uses LLM to score content relevance and generate concise summaries.
Also provides keyword-based fallback ranking when LLM is unavailable.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ai_research_aggregator.config import AggregatorConfig
from ai_research_aggregator.models import ContentItem, ContentType

logger = logging.getLogger(__name__)


# --- Cost tracking ---

# Pricing per 1M tokens (input, output) as of Jan 2025
PRICING = {
    # Anthropic
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-5-haiku-20241022": (0.80, 4.0),
    "claude-3-opus-20240229": (15.0, 75.0),
    # OpenAI
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.0, 30.0),
}


@dataclass
class LLMCostTracker:
    """Tracks token usage and estimated cost across LLM calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    failed_calls: int = 0
    model: str = ""
    call_details: List[Dict] = field(default_factory=list)

    def record_call(self, input_tokens: int, output_tokens: int, purpose: str = ""):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1
        self.call_details.append({
            "purpose": purpose,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })

    def record_failure(self):
        self.failed_calls += 1

    @property
    def estimated_cost(self) -> float:
        """Estimate USD cost based on model pricing."""
        if self.model in PRICING:
            input_price, output_price = PRICING[self.model]
        else:
            # Default to Claude Sonnet pricing as fallback
            input_price, output_price = (3.0, 15.0)
        input_cost = (self.total_input_tokens / 1_000_000) * input_price
        output_cost = (self.total_output_tokens / 1_000_000) * output_price
        return input_cost + output_cost

    def summary(self) -> str:
        """Return a human-readable cost summary."""
        lines = [
            f"LLM Cost Report ({self.model})",
            f"  Calls: {self.total_calls} ({self.failed_calls} failed)",
            f"  Input tokens:  {self.total_input_tokens:,}",
            f"  Output tokens: {self.total_output_tokens:,}",
            f"  Estimated cost: ${self.estimated_cost:.4f}",
        ]
        if self.call_details:
            lines.append("  Breakdown:")
            for detail in self.call_details:
                lines.append(
                    f"    {detail['purpose']}: "
                    f"{detail['input_tokens']:,} in / {detail['output_tokens']:,} out"
                )
        return "\n".join(lines)


# Global tracker for the current run — reset by generate_digest()
_cost_tracker: Optional[LLMCostTracker] = None


def get_cost_tracker() -> Optional[LLMCostTracker]:
    """Get the current cost tracker (may be None if no LLM calls made)."""
    return _cost_tracker


def reset_cost_tracker(model: str = "") -> LLMCostTracker:
    """Reset and return a fresh cost tracker."""
    global _cost_tracker
    _cost_tracker = LLMCostTracker(model=model)
    return _cost_tracker


# --- API key detection ---

def check_api_key(provider: str) -> bool:
    """Check whether the required API key is set for the given provider."""
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "ollama": None,  # Local, no key needed
    }
    env_var = key_map.get(provider)
    if env_var is None:
        return True  # Local provider, always available
    return bool(os.environ.get(env_var))

# --- Prompt loading ---
# Prompts are loaded from .txt files in the prompts/ directory.
# The editorial persona is automatically injected via {persona} placeholder.

def _get_ranking_prompt(interests: str, key_figures: str, items_json: str) -> str:
    """Build the ranking prompt from template files."""
    from ai_research_aggregator.prompts import get_prompt
    return get_prompt(
        "ranking",
        interests=interests,
        key_figures=key_figures,
        items_json=items_json,
    )


def _get_opportunity_prompt(interests: str, highlights: str) -> str:
    """Build the opportunity analysis prompt from template files."""
    from ai_research_aggregator.prompts import get_prompt
    return get_prompt(
        "opportunity",
        interests=interests,
        highlights=highlights,
    )


def rank_with_llm(
    items: List[ContentItem],
    config: AggregatorConfig,
) -> List[ContentItem]:
    """
    Use LLM to rank and summarize content items.

    Items are processed in batches and scored 0-100 for relevance.
    Includes retry with titles-only fallback and cost tracking.
    """
    try:
        from harness import llm_call
    except ImportError:
        logger.warning("Harness not available, falling back to keyword ranking")
        return rank_with_keywords(items, config)

    # Check API key before making any calls
    if not check_api_key(config.llm.provider):
        logger.warning(
            f"No API key found for {config.llm.provider}. Using keyword ranking."
        )
        print(f"  Warning: No API key for {config.llm.provider}. Using keyword ranking.")
        return rank_with_keywords(items, config)

    if not items:
        return []

    # Ensure cost tracker is initialized
    tracker = _cost_tracker or reset_cost_tracker(config.llm.model)

    batch_size = config.llm.ranking_batch_size
    all_scored = []

    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1

        # Prepare full items for the prompt
        items_for_prompt = _prepare_items_for_prompt(batch, include_abstracts=True)

        interests_str = "\n".join(f"- {t}" for t in config.interests.topics)
        figures_str = ", ".join(config.interests.key_figures)
        prompt = _get_ranking_prompt(
            interests=interests_str,
            key_figures=figures_str,
            items_json=json.dumps(items_for_prompt, indent=2),
        )

        rankings = _call_llm_for_rankings(
            llm_call, prompt, batch, batch_num, config, tracker
        )

        if rankings is None:
            # First attempt failed — retry with titles-only (shorter prompt)
            logger.info(f"Retrying batch {batch_num} with titles-only prompt")
            items_for_prompt_short = _prepare_items_for_prompt(batch, include_abstracts=False)
            short_prompt = _get_ranking_prompt(
                interests=interests_str,
                key_figures=figures_str,
                items_json=json.dumps(items_for_prompt_short, indent=2),
            )

            rankings = _call_llm_for_rankings(
                llm_call, short_prompt, batch, batch_num, config, tracker,
                purpose_suffix=" (retry, titles-only)",
            )

        if rankings is not None:
            for ranking in rankings:
                idx = ranking.get("index", -1)
                if 0 <= idx < len(batch):
                    batch[idx].relevance_score = ranking.get("relevance_score", 0)
                    batch[idx].summary = ranking.get("summary", "")
                    batch[idx].relevance_reason = ranking.get("relevance_reason", "")
        else:
            # Both attempts failed — keyword fallback
            logger.warning(f"LLM ranking failed for batch {batch_num} after retry. Using keyword fallback.")
            batch = _keyword_score_batch(batch, config)

        all_scored.extend(batch)

    # Sort by relevance score descending
    all_scored.sort(key=lambda x: x.relevance_score, reverse=True)
    return all_scored


def _prepare_items_for_prompt(
    batch: List[ContentItem], include_abstracts: bool = True
) -> List[Dict]:
    """Prepare batch items for the LLM prompt."""
    items_for_prompt = []
    for i, item in enumerate(batch):
        entry = {
            "index": i,
            "title": item.title,
            "source": item.source.value,
            "type": item.content_type.value,
        }
        if include_abstracts:
            entry["abstract"] = item.abstract[:300] if item.abstract else ""
            entry["authors"] = item.authors[:5]
            entry["tags"] = item.tags[:10]
        # Add engagement metrics for community posts
        if item.content_type == ContentType.COMMUNITY_POST:
            entry["points"] = item.metadata.get("points", 0) + item.metadata.get("score", 0)
            entry["comments"] = item.metadata.get("num_comments", 0)
        items_for_prompt.append(entry)
    return items_for_prompt


def _call_llm_for_rankings(
    llm_call,
    prompt: str,
    batch: List[ContentItem],
    batch_num: int,
    config: AggregatorConfig,
    tracker: LLMCostTracker,
    purpose_suffix: str = "",
) -> Optional[List[Dict]]:
    """
    Call LLM and parse rankings. Returns parsed rankings list, or None on failure.
    Records usage in the cost tracker.
    """
    purpose = f"ranking batch {batch_num}{purpose_suffix}"
    try:
        response = llm_call(
            prompt=prompt,
            provider=config.llm.provider,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )

        # Track token usage if available on the response
        input_tokens = getattr(response, "input_tokens", 0) or 0
        output_tokens = getattr(response, "output_tokens", 0) or 0
        tracker.record_call(input_tokens, output_tokens, purpose=purpose)

        rankings = _parse_ranking_response(response.text, expected_count=len(batch))

        if not rankings:
            logger.warning(f"Empty rankings from LLM for {purpose}")
            tracker.record_failure()
            return None

        return rankings

    except Exception as e:
        logger.warning(f"LLM call failed for {purpose}: {e}")
        tracker.record_failure()
        return None


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

    if not check_api_key(config.llm.provider):
        logger.warning(f"No API key for {config.llm.provider}, skipping opportunity analysis")
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

    prompt = _get_opportunity_prompt(
        interests="\n".join(f"- {t}" for t in config.interests.topics),
        highlights=json.dumps(highlights, indent=2),
    )

    tracker = _cost_tracker

    try:
        response = llm_call(
            prompt=prompt,
            provider=config.llm.provider,
            model=config.llm.model,
            temperature=0.5,
            max_tokens=1024,
        )

        # Track cost
        if tracker:
            input_tokens = getattr(response, "input_tokens", 0) or 0
            output_tokens = getattr(response, "output_tokens", 0) or 0
            tracker.record_call(input_tokens, output_tokens, purpose="opportunity analysis")

        return response.text.strip()
    except Exception as e:
        logger.warning(f"Opportunity analysis generation failed: {e}")
        if tracker:
            tracker.record_failure()
        return ""


# --- Pass 2: Editorial summaries, section intros, headline ---

def generate_editorial_summaries(
    items: List[ContentItem],
    all_top_items: List[ContentItem],
    config: "AggregatorConfig",
) -> List[ContentItem]:
    """
    Second-pass LLM: generate richer editorial summaries for top items.

    Each item gets a 3-4 sentence summary that cross-references other items
    in today's briefing and uses the editorial persona.

    Args:
        items: The top items to generate editorial summaries for.
        all_top_items: All top items across sections (for cross-referencing).
        config: Aggregator config.

    Returns:
        The same items with updated summary and relevance_reason fields.
    """
    try:
        from harness import llm_call
    except ImportError:
        return items

    if not check_api_key(config.llm.provider):
        return items

    from ai_research_aggregator.prompts import get_prompt

    tracker = _cost_tracker

    # Build cross-reference context (other items)
    other_items_json = json.dumps([
        {"title": it.title, "source": it.source.value, "summary": it.summary or ""}
        for it in all_top_items
    ], indent=2)

    interests_str = "\n".join(f"- {t}" for t in config.interests.topics)

    for i, item in enumerate(items):
        item_json = json.dumps({
            "title": item.title,
            "source": item.source.value,
            "type": item.content_type.value,
            "abstract": item.abstract[:500] if item.abstract else "",
            "authors": item.authors[:5],
            "tags": item.tags[:10],
            "current_summary": item.summary,
        }, indent=2)

        try:
            prompt = get_prompt(
                "editorial_summary",
                interests=interests_str,
                other_items=other_items_json,
                item_json=item_json,
            )

            response = llm_call(
                prompt=prompt,
                provider=config.llm.provider,
                model=config.llm.model,
                temperature=0.4,
                max_tokens=512,
            )

            if tracker:
                input_tokens = getattr(response, "input_tokens", 0) or 0
                output_tokens = getattr(response, "output_tokens", 0) or 0
                tracker.record_call(input_tokens, output_tokens,
                                    purpose=f"editorial summary {i+1}/{len(items)}")

            parsed = _parse_json_object(response.text)
            if parsed:
                if parsed.get("summary"):
                    item.summary = parsed["summary"]
                if parsed.get("relevance_reason"):
                    item.relevance_reason = parsed["relevance_reason"]

        except Exception as e:
            logger.warning(f"Editorial summary failed for '{item.title[:50]}': {e}")
            if tracker:
                tracker.record_failure()
            # Keep existing pass-1 summary

    return items


def generate_section_intro(
    section_title: str,
    items: List[ContentItem],
    config: "AggregatorConfig",
) -> str:
    """
    Generate a 2-3 sentence editorial intro for a digest section.

    Returns the intro text, or empty string on failure.
    """
    try:
        from harness import llm_call
    except ImportError:
        return ""

    if not check_api_key(config.llm.provider):
        return ""

    from ai_research_aggregator.prompts import get_prompt

    items_summary = json.dumps([
        {"title": it.title, "source": it.source.value}
        for it in items
    ], indent=2)

    try:
        prompt = get_prompt(
            "section_intro",
            section_title=section_title,
            items_summary=items_summary,
        )

        response = llm_call(
            prompt=prompt,
            provider=config.llm.provider,
            model=config.llm.model,
            temperature=0.6,
            max_tokens=256,
        )

        tracker = _cost_tracker
        if tracker:
            input_tokens = getattr(response, "input_tokens", 0) or 0
            output_tokens = getattr(response, "output_tokens", 0) or 0
            tracker.record_call(input_tokens, output_tokens,
                                purpose=f"section intro: {section_title}")

        return response.text.strip().strip('"')

    except Exception as e:
        logger.warning(f"Section intro failed for '{section_title}': {e}")
        if _cost_tracker:
            _cost_tracker.record_failure()
        return ""


def generate_headline(
    top_items: List[ContentItem],
    date_str: str,
    config: "AggregatorConfig",
) -> str:
    """
    Generate a compelling newsletter subject line from today's top items.

    Returns the headline text, or empty string on failure.
    """
    try:
        from harness import llm_call
    except ImportError:
        return ""

    if not check_api_key(config.llm.provider):
        return ""

    from ai_research_aggregator.prompts import get_prompt

    top_items_json = json.dumps([
        {"title": it.title, "summary": it.summary or it.abstract[:150]}
        for it in top_items[:3]
    ], indent=2)

    try:
        prompt = get_prompt(
            "headline",
            top_items=top_items_json,
            date=date_str,
        )

        response = llm_call(
            prompt=prompt,
            provider=config.llm.provider,
            model=config.llm.model,
            temperature=0.7,
            max_tokens=100,
        )

        tracker = _cost_tracker
        if tracker:
            input_tokens = getattr(response, "input_tokens", 0) or 0
            output_tokens = getattr(response, "output_tokens", 0) or 0
            tracker.record_call(input_tokens, output_tokens, purpose="headline")

        headline = response.text.strip().strip('"').strip("'")
        # Enforce max length
        if len(headline) > 70:
            headline = headline[:67] + "..."
        return headline

    except Exception as e:
        logger.warning(f"Headline generation failed: {e}")
        if _cost_tracker:
            _cost_tracker.record_failure()
        return ""


def _parse_json_object(text: str) -> Optional[Dict]:
    """Parse a single JSON object from LLM response text."""
    text = text.strip()
    # Remove markdown code blocks
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try to extract object from surrounding text
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None


def _parse_ranking_response(text: str, expected_count: int = 0) -> List[Dict]:
    """
    Parse LLM ranking response with multi-level fallback chain.

    Fallback order:
    1. Direct JSON array parse (clean response)
    2. Extract JSON array from surrounding text / markdown
    3. Extract individual JSON objects via regex
    4. Extract score/index pairs from semi-structured text
    """
    text = text.strip()

    # --- Fallback 1: Clean JSON array ---
    # Remove markdown code blocks if present
    cleaned = text
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            logger.debug("Parsed rankings via direct JSON parse")
            return _validate_rankings(result)
    except json.JSONDecodeError:
        pass

    # --- Fallback 2: Extract JSON array from surrounding text ---
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                logger.debug("Parsed rankings via JSON array extraction")
                return _validate_rankings(result)
        except json.JSONDecodeError:
            pass

    # --- Fallback 3: Extract individual JSON objects ---
    objects = re.findall(r'\{[^{}]*\}', text)
    if objects:
        parsed = []
        for obj_str in objects:
            try:
                obj = json.loads(obj_str)
                if "index" in obj or "relevance_score" in obj:
                    parsed.append(obj)
            except json.JSONDecodeError:
                continue
        if parsed:
            logger.debug(f"Parsed {len(parsed)} rankings via individual object extraction")
            return _validate_rankings(parsed)

    # --- Fallback 4: Extract score/index pairs from text ---
    # Handles cases like "Item 0: score 85" or "index: 0, relevance_score: 85"
    pattern = r'(?:index|item)\s*[:=]?\s*(\d+)\D+(?:score|relevance)\s*[:=]?\s*(\d+)'
    score_matches = re.findall(pattern, text, re.IGNORECASE)
    if score_matches:
        parsed = []
        for idx_str, score_str in score_matches:
            parsed.append({
                "index": int(idx_str),
                "relevance_score": min(100, max(0, int(score_str))),
                "summary": "",
                "relevance_reason": "",
            })
        logger.debug(f"Parsed {len(parsed)} rankings via score/index extraction")
        return parsed

    logger.warning("Failed to parse LLM ranking response with all fallback strategies")
    return []


def _validate_rankings(rankings: List[Dict]) -> List[Dict]:
    """Validate and normalize parsed ranking entries."""
    validated = []
    for entry in rankings:
        if not isinstance(entry, dict):
            continue
        # Ensure index exists
        if "index" not in entry:
            continue
        # Clamp score
        score = entry.get("relevance_score", 0)
        if isinstance(score, (int, float)):
            entry["relevance_score"] = min(100, max(0, int(score)))
        else:
            entry["relevance_score"] = 0
        # Ensure string fields
        entry.setdefault("summary", "")
        entry.setdefault("relevance_reason", "")
        validated.append(entry)
    return validated
