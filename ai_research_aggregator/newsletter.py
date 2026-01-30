"""
Newsletter formatter for Substack publication.

Converts a DailyDigest into Substack-compatible HTML with clean,
readable formatting that works in both the Substack web reader and
email clients.

Substack accepts standard HTML for post bodies. Their editor renders
it with their own CSS, so we use semantic HTML and minimal inline
styles for email compatibility.
"""

import html
from datetime import datetime
from typing import Optional

from ai_research_aggregator.models import ContentItem, DailyDigest, EventItem


def digest_to_newsletter_html(
    digest: DailyDigest,
    newsletter_title: Optional[str] = None,
    intro_text: Optional[str] = None,
    footer_text: Optional[str] = None,
) -> str:
    """
    Convert a DailyDigest into Substack-ready HTML.

    Substack's editor accepts HTML directly. This produces clean,
    semantic HTML that renders well in both their web view and email.

    Args:
        digest: The generated daily digest.
        newsletter_title: Override the default title.
        intro_text: Optional intro paragraph (supports HTML).
        footer_text: Optional footer text.

    Returns:
        HTML string ready for Substack post body.
    """
    date_str = digest.date.strftime("%A, %B %d, %Y")
    title = newsletter_title or f"AI Research Digest - {date_str}"

    parts = []

    # Intro
    if intro_text:
        parts.append(f"<p>{intro_text}</p>")

    parts.append(
        f'<p style="color: #666; font-size: 14px;">'
        f"Scanned <strong>{digest.total_items_scanned}</strong> items across "
        f"arXiv, AI lab blogs, community forums, and event platforms."
        f"</p>"
    )

    parts.append("<hr>")

    # Sections
    for section in digest.sections:
        parts.append(f"<h2>{_esc(section.title)}</h2>")

        if section.editorial_intro:
            parts.append(
                f'<p style="color: #444; font-style: italic; '
                f'border-left: 3px solid #d1d5db; padding-left: 12px; margin: 8px 0 16px;">'
                f"{_esc(section.editorial_intro)}</p>"
            )
        elif section.description:
            parts.append(
                f'<p style="color: #666; font-style: italic;">'
                f"{_esc(section.description)}</p>"
            )

        if not section.items:
            parts.append("<p><em>No items found for this section today.</em></p>")
            continue

        for i, item in enumerate(section.items, 1):
            parts.append(_render_item(i, item))

        parts.append("<hr>")

    # Opportunity analysis
    if digest.opportunity_analysis:
        parts.append("<h2>Opportunity Spotlight</h2>")
        parts.append(
            '<div style="background: #f0f7ff; border-left: 4px solid #2563eb; '
            'padding: 16px 20px; border-radius: 0 8px 8px 0; margin: 8px 0 24px;">'
        )
        for paragraph in digest.opportunity_analysis.split("\n\n"):
            paragraph = paragraph.strip()
            if paragraph:
                parts.append(f"<p>{_esc(paragraph)}</p>")
        parts.append("</div>")
        parts.append("<hr>")

    # Footer
    if footer_text:
        parts.append(f'<p style="color: #888; font-size: 13px;">{footer_text}</p>')

    parts.append(
        '<p style="color: #999; font-size: 12px; text-align: center;">'
        "Curated by Hidden Layer"
        "</p>"
    )

    return "\n\n".join(parts)


def digest_to_newsletter_subject(digest: DailyDigest) -> str:
    """Generate the email subject / post title for the newsletter."""
    date_str = digest.date.strftime("%b %d, %Y")

    # Use LLM-generated headline if available
    if digest.headline:
        return f"Hidden Layer: {digest.headline}"

    # Fallback: pick a highlight from top item
    highlight = ""
    for section in digest.sections:
        if section.items:
            top = section.items[0]
            highlight = f": {top.title[:60]}"
            if len(top.title) > 60:
                highlight += "..."
            break

    return f"Hidden Layer Digest - {date_str}{highlight}"


def digest_to_newsletter_subtitle(digest: DailyDigest) -> str:
    """Generate the subtitle / preview text."""
    counts = []
    for section in digest.sections:
        if section.items:
            counts.append(f"{len(section.items)} {section.title.lower()}")

    if counts:
        return "Today's picks: " + ", ".join(counts)

    return f"AI research highlights for {digest.date.strftime('%B %d')}"


def _render_item(index: int, item: ContentItem) -> str:
    """Render a single content item as HTML."""
    parts = []

    # Title with link
    parts.append(
        f"<h3>{index}. "
        f'<a href="{_esc_attr(item.url)}">{_esc(item.title)}</a>'
        f"</h3>"
    )

    # Metadata line
    meta = []
    if item.authors:
        authors_str = ", ".join(item.authors[:3])
        if len(item.authors) > 3:
            authors_str += f" +{len(item.authors) - 3} more"
        meta.append(f"<strong>{_esc(authors_str)}</strong>")
    if item.published_date:
        meta.append(item.published_date.strftime("%b %d, %Y"))

    source_label = _source_display_name(item.source.value)
    meta.append(source_label)

    if item.relevance_score > 0:
        score_pct = int(item.relevance_score)
        color = _score_color(score_pct)
        meta.append(
            f'<span style="color: {color}; font-weight: bold;">'
            f"{score_pct}/100</span>"
        )

    if meta:
        parts.append(
            f'<p style="color: #666; font-size: 14px; margin-top: -8px;">'
            f'{" &middot; ".join(meta)}</p>'
        )

    # Summary or abstract as blockquote
    text = item.summary or item.abstract
    if text:
        display_text = text[:400]
        if len(text) > 400:
            display_text += "..."
        parts.append(
            f"<blockquote>"
            f"<p>{_esc(display_text)}</p>"
            f"</blockquote>"
        )

    # Why this matters
    if item.relevance_reason:
        parts.append(
            f'<p><strong>Why this matters:</strong> '
            f'{_esc(item.relevance_reason)}</p>'
        )

    # Tags
    if item.tags:
        display_tags = item.tags[:6]
        tag_spans = " ".join(
            f'<code>{_esc(tag)}</code>' for tag in display_tags
        )
        parts.append(f"<p>{tag_spans}</p>")

    # Event-specific
    if isinstance(item, EventItem):
        event_details = []
        if item.event_date:
            event_details.append(
                f"<strong>When:</strong> {item.event_date.strftime('%B %d, %Y at %I:%M %p')}"
            )
        if item.location:
            event_details.append(f"<strong>Where:</strong> {_esc(item.location)}")
        if item.price:
            event_details.append(f"<strong>Price:</strong> {_esc(item.price)}")
        if item.rsvp_count:
            event_details.append(f"<strong>RSVPs:</strong> {item.rsvp_count}")
        if event_details:
            parts.append(
                '<div style="background: #f7f7f7; padding: 12px; '
                'border-radius: 6px; margin: 8px 0;">'
                + "<br>".join(event_details)
                + "</div>"
            )

    return "\n".join(parts)


def _source_display_name(source_value: str) -> str:
    """Convert source enum value to a display name."""
    mapping = {
        "arxiv": "arXiv",
        "semantic_scholar": "Semantic Scholar",
        "openai_blog": "OpenAI Blog",
        "anthropic_blog": "Anthropic Blog",
        "deepmind_blog": "DeepMind Blog",
        "meta_ai_blog": "Meta AI Blog",
        "hacker_news": "Hacker News",
        "reddit_ml": "Reddit",
        "twitter": "Twitter/X",
        "luma": "Lu.ma",
        "meetup": "Meetup",
        "eventbrite": "Eventbrite",
    }
    return mapping.get(source_value, source_value)


def _score_color(score: int) -> str:
    """Map relevance score to a color."""
    if score >= 80:
        return "#16a34a"  # green
    if score >= 60:
        return "#2563eb"  # blue
    if score >= 40:
        return "#d97706"  # amber
    return "#6b7280"      # gray


def _esc(text: str) -> str:
    """HTML-escape text."""
    return html.escape(text, quote=False)


def _esc_attr(text: str) -> str:
    """HTML-escape for use in attributes."""
    return html.escape(text, quote=True)
