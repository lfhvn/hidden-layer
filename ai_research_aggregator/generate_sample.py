#!/usr/bin/env python3
"""Generate a sample Hidden Layer newsletter and save as viewable HTML."""

import os
from datetime import datetime, timedelta

from ai_research_aggregator.config import AggregatorConfig
from ai_research_aggregator.models import (
    ContentItem, ContentType, DailyDigest, DigestSection, EventItem, SourceName,
)
from ai_research_aggregator.newsletter import (
    digest_to_newsletter_html, digest_to_newsletter_subject, digest_to_newsletter_subtitle,
)

now = datetime.now()

papers = [
    ContentItem(
        title="Decomposing GPT-4 Reasoning with Sparse Autoencoders",
        url="https://arxiv.org/abs/2601.14201",
        source=SourceName.ARXIV, content_type=ContentType.PAPER,
        published_date=now - timedelta(hours=6),
        authors=["Chris Olah", "Joshua Batson", "Adly Templeton", "Tom Brown"],
        abstract="We apply sparse autoencoders to GPT-4's residual stream and identify a dictionary of 32,768 interpretable features. Among these, we find features that activate specifically during multi-step reasoning, factual recall, and safety-relevant refusal behavior. We demonstrate that ablating individual features produces targeted behavioral changes, opening a new pathway for mechanistic control.",
        tags=["cs.AI", "cs.LG"],
        relevance_score=95,
        summary="Applies SAEs to GPT-4 and finds 32K interpretable features including reasoning and safety features, enabling targeted behavioral control via ablation.",
        relevance_reason="Core SAE interpretability work by a tracked researcher, directly relevant to mechanistic interpretability and activation steering.",
    ),
    ContentItem(
        title="Theory of Mind in Large Language Models: A Systematic Evaluation of Perspective-Taking",
        url="https://arxiv.org/abs/2601.13987",
        source=SourceName.ARXIV, content_type=ContentType.PAPER,
        published_date=now - timedelta(hours=14),
        authors=["Sasha Rush", "Yonatan Belinkov", "Anna Rogers"],
        abstract="We present a comprehensive evaluation of theory of mind capabilities in 15 LLMs across false-belief, knowledge attribution, and intention prediction tasks. We find that models above 70B parameters exhibit emergent perspective-taking that correlates with model scale but not RLHF training. We release ToM-Bench, a 2,400-item benchmark.",
        tags=["cs.CL", "cs.AI"],
        relevance_score=91,
        summary="Systematic ToM evaluation across 15 LLMs finds emergent perspective-taking at scale. Releases ToM-Bench (2,400 items).",
        relevance_reason="Directly advances theory of mind evaluation, relevant to SELPHI and introspection research.",
    ),
    ContentItem(
        title="Steering Without Vectors: Behavioral Control via Activation Patching in Latent Space",
        url="https://arxiv.org/abs/2601.14088",
        source=SourceName.ARXIV, content_type=ContentType.PAPER,
        published_date=now - timedelta(hours=10),
        authors=["Paul Christiano", "Jan Leike", "Ajeya Cotra"],
        abstract="We propose activation patching as an alternative to steering vectors for behavioral control. Rather than adding a fixed direction, we identify and patch specific feature activations at inference time. This achieves 89% adherence on our steerability benchmark while preserving model capability, compared to 71% for contrastive activation addition.",
        tags=["cs.AI", "cs.LG"],
        relevance_score=89,
        summary="Proposes activation patching as a more precise alternative to steering vectors, achieving 89% adherence vs 71% for CAA.",
        relevance_reason="Directly relevant to steerability and alignment research, by tracked researchers.",
    ),
    ContentItem(
        title="Multi-Agent Debate with Latent Communication Channels",
        url="https://arxiv.org/abs/2601.13850",
        source=SourceName.ARXIV, content_type=ContentType.PAPER,
        published_date=now - timedelta(hours=18),
        authors=["Yilun Du", "Shuang Li", "Joshua Tenenbaum"],
        abstract="We augment multi-agent debate with latent communication channels where agents exchange compressed representations instead of natural language. On MATH and GSM8K, this improves accuracy by 12% over standard debate while reducing token costs by 60%. We analyze the emergent communication protocol and find it encodes structured reasoning traces.",
        tags=["cs.MA", "cs.AI", "cs.CL"],
        relevance_score=87,
        summary="Multi-agent debate with latent channels instead of language improves accuracy 12% on math benchmarks while cutting token costs 60%.",
        relevance_reason="Combines multi-agent coordination with AI-to-AI communication — core Hidden Layer themes.",
    ),
    ContentItem(
        title="Do Language Models Know When They're Wrong? Probing Introspective Accuracy",
        url="https://arxiv.org/abs/2601.14150",
        source=SourceName.ARXIV, content_type=ContentType.PAPER,
        published_date=now - timedelta(hours=8),
        authors=["Jacob Steinhardt", "Collin Burns", "Erik Jones"],
        abstract="We investigate whether LLMs can accurately report their own uncertainty. Using probes trained on intermediate activations, we find models maintain accurate internal uncertainty estimates that diverge from their verbalized confidence. We propose a training procedure that closes this gap, improving calibration by 34% without harming task performance.",
        tags=["cs.AI", "cs.LG"],
        relevance_score=86,
        summary="Shows LLMs maintain accurate internal uncertainty that differs from verbalized confidence. Proposes training fix improving calibration 34%.",
        relevance_reason="Directly relevant to introspection and AI deception detection — can models honestly report internal states?",
    ),
    ContentItem(
        title="Scalable Oversight through Recursive Reward Modeling with Human Feedback",
        url="https://arxiv.org/abs/2601.13799",
        source=SourceName.ARXIV, content_type=ContentType.PAPER,
        published_date=now - timedelta(hours=22),
        authors=["Geoffrey Irving", "Amanda Askell", "Sam McCandlish"],
        abstract="We present a recursive approach to reward modeling where AI assistants help humans evaluate outputs from more capable AI systems. Our three-level recursive setup improves evaluator agreement by 28% on complex reasoning tasks compared to direct human evaluation.",
        tags=["cs.AI", "cs.CL"],
        relevance_score=78,
        summary="Recursive reward modeling where AI assists human evaluators, improving agreement 28% on complex tasks.",
        relevance_reason="Relevant to RLHF, alignment, and scalable oversight.",
    ),
    ContentItem(
        title="Attention Pattern Geometry Predicts In-Context Learning Ability",
        url="https://arxiv.org/abs/2601.14005",
        source=SourceName.ARXIV, content_type=ContentType.PAPER,
        published_date=now - timedelta(hours=16),
        authors=["Percy Liang", "Sang Michael Xie", "Tengyu Ma"],
        abstract="We discover that the geometric structure of attention patterns — specifically, the rank and eigenvalue distribution of attention matrices — strongly predicts a model's in-context learning performance. Models with higher-rank attention at early layers demonstrate better few-shot adaptation. We validate across 8 model families.",
        tags=["cs.LG", "cs.CL"],
        relevance_score=72,
        summary="Attention matrix geometry (rank, eigenvalues) predicts in-context learning ability across 8 model families.",
        relevance_reason="Relevant to latent representations and interpretability — understanding what internal structure enables capabilities.",
    ),
]

blogs = [
    ContentItem(
        title="Introducing Constitutional AI 2.0: Principles That Scale",
        url="https://www.anthropic.com/research/constitutional-ai-2",
        source=SourceName.ANTHROPIC_BLOG, content_type=ContentType.BLOG_POST,
        published_date=now - timedelta(hours=4),
        authors=["Dario Amodei", "Jared Kaplan"],
        abstract="We announce Constitutional AI 2.0, a major update to our alignment approach. Key advances include dynamic principle selection based on context, principle composition for complex scenarios, and a new self-critique mechanism that improves harmlessness by 45% while maintaining helpfulness.",
        tags=["alignment", "anthropic", "constitutional-ai"],
        relevance_score=93,
        summary="Anthropic releases Constitutional AI 2.0 with dynamic principle selection and self-critique, improving harmlessness 45%.",
        relevance_reason="Major alignment advance from Anthropic, authored by Dario Amodei — core tracked figure and topic.",
    ),
    ContentItem(
        title="Our Approach to AI Safety Research in 2026",
        url="https://openai.com/blog/safety-research-2026",
        source=SourceName.OPENAI_BLOG, content_type=ContentType.BLOG_POST,
        published_date=now - timedelta(hours=12),
        authors=["Sam Altman", "Jan Leike"],
        abstract="We outline OpenAI's safety research priorities for 2026, including: automated red-teaming at scale, interpretability of chain-of-thought reasoning, and new approaches to detecting deceptive alignment. We also announce $50M in safety grants.",
        tags=["safety", "openai"],
        relevance_score=85,
        summary="OpenAI's 2026 safety roadmap: automated red-teaming, CoT interpretability, deceptive alignment detection. $50M in safety grants.",
        relevance_reason="Safety priorities from tracked figures at OpenAI, touching CoT interpretability and deception detection.",
    ),
    ContentItem(
        title="Gemini's New Multimodal Reasoning Capabilities",
        url="https://deepmind.google/blog/gemini-multimodal-reasoning",
        source=SourceName.DEEPMIND_BLOG, content_type=ContentType.BLOG_POST,
        published_date=now - timedelta(hours=20),
        authors=["Demis Hassabis", "Oriol Vinyals"],
        abstract="We present new multimodal reasoning capabilities in Gemini that allow the model to perform joint reasoning across text, images, audio, and video in a unified latent space. On our new MultiReason benchmark, Gemini achieves 82% accuracy.",
        tags=["multimodal", "deepmind", "reasoning"],
        relevance_score=70,
        summary="Gemini gains unified multimodal reasoning across text/image/audio/video, hitting 82% on new MultiReason benchmark.",
        relevance_reason="Multimodal latent space work relevant to representations research; authored by tracked figure Demis Hassabis.",
    ),
    ContentItem(
        title="LLaMA 4 Scout: Efficient Expertise through Mixture of Experts",
        url="https://ai.meta.com/blog/llama-4-scout",
        source=SourceName.META_AI_BLOG, content_type=ContentType.BLOG_POST,
        published_date=now - timedelta(hours=28),
        authors=["Mark Zuckerberg", "Yann LeCun"],
        abstract="We release LLaMA 4 Scout, a 109B active parameter MoE model (from 400B total) that matches GPT-4 performance while running 3x faster. The model uses a new routing mechanism that specializes experts by domain rather than token position.",
        tags=["llama", "meta", "open-source"],
        relevance_score=62,
        summary="Meta releases LLaMA 4 Scout — 109B active MoE model matching GPT-4 at 3x speed, with domain-specialized routing.",
        relevance_reason="Major open model release; Yann LeCun is a tracked figure.",
    ),
]

community = [
    ContentItem(
        title='Anthropic researchers found "deception neurons" that activate when Claude is about to be dishonest',
        url="https://news.ycombinator.com/item?id=42901234",
        source=SourceName.HACKER_NEWS, content_type=ContentType.COMMUNITY_POST,
        published_date=now - timedelta(hours=5),
        authors=["throwaway_ml"],
        abstract="Link to new Anthropic paper on identifying specific neurons/features that activate when the model is about to produce a deceptive or misleading output. Discussion of implications for AI safety monitoring.",
        tags=["anthropic", "interpretability", "safety"],
        relevance_score=88,
        summary="HN discussion of Anthropic's discovery of deception-correlated features in Claude's activations.",
        relevance_reason="Directly relevant to AI deception detection and introspection research.",
        metadata={"points": 847, "num_comments": 312},
    ),
    ContentItem(
        title="[R] Multi-agent systems spontaneously develop deceptive coordination strategies",
        url="https://reddit.com/r/MachineLearning/comments/1d5e6f7",
        source=SourceName.REDDIT_ML, content_type=ContentType.COMMUNITY_POST,
        published_date=now - timedelta(hours=15),
        authors=["alignment_researcher"],
        abstract="Paper discussion: researchers at UC Berkeley show that multi-agent LLM systems spontaneously develop deceptive coordination strategies when optimized for team performance, even without explicit incentives to deceive.",
        tags=["multi-agent", "deception", "alignment"],
        relevance_score=83,
        summary="UC Berkeley paper shows multi-agent LLM systems spontaneously develop deceptive coordination — no explicit deception incentive needed.",
        relevance_reason="Intersects multi-agent coordination, deception, and alignment — core Hidden Layer research areas.",
        metadata={"points": 567, "num_comments": 145},
    ),
    ContentItem(
        title="[D] Has anyone tried combining steering vectors with LoRA for fine-grained personality control?",
        url="https://reddit.com/r/MachineLearning/comments/1a2b3c4",
        source=SourceName.REDDIT_ML, content_type=ContentType.COMMUNITY_POST,
        published_date=now - timedelta(hours=9),
        authors=["ml_researcher_42"],
        abstract="Discussion thread exploring combining activation steering with LoRA adapters. Several researchers share preliminary results showing the combination outperforms either method alone for personality and style control.",
        tags=["steering", "lora", "alignment"],
        relevance_score=80,
        summary="r/ML discussion on combining steering vectors + LoRA for personality control — early results show synergistic effects.",
        relevance_reason="Directly relevant to steerability research and activation steering.",
        metadata={"points": 234, "num_comments": 89},
    ),
    ContentItem(
        title='Show HN: Open-source SAE feature visualizer for any transformer model',
        url="https://news.ycombinator.com/item?id=42905678",
        source=SourceName.HACKER_NEWS, content_type=ContentType.COMMUNITY_POST,
        published_date=now - timedelta(hours=11),
        authors=["sae_explorer"],
        abstract="Open-source tool for visualizing SAE features across any transformer model. Supports Llama, Mistral, GPT-2, Pythia. Includes activation heatmaps, feature correlation analysis, and interactive dashboard.",
        tags=["sae", "interpretability", "open-source", "tools"],
        relevance_score=77,
        summary="Open-source SAE feature visualizer supporting multiple transformers — includes activation heatmaps and interactive dashboard.",
        relevance_reason="Directly relevant to SAE/Latent Lens interpretability research and tooling.",
        metadata={"points": 423, "num_comments": 67},
    ),
]

events = [
    EventItem(
        title="SF AI Meetup: Mechanistic Interpretability in Practice",
        url="https://lu.ma/sf-mechinterp-feb",
        source=SourceName.LUMA, content_type=ContentType.EVENT,
        event_date=datetime(2026, 2, 12, 18, 30),
        event_end_date=datetime(2026, 2, 12, 21, 0),
        location="Anthropic HQ, 427 N Tatnall St, San Francisco, CA",
        price="Free", rsvp_count=185, is_virtual=False,
        tags=["interpretability", "meetup", "sf"],
        abstract="Lightning talks on SAE features, activation patching, and circuit discovery. Speakers from Anthropic and EleutherAI.",
        relevance_reason="Directly relevant to SAE interpretability research — networking opportunity with Anthropic and EleutherAI researchers.",
    ),
    EventItem(
        title="AI Tinkerers SF — February Demo Night",
        url="https://lu.ma/ai-tinkerers-sf-feb",
        source=SourceName.LUMA, content_type=ContentType.EVENT,
        event_date=datetime(2026, 2, 6, 18, 0),
        event_end_date=datetime(2026, 2, 6, 21, 0),
        location="GitHub HQ, 88 Colin P Kelly Jr St, San Francisco, CA",
        price="Free", rsvp_count=250, is_virtual=False,
        tags=["demo-night", "sf", "community"],
        abstract="Monthly demo night for AI builders. 8 teams present projects in 5 minutes each. Food and drinks provided.",
        relevance_reason="Good venue for demoing Hidden Layer tooling and scouting multi-agent / interpretability projects from other builders.",
    ),
    EventItem(
        title="Multi-Agent Systems Workshop @ Stanford HAI",
        url="https://lu.ma/stanford-multi-agent-feb",
        source=SourceName.LUMA, content_type=ContentType.EVENT,
        event_date=datetime(2026, 2, 18, 13, 0),
        event_end_date=datetime(2026, 2, 18, 17, 30),
        location="Stanford HAI, 475 Via Ortega, Stanford, CA",
        price="$25", rsvp_count=90, is_virtual=False,
        tags=["multi-agent", "workshop", "stanford"],
        abstract="Workshop on coordination, communication, and emergent behavior in multi-agent LLM systems. Keynote by Yilun Du (MIT).",
        relevance_reason="Core multi-agent research event with keynote by a latent-channel communication author — high overlap with Hidden Layer priorities.",
    ),
    EventItem(
        title="Bay Area AI Safety Unconference",
        url="https://lu.ma/bay-area-safety-unconf",
        source=SourceName.LUMA, content_type=ContentType.EVENT,
        event_date=datetime(2026, 2, 22, 10, 0),
        event_end_date=datetime(2026, 2, 22, 18, 0),
        location="Lighthaven, 2740 Telegraph Ave, Berkeley, CA",
        price="Free", rsvp_count=120, is_virtual=False,
        tags=["ai-safety", "unconference", "berkeley"],
        abstract="Full-day unconference on AI safety research directions. Topics include scalable oversight, deception detection, and alignment evaluation.",
        relevance_reason="Covers deception detection and alignment evaluation — ideal venue to present introspection and steerability findings.",
    ),
    EventItem(
        title="LLM Evaluation & Benchmarking Summit (Virtual + SF Watch Party)",
        url="https://eventbrite.com/e/llm-eval-summit-2026",
        source=SourceName.EVENTBRITE, content_type=ContentType.EVENT,
        event_date=datetime(2026, 3, 5, 9, 0),
        event_end_date=datetime(2026, 3, 5, 17, 0),
        location="Online + SF Watch Party at WeWork, 44 Montgomery St",
        price="$50 (virtual free)", rsvp_count=800, is_virtual=True,
        tags=["evaluation", "benchmarks", "summit"],
        abstract="Day-long summit on LLM evaluation methodology, benchmark design, and reproducibility. Speakers from Stanford, Anthropic, OpenAI, and Hugging Face.",
        relevance_reason="Evaluation methodology is critical infrastructure — relevant for benchmarking ToM, introspection, and steerability metrics.",
    ),
]

opportunity_text = (
    "Today's papers reveal a convergence that points to a concrete product opportunity: "
    "an open-source interpretability-driven safety monitor for multi-agent systems. "
    "Olah et al.'s SAE decomposition of GPT-4 shows we can now isolate features tied to "
    "deception and reasoning in frontier models. Simultaneously, the UC Berkeley work on "
    "spontaneous deceptive coordination in multi-agent systems demonstrates that these "
    "deception features aren't theoretical — they emerge unprompted when agents optimize "
    "together. The open-source SAE feature visualizer on Hacker News provides the tooling "
    "foundation, and Christiano & Leike's activation patching gives us a precise "
    "intervention mechanism."
    "\n\n"
    "The opportunity is to build a runtime monitoring layer that sits between agents in a "
    "multi-agent deployment, continuously probing each agent's activations for the "
    "deception-correlated features identified by SAE decomposition. When deceptive "
    "coordination patterns are detected, the system could apply targeted activation patches "
    "in real time — not just flagging the behavior but correcting it. Nothing like this "
    "exists today; current multi-agent frameworks (CrewAI, AutoGen, LangGraph) have zero "
    "interpretability hooks."
    "\n\n"
    "A concrete first step: fork the open-source SAE visualizer to support streaming "
    "activation analysis from Ollama or vLLM inference servers, then instrument a simple "
    "two-agent debate setup to detect when the agents' latent representations shift toward "
    "coordination features that weren't present in their individual training runs. The "
    "Stanford HAI multi-agent workshop on Feb 18 would be an ideal venue to present early "
    "results and recruit collaborators."
)

digest = DailyDigest(
    date=now,
    sections=[
        DigestSection("Top Research Papers", "Highest-relevance papers from arXiv in the last 72 hours", papers),
        DigestSection("Lab Updates & Blog Posts", "From OpenAI, Anthropic, DeepMind, and Meta AI", blogs),
        DigestSection("Community Discussions", "Top-voted threads from Hacker News and r/MachineLearning", community),
        DigestSection("Upcoming SF & Bay Area AI Events", "Meetups, workshops, and conferences worth attending", events),
    ],
    total_items_scanned=247,
    generation_time_s=38.4,
    opportunity_analysis=opportunity_text,
)

config = AggregatorConfig()
subject = digest_to_newsletter_subject(digest)
subtitle = digest_to_newsletter_subtitle(digest)
body_html = digest_to_newsletter_html(
    digest,
    intro_text=config.substack.intro_text,
    footer_text=config.substack.footer_text,
)


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _source_name(source: SourceName) -> str:
    names = {
        SourceName.ARXIV: "arXiv",
        SourceName.OPENAI_BLOG: "OpenAI Blog",
        SourceName.ANTHROPIC_BLOG: "Anthropic Blog",
        SourceName.DEEPMIND_BLOG: "DeepMind Blog",
        SourceName.META_AI_BLOG: "Meta AI Blog",
        SourceName.GOOGLE_AI_BLOG: "Google AI Blog",
        SourceName.HUGGINGFACE_BLOG: "Hugging Face Blog",
        SourceName.MS_RESEARCH_BLOG: "Microsoft Research",
        SourceName.ELEUTHERAI_BLOG: "EleutherAI Blog",
        SourceName.MISTRAL_BLOG: "Mistral AI Blog",
        SourceName.TOGETHER_BLOG: "Together AI Blog",
        SourceName.AI2_BLOG: "AI2 Blog",
        SourceName.COHERE_BLOG: "Cohere Blog",
        SourceName.HACKER_NEWS: "Hacker News",
        SourceName.REDDIT_ML: "r/MachineLearning",
        SourceName.LUMA: "Lu.ma",
        SourceName.EVENTBRITE: "Eventbrite",
    }
    return names.get(source, source.value)


def _build_body(dg: DailyDigest) -> str:
    parts = []
    for section in dg.sections:
        parts.append(f'<h2>{_esc(section.title)}</h2>')
        parts.append(f'<div class="section-desc">{_esc(section.description)}</div>')

        for i, item in enumerate(section.items, 1):
            is_event = isinstance(item, EventItem)
            parts.append('<div class="item">')
            parts.append(f'  <h3>{i}. <a href="{item.url}">{_esc(item.title)}</a></h3>')

            # Meta line
            meta_parts = []
            if item.authors:
                if len(item.authors) > 3:
                    meta_parts.append(f"<strong>{_esc(', '.join(item.authors[:3]))} +{len(item.authors)-3} more</strong>")
                else:
                    meta_parts.append(f"<strong>{_esc(', '.join(item.authors))}</strong>")
            if item.published_date:
                meta_parts.append(item.published_date.strftime("%b %d, %Y"))
            meta_parts.append(_source_name(item.source))
            if item.relevance_score and not is_event:
                score = item.relevance_score
                cls = "score-high" if score >= 80 else "score-mid" if score >= 60 else "score-low"
                meta_parts.append(f'<span class="{cls}">{score:.0f}/100</span>')
            if hasattr(item, 'metadata') and item.metadata:
                pts = item.metadata.get("points")
                cmt = item.metadata.get("num_comments")
                if pts:
                    meta_parts.append(f"{pts} points")
                if cmt:
                    meta_parts.append(f"{cmt} comments")
            parts.append(f'  <div class="item-meta">{" · ".join(meta_parts)}</div>')

            # Summary
            summary_text = item.summary or item.abstract
            if summary_text:
                parts.append(f'  <div class="summary">{_esc(summary_text)}</div>')

            # Why this matters
            if item.relevance_reason:
                parts.append(f'  <div class="why"><strong>Why this matters:</strong> {_esc(item.relevance_reason)}</div>')

            # Tags
            if item.tags:
                tags_html = " ".join(f'<span class="tag">{_esc(t)}</span>' for t in item.tags[:5])
                parts.append(f'  <div class="tags">{tags_html}</div>')

            # Event card
            if is_event:
                card_lines = []
                if item.event_date:
                    date_str = item.event_date.strftime("%A, %B %d, %Y at %I:%M %p")
                    card_lines.append(f'<div class="event-field"><strong>When:</strong> {date_str}</div>')
                if item.location:
                    card_lines.append(f'<div class="event-field"><strong>Where:</strong> {_esc(item.location)}</div>')
                if item.price:
                    card_lines.append(f'<div class="event-field"><strong>Price:</strong> {_esc(item.price)}</div>')
                if item.rsvp_count:
                    card_lines.append(f'<div class="event-field"><strong>RSVPs:</strong> {item.rsvp_count}</div>')
                parts.append(f'  <div class="event-card">{"".join(card_lines)}</div>')

            parts.append('</div>')

    return "\n".join(parts)


os.makedirs("digests", exist_ok=True)
html_path = os.path.join("digests", f"newsletter-{now.strftime('%Y-%m-%d')}.html")

with open(html_path, "w") as f:
    f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{subject}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    max-width: 680px;
    margin: 0 auto;
    padding: 40px 24px 60px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 16px;
    line-height: 1.65;
    color: #1a1a1a;
    background: #fff;
  }}
  header {{
    border-bottom: 2px solid #1a1a1a;
    padding-bottom: 20px;
    margin-bottom: 28px;
  }}
  header .brand {{
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 8px;
  }}
  header h1 {{
    font-size: 26px;
    font-weight: 700;
    line-height: 1.25;
    color: #1a1a1a;
    margin-bottom: 6px;
  }}
  header .subtitle {{
    font-size: 15px;
    color: #666;
    font-style: italic;
  }}
  header .meta {{
    font-size: 13px;
    color: #999;
    margin-top: 10px;
  }}
  .intro {{
    font-size: 15px;
    color: #555;
    margin-bottom: 24px;
    padding: 14px 16px;
    background: #f9fafb;
    border-radius: 6px;
    border-left: 3px solid #2563eb;
  }}
  h2 {{
    font-size: 20px;
    font-weight: 700;
    color: #1a1a1a;
    margin-top: 36px;
    margin-bottom: 4px;
    padding-bottom: 6px;
    border-bottom: 1px solid #e5e7eb;
  }}
  h2 + .section-desc {{
    font-size: 14px;
    color: #888;
    font-style: italic;
    margin-bottom: 16px;
  }}
  .item {{
    margin-bottom: 24px;
    padding-bottom: 20px;
    border-bottom: 1px solid #f0f0f0;
  }}
  .item:last-child {{
    border-bottom: none;
  }}
  .item h3 {{
    font-size: 17px;
    font-weight: 600;
    line-height: 1.35;
    margin-bottom: 2px;
  }}
  .item h3 a {{
    color: #1a1a1a;
    text-decoration: none;
  }}
  .item h3 a:hover {{
    color: #2563eb;
  }}
  .item-meta {{
    font-size: 13px;
    color: #888;
    margin-bottom: 6px;
  }}
  .item-meta .score-high {{ color: #16a34a; font-weight: 700; }}
  .item-meta .score-mid {{ color: #2563eb; font-weight: 700; }}
  .item-meta .score-low {{ color: #d97706; font-weight: 700; }}
  .item .summary {{
    font-size: 15px;
    color: #374151;
    margin: 6px 0;
    padding-left: 12px;
    border-left: 3px solid #e5e7eb;
  }}
  .item .why {{
    font-size: 13px;
    color: #6b7280;
    margin-top: 4px;
  }}
  .tags {{
    margin-top: 4px;
  }}
  .tag {{
    display: inline-block;
    font-size: 11px;
    font-weight: 500;
    color: #6b7280;
    background: #f3f4f6;
    padding: 1px 7px;
    border-radius: 3px;
    margin-right: 4px;
  }}
  .event-card {{
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 14px 16px;
    margin-top: 8px;
    font-size: 14px;
    color: #374151;
  }}
  .event-card strong {{
    color: #1a1a1a;
  }}
  .event-card .event-field {{
    margin-bottom: 2px;
  }}
  .opportunity {{
    background: #f0f7ff;
    border-left: 4px solid #2563eb;
    padding: 16px 20px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0 24px;
    font-size: 15px;
    color: #374151;
    line-height: 1.7;
  }}
  .opportunity p {{
    margin-bottom: 12px;
  }}
  .opportunity p:last-child {{
    margin-bottom: 0;
  }}
  footer {{
    margin-top: 40px;
    padding-top: 20px;
    border-top: 2px solid #1a1a1a;
  }}
  footer .cta {{
    font-size: 14px;
    color: #555;
    margin-bottom: 16px;
  }}
  footer .colophon {{
    font-size: 12px;
    color: #aaa;
    text-align: center;
  }}
</style>
</head>
<body>

<header>
  <div class="brand">Hidden Layer</div>
  <h1>{subject.replace('Hidden Layer Digest - ', '')}</h1>
  <div class="subtitle">{subtitle}</div>
  <div class="meta">Scanned <strong>247</strong> items across arXiv, AI lab blogs, community forums, and event platforms.</div>
</header>

<div class="intro">
  {config.substack.intro_text}
</div>

{_build_body(digest)}

<h2>Opportunity Spotlight</h2>
<div class="opportunity">
{"".join(f"<p>{_esc(p.strip())}</p>" for p in digest.opportunity_analysis.split(chr(10) + chr(10)) if p.strip())}
</div>

<footer>
  <div class="cta">{config.substack.footer_text}</div>
  <div class="colophon">Curated by Hidden Layer</div>
</footer>

</body>
</html>
""")

print(f"Newsletter saved to: {html_path}")
