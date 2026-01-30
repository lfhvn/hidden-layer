# Two-pass LLM summarization for richer editorial summaries

**Labels:** phase-2, content-quality, P1

## Problem

Currently there is one LLM pass that ranks items and generates 1-2 sentence summaries simultaneously. The summaries are generic and don't cross-reference other items in the digest. The result reads like a list of abstracts, not curated analysis.

## Acceptance Criteria

### Pass 1: Rank and score (existing, improved)
- [ ] Rank all items and assign relevance scores (current behavior)
- [ ] Generate brief 1-sentence summaries (for sorting purposes)

### Pass 2: Editorial summaries (new)
- [ ] For the top-N items that make the final digest, generate richer 3-4 sentence editorial summaries
- [ ] Second pass receives the item's abstract PLUS the list of other top items for cross-referencing
- [ ] Summaries explain **what it is**, **why it matters**, and **what it connects to** from other items in the digest
- [ ] Uses the editorial persona from the prompts system

### Section introductions (new)
- [ ] Before each section's items, generate a 2-3 sentence section intro
- [ ] Intro contextualizes the day's batch for that section (e.g., "Strong day for interpretability...")
- [ ] `DigestSection` model gets a new `editorial_intro` field
- [ ] Section intro renders in all three output formats

### Better subject lines
- [ ] Use LLM to generate a compelling newsletter subject line from the top 3 items
- [ ] Replace current approach (first item's title) with the generated headline

## Cost Impact

Adds ~3-5 additional LLM calls per digest (section intros + headline). Estimated increase: $0.10-0.20/run.

## Files to Modify

- `ai_research_aggregator/ranking.py` — two-pass logic, section intro generation
- `ai_research_aggregator/models.py` — `editorial_intro` on `DigestSection`
- `ai_research_aggregator/digest.py` — orchestrate second pass after ranking
- `ai_research_aggregator/newsletter.py` — render section intros, use LLM subject line
- `ai_research_aggregator/prompts/editorial_summary.txt` — new prompt template
- `ai_research_aggregator/prompts/section_intro.txt` — new prompt template
- `ai_research_aggregator/prompts/headline.txt` — new prompt template
