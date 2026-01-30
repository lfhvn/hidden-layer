# Editorial voice and system prompt persona

**Labels:** phase-2, content-quality, P1

## Problem

The LLM ranking prompt asks for generic relevance scoring and 1-2 sentence summaries. The output reads like a database listing, not a newsletter. There is no consistent editorial voice or personality across the digest.

## Acceptance Criteria

- [ ] Create `prompts/` directory with versioned prompt templates as `.txt` files
- [ ] Define a Hidden Layer editorial persona: direct, technically precise, no hype, connects new work to the bigger picture
- [ ] System prompt is prepended to all LLM calls (ranking, summarization, opportunity analysis)
- [ ] Ranking prompt uses the persona to produce summaries that sound like editorial commentary, not abstracts
- [ ] Opportunity analysis prompt uses the persona for consistent tone
- [ ] Prompts are loaded from files at runtime (not hardcoded strings)
- [ ] Config option to customize or override the persona

## Example Persona

> You are the editor of Hidden Layer, a research newsletter for AI practitioners who care about what is actually happening beneath the surface — interpretability, alignment, and the mechanics of how models think. Your tone is direct, technically precise, and occasionally wry. You avoid hype. You connect new work to the bigger picture.

## Files to Create/Modify

- New: `ai_research_aggregator/prompts/` directory
- New: `ai_research_aggregator/prompts/persona.txt`
- New: `ai_research_aggregator/prompts/ranking.txt`
- New: `ai_research_aggregator/prompts/opportunity.txt`
- `ai_research_aggregator/ranking.py` — load prompts from files, use persona
