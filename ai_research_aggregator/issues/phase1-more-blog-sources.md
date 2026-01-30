# Add more blog RSS sources

**Labels:** phase-1, sources, P0

## Problem

Only four blog sources are configured (OpenAI, Anthropic, DeepMind, Meta AI). Many important AI labs and organizations publish research blogs that are missed.

## Acceptance Criteria

- [ ] Add RSS feeds for:
  - Hugging Face blog
  - Google AI blog (separate from DeepMind)
  - Microsoft Research blog
  - EleutherAI blog
  - Mistral blog
  - Together AI blog
  - AI2 (Allen Institute for AI) blog
  - Cohere blog
- [ ] Each feed has a corresponding `SourceName` enum entry
- [ ] All new feeds are tested with `ai-digest sources test`
- [ ] New feeds are enabled by default in `default_config.yaml`
- [ ] Config supports disabling individual feeds

## Files to Modify

- `ai_research_aggregator/models.py` — add `SourceName` enum entries
- `ai_research_aggregator/sources/blogs.py` — add feeds to `BLOG_FEEDS`
- `ai_research_aggregator/default_config.yaml` — add enable/disable toggles
