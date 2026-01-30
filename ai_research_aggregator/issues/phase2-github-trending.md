# GitHub trending AI/ML repositories source

**Labels:** phase-2, sources, P2

## Problem

Open-source tool and model releases are a major category of AI news. Currently they only surface if someone posts them to HN or Reddit. Directly tracking GitHub trending repos would catch releases earlier and more reliably.

## Acceptance Criteria

- [ ] New source class `GitHubTrendingSource` in `sources/github_trending.py`
- [ ] Fetches trending repositories filtered to AI/ML topics
- [ ] Uses GitHub API (`/search/repositories?q=...&sort=stars&order=desc`) with date filtering for recently created/updated
- [ ] Filters by language (Python primarily) and topic tags (machine-learning, deep-learning, llm, nlp, etc.)
- [ ] Maps repos to `ContentItem` with: title = repo name, abstract = repo description, authors = owner, URL = repo URL, metadata includes star count and fork count
- [ ] New `SourceName.GITHUB_TRENDING` enum entry
- [ ] Configurable topic filters in `default_config.yaml`
- [ ] Respects GitHub API rate limits (60 requests/hour unauthenticated, 5000 with token)

## Files to Create/Modify

- New: `ai_research_aggregator/sources/github_trending.py`
- `ai_research_aggregator/models.py` — add `SourceName.GITHUB_TRENDING`
- `ai_research_aggregator/digest.py` — integrate new source, new section
- `ai_research_aggregator/config.py` — GitHub settings (optional token)
- `ai_research_aggregator/default_config.yaml` — enable/disable and config
