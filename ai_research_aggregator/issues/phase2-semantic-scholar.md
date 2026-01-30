# Semantic Scholar source integration

**Labels:** phase-2, sources, P2

## Problem

arXiv provides papers but no citation data or influence signals. A paper cited 50 times in a week is far more important than one with zero citations, but the current system can't distinguish them. Also, tracking papers by specific authors (the `key_figures` list) is only done via keyword matching.

## Acceptance Criteria

- [ ] New source class `SemanticScholarSource` in `sources/semantic_scholar.py`
- [ ] Uses the free Semantic Scholar API (`api.semanticscholar.org/graph/v1/paper/search`)
- [ ] Fetches recent papers matching user interests and search terms
- [ ] Retrieves citation counts, influence scores, and related papers for each item
- [ ] Tracks papers by authors in the `key_figures` config list
- [ ] Citation count and influence score are stored in `ContentItem.metadata`
- [ ] Ranking engine uses citation/influence signals as bonus scoring factors
- [ ] New `SourceName.SEMANTIC_SCHOLAR` enum entry
- [ ] Rate limiting: Semantic Scholar allows 100 requests/5 minutes on free tier
- [ ] Deduplication with arXiv source (same paper appears on both)

## API Details

- Search endpoint: `GET /graph/v1/paper/search?query={query}&fields=title,abstract,authors,citationCount,influentialCitationCount,publicationDate,externalIds`
- Author endpoint: `GET /graph/v1/author/{authorId}/papers?fields=title,abstract,citationCount,publicationDate`
- Free tier: 100 requests per 5 minutes, no API key needed
- Optional API key for higher rate limits

## Files to Create/Modify

- New: `ai_research_aggregator/sources/semantic_scholar.py`
- `ai_research_aggregator/models.py` — add `SourceName.SEMANTIC_SCHOLAR`
- `ai_research_aggregator/ranking.py` — use citation signals in scoring
- `ai_research_aggregator/digest.py` — integrate new source
- `ai_research_aggregator/config.py` — Semantic Scholar settings
- `ai_research_aggregator/default_config.yaml` — enable/disable and config
