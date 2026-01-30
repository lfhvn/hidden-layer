# Retry with exponential backoff for all source fetchers

**Labels:** phase-1, reliability, P0

## Problem

Every source fetcher does a single `requests.get` with a timeout and nothing else. The `fetch_safe` method in `sources/base.py` catches all exceptions and returns `[]`, which means a transient 429 or 503 silently drops an entire source. arXiv returns 503 under load, Reddit returns 429 if hit too fast, and Luma/Eventbrite can be intermittently unavailable.

## Acceptance Criteria

- [ ] `BaseSource` has a `_request_with_retry(url, params, headers, max_retries=3, backoff_base=2)` method
- [ ] Retries on 429, 500, 502, 503, 504 status codes and on `ConnectionError`/`Timeout`
- [ ] Exponential backoff: 2s, 4s, 8s between retries
- [ ] Respects `Retry-After` header when present (arXiv sends this)
- [ ] All four source classes (`arxiv.py`, `blogs.py`, `communities.py`, `events.py`) use the new method
- [ ] Failed requests after all retries still return gracefully (empty list, logged warning)

## Files to Modify

- `ai_research_aggregator/sources/base.py` — add retry method
- `ai_research_aggregator/sources/arxiv.py` — use retry method
- `ai_research_aggregator/sources/blogs.py` — use retry method
- `ai_research_aggregator/sources/communities.py` — use retry method
- `ai_research_aggregator/sources/events.py` — use retry method
