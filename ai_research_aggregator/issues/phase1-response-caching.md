# File-based response caching

**Labels:** phase-1, reliability, P0

## Problem

Every run fetches all sources from scratch. During development this hammers APIs unnecessarily and wastes time. In production, if the tool is re-run after a partial failure, it repeats all the work. There is no caching layer.

## Acceptance Criteria

- [ ] Cache directory at `~/.cache/ai-research-aggregator/`
- [ ] `BaseSource` has a `_cached_get(url, params, cache_ttl_s)` method
- [ ] Cache key is hash of (source_name + URL + sorted params)
- [ ] Cache TTLs: 1 hour for community/events, 4 hours for arXiv/blogs
- [ ] Expired cache entries are cleaned up on read
- [ ] `--no-cache` CLI flag bypasses the cache entirely
- [ ] Cache hit/miss is logged at DEBUG level
- [ ] Cache works correctly when params change (different search terms produce different cache entries)

## Files to Modify

- `ai_research_aggregator/sources/base.py` — add caching method
- `ai_research_aggregator/cli.py` — add `--no-cache` flag
- All four source files — use `_cached_get` instead of raw `requests.get`
