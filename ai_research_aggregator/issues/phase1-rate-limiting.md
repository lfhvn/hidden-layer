# Per-source rate limiting

**Labels:** phase-1, reliability, P0

## Problem

The HN source in `communities.py` loops over 8 search terms making an HTTP request each with no delay. The Luma source in `events.py` loops over 3 queries similarly. Reddit explicitly requires rate limiting. Hitting APIs too fast causes 429 responses and potential IP bans.

## Acceptance Criteria

- [ ] `BaseSource` has a configurable `request_delay_s` attribute (default 0.5s)
- [ ] Delay is applied between consecutive requests within a single source's `fetch()` call
- [ ] HN source respects delay between its 8 search term requests
- [ ] Reddit source respects delay between subreddit/search requests
- [ ] Luma source respects delay between query requests
- [ ] Delay is configurable per-source in `default_config.yaml`

## Files to Modify

- `ai_research_aggregator/sources/base.py` — add delay support
- `ai_research_aggregator/sources/communities.py` — apply delay in loops
- `ai_research_aggregator/sources/events.py` — apply delay in loops
- `ai_research_aggregator/default_config.yaml` — optional per-source delay config
