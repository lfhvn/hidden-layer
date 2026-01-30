# Source health reporting

**Labels:** phase-1, reliability, P0

## Problem

Currently `fetch_safe` logs warnings but the CLI has no visibility into which sources succeeded, failed, or were slow. A silently failing source means the digest is missing content with no indication to the user.

## Acceptance Criteria

- [ ] New `SourceResult` dataclass: `source_name`, `items_count`, `error` (optional), `latency_s`, `cache_hit`
- [ ] `fetch_safe` returns `SourceResult` alongside the items list
- [ ] `generate_digest` collects all `SourceResult` objects
- [ ] `DailyDigest` has a `source_health` field with the collected results
- [ ] Terminal output shows a source health summary (e.g., "arXiv: 47 items (2.3s) | Blogs: 4 items (1.1s) | HN: failed (timeout)")
- [ ] Markdown output includes source health in the metadata section
- [ ] Newsletter HTML includes source count in the header metadata line

## Files to Modify

- `ai_research_aggregator/sources/base.py` — `SourceResult` dataclass, update `fetch_safe`
- `ai_research_aggregator/models.py` — add `source_health` to `DailyDigest`
- `ai_research_aggregator/digest.py` — collect and propagate health data, terminal display
- `ai_research_aggregator/newsletter.py` — show source counts in header
