# Content deduplication (within-run and cross-day)

**Labels:** phase-1, reliability, P0

## Problem

There is no deduplication. The same arXiv paper can appear on HN, Reddit, and in a blog post. Running the tool twice produces identical output. HN source has `seen_ids` within a single run but nothing across sources or days.

## Acceptance Criteria

### Within-run dedup
- [ ] After all sources return, deduplicate by normalized URL (strip trailing slashes, query params, utm tags)
- [ ] Deduplicate by title similarity using Jaccard similarity on lowercased word sets (threshold 0.8)
- [ ] When merging duplicates, prefer the item with the most metadata (abstract, authors, summary)
- [ ] arXiv IDs are normalized (strip version suffix like `v1`, `v2`)

### Cross-day dedup
- [ ] SQLite database at `~/.config/ai-research-aggregator/history.db`
- [ ] Table: `seen_items(url_hash TEXT PRIMARY KEY, title TEXT, source TEXT, first_seen DATE, last_seen DATE, published_in_digest BOOLEAN)`
- [ ] Before assembling digest, filter items published in the last 7 days
- [ ] After publishing, mark included items as published
- [ ] `ai-digest history clear` command to reset the database
- [ ] `ai-digest history stats` command to show database size and recent activity

## Files to Create/Modify

- New: `ai_research_aggregator/storage.py` — SQLite history management
- `ai_research_aggregator/digest.py` — dedup logic before ranking
- `ai_research_aggregator/sources/arxiv.py` — normalize arXiv IDs
- `ai_research_aggregator/cli.py` — history subcommand
