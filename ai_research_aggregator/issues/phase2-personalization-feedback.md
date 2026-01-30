# Engagement-based ranking feedback and interest evolution

**Labels:** phase-2, personalization, P3

## Problem

The config has a good interests structure (topics, search terms, key figures) but it is static. There is no feedback loop from reader engagement. The system can't learn what the reader actually finds valuable vs. what the keyword/LLM scorer predicts.

## Acceptance Criteria

### Engagement tracking
- [ ] After publishing, retrieve Substack engagement data (open rate, click-through per link) via the Substack API
- [ ] Store engagement data in the SQLite history database: `engagement(digest_date DATE, item_url_hash TEXT, clicked BOOLEAN, open_rate FLOAT)`
- [ ] `ai-digest analytics` command shows engagement trends

### Feedback-weighted ranking
- [ ] Items similar to previously clicked items get a configurable bonus (default +10 points)
- [ ] Similarity measured by: same source, same authors, tag overlap, title keyword overlap
- [ ] Feedback weight decays over time (30 days half-life)
- [ ] `--no-feedback` flag disables engagement weighting

### Interest evolution
- [ ] Track which topics appear in top-ranked items over time
- [ ] `ai-digest config suggest` analyzes the last 30 days and suggests new topics/terms
- [ ] Example: "test-time compute appeared in 8 top-ranked items but isn't in your interests. Add it?"

## Files to Create/Modify

- New: `ai_research_aggregator/analytics.py` — engagement tracking and analysis
- `ai_research_aggregator/storage.py` — engagement tables in SQLite
- `ai_research_aggregator/ranking.py` — feedback-weighted scoring
- `ai_research_aggregator/substack.py` — retrieve engagement data
- `ai_research_aggregator/cli.py` — `analytics` and `config suggest` commands
