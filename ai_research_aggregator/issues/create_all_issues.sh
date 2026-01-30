#!/usr/bin/env bash
# Create all Phase 1 and Phase 2 GitHub issues for Hidden Layer AI Research Aggregator.
# Prerequisites: gh CLI installed and authenticated (gh auth login)
#
# Usage: ./create_all_issues.sh
#
set -euo pipefail

if ! command -v gh &> /dev/null; then
    echo "Error: gh CLI not found. Install it:"
    echo "  macOS:  brew install gh"
    echo "  Linux:  sudo apt install gh"
    exit 1
fi

if ! gh auth status &> /dev/null; then
    echo "Error: gh not authenticated. Run: gh auth login"
    exit 1
fi

echo "Creating Phase 1 and Phase 2 issues..."
echo ""

# --- Phase 1: Foundation ---

gh issue create \
  --title "P0: Retry with exponential backoff for all source fetchers" \
  --label "phase-1,reliability,P0" \
  --body "$(cat <<'EOF'
## Problem

Every source fetcher does a single `requests.get` with a timeout and nothing else. The `fetch_safe` method in `sources/base.py` catches all exceptions and returns `[]`, which means a transient 429 or 503 silently drops an entire source.

## Acceptance Criteria

- [ ] `BaseSource` has a `_request_with_retry(url, params, headers, max_retries=3, backoff_base=2)` method
- [ ] Retries on 429, 500, 502, 503, 504 status codes and on `ConnectionError`/`Timeout`
- [ ] Exponential backoff: 2s, 4s, 8s between retries
- [ ] Respects `Retry-After` header when present (arXiv sends this)
- [ ] All four source classes use the new method
- [ ] Failed requests after all retries still return gracefully (empty list, logged warning)

## Files

`sources/base.py`, `sources/arxiv.py`, `sources/blogs.py`, `sources/communities.py`, `sources/events.py`
EOF
)"
echo "✓ Created: Retry with exponential backoff"

gh issue create \
  --title "P0: Per-source rate limiting" \
  --label "phase-1,reliability,P0" \
  --body "$(cat <<'EOF'
## Problem

The HN source loops over 8 search terms with no delay between requests. Reddit and Luma have similar patterns. This causes 429 responses and potential IP bans.

## Acceptance Criteria

- [ ] `BaseSource` has a configurable `request_delay_s` attribute (default 0.5s)
- [ ] Delay is applied between consecutive requests within a source's `fetch()` call
- [ ] HN, Reddit, and Luma sources respect the delay
- [ ] Delay is configurable per-source in `default_config.yaml`

## Files

`sources/base.py`, `sources/communities.py`, `sources/events.py`, `default_config.yaml`
EOF
)"
echo "✓ Created: Per-source rate limiting"

gh issue create \
  --title "P0: File-based response caching" \
  --label "phase-1,reliability,P0" \
  --body "$(cat <<'EOF'
## Problem

Every run fetches all sources from scratch. No caching layer exists.

## Acceptance Criteria

- [ ] Cache directory at `~/.cache/ai-research-aggregator/`
- [ ] `BaseSource._cached_get(url, params, cache_ttl_s)` method
- [ ] Cache key = hash of (source_name + URL + sorted params)
- [ ] TTLs: 1hr for community/events, 4hr for arXiv/blogs
- [ ] `--no-cache` CLI flag bypasses cache
- [ ] Cache hit/miss logged at DEBUG level

## Files

`sources/base.py`, `cli.py`, all source files
EOF
)"
echo "✓ Created: File-based response caching"

gh issue create \
  --title "P0: Source health reporting" \
  --label "phase-1,reliability,P0" \
  --body "$(cat <<'EOF'
## Problem

`fetch_safe` logs warnings but the CLI has no visibility into which sources succeeded, failed, or were slow.

## Acceptance Criteria

- [ ] `SourceResult` dataclass: `source_name`, `items_count`, `error`, `latency_s`, `cache_hit`
- [ ] `generate_digest` collects all `SourceResult` objects
- [ ] `DailyDigest` has a `source_health` field
- [ ] Terminal output shows source health summary
- [ ] Newsletter HTML includes source count in header

## Files

`sources/base.py`, `models.py`, `digest.py`, `newsletter.py`
EOF
)"
echo "✓ Created: Source health reporting"

gh issue create \
  --title "P0: Content deduplication (within-run and cross-day)" \
  --label "phase-1,reliability,P0" \
  --body "$(cat <<'EOF'
## Problem

No deduplication. The same paper can appear from arXiv, HN, Reddit, and a blog post. Running twice produces identical output.

## Acceptance Criteria

### Within-run
- [ ] Deduplicate by normalized URL (strip trailing slashes, query params, utm tags)
- [ ] Deduplicate by title Jaccard similarity (threshold 0.8)
- [ ] Merge: prefer item with most metadata
- [ ] Normalize arXiv IDs (strip version suffix)

### Cross-day
- [ ] SQLite at `~/.config/ai-research-aggregator/history.db`
- [ ] `seen_items` table with URL hash, title, source, dates, published flag
- [ ] Filter items published in last 7 days
- [ ] `ai-digest history clear` and `history stats` commands

## Files

New: `storage.py` | Modify: `digest.py`, `sources/arxiv.py`, `cli.py`
EOF
)"
echo "✓ Created: Content deduplication"

gh issue create \
  --title "P0: LLM ranking robustness and cost tracking" \
  --label "phase-1,reliability,P0" \
  --body "$(cat <<'EOF'
## Problem

LLM JSON parsing failures drop entire batches to keyword scoring. No cost tracking. Missing API keys cause mid-run failures.

## Acceptance Criteria

- [ ] Structured output mode if provider supports it
- [ ] Fallback parsing: regex individual objects → extract score/index pairs → keyword scoring
- [ ] Failed batch retried once with shorter prompt (titles only)
- [ ] Token usage logged, cumulative cost tracked, `--cost-report` flag
- [ ] Detect missing API key upfront, warn, fall back gracefully
- [ ] `--no-llm` flag forces keyword mode

## Files

`ranking.py`, `digest.py`, `cli.py`, `models.py`
EOF
)"
echo "✓ Created: LLM ranking robustness"

gh issue create \
  --title "P0: Add more blog RSS sources" \
  --label "phase-1,sources,P0" \
  --body "$(cat <<'EOF'
## Problem

Only 4 blog sources. Missing: Hugging Face, Google AI, Microsoft Research, EleutherAI, Mistral, Together AI, AI2, Cohere.

## Acceptance Criteria

- [ ] Add RSS feeds for all listed blogs
- [ ] Each has a `SourceName` enum entry
- [ ] All tested with `ai-digest sources test`
- [ ] Enabled by default, individually disableable

## Files

`models.py`, `sources/blogs.py`, `default_config.yaml`
EOF
)"
echo "✓ Created: Add more blog RSS sources"

gh issue create \
  --title "P0: Parallel source fetching with ThreadPoolExecutor" \
  --label "phase-1,performance,P0" \
  --body "$(cat <<'EOF'
## Problem

All sources fetched sequentially (~30s). The four source categories are independent.

## Acceptance Criteria

- [ ] `ThreadPoolExecutor` fetches all source categories in parallel
- [ ] Internal requests within a source remain sequential (respects rate limits)
- [ ] Source health reporting works correctly under parallelism
- [ ] `--sequential` flag for debugging
- [ ] Total fetch time reduced ~50%

## Files

`digest.py`, `cli.py`
EOF
)"
echo "✓ Created: Parallel source fetching"

gh issue create \
  --title "P1: Scheduling, run logging, and failure notification" \
  --label "phase-1,operations,P1" \
  --body "$(cat <<'EOF'
## Problem

No error notification, no run history, no way to know if the daily run failed.

## Acceptance Criteria

- [ ] Run log at `~/.config/ai-research-aggregator/run_log.jsonl` with timestamp, status, items, errors, duration
- [ ] `ai-digest history runs` shows last 10 runs
- [ ] `ai-digest run` wraps generation with retry (2 retries, 5-min gaps)
- [ ] `--notify email@example.com` sends success/failure email via SMTP
- [ ] `ai-digest install-schedule --time 07:00` generates systemd timer (Linux) or launchd plist (macOS)

## Files

New: `scheduler.py` | Modify: `cli.py`, `default_config.yaml`
EOF
)"
echo "✓ Created: Scheduling and notifications"

# --- Phase 2: Differentiation ---

gh issue create \
  --title "P1: Editorial voice and system prompt persona" \
  --label "phase-2,content-quality,P1" \
  --body "$(cat <<'EOF'
## Problem

LLM output reads like a database listing. No consistent editorial voice.

## Acceptance Criteria

- [ ] `prompts/` directory with versioned `.txt` prompt templates
- [ ] Hidden Layer editorial persona: direct, technically precise, no hype
- [ ] System prompt prepended to all LLM calls
- [ ] Prompts loaded from files at runtime (not hardcoded strings)
- [ ] Config option to customize persona

## Files

New: `prompts/persona.txt`, `prompts/ranking.txt`, `prompts/opportunity.txt` | Modify: `ranking.py`
EOF
)"
echo "✓ Created: Editorial voice"

gh issue create \
  --title "P1: Two-pass LLM summarization with section intros" \
  --label "phase-2,content-quality,P1" \
  --body "$(cat <<'EOF'
## Problem

Single LLM pass produces generic 1-2 sentence summaries. No cross-referencing. No section context.

## Acceptance Criteria

- [ ] Pass 1: rank and score (existing)
- [ ] Pass 2: for top-N items, generate 3-4 sentence editorial summaries with cross-references
- [ ] Section intros: 2-3 sentence contextual introduction per section
- [ ] `DigestSection.editorial_intro` field, rendered in all output formats
- [ ] LLM-generated subject lines from top 3 items
- [ ] New prompt templates: `editorial_summary.txt`, `section_intro.txt`, `headline.txt`

## Cost Impact

+$0.10-0.20/run (3-5 additional LLM calls)

## Files

`ranking.py`, `models.py`, `digest.py`, `newsletter.py`, new prompt templates
EOF
)"
echo "✓ Created: Two-pass summarization"

gh issue create \
  --title "P1: RSS feed output and static HTML archive" \
  --label "phase-2,distribution,P1" \
  --body "$(cat <<'EOF'
## Problem

No RSS feed. No web archive. No SEO surface area.

## Acceptance Criteria

- [ ] Atom feed from last 30 digests (`digests/feed.xml`), validates against spec
- [ ] Static HTML archive: index page + individual digest pages
- [ ] SEO tags: `<title>`, `<meta description>`, Open Graph, JSON-LD
- [ ] Deployable to GitHub Pages / Cloudflare Pages
- [ ] `ai-digest archive build` and `archive serve` commands
- [ ] Digest generation auto-updates the feed

## Files

New: `feeds.py`, `archive.py` | Modify: `cli.py`, `digest.py`
EOF
)"
echo "✓ Created: RSS feed and archive"

gh issue create \
  --title "P2: Semantic Scholar source integration" \
  --label "phase-2,sources,P2" \
  --body "$(cat <<'EOF'
## Problem

No citation data or influence signals. Can't distinguish a heavily-cited paper from one with zero citations.

## Acceptance Criteria

- [ ] `SemanticScholarSource` using free API (100 req/5 min)
- [ ] Fetches papers matching interests + key_figures author tracking
- [ ] Citation counts and influence scores in `ContentItem.metadata`
- [ ] Ranking engine uses citation/influence as bonus factors
- [ ] Deduplication with arXiv source

## API

- `GET /graph/v1/paper/search?query={q}&fields=title,abstract,authors,citationCount,influentialCitationCount,publicationDate,externalIds`
- `GET /graph/v1/author/{id}/papers`

## Files

New: `sources/semantic_scholar.py` | Modify: `models.py`, `ranking.py`, `digest.py`, `config.py`, `default_config.yaml`
EOF
)"
echo "✓ Created: Semantic Scholar"

gh issue create \
  --title "P2: GitHub trending AI/ML repositories source" \
  --label "phase-2,sources,P2" \
  --body "$(cat <<'EOF'
## Problem

Open-source tool/model releases only surface if posted to HN/Reddit.

## Acceptance Criteria

- [ ] `GitHubTrendingSource` using GitHub API search
- [ ] Filtered by ML/AI topics and languages (Python)
- [ ] Maps repos to `ContentItem`: title=name, abstract=description, metadata=stars/forks
- [ ] Configurable topic filters
- [ ] Respects GitHub API rate limits

## Files

New: `sources/github_trending.py` | Modify: `models.py`, `digest.py`, `config.py`, `default_config.yaml`
EOF
)"
echo "✓ Created: GitHub trending"

gh issue create \
  --title "P2: Publisher abstraction and draft-first workflow" \
  --label "phase-2,distribution,P2" \
  --body "$(cat <<'EOF'
## Problem

Substack client uses unofficial API. No token refresh. No abstraction for alternative platforms. Auto-publishes with no review step.

## Acceptance Criteria

- [ ] Abstract `NewsletterPublisher` interface: `create_draft()`, `publish()`, `validate_credentials()`
- [ ] `SubstackPublisher` implements interface (refactored from `substack.py`)
- [ ] Token validation and refresh
- [ ] Draft-first by default, `--auto-publish` for fully automated
- [ ] `--notify` sends preview link
- [ ] Email fallback via SMTP if Substack fails
- [ ] Config: `publisher: substack` (extensible)

## Files

New: `publisher.py` | Modify: `substack.py`, `cli.py`, `config.py`, `default_config.yaml`
EOF
)"
echo "✓ Created: Publisher abstraction"

gh issue create \
  --title "P2: Social media cross-posting (Twitter/X, LinkedIn, Bluesky)" \
  --label "phase-2,distribution,P2" \
  --body "$(cat <<'EOF'
## Problem

No automated social media distribution for digest content.

## Acceptance Criteria

- [ ] Twitter/X: thread of top 3 items + opportunity spotlight (API v2)
- [ ] LinkedIn: single post with top 3 items + digest link
- [ ] Bluesky: cross-post thread (AT Protocol)
- [ ] Credentials in config, per-platform enable/disable
- [ ] `--dry-run` flag, `ai-digest social post-all` command

## Files

New: `social.py` | Modify: `cli.py`, `config.py`, `default_config.yaml`
EOF
)"
echo "✓ Created: Social media cross-posting"

gh issue create \
  --title "P3: Engagement-based ranking feedback and interest evolution" \
  --label "phase-2,personalization,P3" \
  --body "$(cat <<'EOF'
## Problem

Static interests config. No feedback loop from reader engagement.

## Acceptance Criteria

- [ ] Retrieve Substack engagement data (opens, clicks) post-publish
- [ ] Store in SQLite: `engagement(digest_date, item_url_hash, clicked, open_rate)`
- [ ] Items similar to clicked items get +10 bonus (decays over 30 days)
- [ ] `--no-feedback` flag disables engagement weighting
- [ ] `ai-digest config suggest` analyzes 30 days and suggests new topics
- [ ] `ai-digest analytics` shows engagement trends

## Files

New: `analytics.py` | Modify: `storage.py`, `ranking.py`, `substack.py`, `cli.py`
EOF
)"
echo "✓ Created: Engagement feedback"

echo ""
echo "Done! All 17 issues created."
echo "View them at: gh issue list"
