# AI Research Aggregator

Daily AI research digest tool that scrapes papers, blogs, community posts, and events, then ranks them by relevance to your interests. Outputs to your terminal, saves to markdown, or publishes to Substack as the **Hidden Layer** newsletter.

## Quick Start

There are two ways to use the aggregator — as a **local digest** you read yourself, or as a **Substack newsletter** others can subscribe to. Both start with the same setup.

### 1. Install

From the `hidden-layer` repo root:

```bash
pip install -e .
```

This registers the `ai-digest` CLI command.

### 2. Create your config

```bash
ai-digest config init
```

This writes a default config to `~/.config/ai-research-aggregator/config.yaml`. Open it and customize your interests, key figures, and arXiv categories. The defaults are tuned for Hidden Layer's research areas (interpretability, alignment, multi-agent systems, theory of mind, etc.).

### 3. Run your first digest

```bash
ai-digest
```

That's it. This fetches from all sources, ranks by keyword relevance, and prints the digest to your terminal.

---

## Local Digest (No Newsletter Required)

The digest works entirely standalone — no Substack account or API keys needed.

### Terminal output (default)

```bash
# Full digest with LLM-powered ranking and summaries
ai-digest

# Fast mode: keyword ranking only, no API calls
ai-digest digest --no-llm

# Skip event scraping (faster if you just want papers/blogs)
ai-digest digest --skip-events

# Combine flags
ai-digest digest --no-llm --skip-events
```

### Save to markdown

```bash
# Print to terminal AND save to digests/digest-2026-01-30.md
ai-digest digest --save

# Fast mode + save
ai-digest digest --no-llm --save
```

Output goes to the `digests/` directory (configurable via `output.output_dir` in your config).

### With LLM ranking

When run without `--no-llm`, the aggregator sends content batches to an LLM (default: Claude via Anthropic API) for intelligent relevance scoring and one-line summaries. This requires an `ANTHROPIC_API_KEY` environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
ai-digest
```

You can switch to OpenAI or a local model by editing `llm.provider` and `llm.model` in your config:

```yaml
llm:
  provider: openai          # or "ollama" for local models
  model: gpt-4o             # or "llama3.2:latest" for Ollama
```

### Verify sources are reachable

```bash
# Test all sources
ai-digest sources test

# Test a specific source
ai-digest sources test --source arxiv
ai-digest sources test --source blogs
ai-digest sources test --source communities
ai-digest sources test --source events
```

---

## Substack Newsletter (Hidden Layer)

The aggregator can publish directly to **hidden-layer.substack.com** (or any Substack publication). This is the path for creating a subscribable newsletter.

### One-time setup

#### A. Create your Substack

1. Go to [substack.com](https://substack.com) and create a publication
2. Note your publication slug (the part before `.substack.com`)

#### B. Set the slug in your config

The default config already points to `hidden-layer`. If your publication slug is different, edit `~/.config/ai-research-aggregator/config.yaml`:

```yaml
substack:
  publication: "hidden-layer"   # your-slug-here
```

#### C. Authenticate

```bash
ai-digest newsletter login
```

This prompts for your Substack email and password, then saves a session token to `~/.config/ai-research-aggregator/substack_token` (permissions 600). You only need to do this once (or when the session expires).

### Publishing workflow

#### Preview locally first

```bash
ai-digest newsletter preview
```

This generates the full digest and saves two files:
- `digests/newsletter-2026-01-30.html` — full page preview (open in browser)
- `digests/newsletter-2026-01-30-body.html` — raw body HTML (for manual paste into Substack editor)

#### Create a draft on Substack

```bash
ai-digest newsletter draft
```

This generates the digest, formats it as HTML, and pushes it as a **draft** to your Substack. You'll get a URL to review and manually publish from the Substack dashboard.

#### Publish directly

```bash
# With confirmation prompt
ai-digest newsletter publish

# Skip confirmation
ai-digest newsletter publish --yes
```

This generates, formats, and publishes the post live. By default it also emails all subscribers (configurable via `substack.send_email`).

All newsletter commands accept `--no-llm` and `--skip-events`:

```bash
ai-digest newsletter draft --no-llm --skip-events
```

### Automation (daily cron)

To publish automatically every morning:

```bash
# crontab -e
0 7 * * * ANTHROPIC_API_KEY=sk-ant-... /path/to/ai-digest newsletter publish --yes >> /var/log/ai-digest.log 2>&1
```

Or for a draft-based workflow where you review before sending:

```bash
0 7 * * * ANTHROPIC_API_KEY=sk-ant-... /path/to/ai-digest newsletter draft >> /var/log/ai-digest.log 2>&1
```

---

## Configuration Reference

Config file: `~/.config/ai-research-aggregator/config.yaml`

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| **interests** | `topics` | 15 AI research topics | Used for relevance ranking |
| | `search_terms` | 10 terms | Extra arXiv search filters |
| | `key_figures` | 16 names | Boosts score when mentioned |
| | `arxiv_categories` | 7 categories | arXiv categories to monitor |
| **sources** | `max_papers` | 50 | Max arXiv papers to fetch |
| | `max_blog_posts` | 30 | Max blog posts to fetch |
| | `max_community_posts` | 30 | Max HN/Reddit posts |
| | `max_events` | 20 | Max SF events |
| | `papers_days_back` | 3 | How far back to search |
| | `subreddits` | ML, artificial, LocalLLaMA | Reddit subs to monitor |
| | `enable_arxiv` | true | Toggle arXiv source |
| | `enable_blogs` | true | Toggle blog sources |
| | `enable_communities` | true | Toggle HN/Reddit |
| | `enable_events` | true | Toggle SF events |
| **llm** | `provider` | anthropic | LLM provider for ranking |
| | `model` | claude-sonnet-4-20250514 | Model for ranking |
| | `temperature` | 0.3 | Sampling temperature |
| | `ranking_batch_size` | 15 | Items per LLM call |
| | `top_papers` | 10 | Papers in final digest |
| | `top_blog_posts` | 8 | Blog posts in final digest |
| | `top_community` | 8 | Community posts in final digest |
| | `top_events` | 10 | Events in final digest |
| **output** | `output_dir` | digests | Where files are saved |
| **substack** | `publication` | hidden-layer | Substack slug |
| | `email` | "" | Login email |
| | `auto_publish` | false | Auto-publish vs. draft |
| | `send_email` | true | Email subscribers on publish |
| | `intro_text` | "Welcome to..." | Top of every newsletter |
| | `footer_text` | "Thanks for reading..." | Bottom of every newsletter |

---

## Sources

### Papers
- **arXiv API** — categories: cs.AI, cs.CL, cs.LG, cs.CV, cs.MA, cs.NE, stat.ML
  - Searches by category + your configured search terms
  - Sorted by submission date (newest first)

### Blogs (RSS)
- **OpenAI Blog** — openai.com/blog/rss.xml
- **Anthropic Research** — anthropic.com/research/rss.xml
- **Google DeepMind** — deepmind.google/blog/rss.xml
- **Meta AI** — ai.meta.com/blog/rss/

Auto-detects mentions of tracked key figures in post titles and descriptions.

### Community
- **Hacker News** — Algolia search API, filtered by AI keywords, sorted by points
- **Reddit** — r/MachineLearning, r/artificial, r/LocalLLaMA (configurable), sorted by score

### Events
- **Lu.ma** — searches for AI events in San Francisco
- **Eventbrite** — searches SF for AI/ML events, parses JSON-LD structured data

---

## Ranking

Two ranking modes:

### LLM ranking (default)

Sends content in batches to an LLM with your interest profile. The LLM returns:
- **Relevance score** (0-100): how relevant the item is to your topics
- **Summary**: 1-2 sentence summary
- **Relevance reason**: why this matters to you

### Keyword ranking (`--no-llm`)

Scores items based on:
- Keyword overlap with your topics and search terms (up to 60 pts)
- Key figure mentions (up to 20 pts)
- Recency bonus (up to 10 pts)
- Community engagement — HN points, Reddit score (up to 10 pts)

No API calls required. Useful for fast iterations or when you don't have an API key set up.

---

## CLI Reference

```
ai-digest                                  Print today's digest to terminal
ai-digest digest --save                    Also save as markdown
ai-digest digest --no-llm                  Keyword ranking only (no API)
ai-digest digest --skip-events             Skip SF event scraping
ai-digest config init                      Create default config file
ai-digest config init --force              Overwrite existing config
ai-digest config show                      Print current config
ai-digest config path                      Show config file location
ai-digest sources test                     Test all content sources
ai-digest sources test --source arxiv      Test a specific source
ai-digest newsletter preview               Save HTML preview locally
ai-digest newsletter login                 Authenticate with Substack
ai-digest newsletter draft                 Push digest as Substack draft
ai-digest newsletter publish               Publish live to Substack
ai-digest newsletter publish --yes         Publish without confirmation

Global flags (work with any command):
  -c, --config PATH                        Custom config file path
  -v, --verbose                            Debug logging
```

---

## Architecture

```
ai_research_aggregator/
├── __init__.py              # Package metadata
├── cli.py                   # CLI entry point (ai-digest command)
├── config.py                # YAML configuration system
├── default_config.yaml      # Default config template
├── digest.py                # Orchestrator: fetch -> rank -> output
├── models.py                # Data models (ContentItem, EventItem, DailyDigest)
├── ranking.py               # LLM + keyword relevance ranking
├── newsletter.py            # DailyDigest -> Substack HTML formatter
├── substack.py              # Substack API client (auth, drafts, publish)
├── README.md                # This file
└── sources/
    ├── base.py              # BaseSource ABC
    ├── arxiv.py             # arXiv API
    ├── blogs.py             # RSS feeds (OpenAI, Anthropic, DeepMind, Meta AI)
    ├── communities.py       # Hacker News + Reddit
    └── events.py            # Lu.ma + Eventbrite
```

### Data flow

```
Sources (fetch)
    arXiv API  ─┐
    Blog RSS   ─┤
    HN/Reddit  ─┼──> List[ContentItem]
    Lu.ma      ─┤
    Eventbrite ─┘
                    │
                    v
Ranking Engine (score + summarize)
    LLM ranking ──or── Keyword ranking
                    │
                    v
DailyDigest (sections with ranked items)
                    │
         ┌──────────┼──────────┐
         v          v          v
    Terminal    Markdown    Substack
    output      file       newsletter
```

### Integration with harness

The LLM ranking uses the `harness.llm_call()` abstraction, so it works with any configured provider (Anthropic, OpenAI, Ollama, MLX). Switch providers by editing `llm.provider` and `llm.model` in your config.
