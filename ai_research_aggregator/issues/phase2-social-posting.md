# Social media cross-posting (Twitter/X, LinkedIn, Bluesky)

**Labels:** phase-2, distribution, P2

## Problem

Each digest contains high-value content summaries that would perform well on social media, but there is no automated posting. Manual cross-posting is tedious and inconsistent.

## Acceptance Criteria

### Twitter/X
- [ ] Post a thread with the top 3 items + opportunity spotlight when a digest is published
- [ ] Each tweet: item title, 1-sentence summary, link, relevance score
- [ ] Final tweet: opportunity spotlight excerpt + link to full digest
- [ ] Uses Twitter API v2 (free tier: 1,500 tweets/month)
- [ ] Thread is posted as a reply chain
- [ ] `ai-digest social twitter` command posts the latest digest

### LinkedIn
- [ ] Post a single update with the top 3 items formatted as a list
- [ ] Include link to the full Substack digest
- [ ] `ai-digest social linkedin` command posts the latest digest

### Bluesky
- [ ] Cross-post the Twitter thread content
- [ ] Uses the AT Protocol API (free, stable)
- [ ] `ai-digest social bluesky` command posts the latest digest

### Configuration
- [ ] Social media credentials stored in config (API keys, tokens)
- [ ] Per-platform enable/disable in `default_config.yaml`
- [ ] `--dry-run` flag that prints what would be posted without posting
- [ ] `ai-digest social post-all` posts to all enabled platforms

## Files to Create/Modify

- New: `ai_research_aggregator/social.py` — social media posting for all platforms
- `ai_research_aggregator/cli.py` — `social` subcommand
- `ai_research_aggregator/config.py` — social media credentials and settings
- `ai_research_aggregator/default_config.yaml` — platform configs
