# RSS feed output and static HTML archive

**Labels:** phase-2, distribution, P1

## Problem

The only distribution channels are terminal output, local markdown files, and Substack. There is no RSS feed for people who prefer feed readers, no web archive for SEO, and no way for search engines to discover the content.

## Acceptance Criteria

### RSS/Atom feed
- [ ] Generate an Atom feed from the last 30 digests
- [ ] Feed includes: title, summary, link to HTML archive page, publication date, author
- [ ] Each digest entry includes the full HTML content as `<content>` element
- [ ] Feed is saved alongside the digest output (e.g., `digests/feed.xml`)
- [ ] Feed validates against the Atom specification

### Static HTML archive
- [ ] Generate a browsable HTML archive from all digests in the output directory
- [ ] Index page lists all digests by date with title and summary
- [ ] Each digest page is the newsletter HTML with proper SEO tags
- [ ] Pages include: `<title>`, `<meta description>`, Open Graph tags, `Article` JSON-LD
- [ ] Archive can be deployed to GitHub Pages or Cloudflare Pages (static files only)
- [ ] `ai-digest archive build` command generates the full archive
- [ ] `ai-digest archive serve` starts a local preview server

### Integration
- [ ] `ai-digest digest` automatically updates the RSS feed after generating a new digest
- [ ] Archive build can be triggered as a post-publish hook

## Files to Create/Modify

- New: `ai_research_aggregator/feeds.py` — Atom feed generation
- New: `ai_research_aggregator/archive.py` — static HTML archive generation
- `ai_research_aggregator/cli.py` — `archive` subcommand
- `ai_research_aggregator/digest.py` — trigger feed update after digest generation
