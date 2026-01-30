# Parallel source fetching with ThreadPoolExecutor

**Labels:** phase-1, performance, P0

## Problem

All sources are fetched sequentially in `digest.py`. The four source categories (arXiv, blogs, community, events) are independent and could run concurrently. Current fetch time is ~30s; parallel execution would cut this roughly in half.

## Acceptance Criteria

- [ ] Use `concurrent.futures.ThreadPoolExecutor` to fetch all four source categories in parallel
- [ ] Each source category still executes its internal requests sequentially (respecting rate limits)
- [ ] Errors in one source don't affect others (already true, but verify under parallelism)
- [ ] Source health reporting works correctly with parallel execution
- [ ] `--sequential` flag to disable parallelism for debugging
- [ ] Total fetch time is logged and shown to be ~50% faster

## Files to Modify

- `ai_research_aggregator/digest.py` — wrap source fetching in ThreadPoolExecutor
- `ai_research_aggregator/cli.py` — add `--sequential` flag
