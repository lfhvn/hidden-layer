# LLM ranking robustness and cost tracking

**Labels:** phase-1, reliability, P0

## Problem

The ranking in `ranking.py` parses JSON from LLM output. If the LLM returns malformed JSON, the entire batch gets zero scores. The fallback to keyword scoring produces much worse results. There is no cost tracking. Missing API keys cause mid-run failures instead of graceful degradation.

## Acceptance Criteria

### Robust parsing
- [ ] Primary: use structured output / tool use mode if the provider supports it
- [ ] Fallback 1: regex extraction of individual `{...}` objects from response
- [ ] Fallback 2: extract score numbers paired with indices
- [ ] Fallback 3: keyword scoring (current behavior)

### Partial failure recovery
- [ ] Failed batch is retried once with a shorter prompt (titles only, no abstracts)
- [ ] Only falls back to keyword scoring after retry also fails

### Cost tracking
- [ ] Log token usage (input + output) from each `llm_call` response
- [ ] Track cumulative cost across all LLM calls in a digest run
- [ ] `--cost-report` flag prints estimated API cost at the end
- [ ] Cost estimate uses current Anthropic/OpenAI pricing (configurable)

### Graceful degradation
- [ ] Detect missing API key before starting LLM ranking
- [ ] Print clear warning: "No API key found for {provider}. Using keyword ranking."
- [ ] Don't attempt LLM calls if key is missing (avoid mid-batch failures)
- [ ] `--no-llm` flag explicitly forces keyword mode

## Files to Modify

- `ai_research_aggregator/ranking.py` — structured output, retry, cost tracking, parsing
- `ai_research_aggregator/digest.py` — detect API key early, propagate cost data
- `ai_research_aggregator/cli.py` — `--cost-report` and `--no-llm` flags
- `ai_research_aggregator/models.py` — add `llm_cost_estimate` to `DailyDigest`
