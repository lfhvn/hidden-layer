# Publisher abstraction and draft-first workflow

**Labels:** phase-2, distribution, P2

## Problem

The Substack client uses unofficial API endpoints that could break at any time. Authentication is via session cookie with no expiry handling. There is no abstraction layer — switching to another platform would require rewriting the publishing flow. The current workflow auto-publishes, with no human review step.

## Acceptance Criteria

### Publisher interface
- [ ] Abstract `NewsletterPublisher` base class with methods: `create_draft()`, `publish()`, `validate_credentials()`
- [ ] `SubstackPublisher` implements the interface (refactored from current `substack.py`)
- [ ] Config specifies publisher type: `publisher: substack` (extensible to buttondown, ghost, etc.)

### Substack hardening
- [ ] Token validation check before attempting API calls (lightweight GET to test session)
- [ ] Token refresh: if expired, attempt re-login with saved credentials
- [ ] Clear error messages when authentication fails

### Draft-first workflow (default)
- [ ] `ai-digest newsletter publish` creates a draft by default (not auto-publish)
- [ ] Sends notification (email/stdout) with the Substack preview link
- [ ] Human reviews in Substack dashboard and clicks publish
- [ ] `--auto-publish` flag for fully automated workflow (explicit opt-in)
- [ ] `--notify` flag sends notification with preview link

### Email fallback
- [ ] If Substack API fails after retries, save HTML locally
- [ ] Optionally send the HTML via SMTP to a configured fallback email
- [ ] The newsletter must never silently fail to produce output

## Files to Create/Modify

- New: `ai_research_aggregator/publisher.py` — abstract interface
- `ai_research_aggregator/substack.py` — refactor to implement interface, add token validation
- `ai_research_aggregator/cli.py` — draft-first workflow, `--auto-publish`, `--notify`
- `ai_research_aggregator/config.py` — publisher type config, SMTP settings
- `ai_research_aggregator/default_config.yaml` — publisher and notification settings
