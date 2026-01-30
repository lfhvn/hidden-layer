# Scheduling, run logging, and failure notification

**Labels:** phase-1, operations, P1

## Problem

The README suggests a raw crontab entry. There is no error notification, no run history, and no way to know if the daily run failed.

## Acceptance Criteria

### Run logging
- [ ] Each run appends to `~/.config/ai-research-aggregator/run_log.jsonl`
- [ ] Log fields: `timestamp`, `status` (success/failure), `items_scanned`, `items_published`, `sources_status`, `llm_cost_estimate`, `errors`, `duration_s`
- [ ] `ai-digest history runs` shows the last 10 runs with status

### Failure handling
- [ ] `ai-digest run` wraps digest generation with: error capture, retry on failure (up to 2 retries with 5-min gaps)
- [ ] On final failure, log the error and optionally send a notification

### Notification
- [ ] `--notify email@example.com` sends success/failure email via SMTP
- [ ] SMTP settings configurable in `default_config.yaml`
- [ ] Notification includes: digest summary on success, error details on failure

### Schedule installation
- [ ] `ai-digest install-schedule --time 07:00` generates a systemd timer (Linux) or launchd plist (macOS)
- [ ] Generated config runs `ai-digest run` at the specified time daily
- [ ] Prints the generated config and install instructions

## Files to Create/Modify

- New: `ai_research_aggregator/scheduler.py` — run wrapper, notification, schedule generation
- `ai_research_aggregator/cli.py` — `run`, `install-schedule`, `history runs` commands
- `ai_research_aggregator/default_config.yaml` — notification/SMTP settings
