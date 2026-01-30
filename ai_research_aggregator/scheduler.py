"""
Run wrapper, logging, notification, and schedule generation.

Provides:
- Run logging to JSONL file
- Retry wrapper for digest generation
- Email notification on success/failure
- systemd timer / launchd plist generation
"""

import json
import logging
import os
import platform
import smtplib
import sys
import time
import traceback
from datetime import datetime
from email.mime.text import MIMEText
from typing import List, Optional

from ai_research_aggregator.config import AggregatorConfig
from ai_research_aggregator.digest import generate_digest, save_digest
from ai_research_aggregator.models import DailyDigest

logger = logging.getLogger(__name__)

LOG_DIR = os.path.join(
    os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
    "ai-research-aggregator",
)
RUN_LOG_PATH = os.path.join(LOG_DIR, "run_log.jsonl")


# --- Run logging ---

def _log_run(entry: dict):
    """Append a run log entry to the JSONL file."""
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(RUN_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_recent_runs(count: int = 10) -> List[dict]:
    """Read the last N run log entries."""
    if not os.path.exists(RUN_LOG_PATH):
        return []
    entries = []
    with open(RUN_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries[-count:]


# --- Run wrapper ---

def run_digest(
    config: Optional[AggregatorConfig] = None,
    use_llm: bool = True,
    skip_events: bool = False,
    save: bool = True,
    max_retries: int = 2,
    retry_delay_s: int = 300,
    notify_email: Optional[str] = None,
) -> Optional[DailyDigest]:
    """
    Run digest generation with retry, logging, and notification.

    Args:
        config: Aggregator config.
        use_llm: Whether to use LLM ranking.
        skip_events: Skip event fetching.
        save: Save digest to markdown file.
        max_retries: Number of retries on failure (0 = no retries).
        retry_delay_s: Seconds between retries.
        notify_email: Email address for notifications.

    Returns:
        The generated digest, or None if all attempts failed.
    """
    if config is None:
        config = AggregatorConfig()

    last_error = None
    attempt = 0

    while attempt <= max_retries:
        attempt += 1
        start_time = time.time()

        try:
            print(f"Digest run attempt {attempt}/{max_retries + 1}...")
            digest = generate_digest(
                config=config,
                use_llm=use_llm,
                skip_events=skip_events,
            )

            duration = time.time() - start_time

            # Save if requested
            filepath = None
            if save:
                filepath = save_digest(digest, config)

            # Log success
            sources_status = {}
            for sh in digest.source_health:
                sources_status[sh.source_name] = {
                    "items": sh.items_count,
                    "error": sh.error,
                    "latency_s": sh.latency_s,
                }

            items_published = sum(len(s.items) for s in digest.sections)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "attempt": attempt,
                "items_scanned": digest.total_items_scanned,
                "items_published": items_published,
                "sections": len(digest.sections),
                "sources_status": sources_status,
                "llm_cost_estimate": digest.llm_cost_estimate,
                "duration_s": round(duration, 1),
                "saved_to": filepath,
                "errors": [],
            }
            _log_run(log_entry)

            # Notify on success
            if notify_email:
                _send_notification(
                    config, notify_email,
                    subject="[Hidden Layer] Digest generated successfully",
                    body=_success_body(digest, filepath, duration),
                )

            return digest

        except Exception as e:
            duration = time.time() - start_time
            last_error = str(e)
            error_tb = traceback.format_exc()
            logger.error(f"Digest run attempt {attempt} failed: {e}")

            if attempt <= max_retries:
                print(f"  Attempt {attempt} failed: {e}")
                print(f"  Retrying in {retry_delay_s}s...")
                time.sleep(retry_delay_s)
            else:
                # Final failure — log and notify
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "status": "failure",
                    "attempt": attempt,
                    "items_scanned": 0,
                    "items_published": 0,
                    "sections": 0,
                    "sources_status": {},
                    "llm_cost_estimate": 0.0,
                    "duration_s": round(duration, 1),
                    "saved_to": None,
                    "errors": [last_error],
                }
                _log_run(log_entry)

                if notify_email:
                    _send_notification(
                        config, notify_email,
                        subject="[Hidden Layer] Digest generation FAILED",
                        body=_failure_body(last_error, error_tb, attempt),
                    )

    return None


def _success_body(digest: DailyDigest, filepath: Optional[str], duration: float) -> str:
    items_published = sum(len(s.items) for s in digest.sections)
    lines = [
        f"Daily digest generated successfully.",
        f"",
        f"Items scanned: {digest.total_items_scanned}",
        f"Items published: {items_published}",
        f"Sections: {len(digest.sections)}",
        f"Generation time: {duration:.1f}s",
        f"LLM cost estimate: ${digest.llm_cost_estimate:.4f}",
    ]
    if filepath:
        lines.append(f"Saved to: {filepath}")
    if digest.source_health:
        lines.append("")
        lines.append("Source health:")
        for sh in digest.source_health:
            status = f"FAILED ({sh.error})" if sh.error else f"{sh.items_count} items ({sh.latency_s:.1f}s)"
            lines.append(f"  {sh.source_name}: {status}")
    return "\n".join(lines)


def _failure_body(error: str, tb: str, attempts: int) -> str:
    return (
        f"Digest generation failed after {attempts} attempt(s).\n\n"
        f"Error: {error}\n\n"
        f"Traceback:\n{tb}"
    )


# --- Email notification ---

def _send_notification(
    config: AggregatorConfig,
    to_email: str,
    subject: str,
    body: str,
):
    """Send an email notification via SMTP."""
    # Get SMTP settings from environment or config
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASSWORD", "")
    from_email = os.environ.get("SMTP_FROM", smtp_user)

    if not smtp_user or not smtp_pass:
        logger.warning("SMTP credentials not configured. Set SMTP_USER and SMTP_PASSWORD env vars.")
        print("  Warning: Email notification skipped (SMTP credentials not set)")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        logger.info(f"Notification sent to {to_email}")
        print(f"  Notification sent to {to_email}")
    except Exception as e:
        logger.warning(f"Failed to send notification: {e}")
        print(f"  Warning: Failed to send notification: {e}")


# --- Schedule generation ---

def generate_systemd_timer(time_str: str = "07:00") -> str:
    """Generate a systemd timer and service unit for daily digest runs."""
    python_path = sys.executable
    service = f"""[Unit]
Description=AI Research Aggregator - Daily Digest
After=network-online.target

[Service]
Type=oneshot
ExecStart={python_path} -m ai_research_aggregator.cli run
Environment=PATH={os.environ.get('PATH', '/usr/bin')}
StandardOutput=journal
StandardError=journal
"""

    timer = f"""[Unit]
Description=AI Research Aggregator - Daily Timer

[Timer]
OnCalendar=*-*-* {time_str}:00
Persistent=true

[Install]
WantedBy=timers.target
"""
    return service, timer


def generate_launchd_plist(time_str: str = "07:00") -> str:
    """Generate a macOS launchd plist for daily digest runs."""
    python_path = sys.executable
    hour, minute = time_str.split(":")

    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hidden-layer.ai-research-aggregator</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>ai_research_aggregator.cli</string>
        <string>run</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{int(hour)}</integer>
        <key>Minute</key>
        <integer>{int(minute)}</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{os.path.expanduser('~/Library/Logs/ai-research-aggregator.log')}</string>
    <key>StandardErrorPath</key>
    <string>{os.path.expanduser('~/Library/Logs/ai-research-aggregator.log')}</string>
</dict>
</plist>
"""
    return plist


def print_schedule_instructions(time_str: str = "07:00"):
    """Print schedule installation instructions for the current platform."""
    system = platform.system()

    if system == "Linux":
        service_content, timer_content = generate_systemd_timer(time_str)
        service_path = os.path.expanduser("~/.config/systemd/user/ai-digest.service")
        timer_path = os.path.expanduser("~/.config/systemd/user/ai-digest.timer")

        print("systemd Timer Configuration")
        print("=" * 50)
        print(f"\n--- {service_path} ---")
        print(service_content)
        print(f"\n--- {timer_path} ---")
        print(timer_content)
        print("\nInstall steps:")
        print(f"  mkdir -p ~/.config/systemd/user")
        print(f"  # Save the above files to the paths shown")
        print(f"  systemctl --user daemon-reload")
        print(f"  systemctl --user enable --now ai-digest.timer")
        print(f"\nCheck status:")
        print(f"  systemctl --user status ai-digest.timer")
        print(f"  journalctl --user -u ai-digest.service -f")

    elif system == "Darwin":
        plist_content = generate_launchd_plist(time_str)
        plist_path = os.path.expanduser(
            "~/Library/LaunchAgents/com.hidden-layer.ai-research-aggregator.plist"
        )

        print("macOS launchd Configuration")
        print("=" * 50)
        print(f"\n--- {plist_path} ---")
        print(plist_content)
        print("\nInstall steps:")
        print(f"  # Save the above to {plist_path}")
        print(f"  launchctl load {plist_path}")
        print(f"\nCheck status:")
        print(f"  launchctl list | grep ai-research")
        print(f"  cat ~/Library/Logs/ai-research-aggregator.log")

    else:
        print(f"Platform '{system}' not directly supported.")
        print(f"Use cron or Task Scheduler to run:")
        print(f"  {sys.executable} -m ai_research_aggregator.cli run")
        print(f"\nCrontab example (daily at {time_str}):")
        hour, minute = time_str.split(":")
        print(f"  {minute} {hour} * * * {sys.executable} -m ai_research_aggregator.cli run")
