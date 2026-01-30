#!/usr/bin/env python3
"""
AI Research Aggregator CLI

Usage:
    ai-digest                              # Generate today's digest (terminal output)
    ai-digest digest --save                # Also save to markdown file
    ai-digest digest --no-llm              # Skip LLM ranking (keyword-only, faster)
    ai-digest digest --skip-events         # Skip event fetching
    ai-digest digest --cost-report         # Show LLM token usage and cost estimate
    ai-digest run                              # Run with retry, logging, and notification
    ai-digest run --notify email@example.com   # Send notification on success/failure
    ai-digest config init                  # Create default config file
    ai-digest config show                  # Show current configuration
    ai-digest config path                  # Show config file path
    ai-digest sources test                 # Test all sources
    ai-digest sources test --source arxiv  # Test a specific source
    ai-digest history runs                 # Show recent run history
    ai-digest install-schedule --time 07:00  # Generate systemd/launchd schedule
    ai-digest newsletter preview           # Generate & preview newsletter HTML
    ai-digest newsletter draft             # Create a Substack draft
    ai-digest newsletter publish           # Create & publish to Substack
    ai-digest newsletter login             # Log in to Substack
"""

import argparse
import logging
import sys
from pathlib import Path

from ai_research_aggregator.config import AggregatorConfig, USER_CONFIG_PATH
from ai_research_aggregator.digest import (
    generate_digest,
    print_digest_terminal,
    save_digest,
)


def cmd_digest(args):
    """Generate the daily digest."""
    config = AggregatorConfig.load(args.config) if args.config else AggregatorConfig.load()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if getattr(args, "no_cache", False):
        from ai_research_aggregator.sources.base import BaseSource
        BaseSource.clear_cache()

    digest = generate_digest(
        config=config,
        use_llm=not args.no_llm,
        skip_events=args.skip_events,
    )

    # Terminal output
    print_digest_terminal(digest)

    # Cost report
    if getattr(args, "cost_report", False) and digest.llm_calls > 0:
        from ai_research_aggregator.ranking import get_cost_tracker
        tracker = get_cost_tracker()
        if tracker:
            print("")
            print(tracker.summary())

    # Save to file if requested
    if args.save:
        filepath = save_digest(digest, config)
        print(f"\nDigest saved to: {filepath}")


def cmd_config_init(args):
    """Create default configuration file."""
    config = AggregatorConfig()

    if args.path:
        save_path = args.path
    else:
        save_path = USER_CONFIG_PATH

    if Path(save_path).exists() and not args.force:
        print(f"Config already exists at: {save_path}")
        print("Use --force to overwrite.")
        sys.exit(1)

    config.save(save_path)
    print(f"Default config created at: {save_path}")
    print("Edit this file to customize your interests and settings.")


def cmd_config_show(args):
    """Show current configuration."""
    config = AggregatorConfig.load(args.config) if hasattr(args, "config") and args.config else AggregatorConfig.load()

    import yaml
    print("Current Configuration:")
    print("-" * 40)
    print(yaml.dump(config.to_dict(), default_flow_style=False, sort_keys=False))


def cmd_config_path(args):
    """Show config file path."""
    print(f"User config: {USER_CONFIG_PATH}")
    exists = Path(USER_CONFIG_PATH).exists()
    print(f"Exists: {'yes' if exists else 'no (using defaults)'}")


def cmd_history_stats(args):
    """Show deduplication history stats."""
    from ai_research_aggregator.storage import HistoryDB
    db = HistoryDB()
    stats = db.stats()
    print("Deduplication History:")
    print(f"  Database: {stats['db_path']}")
    print(f"  Total items seen: {stats['total_items']}")
    print(f"  Published in digest: {stats['published_items']}")
    print(f"  Seen in last 7 days: {stats['seen_last_7_days']}")


def cmd_history_clear(args):
    """Clear all deduplication history."""
    from ai_research_aggregator.storage import HistoryDB
    db = HistoryDB()
    db.clear()
    print("History cleared.")


def cmd_history_runs(args):
    """Show recent run history."""
    from ai_research_aggregator.scheduler import get_recent_runs
    runs = get_recent_runs(count=args.count if hasattr(args, "count") else 10)
    if not runs:
        print("No run history found.")
        return

    print(f"Last {len(runs)} runs:")
    print("-" * 70)
    for run in reversed(runs):
        ts = run.get("timestamp", "unknown")
        status = run.get("status", "unknown")
        duration = run.get("duration_s", 0)
        scanned = run.get("items_scanned", 0)
        published = run.get("items_published", 0)
        cost = run.get("llm_cost_estimate", 0)
        attempt = run.get("attempt", 1)
        errors = run.get("errors", [])

        status_icon = "OK" if status == "success" else "FAIL"
        line = f"  [{status_icon}] {ts}  {duration:.0f}s  {scanned} scanned / {published} published"
        if cost > 0:
            line += f"  ${cost:.4f}"
        if attempt > 1:
            line += f"  (attempt {attempt})"
        print(line)

        if errors:
            for err in errors:
                print(f"       Error: {err[:100]}")
    print("-" * 70)


def cmd_run(args):
    """Run digest with retry, logging, and optional notification."""
    config = AggregatorConfig.load(args.config) if args.config else AggregatorConfig.load()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if getattr(args, "no_cache", False):
        from ai_research_aggregator.sources.base import BaseSource
        BaseSource.clear_cache()

    from ai_research_aggregator.scheduler import run_digest
    from ai_research_aggregator.digest import print_digest_terminal

    should_save = args.save and not getattr(args, "no_save", False)

    digest = run_digest(
        config=config,
        use_llm=not args.no_llm,
        skip_events=args.skip_events,
        save=should_save,
        max_retries=args.retries,
        retry_delay_s=args.retry_delay,
        notify_email=args.notify if hasattr(args, "notify") else None,
    )

    if digest:
        print_digest_terminal(digest)

        if getattr(args, "cost_report", False) and digest.llm_calls > 0:
            from ai_research_aggregator.ranking import get_cost_tracker
            tracker = get_cost_tracker()
            if tracker:
                print("")
                print(tracker.summary())
    else:
        print("\nAll attempts failed. Check 'ai-digest history runs' for details.")
        sys.exit(1)


def cmd_install_schedule(args):
    """Generate and print schedule installation instructions."""
    from ai_research_aggregator.scheduler import print_schedule_instructions
    print_schedule_instructions(time_str=args.time)


def cmd_sources_test(args):
    """Test content sources."""
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = AggregatorConfig()

    sources_map = {}
    if not args.source or args.source == "arxiv":
        from ai_research_aggregator.sources.arxiv import ArxivSource
        sources_map["arxiv"] = ArxivSource(
            categories=config.interests.arxiv_categories,
            search_terms=config.interests.search_terms[:3],
        )

    if not args.source or args.source == "blogs":
        from ai_research_aggregator.sources.blogs import BlogAggregatorSource
        sources_map["blogs"] = BlogAggregatorSource()

    if not args.source or args.source == "communities":
        from ai_research_aggregator.sources.communities import CommunitySource
        sources_map["communities"] = CommunitySource()

    if not args.source or args.source == "events":
        from ai_research_aggregator.sources.events import SFEventsSource
        sources_map["events"] = SFEventsSource()

    if not sources_map:
        print(f"Unknown source: {args.source}")
        print("Available: arxiv, blogs, communities, events")
        sys.exit(1)

    for name, source in sources_map.items():
        print(f"\nTesting {name}...")
        print("-" * 40)
        items = source.fetch_safe(max_items=5)
        if items:
            print(f"  Fetched {len(items)} items:")
            for item in items[:5]:
                print(f"  - {item.title[:80]}")
                if item.authors:
                    print(f"    Authors: {', '.join(item.authors[:3])}")
                print(f"    URL: {item.url}")
        else:
            print(f"  No items fetched (source may be unavailable)")


def cmd_newsletter_preview(args):
    """Generate newsletter HTML and preview it."""
    config = AggregatorConfig.load(args.config) if args.config else AggregatorConfig.load()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    from ai_research_aggregator.newsletter import (
        digest_to_newsletter_html,
        digest_to_newsletter_subject,
        digest_to_newsletter_subtitle,
    )

    digest = generate_digest(
        config=config,
        use_llm=not args.no_llm,
        skip_events=args.skip_events,
    )

    subject = digest_to_newsletter_subject(digest)
    subtitle = digest_to_newsletter_subtitle(digest)
    body_html = digest_to_newsletter_html(
        digest,
        intro_text=config.substack.intro_text or None,
        footer_text=config.substack.footer_text or None,
    )

    # Save HTML preview
    output_dir = config.output.output_dir
    import os
    os.makedirs(output_dir, exist_ok=True)
    date_str = digest.date.strftime("%Y-%m-%d")
    preview_path = os.path.join(output_dir, f"newsletter-{date_str}.html")

    # Wrap in a minimal HTML document for browser preview
    full_html = (
        "<!DOCTYPE html>\n<html><head>"
        '<meta charset="utf-8">'
        f"<title>{subject}</title>"
        '<style>'
        'body { max-width: 680px; margin: 40px auto; padding: 0 20px; '
        'font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; '
        'line-height: 1.6; color: #1a1a1a; } '
        'h1 { font-size: 28px; } h2 { font-size: 22px; margin-top: 32px; } '
        'h3 { font-size: 17px; margin-bottom: 4px; } '
        'a { color: #2563eb; text-decoration: none; } a:hover { text-decoration: underline; } '
        'blockquote { border-left: 3px solid #e5e7eb; margin: 8px 0; padding: 4px 16px; color: #4b5563; } '
        'code { background: #f3f4f6; padding: 2px 6px; border-radius: 3px; font-size: 13px; } '
        'hr { border: none; border-top: 1px solid #e5e7eb; margin: 24px 0; } '
        'img { max-width: 100%; } '
        '</style>'
        "</head><body>\n"
        f"<h1>{subject}</h1>\n"
        f"<p><em>{subtitle}</em></p>\n"
        f"{body_html}\n"
        "</body></html>"
    )

    with open(preview_path, "w") as f:
        f.write(full_html)

    print(f"Newsletter preview saved to: {preview_path}")
    print(f"Title: {subject}")
    print(f"Subtitle: {subtitle}")
    print(f"Body size: {len(body_html):,} chars")
    print(f"\nOpen in your browser to preview the formatting.")

    # Also save raw body HTML for copy-paste into Substack editor
    raw_path = os.path.join(output_dir, f"newsletter-{date_str}-body.html")
    with open(raw_path, "w") as f:
        f.write(body_html)
    print(f"Raw HTML body (for Substack paste): {raw_path}")


def cmd_newsletter_draft(args):
    """Create a Substack draft from today's digest."""
    config = AggregatorConfig.load(args.config) if args.config else AggregatorConfig.load()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if not config.substack.publication:
        print("Error: No Substack publication configured.")
        print("Set 'substack.publication' in your config file, or use:")
        print("  ai-digest newsletter draft --publication your-slug")
        sys.exit(1)

    publication = args.publication or config.substack.publication

    from ai_research_aggregator.newsletter import (
        digest_to_newsletter_html,
        digest_to_newsletter_subject,
        digest_to_newsletter_subtitle,
    )
    from ai_research_aggregator.substack import SubstackClient

    # Generate digest
    digest = generate_digest(
        config=config,
        use_llm=not args.no_llm,
        skip_events=args.skip_events,
    )

    subject = digest_to_newsletter_subject(digest)
    subtitle = digest_to_newsletter_subtitle(digest)
    body_html = digest_to_newsletter_html(
        digest,
        intro_text=config.substack.intro_text or None,
        footer_text=config.substack.footer_text or None,
    )

    # Connect to Substack
    client = SubstackClient(
        publication_url=publication,
        email=config.substack.email or None,
        password=config.substack.password or None,
    )

    if not client.is_authenticated:
        print("Error: Not authenticated with Substack.")
        print("Run: ai-digest newsletter login")
        sys.exit(1)

    # Create draft
    section_id = config.substack.section_id if config.substack.section_id else None
    post = client.create_draft(
        title=subject,
        body_html=body_html,
        subtitle=subtitle,
        section_id=section_id,
    )

    print(f"\nDraft created on Substack!")
    print(f"  Title: {post.title}")
    print(f"  Draft URL: {post.draft_url}")
    print(f"\nOpen the URL above to review and publish.")


def cmd_newsletter_publish(args):
    """Generate digest and publish directly to Substack."""
    config = AggregatorConfig.load(args.config) if args.config else AggregatorConfig.load()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if not config.substack.publication:
        print("Error: No Substack publication configured.")
        print("Set 'substack.publication' in your config file.")
        sys.exit(1)

    publication = args.publication or config.substack.publication

    from ai_research_aggregator.newsletter import (
        digest_to_newsletter_html,
        digest_to_newsletter_subject,
        digest_to_newsletter_subtitle,
    )
    from ai_research_aggregator.substack import SubstackClient

    # Generate digest
    digest = generate_digest(
        config=config,
        use_llm=not args.no_llm,
        skip_events=args.skip_events,
    )

    subject = digest_to_newsletter_subject(digest)
    subtitle = digest_to_newsletter_subtitle(digest)
    body_html = digest_to_newsletter_html(
        digest,
        intro_text=config.substack.intro_text or None,
        footer_text=config.substack.footer_text or None,
    )

    # Confirm before publishing
    send_email = config.substack.send_email
    if not args.yes:
        print(f"About to publish to: {publication}.substack.com")
        print(f"  Title: {subject}")
        print(f"  Send email: {'yes' if send_email else 'no'}")
        print(f"  Body size: {len(body_html):,} chars")
        response = input("\nProceed? (yes/no): ")
        if response.lower() not in ("yes", "y"):
            print("Cancelled.")
            return

    # Connect and publish
    client = SubstackClient(
        publication_url=publication,
        email=config.substack.email or None,
        password=config.substack.password or None,
    )

    if not client.is_authenticated:
        print("Error: Not authenticated with Substack.")
        print("Run: ai-digest newsletter login")
        sys.exit(1)

    section_id = config.substack.section_id if config.substack.section_id else None
    post = client.create_and_publish(
        title=subject,
        body_html=body_html,
        subtitle=subtitle,
        send_email=send_email,
        section_id=section_id,
    )

    print(f"\nPublished to Substack!")
    print(f"  URL: {post.published_url}")
    if send_email:
        print(f"  Email sent to subscribers.")


def cmd_newsletter_login(args):
    """Log in to Substack and save session."""
    publication = args.publication
    if not publication:
        config = AggregatorConfig.load(args.config) if hasattr(args, "config") and args.config else AggregatorConfig.load()
        publication = config.substack.publication

    if not publication:
        publication = input("Substack publication slug (e.g., 'myresearch'): ").strip()

    if not publication:
        print("Error: Publication slug is required.")
        sys.exit(1)

    email = input("Email: ").strip()
    if not email:
        print("Error: Email is required.")
        sys.exit(1)

    import getpass
    password = getpass.getpass("Password: ")
    if not password:
        print("Error: Password is required.")
        sys.exit(1)

    from ai_research_aggregator.substack import SubstackClient

    client = SubstackClient(publication_url=publication)
    success = client.login(email, password)

    if success:
        print(f"\nLogged in to {publication}.substack.com")
        print("Session token saved. You can now use newsletter commands.")
    else:
        print("\nLogin failed. Check your email and password.")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Research Aggregator - Daily AI research digest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", "-c",
        help="Path to config file",
        default=None,
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Default: generate digest
    digest_parser = subparsers.add_parser("digest", help="Generate daily digest (default)")
    digest_parser.add_argument("--save", "-s", action="store_true", help="Save digest to markdown file")
    digest_parser.add_argument("--no-llm", action="store_true", help="Skip LLM ranking (keyword-only)")
    digest_parser.add_argument("--no-cache", action="store_true", help="Bypass response cache")
    digest_parser.add_argument("--skip-events", action="store_true", help="Skip event fetching")
    digest_parser.add_argument("--cost-report", action="store_true", help="Print LLM cost estimate after generation")
    digest_parser.set_defaults(func=cmd_digest)

    # Run command (digest with retry + logging + notification)
    run_parser = subparsers.add_parser("run", help="Run digest with retry, logging, and notification")
    run_parser.add_argument("--save", "-s", action="store_true", default=True, help="Save digest to markdown file (default: yes)")
    run_parser.add_argument("--no-save", action="store_true", help="Don't save digest to file")
    run_parser.add_argument("--no-llm", action="store_true", help="Skip LLM ranking (keyword-only)")
    run_parser.add_argument("--no-cache", action="store_true", help="Bypass response cache")
    run_parser.add_argument("--skip-events", action="store_true", help="Skip event fetching")
    run_parser.add_argument("--cost-report", action="store_true", help="Print LLM cost estimate")
    run_parser.add_argument("--retries", type=int, default=2, help="Max retries on failure (default: 2)")
    run_parser.add_argument("--retry-delay", type=int, default=300, help="Seconds between retries (default: 300)")
    run_parser.add_argument("--notify", metavar="EMAIL", help="Send notification to this email")
    run_parser.set_defaults(func=cmd_run)

    # Install schedule command
    schedule_parser = subparsers.add_parser("install-schedule", help="Generate daily schedule config")
    schedule_parser.add_argument("--time", default="07:00", help="Daily run time in HH:MM format (default: 07:00)")
    schedule_parser.set_defaults(func=cmd_install_schedule)

    # Config commands
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_subparsers = config_parser.add_subparsers(dest="config_command", help="Config commands")

    init_parser = config_subparsers.add_parser("init", help="Create default config file")
    init_parser.add_argument("--path", help="Config file path")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing config")
    init_parser.set_defaults(func=cmd_config_init)

    show_parser = config_subparsers.add_parser("show", help="Show current configuration")
    show_parser.set_defaults(func=cmd_config_show)

    path_parser = config_subparsers.add_parser("path", help="Show config file path")
    path_parser.set_defaults(func=cmd_config_path)

    # History commands
    history_parser = subparsers.add_parser("history", help="Deduplication history")
    history_subparsers = history_parser.add_subparsers(dest="history_command", help="History commands")

    history_stats_parser = history_subparsers.add_parser("stats", help="Show history stats")
    history_stats_parser.set_defaults(func=cmd_history_stats)

    history_clear_parser = history_subparsers.add_parser("clear", help="Clear all history")
    history_clear_parser.set_defaults(func=cmd_history_clear)

    history_runs_parser = history_subparsers.add_parser("runs", help="Show recent run history")
    history_runs_parser.add_argument("--count", "-n", type=int, default=10, help="Number of runs to show")
    history_runs_parser.set_defaults(func=cmd_history_runs)

    # Sources commands
    sources_parser = subparsers.add_parser("sources", help="Source management")
    sources_subparsers = sources_parser.add_subparsers(dest="sources_command", help="Source commands")

    test_parser = sources_subparsers.add_parser("test", help="Test content sources")
    test_parser.add_argument("--source", help="Test a specific source (arxiv, blogs, communities, events)")
    test_parser.set_defaults(func=cmd_sources_test)

    # Newsletter commands
    newsletter_parser = subparsers.add_parser("newsletter", help="Substack newsletter management")
    newsletter_subparsers = newsletter_parser.add_subparsers(
        dest="newsletter_command", help="Newsletter commands"
    )

    # newsletter preview
    preview_parser = newsletter_subparsers.add_parser(
        "preview", help="Generate newsletter HTML preview"
    )
    preview_parser.add_argument("--no-llm", action="store_true", help="Skip LLM ranking")
    preview_parser.add_argument("--skip-events", action="store_true", help="Skip events")
    preview_parser.set_defaults(func=cmd_newsletter_preview)

    # newsletter draft
    draft_parser = newsletter_subparsers.add_parser(
        "draft", help="Create a Substack draft"
    )
    draft_parser.add_argument("--publication", help="Substack publication slug")
    draft_parser.add_argument("--no-llm", action="store_true", help="Skip LLM ranking")
    draft_parser.add_argument("--skip-events", action="store_true", help="Skip events")
    draft_parser.set_defaults(func=cmd_newsletter_draft)

    # newsletter publish
    publish_parser = newsletter_subparsers.add_parser(
        "publish", help="Generate and publish to Substack"
    )
    publish_parser.add_argument("--publication", help="Substack publication slug")
    publish_parser.add_argument("--no-llm", action="store_true", help="Skip LLM ranking")
    publish_parser.add_argument("--skip-events", action="store_true", help="Skip events")
    publish_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    publish_parser.set_defaults(func=cmd_newsletter_publish)

    # newsletter login
    login_parser = newsletter_subparsers.add_parser(
        "login", help="Log in to Substack"
    )
    login_parser.add_argument("--publication", help="Substack publication slug")
    login_parser.set_defaults(func=cmd_newsletter_login)

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Default to digest if no command given
    if not args.command:
        # Run digest with defaults
        args.save = False
        args.no_llm = False
        args.skip_events = False
        args.cost_report = False
        cmd_digest(args)
        return

    # If subcommand has no function, show help
    if not hasattr(args, "func"):
        if args.command == "config":
            parser.parse_args(["config", "-h"])
        elif args.command == "history":
            parser.parse_args(["history", "-h"])
        elif args.command == "sources":
            parser.parse_args(["sources", "-h"])
        elif args.command == "newsletter":
            parser.parse_args(["newsletter", "-h"])
        sys.exit(0)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
