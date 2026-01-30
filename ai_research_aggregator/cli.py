#!/usr/bin/env python3
"""
AI Research Aggregator CLI

Usage:
    ai-digest                              # Generate today's digest (terminal output)
    ai-digest --save                       # Also save to markdown file
    ai-digest --no-llm                     # Skip LLM ranking (keyword-only, faster)
    ai-digest --skip-events                # Skip event fetching
    ai-digest config init                  # Create default config file
    ai-digest config show                  # Show current configuration
    ai-digest config path                  # Show config file path
    ai-digest sources test                 # Test all sources
    ai-digest sources test --source arxiv  # Test a specific source
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

    digest = generate_digest(
        config=config,
        use_llm=not args.no_llm,
        skip_events=args.skip_events,
    )

    # Terminal output
    print_digest_terminal(digest)

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
    digest_parser.add_argument("--skip-events", action="store_true", help="Skip event fetching")
    digest_parser.set_defaults(func=cmd_digest)

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

    # Sources commands
    sources_parser = subparsers.add_parser("sources", help="Source management")
    sources_subparsers = sources_parser.add_subparsers(dest="sources_command", help="Source commands")

    test_parser = sources_subparsers.add_parser("test", help="Test content sources")
    test_parser.add_argument("--source", help="Test a specific source (arxiv, blogs, communities, events)")
    test_parser.set_defaults(func=cmd_sources_test)

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
        cmd_digest(args)
        return

    # If subcommand has no function, show help
    if not hasattr(args, "func"):
        if args.command == "config":
            parser.parse_args(["config", "-h"])
        elif args.command == "sources":
            parser.parse_args(["sources", "-h"])
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
