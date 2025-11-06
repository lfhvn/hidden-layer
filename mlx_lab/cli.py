#!/usr/bin/env python3
"""
MLX Lab CLI - Command-line interface for MLX model management and research

Usage:
    mlx-lab setup                          # Run setup wizard
    mlx-lab models list                    # List downloaded models
    mlx-lab models download <name>         # Download a model
    mlx-lab models remove <name>           # Remove a model
    mlx-lab models info <name>             # Show model details
    mlx-lab models test <name>             # Test model performance
    mlx-lab models compare <name1> <name2> # Compare models
    mlx-lab concepts list                  # List concept vectors
    mlx-lab concepts info <name>           # Show concept details
    mlx-lab config show                    # Show configuration
    mlx-lab config validate                # Validate setup
"""

import argparse
import sys
from typing import List

from mlx_lab.models import ModelManager
from mlx_lab.benchmark import PerformanceBenchmark
from mlx_lab.concepts import ConceptBrowser
from mlx_lab.config import ConfigManager
from mlx_lab.setup import SetupWizard


def cmd_setup(args):
    """Run setup wizard"""
    wizard = SetupWizard()
    success = wizard.run_setup(interactive=not args.non_interactive)
    sys.exit(0 if success else 1)


def cmd_models_list(args):
    """List downloaded models"""
    manager = ModelManager()
    models = manager.list_models()
    print(manager.format_model_list(models))


def cmd_models_download(args):
    """Download a model"""
    manager = ModelManager()

    # Check if it's a short name from recommended models
    recommended = manager.get_recommended_models()
    repo_id = args.name

    # If short name, convert to full repo_id
    if args.name in recommended:
        repo_id = recommended[args.name]["repo_id"]
        print(f"Downloading {args.name} ({repo_id})...")
    else:
        print(f"Downloading {repo_id}...")

    print("")
    print("This may take several minutes depending on model size...")
    print("")

    success = manager.download_model(repo_id, progress_callback=print)

    if success:
        print("")
        print("To test this model's performance, run:")
        print(f"  mlx-lab models test {args.name}")
    else:
        sys.exit(1)


def cmd_models_remove(args):
    """Remove a model"""
    manager = ModelManager()

    # Confirm first
    if not args.yes:
        model_info = manager.get_model_info(args.name)
        if model_info:
            print(f"About to remove: {model_info.name}")
            print(f"Size: {model_info.size_bytes / (1024**3):.1f} GB")
            response = input("Are you sure? (yes/no): ")
            if response.lower() not in ["yes", "y"]:
                print("Cancelled")
                return

    success = manager.remove_model(args.name)
    if success:
        print(f"✅ Removed {args.name}")
    else:
        print(f"❌ Failed to remove {args.name}")
        print(f"Model not found. Run: mlx-lab models list")
        sys.exit(1)


def cmd_models_info(args):
    """Show model info"""
    manager = ModelManager()
    model_info = manager.get_model_info(args.name)

    if model_info:
        print(manager.format_model_info(model_info))
    else:
        print(f"Model '{args.name}' not found")
        print("")
        print("Available models:")

        # Show recommended models
        recommended = manager.get_recommended_models()
        for key, info in recommended.items():
            print(f"  • {key} - {info['description']}")

        sys.exit(1)


def cmd_models_test(args):
    """Test model performance"""
    manager = ModelManager()
    benchmark = PerformanceBenchmark()

    # Get model info
    model_info = manager.get_model_info(args.name)
    if not model_info:
        print(f"Model '{args.name}' not found")
        print("Run: mlx-lab models list")
        sys.exit(1)

    print(f"Testing {model_info.name}...")
    print("")

    # Run benchmark
    result = benchmark.benchmark_model(
        model_info.repo_id, use_cache=not args.no_cache, progress_callback=print
    )

    if result:
        print("")
        print(benchmark.format_result(result))
    else:
        sys.exit(1)


def cmd_models_compare(args):
    """Compare models"""
    manager = ModelManager()
    benchmark = PerformanceBenchmark()

    # Resolve model names to repo_ids
    repo_ids = []
    for name in args.models:
        model_info = manager.get_model_info(name)
        if model_info:
            repo_ids.append(model_info.repo_id)
        else:
            print(f"Model '{name}' not found")
            sys.exit(1)

    print(f"Comparing {len(repo_ids)} models...")
    print("")

    # Benchmark each model
    results = benchmark.compare_models(repo_ids, use_cache=not args.no_cache)

    if results:
        print("")
        print(benchmark.format_comparison(results))
    else:
        print("No benchmark results available")
        sys.exit(1)


def cmd_concepts_list(args):
    """List concept vectors"""
    browser = ConceptBrowser()
    concepts = browser.list_concepts()
    print(browser.format_concept_list(concepts))


def cmd_concepts_info(args):
    """Show concept info"""
    browser = ConceptBrowser()
    concept_info = browser.get_concept_info(args.name)

    if concept_info:
        print(browser.format_concept_info(concept_info))
    else:
        print(f"Concept '{args.name}' not found")
        print("")
        print("Run: mlx-lab concepts list")
        sys.exit(1)


def cmd_config_show(args):
    """Show configuration"""
    manager = ConfigManager()
    config = manager.get_current_config()
    print(manager.format_config_display(config))


def cmd_config_validate(args):
    """Validate setup"""
    manager = ConfigManager()
    is_valid, issues = manager.validate_setup()
    print(manager.format_validation_report(is_valid, issues))
    sys.exit(0 if is_valid else 1)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser"""
    parser = argparse.ArgumentParser(
        description="MLX Lab - CLI tool for MLX model management and research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Run setup wizard")
    setup_parser.add_argument(
        "--non-interactive", action="store_true", help="Run without prompts"
    )
    setup_parser.set_defaults(func=cmd_setup)

    # Models commands
    models_parser = subparsers.add_parser("models", help="Model management")
    models_subparsers = models_parser.add_subparsers(
        dest="models_command", help="Model commands"
    )

    # models list
    list_parser = models_subparsers.add_parser("list", help="List downloaded models")
    list_parser.set_defaults(func=cmd_models_list)

    # models download
    download_parser = models_subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("name", help="Model name or repo ID")
    download_parser.set_defaults(func=cmd_models_download)

    # models remove
    remove_parser = models_subparsers.add_parser("remove", help="Remove a model")
    remove_parser.add_argument("name", help="Model name")
    remove_parser.add_argument(
        "-y", "--yes", action="store_true", help="Skip confirmation"
    )
    remove_parser.set_defaults(func=cmd_models_remove)

    # models info
    info_parser = models_subparsers.add_parser("info", help="Show model details")
    info_parser.add_argument("name", help="Model name")
    info_parser.set_defaults(func=cmd_models_info)

    # models test
    test_parser = models_subparsers.add_parser(
        "test", help="Test model performance"
    )
    test_parser.add_argument("name", help="Model name")
    test_parser.add_argument(
        "--no-cache", action="store_true", help="Ignore cached results"
    )
    test_parser.set_defaults(func=cmd_models_test)

    # models compare
    compare_parser = models_subparsers.add_parser("compare", help="Compare models")
    compare_parser.add_argument("models", nargs="+", help="Model names to compare")
    compare_parser.add_argument(
        "--no-cache", action="store_true", help="Ignore cached results"
    )
    compare_parser.set_defaults(func=cmd_models_compare)

    # Concepts commands
    concepts_parser = subparsers.add_parser("concepts", help="Concept management")
    concepts_subparsers = concepts_parser.add_subparsers(
        dest="concepts_command", help="Concept commands"
    )

    # concepts list
    concepts_list_parser = concepts_subparsers.add_parser(
        "list", help="List concept vectors"
    )
    concepts_list_parser.set_defaults(func=cmd_concepts_list)

    # concepts info
    concepts_info_parser = concepts_subparsers.add_parser(
        "info", help="Show concept details"
    )
    concepts_info_parser.add_argument("name", help="Concept name")
    concepts_info_parser.set_defaults(func=cmd_concepts_info)

    # Config commands
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_subparsers = config_parser.add_subparsers(
        dest="config_command", help="Config commands"
    )

    # config show
    config_show_parser = config_subparsers.add_parser(
        "show", help="Show configuration"
    )
    config_show_parser.set_defaults(func=cmd_config_show)

    # config validate
    config_validate_parser = config_subparsers.add_parser(
        "validate", help="Validate setup"
    )
    config_validate_parser.set_defaults(func=cmd_config_validate)

    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # If subcommand has no function, show help for that subcommand
    if not hasattr(args, "func"):
        if args.command == "models":
            parser.parse_args(["models", "-h"])
        elif args.command == "concepts":
            parser.parse_args(["concepts", "-h"])
        elif args.command == "config":
            parser.parse_args(["config", "-h"])
        sys.exit(0)

    # Execute command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
