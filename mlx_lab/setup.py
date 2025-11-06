"""
Setup wizard for MLX Lab

Interactive first-time setup and validation.
"""

from typing import Optional, Callable

from mlx_lab.config import ConfigManager
from mlx_lab.models import ModelManager
from mlx_lab.benchmark import PerformanceBenchmark


class SetupWizard:
    """Interactive setup wizard for MLX Lab"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.model_manager = ModelManager()
        self.benchmark = PerformanceBenchmark()

    def run_setup(
        self, interactive: bool = True, output_callback: Optional[Callable] = None
    ) -> bool:
        """
        Run the setup wizard

        Args:
            interactive: Whether to prompt for user input
            output_callback: Optional callback for output messages

        Returns:
            True if setup successful, False otherwise
        """

        def output(msg: str):
            if output_callback:
                output_callback(msg)
            else:
                print(msg)

        output("=" * 70)
        output("MLX Lab Setup Wizard")
        output("=" * 70)
        output("")

        # Step 1: Validate environment
        output("Step 1/4: Validating environment...")
        output("")

        is_valid, issues = self.config_manager.validate_setup()

        if is_valid:
            output("✅ Environment validation passed")
        else:
            output("❌ Environment validation failed:")
            output("")
            for issue in issues:
                output(f"  • {issue}")
            output("")

            if not interactive:
                return False

            output("Please fix the issues above and run setup again.")
            return False

        output("")

        # Step 2: Check for downloaded models
        output("Step 2/4: Checking for MLX models...")
        output("")

        models = self.model_manager.list_models()

        if models:
            output(f"✅ Found {len(models)} downloaded model(s):")
            for model in models[:3]:  # Show first 3
                output(f"  • {model.name}")
            if len(models) > 3:
                output(f"  ... and {len(models) - 3} more")
        else:
            output("ℹ️  No models found in cache")

            if interactive:
                output("")
                output("Recommended models for getting started:")
                output("")

                for i, (key, info) in enumerate(
                    self.model_manager.RECOMMENDED_MODELS.items(), 1
                ):
                    output(f"  {i}. {key}")
                    output(f"     {info['description']}")
                    output(f"     RAM: {info['ram_estimate']}")
                    output("")

                # For non-interactive setup, we'll skip downloading
                output(
                    "To download a model, run: mlx-lab models download <model-name>"
                )
                output("")

        output("")

        # Step 3: Test inference (only if models exist)
        if models:
            output("Step 3/4: Testing inference...")
            output("")

            # Use first available model for testing
            test_model = models[0]
            success, message = self.config_manager.test_inference(
                provider="mlx", model=test_model.repo_id
            )

            output(message)
            output("")

            if not success and not interactive:
                return False
        else:
            output("Step 3/4: Skipping inference test (no models available)")
            output("")

        # Step 4: Summary
        output("Step 4/4: Setup complete!")
        output("")
        output("=" * 70)
        output("Next Steps:")
        output("=" * 70)
        output("")

        if not models:
            output("1. Download a model:")
            output("   mlx-lab models download mlx-community/Qwen3-8B-4bit")
            output("")

        output("2. Explore available commands:")
        output("   mlx-lab models list        # List downloaded models")
        output("   mlx-lab models test <name> # Test model performance")
        output("   mlx-lab concepts list      # Browse concept vectors")
        output("   mlx-lab config show        # View configuration")
        output("")

        output("3. Start experimenting:")
        output("   See notebooks in: theory-of-mind/introspection/notebooks/")
        output("")

        return True

    def quick_check(self) -> str:
        """Quick setup check (non-interactive)"""
        lines = []
        lines.append("MLX Lab Quick Check")
        lines.append("=" * 70)
        lines.append("")

        # Validate
        is_valid, issues = self.config_manager.validate_setup()

        if is_valid:
            lines.append("✅ Environment: OK")
        else:
            lines.append(f"❌ Environment: {len(issues)} issue(s)")

        # Check models
        models = self.model_manager.list_models()
        lines.append(f"ℹ️  Downloaded models: {len(models)}")

        # Check concepts
        from mlx_lab.concepts import ConceptBrowser

        concepts = ConceptBrowser().list_concepts()
        lines.append(f"ℹ️  Concept vectors: {len(concepts)}")

        lines.append("")

        if not is_valid:
            lines.append("Run: mlx-lab setup (for detailed setup)")
        elif not models:
            lines.append("Run: mlx-lab models download <name> (to download a model)")
        else:
            lines.append("✅ Ready to use!")

        return "\n".join(lines)
