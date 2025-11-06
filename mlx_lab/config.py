"""
Configuration management for MLX Lab

Integrates with harness configuration and validates setup.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

from mlx_lab.utils import (
    get_repo_root,
    check_mlx_installed,
    check_mlx_lm_installed,
    check_harness_installed,
)


class ConfigManager:
    """Manage MLX Lab and harness configuration"""

    def __init__(self):
        self.repo_root = get_repo_root()
        self.config_dir = self.repo_root / "config"

    def validate_setup(self) -> Tuple[bool, List[str]]:
        """
        Validate the MLX Lab setup

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 9):
            issues.append(
                f"Python 3.9+ required, found {python_version.major}.{python_version.minor}"
            )

        # Check MLX installation
        if not check_mlx_installed():
            issues.append("MLX not installed. Run: pip install mlx")

        # Check mlx-lm installation
        if not check_mlx_lm_installed():
            issues.append("mlx-lm not installed. Run: pip install mlx-lm")

        # Check harness installation
        if not check_harness_installed():
            issues.append("harness not installed. Run: pip install -e .")

        # Check repository structure
        required_dirs = [
            "harness",
            "shared/concepts",
            "config",
            "theory-of-mind/introspection",
        ]

        for dir_path in required_dirs:
            full_path = self.repo_root / dir_path
            if not full_path.exists():
                issues.append(f"Missing directory: {dir_path}")

        # Check if models.yaml exists
        models_yaml = self.config_dir / "models.yaml"
        if not models_yaml.exists():
            issues.append(f"Missing config file: {models_yaml}")

        return len(issues) == 0, issues

    def get_current_config(self) -> Dict[str, Any]:
        """Get current harness configuration"""
        config = {}

        # Try to import harness and get defaults
        try:
            # Add repo root to path temporarily
            sys.path.insert(0, str(self.repo_root))

            from harness.defaults import DEFAULT_PROVIDER, DEFAULT_MODEL

            config["default_provider"] = DEFAULT_PROVIDER
            config["default_model"] = DEFAULT_MODEL

            # Clean up path
            sys.path.pop(0)

        except Exception as e:
            config["error"] = str(e)

        # Check environment variables
        config["env_vars"] = {
            "ANTHROPIC_API_KEY": "set"
            if os.environ.get("ANTHROPIC_API_KEY")
            else "not set",
            "OPENAI_API_KEY": "set"
            if os.environ.get("OPENAI_API_KEY")
            else "not set",
            "HF_HOME": os.environ.get("HF_HOME", "not set (using default)"),
        }

        return config

    def format_validation_report(self, is_valid: bool, issues: List[str]) -> str:
        """Format validation report for display"""
        lines = []
        lines.append("MLX Lab Setup Validation")
        lines.append("=" * 70)
        lines.append("")

        if is_valid:
            lines.append("✅ All checks passed!")
            lines.append("")
            lines.append("Your MLX Lab setup is ready to use.")
        else:
            lines.append("❌ Setup validation failed")
            lines.append("")
            lines.append(f"Found {len(issues)} issue(s):")
            lines.append("")

            for i, issue in enumerate(issues, 1):
                lines.append(f"  {i}. {issue}")

            lines.append("")
            lines.append("To fix these issues:")
            lines.append("  • Install missing packages: pip install -e .[mlx]")
            lines.append("  • Ensure you're in the correct directory")
            lines.append("  • Run: mlx-lab setup (for guided setup)")

        return "\n".join(lines)

    def format_config_display(self, config: Dict[str, Any]) -> str:
        """Format configuration for display"""
        lines = []
        lines.append("Current MLX Lab Configuration")
        lines.append("=" * 70)
        lines.append("")

        if "error" in config:
            lines.append(f"⚠️  Error reading config: {config['error']}")
            lines.append("")
            lines.append("Run: mlx-lab config validate")
            return "\n".join(lines)

        lines.append("Harness Defaults:")
        lines.append(f"  Provider: {config.get('default_provider', 'unknown')}")
        lines.append(f"  Model: {config.get('default_model', 'unknown')}")
        lines.append("")

        lines.append("Environment Variables:")
        env_vars = config.get("env_vars", {})
        for var, value in env_vars.items():
            lines.append(f"  {var}: {value}")

        lines.append("")
        lines.append("To modify configuration:")
        lines.append("  • Edit: config/models.yaml")
        lines.append("  • Edit: harness/defaults.py")
        lines.append("  • Set environment variables in .env")

        return "\n".join(lines)

    def test_inference(self, provider: str = "mlx", model: str = None) -> Tuple[bool, str]:
        """
        Test inference with the specified provider

        Returns:
            Tuple of (success, message)
        """
        try:
            sys.path.insert(0, str(self.repo_root))

            from harness import LLMProvider

            llm = LLMProvider()

            # Simple test prompt
            test_prompt = "Say 'Hello, MLX Lab!' and nothing else."

            response = llm.call(
                test_prompt, provider=provider, model=model, max_tokens=20
            )

            sys.path.pop(0)

            if response and response.text:
                return True, f"✅ Inference test passed\n\nResponse: {response.text}"
            else:
                return False, "❌ Inference test failed: Empty response"

        except Exception as e:
            if sys.path and sys.path[0] == str(self.repo_root):
                sys.path.pop(0)
            return False, f"❌ Inference test failed: {str(e)}"
