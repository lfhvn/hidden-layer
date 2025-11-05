#!/usr/bin/env python3
"""
System check script for Hidden Layer research harness.

Verifies that all dependencies and services are properly configured.
"""
import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version is 3.10+"""
    print("Checking Python version...", end=" ")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} (need 3.10+)")
        return False


def check_package(package_name, import_name=None):
    """Check if a Python package is installed"""
    import_name = import_name or package_name
    try:
        __import__(import_name)
        print(f"  ✓ {package_name}")
        return True
    except ImportError:
        print(f"  ✗ {package_name} (run: pip install {package_name})")
        return False


def check_python_packages():
    """Check all required Python packages"""
    print("\nChecking Python packages...")

    # MLX is only for Apple Silicon
    import platform
    is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"

    required = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("jupyter", "jupyter"),
    ]

    # Add MLX packages only on Apple Silicon
    if is_apple_silicon:
        required.extend([
            ("mlx", "mlx.core"),
            ("mlx-lm", "mlx_lm"),
        ])

    optional = [
        ("anthropic", "anthropic"),
        ("openai", "openai"),
        ("ollama", "ollama"),
    ]

    all_good = True

    print("  Required:")
    for pkg, imp in required:
        if not check_package(pkg, imp):
            all_good = False

    print("  Optional (for API providers):")
    for pkg, imp in optional:
        check_package(pkg, imp)  # Don't fail on optional

    return all_good


def check_ollama():
    """Check if Ollama is running"""
    print("\nChecking Ollama...")
    try:
        import ollama
        # Try to list models
        models = ollama.list()
        print(f"  ✓ Ollama is running")
        print(f"  Models available: {len(models.get('models', []))}")

        if models.get('models'):
            print("    Available models:")
            for model in models['models'][:5]:  # Show first 5
                print(f"      - {model['name']}")
            if len(models['models']) > 5:
                print(f"      ... and {len(models['models']) - 5} more")
        else:
            print("    ⚠️  No models pulled yet. Run: ollama pull llama3.2:latest")

        return True
    except ImportError:
        print("  ✗ Ollama package not installed (run: pip install ollama)")
        return False
    except Exception as e:
        print(f"  ✗ Ollama not running or unreachable")
        print(f"    Error: {e}")
        print("    Start Ollama with: ollama serve")
        return False


def check_mlx():
    """Check if MLX is working (Apple Silicon only)"""
    import platform
    is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"

    if not is_apple_silicon:
        print("\nSkipping MLX check (Apple Silicon only, you're on Linux/x86)")
        return True  # Not a failure on non-Apple Silicon

    print("\nChecking MLX (Apple Silicon)...")
    try:
        import mlx.core as mx
        # Try a simple operation
        a = mx.array([1, 2, 3])
        b = a + 1
        print(f"  ✓ MLX is working (version {mx.__version__})")
        return True
    except ImportError:
        print("  ✗ MLX not installed (run: pip install mlx mlx-lm)")
        return False
    except Exception as e:
        print(f"  ✗ MLX error: {e}")
        return False


def check_env_file():
    """Check for .env file and API keys"""
    print("\nChecking API keys (.env file)...")
    env_path = Path(".env")

    if not env_path.exists():
        print("  ℹ️  No .env file found (optional for local-only usage)")
        print("     Copy .env.example to .env if you want to use API providers")
        return True

    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("  ⚠️  python-dotenv not installed (run: pip install python-dotenv)")
        return True

    # Check for API keys
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if anthropic_key and anthropic_key != "your_anthropic_key_here":
        print("  ✓ Anthropic API key found")
    else:
        print("  ℹ️  No Anthropic API key (optional)")

    if openai_key and openai_key != "your_openai_key_here":
        print("  ✓ OpenAI API key found")
    else:
        print("  ℹ️  No OpenAI API key (optional)")

    return True


def check_harness():
    """Check if harness package is importable"""
    print("\nChecking harness package...")

    # Add code directory to path
    code_path = Path(__file__).parent / "code"
    sys.path.insert(0, str(code_path))

    try:
        from harness import (
            llm_call,
            run_strategy,
            get_tracker,
            STRATEGIES
        )
        print(f"  ✓ Harness package loaded")
        print(f"  Available strategies: {', '.join(STRATEGIES.keys())}")
        return True
    except ImportError as e:
        print(f"  ✗ Cannot import harness: {e}")
        return False


def check_directories():
    """Check required directories exist"""
    print("\nChecking directories...")

    dirs = [
        ("code/harness", True),
        ("notebooks", True),
        ("experiments", False),  # Created automatically
    ]

    all_good = True
    for dir_path, required in dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}/")
        elif required:
            print(f"  ✗ {dir_path}/ (missing!)")
            all_good = False
        else:
            print(f"  ℹ️  {dir_path}/ (will be created automatically)")

    return all_good


def test_basic_functionality():
    """Run a simple end-to-end test"""
    print("\n" + "="*60)
    print("Running basic functionality test...")
    print("="*60)

    code_path = Path(__file__).parent / "code"
    sys.path.insert(0, str(code_path))

    try:
        from harness import llm_call
        from harness.defaults import DEFAULT_MODEL, DEFAULT_PROVIDER

        print(f"\nTesting {DEFAULT_PROVIDER} with model {DEFAULT_MODEL}...")
        response = llm_call(
            "Say 'Hello from Hidden Layer!' and nothing else.",
            max_tokens=50
        )

        print(f"  Response: {response.text[:100]}")
        print(f"  Latency: {response.latency_s:.2f}s")
        print("  ✓ Basic functionality test passed!")
        return True

    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        print("\n  Possible issues:")
        print("    - Ollama not running (run: ollama serve)")
        print("    - Model not pulled (run: ollama pull llama3.2:latest)")
        return False


def main():
    """Run all checks"""
    print("="*60)
    print("Hidden Layer Setup Check")
    print("="*60)

    checks = [
        check_python_version(),
        check_python_packages(),
        check_directories(),
        check_harness(),
        check_ollama(),
        check_mlx(),
        check_env_file(),
    ]

    # Only run test if basic checks pass
    if all(checks):
        test_basic_functionality()

    print("\n" + "="*60)
    if all(checks):
        print("✓ All critical checks passed!")
        print("\nYou're ready to start experimenting!")
        print("\nNext steps:")
        print("  1. Start Ollama: ollama serve &")
        print("  2. Open Jupyter: jupyter notebook notebooks/")
        print("  3. Run: 01_baseline_experiments.ipynb")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nRefer to SETUP.md for detailed installation instructions.")
    print("="*60)


if __name__ == "__main__":
    main()
