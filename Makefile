.PHONY: help setup test lint format clean run-ollama install-dev docs notebook

# Default target
help:
	@echo "Hidden Layer - Makefile Commands"
	@echo "================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup        - Set up virtual environment and install dependencies"
	@echo "  make install-dev  - Install development dependencies (testing, linting)"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run all tests"
	@echo "  make test-imports - Run import tests only"
	@echo "  make test-core    - Run core functionality tests"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         - Run linters (flake8)"
	@echo "  make format       - Format code with black and isort"
	@echo "  make format-check - Check formatting without modifying files"
	@echo "  make type-check   - Run mypy type checking"
	@echo ""
	@echo "Utilities:"
	@echo "  make run-ollama   - Start Ollama server in background"
	@echo "  make clean        - Remove cache and generated files"
	@echo "  make docs         - Verify documentation files"
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup && make run-ollama && make test"

PYTHON ?= python3

# Setup virtual environment
setup:
	@echo "Setting up virtual environment..."
	$(PYTHON) -c "import sys; major, minor = sys.version_info[:2]; assert (major, minor) >= (3, 10), 'Python 3.10+ required (found %s.%s)' % (major, minor)"
	@echo "Python version check passed"
	@echo "Note: MLX requires Python 3.10-3.12 on Apple Silicon. If you have Python 3.13+, MLX will be skipped automatically."
	$(PYTHON) -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt || (echo "⚠️  Some optional dependencies (e.g., MLX) may not be available for your Python version." && echo "This is expected for Python 3.13+ or non-Apple Silicon systems." && echo "Continuing with available packages...")
	@echo "✓ Setup complete! Activate with: source venv/bin/activate"
	@echo "Run 'python3 check_setup.py' to verify your installation."

# Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	@echo "✓ Development dependencies installed"

# Run all tests
test:
	@echo "Running all tests..."
	pytest tests/ -v

# Run import tests only
test-imports:
	@echo "Running import tests..."
	pytest tests/test_imports.py -v

# Run core tests only
test-core:
	@echo "Running core tests..."
	pytest tests/test_core.py -v

# Run tests with coverage
test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=code --cov-report=html --cov-report=term
	@echo "✓ Coverage report generated in htmlcov/"

# Lint code
lint:
	@echo "Running linters..."
	flake8 code/ tests/ --max-line-length=120 --exclude=venv,__pycache__,.git --extend-ignore=E203
	@echo "✓ Linting complete"

# Format code
format:
	@echo "Formatting code..."
	black code/ tests/
	isort code/ tests/
	@echo "✓ Code formatted"

# Check formatting without modifying
format-check:
	@echo "Checking code formatting..."
	black --check code/ tests/
	isort --check-only code/ tests/
	@echo "✓ Format check complete"

# Type checking
type-check:
	@echo "Running type checks..."
	mypy code/ --ignore-missing-imports
	@echo "✓ Type checking complete"

# Start Ollama server
run-ollama:
	@echo "Starting Ollama server..."
	@if pgrep ollama > /dev/null; then \
		echo "✓ Ollama already running"; \
	else \
		ollama serve & \
		sleep 2; \
		echo "✓ Ollama server started"; \
	fi

# Launch Jupyter Lab in notebooks/
notebook:
	@if [ ! -d "venv" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Launching Jupyter Lab (Ctrl+C to stop)..."
	@. venv/bin/activate && jupyter lab --notebook-dir=notebooks

# Clean up cache and generated files
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
	@echo "✓ Cleanup complete"

# Verify documentation
docs:
	@echo "Verifying documentation files..."
	@test -f README.md && echo "  ✓ README.md"
	@test -f ARCHITECTURE.md && echo "  ✓ ARCHITECTURE.md"
	@test -f QUICKSTART.md && echo "  ✓ QUICKSTART.md"
	@test -f SETUP.md && echo "  ✓ SETUP.md"
	@test -f CLAUDE.md && echo "  ✓ CLAUDE.md"
	@test -f code/crit/README.md && echo "  ✓ code/crit/README.md"
	@test -f code/selphi/README.md && echo "  ✓ code/selphi/README.md"
	@test -f config/README.md && echo "  ✓ config/README.md"
	@echo "✓ All documentation files present"

# Quick integration test
integration-test: run-ollama test
	@echo "✓ Integration test complete"

# Pre-commit checks
pre-commit: format-check lint test
	@echo "✓ Pre-commit checks passed"

# Full CI pipeline (local)
ci: clean format lint type-check test-cov docs
	@echo "✓ Full CI pipeline complete"
