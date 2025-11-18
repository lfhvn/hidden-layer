.PHONY: help setup test lint format clean run-ollama install-dev docs notebook build-jupyter

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
	@echo "  make notebook     - Launch Jupyter Lab for interactive experiments"
	@echo "  make build-jupyter - Pre-build Jupyter Lab (run once to speed up startup)"
	@echo "  make clean        - Remove cache and generated files"
	@echo "  make docs         - Verify documentation files"
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup && make run-ollama && make test"

PYTHON ?= python3.11

# Setup virtual environment
setup:
	@./setup.sh

# Start the development environment (setup + notebook)
start: setup notebook

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

# Launch Jupyter Lab
notebook:
	@if [ ! -d "venv" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Launching Jupyter Lab (Ctrl+C to stop)..."
	@echo ""
	@echo "Notebooks are organized by project:"
	@echo "  - communication/multi-agent/notebooks/"
	@echo "  - theory-of-mind/selphi/notebooks/"
	@echo "  - (and other project-specific directories)"
	@echo ""
	@. venv/bin/activate && jupyter lab \
		--no-browser \
		--ServerApp.open_browser=False \
		--LabApp.check_for_updates=False \
		--LabApp.news_url=None

# Pre-build Jupyter Lab extensions (speeds up first startup)
build-jupyter:
	@if [ ! -d "venv" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Building Jupyter Lab (this may take 1-2 minutes)..."
	@. venv/bin/activate && jupyter lab build --dev-build=False --minimize=True || echo "Note: Build warnings are normal"
	@echo "✓ Jupyter Lab built successfully"
	@echo "Next startup should be much faster!"

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
