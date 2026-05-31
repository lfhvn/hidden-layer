.PHONY: help setup test test-offline lint format format-check type-check test-cov clean run-ollama install-dev docs notebook build-jupyter papers ci pre-commit integration-test

# Invoke pytest through the interpreter so it uses the env that has the deps
# (a bare `pytest` console script may point at a different Python).
PYTHON ?= python3

# Offline, deterministic test suites (no API keys, no network) — used by CI.
# Web backends (torch/fastapi) and the stale multi-agent suite are intentionally excluded.
OFFLINE_TESTS := tests theory-of-mind/metacognition/tests agentmesh/tests communication/multi-agent/tests theory-of-mind/selphi/tests memory/lifelog-personalization/tests ai_research_aggregator/tests mlx_lab/tests
# Dirs kept formatting-clean (black/isort/flake8 gate). Uses the repo .flake8 config.
LINT_PATHS := harness tests theory-of-mind/metacognition agentmesh/tests communication/multi-agent/tests theory-of-mind/selphi/tests memory/lifelog-personalization/tests ai_research_aggregator/tests mlx_lab/tests papers/generate_results.py scripts/new_project.py

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
	@echo "CI & Papers (offline, no API keys):"
	@echo "  make ci           - Full local CI (format-check, lint, test, papers, docs)"
	@echo "  make papers       - Regenerate & verify bound-paper results"
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
	@echo "Running offline test suites..."
	DEFAULT_PROVIDER=sim $(PYTHON) -m pytest $(OFFLINE_TESTS) -v

test-offline: test

# Run import tests only
test-imports:
	@echo "Running import tests..."
	$(PYTHON) -m pytest tests/test_imports.py -v

# Run core tests only
test-core:
	@echo "Running core tests..."
	$(PYTHON) -m pytest tests/test_core.py -v

# Run tests with coverage
test-cov:
	@echo "Running offline tests with coverage..."
	DEFAULT_PROVIDER=sim $(PYTHON) -m pytest $(OFFLINE_TESTS) --cov=harness --cov-report=term
	@echo "✓ Coverage report generated"

# Lint code (uses the repo .flake8 config)
lint:
	@echo "Running linters..."
	flake8 $(LINT_PATHS)
	flake8 --select=E9,F63,F7,F82 --exclude=node_modules,.git,venv,build,dist .
	@echo "✓ Linting complete"

# Format code
format:
	@echo "Formatting code..."
	black --line-length 120 $(LINT_PATHS)
	isort --profile black --line-length 120 $(LINT_PATHS)
	@echo "✓ Code formatted"

# Check formatting without modifying
format-check:
	@echo "Checking code formatting..."
	black --check --line-length 120 $(LINT_PATHS)
	isort --check-only --profile black --line-length 120 $(LINT_PATHS)
	@echo "✓ Format check complete"

# Type checking
type-check:
	@echo "Running type checks..."
	mypy harness --ignore-missing-imports
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
	@test -f QUICKSTART.md && echo "  ✓ QUICKSTART.md"
	@test -f RUNBOOK.md && echo "  ✓ RUNBOOK.md"
	@test -f CLAUDE.md && echo "  ✓ CLAUDE.md"
	@test -f RESEARCH.md && echo "  ✓ RESEARCH.md"
	@test -f FAQ.md && echo "  ✓ FAQ.md"
	@test -f harness/README.md && echo "  ✓ harness/README.md"
	@test -f config/README.md && echo "  ✓ config/README.md"
	@test -f docs/ARCHITECTURE.md && echo "  ✓ docs/ARCHITECTURE.md"
	@echo "✓ All documentation files present"

papers:
	@$(MAKE) -C papers check

# Quick integration test
integration-test: run-ollama test
	@echo "✓ Integration test complete"

# Pre-commit checks
pre-commit: format-check lint test
	@echo "✓ Pre-commit checks passed"

# Full CI pipeline (local) — mirrors .github/workflows/ci.yml, fully offline.
ci: format-check lint test papers docs
	@echo "✓ Full CI pipeline complete"
