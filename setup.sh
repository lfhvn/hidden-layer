#!/bin/bash
# Quick setup script for M4 Max Research Lab

set -e

echo "🚀 Setting up Research Lab environment..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This setup is optimized for macOS (M4 Max)"
fi

# Determine python executable
PYTHON_BIN=${PYTHON:-python3}

if ! command -v "$PYTHON_BIN" > /dev/null 2>&1; then
    echo "✗ Python executable '$PYTHON_BIN' not found. Install Python 3.10+ (e.g., 'brew install python@3.11')."
    exit 1
fi

# Verify version
PY_MAJOR=$($PYTHON_BIN -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$($PYTHON_BIN -c 'import sys; print(sys.version_info.minor)')
PY_VERSION="${PY_MAJOR}.${PY_MINOR}"

if [ "$PY_MAJOR" -lt 3 ] || [ "$PY_MAJOR" -gt 3 ]; then
    echo "✗ Python $PY_VERSION detected. Hidden Layer requires Python 3.10–3.12."
    exit 1
fi

if [ "$PY_MINOR" -lt 10 ] || [ "$PY_MINOR" -gt 12 ]; then
    echo "✗ Python $PY_VERSION detected. Hidden Layer currently supports Python 3.10–3.12 to match MLX wheels."
    echo "  Install a supported Python (e.g., 'brew install python@3.11') and re-run with:"
    echo "    PYTHON=python3.11 ./setup.sh"
    exit 1
fi

echo "📦 Creating virtual environment with $PYTHON_BIN..."
$PYTHON_BIN -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python packages..."
pip install -r requirements.txt

echo ""
echo "✅ Python environment ready!"
echo ""

# Check for Ollama
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is installed"
else
    echo "⚠️  Ollama not found. Install it with:"
    echo "    brew install ollama"
fi

echo ""
echo "🎯 Next steps:"
echo "1. Start Ollama: ollama serve &"
echo "2. Pull a model: ollama pull llama3.2:latest"
echo "3. Test the CLI: cd code && python cli.py 'Hello!' --strategy single"
echo "4. Start Jupyter: cd notebooks && jupyter notebook"
echo ""
echo "📖 Read QUICKSTART.md for more examples"
