#!/bin/bash
# Quick setup script for M4 Max Research Lab

set -e

echo "🚀 Setting up Research Lab environment..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This setup is optimized for macOS (M4 Max)"
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
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
