#!/bin/bash
# Unified Setup Script for Hidden Layer
# Handles: Python check, venv creation, dependencies, Ollama setup, model pulling

set -e

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Hidden Layer Setup${NC}"
echo "================================="

# --- 1. Python Version Check ---
echo -e "\n${BLUE}[1/5] Checking Python environment...${NC}"

PYTHON_BIN=${PYTHON:-python3}

if ! command -v "$PYTHON_BIN" > /dev/null 2>&1; then
    echo -e "${RED}✗ Python executable '$PYTHON_BIN' not found.${NC}"
    echo "Please install Python 3.10+ (e.g., 'brew install python@3.11') or set PYTHON variable."
    exit 1
fi

PY_VERSION=$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$($PYTHON_BIN -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$($PYTHON_BIN -c 'import sys; print(sys.version_info.minor)')

echo "Found Python $PY_VERSION at $(which $PYTHON_BIN)"

if [ "$PY_MAJOR" -lt 3 ] || [ "$PY_MINOR" -lt 10 ]; then
    echo -e "${RED}✗ Python 3.10+ is required.${NC}"
    exit 1
fi

# --- 2. Virtual Environment ---
echo -e "\n${BLUE}[2/5] Setting up virtual environment...${NC}"

if [ ! -d "venv" ]; then
    echo "Creating venv..."
    "$PYTHON_BIN" -m venv venv
else
    echo "venv already exists."
fi

# Activate venv for the script
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --quiet --upgrade pip

# --- 3. Dependencies ---
echo -e "\n${BLUE}[3/5] Installing dependencies...${NC}"
echo "This may take a minute."

# Check for Apple Silicon for MLX warning
if [[ "$(uname -m)" == "arm64" && "$OSTYPE" == "darwin"* ]]; then
    if [ "$PY_MINOR" -gt 12 ]; then
         echo -e "${YELLOW}⚠️  Python $PY_VERSION detected on Apple Silicon.${NC}"
         echo "   MLX requires Python 3.10-3.12. MLX installation will be skipped."
    fi
fi

pip install -r requirements.txt

echo -e "${GREEN}✓ Dependencies installed.${NC}"

# --- 4. Ollama Setup ---
echo -e "\n${BLUE}[4/5] Checking Ollama...${NC}"

if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Ollama not found.${NC}"
    read -p "Install Ollama via Homebrew? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo -e "${RED}Homebrew not found. Please install Ollama manually: https://ollama.ai${NC}"
        fi
    else
        echo "Skipping Ollama installation. Some features may not work."
    fi
else
    echo -e "${GREEN}✓ Ollama is installed.${NC}"
fi

# Start Ollama if not running
if command -v ollama &> /dev/null; then
    if ! pgrep -x "ollama" > /dev/null; then
        echo "Starting Ollama server in background..."
        ollama serve > /dev/null 2>&1 &
        sleep 2
    else
        echo "Ollama server is already running."
    fi
fi

# --- 5. Model Setup ---
echo -e "\n${BLUE}[5/5] Checking Models...${NC}"

DEFAULT_MODEL="llama3.2:latest"

if command -v ollama &> /dev/null; then
    if ollama list | grep -q "llama3.2"; then
        echo -e "${GREEN}✓ Model $DEFAULT_MODEL found.${NC}"
    else
        echo -e "${YELLOW}Model $DEFAULT_MODEL not found.${NC}"
        read -p "Pull $DEFAULT_MODEL now? (Recommended) (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Pulling model (this requires internet)..."
            ollama pull "$DEFAULT_MODEL"
        fi
    fi
fi

echo -e "\n${GREEN}=================================${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${GREEN}=================================${NC}"
echo ""
echo "To start the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the quickstart notebook:"
echo "  make notebook"
echo ""
