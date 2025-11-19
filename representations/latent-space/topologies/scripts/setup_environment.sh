#!/bin/bash
# Setup script for Latent Topologies development environment
# This installs all Python dependencies needed for data pipeline

set -e  # Exit on error

echo "=========================================="
echo "Latent Topologies - Environment Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
echo ""

# Check if in topologies directory
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: Please run this script from the topologies directory"
    exit 1
fi

echo "Installing Python dependencies..."
echo "This will download ~2.3GB of packages (PyTorch + CUDA libraries)"
echo "Estimated time: 10-15 minutes depending on network speed"
echo ""

# Install with verbose output and longer timeout
pip install --timeout 2000 --verbose \
    sentence-transformers \
    umap-learn \
    2>&1 | tee /tmp/topologies_install.log

echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="

# Verify imports
python3 << 'EOF'
import sys
try:
    import numpy
    print("✓ numpy installed")
    import pandas
    print("✓ pandas installed")
    import sklearn
    print("✓ scikit-learn installed")
    import sentence_transformers
    print("✓ sentence-transformers installed")
    import umap
    print("✓ umap-learn installed")
    print("\n✓✓✓ ALL DEPENDENCIES INSTALLED SUCCESSFULLY! ✓✓✓")
    sys.exit(0)
except ImportError as e:
    print(f"\n✗ Installation incomplete: {e}")
    sys.exit(1)
EOF

INSTALL_STATUS=$?

if [ $INSTALL_STATUS -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Generate embeddings:"
    echo "     python3 scripts/embed_corpus.py --input data/corpus_2k.csv --embeddings data/embeddings_2k.npy --coords data/coords_2k.npy --umap --save-umap-model data/umap_model_2k.pkl"
    echo ""
    echo "  2. Build database:"
    echo "     python3 scripts/build_seed_db.py --corpus data/corpus_2k.csv --embeddings data/embeddings_2k.npy --coords data/coords_2k.npy --out data/seed.db"
    echo ""
    echo "Full log available at: /tmp/topologies_install.log"
else
    echo ""
    echo "=========================================="
    echo "Installation Failed"
    echo "=========================================="
    echo "Check the log at: /tmp/topologies_install.log"
    exit 1
fi
