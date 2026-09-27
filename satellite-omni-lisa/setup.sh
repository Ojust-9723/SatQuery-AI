#!/bin/bash
# Setup script for LISA Satellite Omni Model training
# Handles the transformers version requirement automatically.

set -e

echo "=== LISA Satellite Omni Model - Setup ==="

# Check if transformers has Qwen3.5 support
python -c "from transformers import Qwen3_5ForConditionalGeneration; print('✅ transformers already has Qwen3.5 support')" 2>/dev/null || {
    echo "⚠️  Installed transformers lacks Qwen3.5 support. Installing from git main..."
    pip install "transformers @ git+https://github.com/huggingface/transformers.git"
}

# Install remaining deps
pip install -r requirements.txt

echo ""
echo "=== Setup complete! Run training with: ==="
echo "  python train.py"
echo ""
echo "For custom config, edit the Configuration section at the top of train.py."
