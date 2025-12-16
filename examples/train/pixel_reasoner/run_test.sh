#!/bin/bash
# Wrapper script to run test with correct environment

echo "Activating conda environment..."

# Try to find conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi

# Activate environment
if command -v conda &> /dev/null; then
    conda activate verl-tool-env 2>/dev/null || conda activate base
    echo "✓ Conda environment activated: $(conda info --envs | grep '*' | awk '{print $1}')"
else
    echo "Warning: conda not found, using system Python"
fi

# Run the test
echo ""
python3 examples/train/pixel_reasoner/test_single_sample.py
