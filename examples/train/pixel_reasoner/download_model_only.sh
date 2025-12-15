#!/bin/bash
# Download only the Pixel Reasoner model (for quick testing)
# This is useful if you want to start with model verification first

set -e

echo "=========================================="
echo "Downloading Pixel Reasoner Model Only"
echo "=========================================="
echo ""

# Configuration
LOCAL_MODEL_DIR="./models"
MODEL_NAME="VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100"

# Create directory
mkdir -p "$LOCAL_MODEL_DIR"

echo "Model: $MODEL_NAME"
echo "Destination: $LOCAL_MODEL_DIR"
echo "Expected size: ~6GB"
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found!"
    echo "Install with: pip install -U huggingface_hub[cli]"
    echo ""
    echo "Alternative: Use huggingface_hub in Python"
    echo "  from huggingface_hub import snapshot_download"
    echo "  snapshot_download('$MODEL_NAME', local_dir='$LOCAL_MODEL_DIR')"
    exit 1
fi

# Check if HF_TOKEN is set (for private models or faster downloads)
if [ -z "$HF_TOKEN" ]; then
    echo "Note: HF_TOKEN not set. If download fails, try:"
    echo "  export HF_TOKEN=your_huggingface_token"
    echo "  or run: huggingface-cli login"
    echo ""
fi

# Download
MODEL_LOCAL_DIR="$LOCAL_MODEL_DIR/$(echo $MODEL_NAME | tr '/' '_')"

echo "Starting download..."
echo ""

huggingface-cli download "$MODEL_NAME" \
    --repo-type model \
    --local-dir "$MODEL_LOCAL_DIR" \
    --resume-download

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Download Successful!"
    echo "=========================================="
    echo ""
    echo "Model location: $MODEL_LOCAL_DIR"
    echo ""
    echo "To use this model, update your inference script:"
    echo "  model_name=$MODEL_LOCAL_DIR"
    echo ""
    echo "Or export as environment variable:"
    echo "  export PIXEL_REASONER_MODEL=\"$MODEL_LOCAL_DIR\""
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Download Failed"
    echo "=========================================="
    echo ""
    echo "Troubleshooting:"
    echo "1. Check your internet connection"
    echo "2. Try with HuggingFace token: huggingface-cli login"
    echo "3. Use HuggingFace mirror (China): export HF_ENDPOINT=https://hf-mirror.com"
    echo "4. Manual download via browser and extract to $MODEL_LOCAL_DIR"
    echo ""
    exit 1
fi
