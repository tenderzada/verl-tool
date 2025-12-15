#!/bin/bash
# Download Pixel Reasoner Model and Datasets
# This script downloads all required models and datasets to local directories

set -e

echo "=========================================="
echo "Pixel Reasoner - Model & Data Download"
echo "=========================================="
echo ""

# Configuration
LOCAL_MODEL_DIR="./models"
LOCAL_DATA_DIR="./data/pixel_reasoner"

# Model to download
MODEL_NAME="VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100"

# Datasets to download
DATASETS=(
    "TIGER-Lab/PixelReasoner-RL-Data"
    "JasperHaozhe/InfoVQA-EvalData-PixelReasoner"
    "JasperHaozhe/TallyQA-EvalData-PixelReasoner"
    "JasperHaozhe/VStar-EvalData-PixelReasoner"
    "JasperHaozhe/MVBench-EvalData-PixelReasoner"
)

# Create directories
mkdir -p "$LOCAL_MODEL_DIR"
mkdir -p "$LOCAL_DATA_DIR"

echo "Download directories:"
echo "  Models: $LOCAL_MODEL_DIR"
echo "  Datasets: $LOCAL_DATA_DIR"
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found!"
    echo "Please install it with: pip install -U huggingface_hub[cli]"
    exit 1
fi

# Function to download with retry
download_with_retry() {
    local repo_id=$1
    local repo_type=$2
    local local_dir=$3
    local max_retries=3
    local retry_count=0

    while [ $retry_count -lt $max_retries ]; do
        echo "Downloading $repo_id (attempt $((retry_count + 1))/$max_retries)..."

        if huggingface-cli download "$repo_id" \
            --repo-type "$repo_type" \
            --local-dir "$local_dir" \
            --resume-download; then
            echo "✓ Successfully downloaded $repo_id"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                echo "Failed, retrying in 5 seconds..."
                sleep 5
            fi
        fi
    done

    echo "✗ Failed to download $repo_id after $max_retries attempts"
    return 1
}

# Download model
echo "=========================================="
echo "Step 1: Downloading Model"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "This may take a while (model size: ~6GB)"
echo ""

MODEL_LOCAL_DIR="$LOCAL_MODEL_DIR/$(echo $MODEL_NAME | tr '/' '_')"
download_with_retry "$MODEL_NAME" "model" "$MODEL_LOCAL_DIR"

echo ""
echo "Model downloaded to: $MODEL_LOCAL_DIR"
echo ""

# Download datasets
echo "=========================================="
echo "Step 2: Downloading Datasets"
echo "=========================================="
echo ""

for dataset in "${DATASETS[@]}"; do
    echo "Dataset: $dataset"
    DATASET_LOCAL_DIR="$LOCAL_DATA_DIR/$(echo $dataset | tr '/' '_')"

    download_with_retry "$dataset" "dataset" "$DATASET_LOCAL_DIR"

    echo ""
done

# Create a configuration file with local paths
CONFIG_FILE="./local_paths.sh"
echo "=========================================="
echo "Step 3: Creating Configuration File"
echo "=========================================="
echo ""

cat > "$CONFIG_FILE" <<EOF
#!/bin/bash
# Local paths configuration for Pixel Reasoner
# Generated on $(date)

# Model path
export PIXEL_REASONER_MODEL="$MODEL_LOCAL_DIR"

# Dataset paths (raw downloads)
export PIXELREASONER_RL_DATA="$LOCAL_DATA_DIR/TIGER-Lab_PixelReasoner-RL-Data"
export INFOVQA_DATA="$LOCAL_DATA_DIR/JasperHaozhe_InfoVQA-EvalData-PixelReasoner"
export TALLYQA_DATA="$LOCAL_DATA_DIR/JasperHaozhe_TallyQA-EvalData-PixelReasoner"
export VSTAR_DATA="$LOCAL_DATA_DIR/JasperHaozhe_VStar-EvalData-PixelReasoner"
export MVBENCH_DATA="$LOCAL_DATA_DIR/JasperHaozhe_MVBench-EvalData-PixelReasoner"

# Print paths
echo "Model path: \$PIXEL_REASONER_MODEL"
echo "InfoVQA: \$INFOVQA_DATA"
echo "TallyQA: \$TALLYQA_DATA"
echo "VStar: \$VSTAR_DATA"
echo "MVBench: \$MVBENCH_DATA"
EOF

chmod +x "$CONFIG_FILE"

echo "✓ Configuration saved to: $CONFIG_FILE"
echo ""

# Summary
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Model: $MODEL_LOCAL_DIR"
echo "  Datasets: $LOCAL_DATA_DIR/"
echo ""
echo "Next steps:"
echo "1. Load paths: source $CONFIG_FILE"
echo "2. Process datasets with examples/data_preprocess/pixel_reasoner/*.py"
echo "3. Update inference scripts to use local model path"
echo ""
echo "See: examples/train/pixel_reasoner/LOCAL_SETUP.md for detailed instructions"
echo "=========================================="
