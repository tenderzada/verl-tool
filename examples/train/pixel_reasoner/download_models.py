#!/usr/bin/env python3
"""
Download Pixel Reasoner models and datasets using huggingface_hub
This script provides better progress tracking and error handling
"""

import os
import sys
from pathlib import Path
from typing import Optional
import argparse

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError
except ImportError:
    print("Error: huggingface_hub not installed!")
    print("Install with: pip install -U huggingface_hub")
    sys.exit(1)

# Configuration
MODEL_REPO = "VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100"

DATASETS = {
    "train": "TIGER-Lab/PixelReasoner-RL-Data",
    "infovqa": "JasperHaozhe/InfoVQA-EvalData-PixelReasoner",
    "tallyqa": "JasperHaozhe/TallyQA-EvalData-PixelReasoner",
    "vstar": "JasperHaozhe/VStar-EvalData-PixelReasoner",
    "mvbench": "JasperHaozhe/MVBench-EvalData-PixelReasoner",
}

def download_model(local_dir: str = "./models", resume: bool = True):
    """Download the Pixel Reasoner model"""
    print("=" * 60)
    print("Downloading Pixel Reasoner Model")
    print("=" * 60)
    print(f"Repository: {MODEL_REPO}")
    print(f"Local directory: {local_dir}")
    print(f"Expected size: ~6GB")
    print("")

    model_local_dir = Path(local_dir) / MODEL_REPO.replace("/", "_")
    model_local_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Starting download...")
        path = snapshot_download(
            repo_id=MODEL_REPO,
            repo_type="model",
            local_dir=str(model_local_dir),
            resume_download=resume,
            local_dir_use_symlinks=False,
        )

        print("")
        print("✓ Model downloaded successfully!")
        print(f"Location: {path}")
        print("")
        return str(model_local_dir)

    except RepositoryNotFoundError:
        print("")
        print(f"✗ Error: Model repository '{MODEL_REPO}' not found!")
        print("Please check the model name or your HuggingFace access token.")
        return None

    except Exception as e:
        print("")
        print(f"✗ Error downloading model: {e}")
        print("")
        print("Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Login to HuggingFace: huggingface-cli login")
        print("3. Try with HF mirror (China): export HF_ENDPOINT=https://hf-mirror.com")
        return None

def download_dataset(dataset_name: str, local_dir: str = "./data/pixel_reasoner", resume: bool = True):
    """Download a specific dataset"""
    if dataset_name not in DATASETS:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print(f"Available datasets: {', '.join(DATASETS.keys())}")
        return None

    repo_id = DATASETS[dataset_name]

    print("=" * 60)
    print(f"Downloading {dataset_name.upper()} Dataset")
    print("=" * 60)
    print(f"Repository: {repo_id}")
    print(f"Local directory: {local_dir}")
    print("")

    dataset_local_dir = Path(local_dir) / repo_id.replace("/", "_")
    dataset_local_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Starting download...")
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dataset_local_dir),
            resume_download=resume,
            local_dir_use_symlinks=False,
        )

        print("")
        print(f"✓ Dataset downloaded successfully!")
        print(f"Location: {path}")
        print("")
        return str(dataset_local_dir)

    except Exception as e:
        print("")
        print(f"✗ Error downloading dataset: {e}")
        return None

def download_all(
    model_dir: str = "./models",
    data_dir: str = "./data/pixel_reasoner",
    skip_model: bool = False,
    skip_datasets: bool = False,
    datasets: Optional[list] = None,
):
    """Download model and all datasets"""
    results = {
        "model": None,
        "datasets": {}
    }

    # Download model
    if not skip_model:
        model_path = download_model(model_dir)
        results["model"] = model_path
    else:
        print("Skipping model download (--skip-model)")
        print("")

    # Download datasets
    if not skip_datasets:
        datasets_to_download = datasets if datasets else list(DATASETS.keys())

        for dataset_name in datasets_to_download:
            dataset_path = download_dataset(dataset_name, data_dir)
            results["datasets"][dataset_name] = dataset_path
    else:
        print("Skipping dataset download (--skip-datasets)")
        print("")

    # Print summary
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)
    print("")

    if results["model"]:
        print(f"✓ Model: {results['model']}")
    elif not skip_model:
        print("✗ Model: Failed")

    print("")
    print("Datasets:")
    for name, path in results["datasets"].items():
        status = "✓" if path else "✗"
        print(f"  {status} {name}: {path or 'Failed'}")

    # Create local_paths.sh
    if results["model"] or results["datasets"]:
        create_config_file(results, model_dir, data_dir)

    print("")
    print("=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Load paths: source local_paths.sh")
    print("2. Process datasets (if needed)")
    print("3. Update inference scripts to use local paths")
    print("")
    print("See LOCAL_SETUP.md for detailed instructions")
    print("")

    return results

def create_config_file(results: dict, model_dir: str, data_dir: str):
    """Create a shell script with local paths"""
    config_file = Path("local_paths.sh")

    with open(config_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Local paths configuration for Pixel Reasoner\n")
        f.write(f"# Generated by download_models.py\n\n")

        if results["model"]:
            f.write(f"export PIXEL_REASONER_MODEL=\"{results['model']}\"\n\n")

        for name, path in results["datasets"].items():
            if path:
                var_name = f"{name.upper()}_DATA"
                f.write(f"export {var_name}=\"{path}\"\n")

        f.write("\n")
        f.write("echo \"Pixel Reasoner Local Paths:\"\n")
        if results["model"]:
            f.write("echo \"  Model: $PIXEL_REASONER_MODEL\"\n")
        for name in results["datasets"].keys():
            var_name = f"{name.upper()}_DATA"
            f.write(f"echo \"  {name}: ${var_name}\"\n")

    config_file.chmod(0o755)
    print(f"\n✓ Configuration saved to: {config_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Download Pixel Reasoner models and datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download everything
  python download_models.py

  # Download only model
  python download_models.py --skip-datasets

  # Download only specific datasets
  python download_models.py --skip-model --datasets infovqa tallyqa

  # Custom directories
  python download_models.py --model-dir ./my_models --data-dir ./my_data
        """
    )

    parser.add_argument(
        "--model-dir",
        default="./models",
        help="Directory to save models (default: ./models)"
    )
    parser.add_argument(
        "--data-dir",
        default="./data/pixel_reasoner",
        help="Directory to save datasets (default: ./data/pixel_reasoner)"
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model download"
    )
    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Skip datasets download"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        help="Specific datasets to download (default: all)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume partial downloads"
    )

    args = parser.parse_args()

    # Check HF_ENDPOINT for China users
    hf_endpoint = os.getenv("HF_ENDPOINT")
    if hf_endpoint:
        print(f"Using HuggingFace endpoint: {hf_endpoint}")
        print("")

    download_all(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        skip_model=args.skip_model,
        skip_datasets=args.skip_datasets,
        datasets=args.datasets,
    )

if __name__ == "__main__":
    main()
