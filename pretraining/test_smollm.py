#!/usr/bin/env python3
"""
Quick test script for pretraining_advanced.py with smollm-corpus.
This runs a minimal training to verify the script works correctly.
"""

import subprocess
import sys
import os


def run_test():
    """Run a quick test of the training script."""

    print("=" * 80)
    print("Testing pretraining_advanced.py with smollm-corpus")
    print("=" * 80)
    print()

    # Check if datasets is installed
    try:
        import datasets  # noqa: F401

        print("datasets library is installed")
    except ImportError:
        print("datasets library not found")
        print("  Install with: pip install datasets")
        return False

    # Check if torch is installed
    try:
        import torch

        print(f"PyTorch {torch.__version__} is installed")
    except ImportError:
        print("PyTorch not found")
        print("  Install with: pip install torch")
        return False

    # Check if tiktoken is installed
    try:
        import tiktoken  # noqa: F401

        print("tiktoken is installed")
    except ImportError:
        print("tiktoken not found")
        print("  Install with: pip install tiktoken")
        return False

    print()
    print("Running minimal test with 50 samples...")
    print("-" * 80)

    # Run the training script with minimal settings
    cmd = [
        sys.executable,
        "pretraining_advanced.py",
        "--debug",  # Use small model
        "--dataset_name",
        "HuggingFaceTB/smollm-corpus",
        "--dataset_subset",
        "cosmopedia-v2",
        "--max_samples",
        "50",  # Very small for quick test
        "--batch_size",
        "2",
        "--n_epochs",
        "1",
        "--eval_freq",
        "5",  # Frequent evaluation
        "--print_sample_iter",
        "10",
        "--cache_dir",
        "test_cache",
        "--output_dir",
        "test_checkpoints",
    ]

    try:
        subprocess.run(
            cmd, cwd=os.path.dirname(os.path.abspath(__file__)), check=True, text=True
        )

        print("-" * 80)
        print("Test completed successfully!")
        print()
        print("Next steps:")
        print("1. Check test_cache/ for cached tokenized data")
        print("2. Check test_checkpoints/ for model checkpoints")
        print("3. Run with larger --max_samples for real training")
        print()
        print("Clean up test files with:")
        print("  rm -rf test_cache/ test_checkpoints/")

        return True

    except subprocess.CalledProcessError as e:
        print("-" * 80)
        print(f"Test failed with error code {e.returncode}")
        print()
        print("Check the error messages above for details.")
        return False

    except KeyboardInterrupt:
        print()
        print("Test interrupted by user")
        return False


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
