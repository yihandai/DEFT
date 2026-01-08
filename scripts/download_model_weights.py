#!/usr/bin/env python3
"""
Script to download BLIP-2 and Fast-RCNN model weights to local directory.
This avoids network issues on remote servers.
"""

import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "r2r_src"))

from vlnbert.feature_level_eval_utils import (
    BLIP2Captioner,
    FastRCNNDetector,
    DEFAULT_MODEL_CACHE_DIR,
)


def download_blip2(cache_dir=None):
    """Download BLIP-2 model weights with resume support."""
    print("=" * 60)
    print("Downloading BLIP-2 model weights...")
    print("=" * 60)

    cache_dir = cache_dir or DEFAULT_MODEL_CACHE_DIR
    print(f"Cache directory: {cache_dir}")

    try:
        captioner = BLIP2Captioner(cache_dir=cache_dir)

        # 检查是否已经部分下载
        model_path = captioner.model_path
        if os.path.exists(model_path):
            # 检查关键文件是否存在
            config_file = os.path.join(model_path, "config.json")
            if os.path.exists(config_file):
                print(f"  Model files already exist at: {model_path}")
                print("  Checking if download is complete...")
                # 检查是否所有必要的文件都存在
                required_files = [
                    "config.json",
                    "pytorch_model.bin",
                    "tokenizer_config.json",
                ]
                missing_files = []
                for file in required_files:
                    file_path = os.path.join(model_path, file)
                    # 检查文件或对应的分片文件
                    if not os.path.exists(file_path):
                        # 检查是否有分片文件（.bin 文件可能是分片的）
                        bin_files = [
                            f for f in os.listdir(model_path) if f.endswith(".bin")
                        ]
                        if not bin_files and file == "pytorch_model.bin":
                            missing_files.append(file)
                        elif file != "pytorch_model.bin":
                            missing_files.append(file)

                if not missing_files:
                    print("  ✓ Model files are complete, skipping download.")
                    return True
                else:
                    print(f"  ⚠ Some files are missing: {missing_files}")
                    print("  Resuming download...")
            else:
                print("  ⚠ Incomplete download detected, resuming...")

        # 初始化会触发下载（transformers 库自动支持断点续传）
        print("  Starting download (resume supported)...")
        captioner._initialize()
        print("✓ BLIP-2 model downloaded successfully!")
        print(f"  Model saved to: {captioner.model_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download BLIP-2 model: {e}")
        print("\n  Note: If download was interrupted, you can run this script again")
        print("  to resume the download. transformers library supports resume.")
        import traceback

        traceback.print_exc()
        return False


def download_faster_rcnn(cache_dir=None):
    """Download Fast-RCNN model weights with resume support."""
    print("\n" + "=" * 60)
    print("Downloading Fast-RCNN model weights...")
    print("=" * 60)

    cache_dir = cache_dir or DEFAULT_MODEL_CACHE_DIR
    print(f"Cache directory: {cache_dir}")

    try:
        detector = FastRCNNDetector(cache_dir=cache_dir)

        # 检查是否已经部分下载
        torch_home = os.path.join(cache_dir, "torchvision")
        if os.path.exists(torch_home):
            # 检查 torchvision 缓存目录
            hub_dir = os.path.join(torch_home, "hub", "checkpoints")
            if os.path.exists(hub_dir):
                # 查找已下载的模型文件
                model_files = [f for f in os.listdir(hub_dir) if f.endswith(".pth")]
                if model_files:
                    print(f"  Found existing model files in: {hub_dir}")
                    print("  Checking if download is complete...")
                    # torchvision 的模型文件通常是一个 .pth 文件
                    # 如果文件大小合理（> 100MB），认为下载完成
                    for f in model_files:
                        file_path = os.path.join(hub_dir, f)
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        if size_mb > 100:  # Fast-RCNN 模型通常 > 100MB
                            print(f"  ✓ Model file exists: {f} ({size_mb:.1f} MB)")
                            print("  Model appears to be complete, skipping download.")
                            # 仍然需要初始化以加载模型
                            detector._initialize()
                            return True
                    print("  ⚠ Model files seem incomplete, resuming download...")

        # 初始化会触发下载（torchvision 使用标准 HTTP，支持断点续传）
        print("  Starting download (resume supported)...")
        detector._initialize()
        print("✓ Fast-RCNN model downloaded successfully!")
        print(f"  Model saved to: {torch_home}")
        return True
    except Exception as e:
        print(f"✗ Failed to download Fast-RCNN model: {e}")
        print("\n  Note: If download was interrupted, you can run this script again")
        print("  to resume the download.")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download BLIP-2 and Fast-RCNN model weights"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        # default=None,
        default=os.path.join(project_root, "model_checkpoints"),
        help=f"Directory to save model weights (default: {DEFAULT_MODEL_CACHE_DIR})",
    )
    parser.add_argument(
        "--blip2_only",
        action="store_true",
        help="Only download BLIP-2 model",
    )
    parser.add_argument(
        "--faster_rcnn_only",
        action="store_true",
        help="Only download Fast-RCNN model",
    )

    args = parser.parse_args()

    cache_dir = args.cache_dir or DEFAULT_MODEL_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\nModel weights will be saved to: {cache_dir}\n")

    success_count = 0
    total_count = 0

    # Download BLIP-2
    if not args.faster_rcnn_only:
        total_count += 1
        if download_blip2(cache_dir):
            success_count += 1

    # Download Fast-RCNN
    if not args.blip2_only:
        total_count += 1
        if download_faster_rcnn(cache_dir):
            success_count += 1

    print("\n" + "=" * 60)
    print(
        f"Download Summary: {success_count}/{total_count} models downloaded successfully"
    )
    print("=" * 60)

    if success_count == total_count:
        print("\n✓ All models downloaded successfully!")
        print(f"\nYou can now use the models offline from: {cache_dir}")
        return 0
    else:
        print("\n✗ Some models failed to download. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
