import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys
import argparse

from r2r_src.read_file import read_img_features
from r2r_src.env import R2RBatch
from r2r_src.vlnbert.vlnbert_init import get_tokenizer


def load_saliency_map(saliency_map_path):
    """Load saliency map from .npy file"""
    if not os.path.exists(saliency_map_path):
        raise FileNotFoundError(f"Saliency map not found: {saliency_map_path}")

    saliency_map = np.load(saliency_map_path)

    # Handle different shapes: [valid_pano, H, W] or [H, W]
    if len(saliency_map.shape) == 3:
        # If it's [valid_pano, H, W], take the first one or average
        # For panorama, you might want to stitch them together
        # For now, we'll take the first valid panorama
        saliency_map = saliency_map[0]

    # Normalize to [0, 1] if values are in [0, 255]
    if saliency_map.max() > 1.0:
        saliency_map = saliency_map / 255.0

    return saliency_map


def load_original_image(image_path):
    """Load original image from file"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Try loading as image file
    if image_path.endswith(".npy"):
        img = np.load(image_path)
        # Handle different shapes
        if len(img.shape) == 3 and img.shape[0] == 3:
            # [C, H, W] -> [H, W, C]
            img = img.transpose(1, 2, 0)
        # Reverse Caffe normalization if needed
        if img.dtype != np.uint8:
            img = img + np.array([[[103.1, 115.9, 123.2]]])  # BGR mean
            img = img.clip(0, 255).astype(np.uint8)
            # Convert BGR to RGB
            img = img[..., ::-1]
    else:
        # Load as regular image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def find_original_image(base_dir, traj_id, t):
    """Try to find original image in common locations"""
    possible_paths = [
        # Check in same directory as saliency map
        os.path.join(base_dir, "saliency_map_pixel", traj_id, str(t), "original.jpg"),
        os.path.join(base_dir, "saliency_map_pixel", traj_id, str(t), "original.png"),
        os.path.join(base_dir, "saliency_map_pixel", traj_id, str(t), "image.npy"),
        # Check in tmp_img directories
        os.path.join("./tmp_img2", traj_id, "0.jpg"),
        os.path.join("./tmp_img", traj_id, "0.jpg"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


def show_saliency_with_original(
    saliency_map_path, original_image_path=None, save_path=None
):
    """
    Display original image and saliency heatmap side by side

    Args:
        saliency_map_path: Path to saliency map .npy file
        original_image_path: Path to original image (optional, will try to find automatically)
        save_path: Path to save the figure (optional)
    """
    # Load saliency map
    saliency_map = load_saliency_map(saliency_map_path)

    # Get directory info for finding original image
    saliency_dir = os.path.dirname(saliency_map_path)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(saliency_dir)))
    traj_id = os.path.basename(os.path.dirname(saliency_dir))
    t = os.path.basename(saliency_dir)

    # Load or find original image
    if original_image_path is None:
        original_image_path = find_original_image(base_dir, traj_id, t)

    if original_image_path is None or not os.path.exists(original_image_path):
        print(f"Warning: Original image not found. Using zeros as placeholder.")
        print(f"Please specify original image path with --image_path")
        # Create placeholder image with same size as saliency map
        original_img = np.zeros(
            (saliency_map.shape[0], saliency_map.shape[1], 3), dtype=np.uint8
        )
    else:
        original_img = load_original_image(original_image_path)
        # Resize to match saliency map if needed
        if original_img.shape[:2] != saliency_map.shape[:2]:
            original_img = cv2.resize(
                original_img, (saliency_map.shape[1], saliency_map.shape[0])
            )

    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Original image
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Right: Saliency heatmap
    im = axes[1].imshow(saliency_map, cmap="jet", interpolation="bilinear")
    axes[1].set_title("Saliency Heatmap", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # Add colorbar for heatmap
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def get_argparser():
    parser = argparse.ArgumentParser(
        description="Visualize saliency map with original image"
    )

    parser.add_argument(
        "--saliency_map", type=str, default=None, help="Path to saliency map .npy file"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to original image (optional, will try to find automatically)",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./snap/VLNBERT-test-baseline-mapgpt-random",
        help="Base directory for snap files",
    )
    parser.add_argument("--traj_id", type=str, default="", help="Trajectory ID")
    parser.add_argument("--t", type=int, default=0, help="Time step")
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the visualization (optional)",
    )
    parser.add_argument(
        "--vlnbert", type=str, default="prevalent", choices=["prevalent", "oscar"]
    )
    return parser


# def create_env():

#     features = "img_features/ResNet-152-places365.tsv"
#     feat_dict = read_img_features(features, test_only=0)
#     return R2RBatch(
#         feat_dict,
#         batch_size=1,
#         splits=["val_unseen"],
#         tokenizer=get_tokenizer(args),
#     )


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()

    # Determine saliency map path
    if args.saliency_map:
        saliency_map_path = args.saliency_map
    else:
        saliency_map_dir = os.path.join(args.base_dir, "saliency_map_pixel")
        saliency_map_path = os.path.join(
            saliency_map_dir, args.traj_id, str(args.t), "attr_map.npy"
        )

    if not os.path.exists(saliency_map_path):
        print(f"Error: Saliency map not found at {saliency_map_path}")
        sys.exit(1)

    show_saliency_with_original(
        saliency_map_path=saliency_map_path,
        original_image_path=args.image_path,
        save_path=args.save_path,
    )
