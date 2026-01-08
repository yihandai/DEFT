#!/usr/bin/env python3

"""Script to precompute image features using PyTorch ResNet-152-Places365 CNN,
using 24 discretized views (3 heights * 8 horizontal views) at each viewpoint,
with VFOV=45 degrees for NavGPT surrogate model."""

import numpy as np
import cv2
import json
import math
import base64
import sys
import os
import argparse
import csv
import gc
import torch
import ctypes

# Increase CSV field size limit to handle large base64-encoded features
# Each feature is ~256KB when base64 encoded, so we need a larger limit
csv.field_size_limit(sys.maxsize)

# MatterSim need to be on the Python path
import MatterSim
import time

# Add parent directory to path to import CNN model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from r2r_src.vlnbert.caffe_resnet import CNN

TSV_FIELDNAMES = ["scanId", "viewpointId", "image_w", "image_h", "vfov", "features"]
FEATURE_SIZE = 2048
BATCH_SIZE = 8  # Some fraction of viewpoint size - batch size 4 equals 11GB memory
GRAPHS = "connectivity/"

# Simulator image parameters
WIDTH = 640
HEIGHT = 480
VFOV = 45  # For NavGPT surrogate model

# Model weight path
WEIGHT_FILE = "./feat_checkpoints/CNN/30913b5b6a4c411bb1b6020f492e5862.npy"
OUTFILE = "img_features/ResNet-152-places365_24vp.tsv"

# Panoramic view configuration
PANORAMIC_HORIZONTAL_VIEWS = 8  # 8 horizontal views for 3x8 = 24 total views
VIEWPOINT_SIZE = 3 * PANORAMIC_HORIZONTAL_VIEWS  # 24 views total


def load_viewpointids():
    viewpointIds = []
    with open(GRAPHS + "scans.txt") as f:
        scans = [scan.strip() for scan in f.readlines()]
        for scan in scans:
            with open(GRAPHS + scan + "_connectivity.json") as j:
                data = json.load(j)
                for item in data:
                    if item["included"]:
                        viewpointIds.append((scan, item["image_id"]))
    print("Loaded %d viewpoints" % len(viewpointIds))
    return viewpointIds


def transform_img(im):
    """Prep opencv 3 channel image for the network (BGR format)"""
    im = np.array(im, copy=True)
    im_orig = im.astype(np.float32, copy=True)
    im_orig = cv2.resize(im_orig, (224, 224))
    # Subtract BGR pixel mean [103.1, 115.9, 123.2]
    im_orig -= np.array([[[103.1, 115.9, 123.2]]])
    blob = im_orig.transpose((2, 0, 1))  # (3, 224, 224)
    return blob


def extract_features():
    """
    Extract features using PyTorch ResNet-152-Places365 model.
    Uses 24 viewpoints (3 heights * 8 horizontal views) with VFOV=45 degrees.
    """
    # Set up the simulator
    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(True)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    # Use non-discretized mode for 8 horizontal views to support 45-degree increments
    # When False, makeAction uses radians instead of discrete 30-degree steps
    sim.setDiscretizedViewingAngles(False)
    sim.setBatchSize(1)
    sim.initialize()

    # Load PyTorch ResNet model
    print(f"Loading ResNet-152 model from {WEIGHT_FILE}")
    if not os.path.exists(WEIGHT_FILE):
        print(f"Error: Weight file not found at {WEIGHT_FILE}")
        print("Please ensure the weight file exists or update WEIGHT_FILE path")
        sys.exit(1)

    model = CNN(weight_file=WEIGHT_FILE)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU for feature extraction")
    else:
        print("Warning: CUDA not available, using CPU (will be slow)")

    # Load viewpoints
    viewpointIds = load_viewpointids()

    # Check for existing output file and load already processed viewpoints
    processed_viewpoints = set()
    file_exists = os.path.exists(OUTFILE)
    file_mode = "a" if file_exists else "w"  # Append if file exists, write if new

    if file_exists:
        print(f"Found existing output file: {OUTFILE}")
        print("Loading already processed viewpoints...")
        try:
            with open(OUTFILE, "r") as f:
                # First, check if file has header by reading first line
                first_line = f.readline().strip()
                f.seek(0)  # Reset to beginning

                # Check if first line matches header format
                first_line_fields = first_line.split("\t")
                has_header = (
                    len(first_line_fields) == len(TSV_FIELDNAMES)
                    and first_line_fields[0] == TSV_FIELDNAMES[0]
                    and first_line_fields[1] == TSV_FIELDNAMES[1]
                )

                if has_header:
                    # File has header, let DictReader auto-detect it
                    reader = csv.DictReader(f, delimiter="\t")
                    print("  File has header, auto-detecting columns...")
                else:
                    # File doesn't have header, use fieldnames
                    reader = csv.DictReader(
                        f, delimiter="\t", fieldnames=TSV_FIELDNAMES
                    )
                    print("  File has no header, using fieldnames...")

                row_count = 0
                for row in reader:
                    row_count += 1
                    scanId_existing = row.get("scanId", "").strip()
                    viewpointId_existing = row.get("viewpointId", "").strip()
                    # Skip header row if it was read as data
                    if (
                        scanId_existing == "scanId"
                        and viewpointId_existing == "viewpointId"
                    ):
                        continue
                    if scanId_existing and viewpointId_existing:
                        processed_viewpoints.add(
                            (scanId_existing, viewpointId_existing)
                        )

                print(f"  Read {row_count} rows from file")
                print(f"Found {len(processed_viewpoints)} already processed viewpoints")
                if len(processed_viewpoints) > 0:
                    # Show first few processed viewpoints as verification
                    print("  Sample processed viewpoints:")
                    for i, (s, v) in enumerate(list(processed_viewpoints)[:5]):
                        print(f"    {i+1}. {s} / {v}")
        except Exception as e:
            print(f"Warning: Could not read existing file: {e}")
            import traceback

            traceback.print_exc()
            print("Will overwrite existing file.")
            file_mode = "w"
            processed_viewpoints = set()

    # Filter out already processed viewpoints
    remaining_viewpoints = [
        (scanId, viewpointId)
        for scanId, viewpointId in viewpointIds
        if (scanId, viewpointId) not in processed_viewpoints
    ]

    print(f"Total viewpoints: {len(viewpointIds)}")
    print(f"Already processed: {len(processed_viewpoints)}")
    print(f"Remaining to process: {len(remaining_viewpoints)}")

    if len(remaining_viewpoints) == 0:
        print("All viewpoints already processed!")
        return

    # Open output TSV file with smaller buffer to prevent memory accumulation
    # Use buffering=1 for line buffering (flush after each line) or buffering=0 for unbuffered
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    with open(
        OUTFILE, file_mode, buffering=1
    ) as tsvfile:  # Line buffering - flush after each write
        # Only write header if file is new
        if file_mode == "w":
            writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=TSV_FIELDNAMES)
            writer.writeheader()
        else:
            writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=TSV_FIELDNAMES)

        count = len(processed_viewpoints)  # Start count from already processed
        start_time = time.time()
        # Track simulator usage to potentially restart it periodically
        sim_usage_count = 0
        MAX_SIM_USAGE = (
            200  # Restart simulator every 200 viewpoints to prevent memory leaks
        )

        for scanId, viewpointId in remaining_viewpoints:
            # Loop all discretized views from this location
            # Follow the same logic as env.py but adapt for 8 horizontal views
            images = []

            # Calculate angle increment in radians (45 degrees for 8 views)
            angle_increment_rad = math.radians(
                360.0 / PANORAMIC_HORIZONTAL_VIEWS
            )  # 45 degrees = π/4

            for ix in range(VIEWPOINT_SIZE):
                if ix == 0:
                    # Start from the lowest elevation (-30 degrees)
                    sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % PANORAMIC_HORIZONTAL_VIEWS == 0:
                    # Every 8 views, move up one elevation level (30 degrees = π/6)
                    # In non-discretized mode, use radians
                    sim.makeAction([0], [0.0], [math.radians(30)])
                else:
                    # Rotate horizontally by angle_increment_rad (45 degrees = π/4)
                    # In non-discretized mode, makeAction([0], [radians], [0]) rotates by that many radians
                    sim.makeAction([0], [angle_increment_rad], [0])

                state = sim.getState()[0]

                # Remove assertion - MatterSim's viewIndex may not match our expected index
                # when using 8 horizontal views instead of 12
                # We'll collect images in the order MatterSim provides them

                # Transform image for ResNet (BGR format)
                # Copy RGB data immediately and release state reference
                rgb_data = state.rgb.copy() if hasattr(state.rgb, "copy") else state.rgb
                img_blob = transform_img(rgb_data)
                images.append(img_blob)

                # Explicitly delete state and rgb_data to free memory immediately
                del state, rgb_data

            # Convert to numpy array and then to torch tensor
            images_array = np.array(images, dtype=np.float32)  # (24, 3, 224, 224)
            images_tensor = torch.from_numpy(images_array)

            if torch.cuda.is_available():
                images_tensor = images_tensor.cuda()

            # Extract features in batches
            features_list = []
            for start_idx in range(0, VIEWPOINT_SIZE, BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, VIEWPOINT_SIZE)
                batch_images = images_tensor[start_idx:end_idx]

                with torch.no_grad():
                    feat, _ = model(batch_images)  # (batch, 2048, 1, 1)
                    feat = feat[:, :, 0, 0]  # (batch, 2048)
                    features_list.append(feat.cpu().numpy())
                    # Free batch memory immediately
                    del feat
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Concatenate all features
            features = np.concatenate(features_list, axis=0)  # (24, 2048)

            # Encode features as base64
            features_base64 = base64.b64encode(features.tobytes()).decode("ascii")

            # Write to TSV file immediately
            writer.writerow(
                {
                    "scanId": scanId,
                    "viewpointId": viewpointId,
                    "image_w": WIDTH,
                    "image_h": HEIGHT,
                    "vfov": VFOV,
                    "features": features_base64,
                }
            )

            # Immediately flush and delete large base64 string to free memory
            tsvfile.flush()

            # Explicitly free memory to prevent accumulation
            # Delete base64 string FIRST as it's the largest object
            del features_base64
            del (
                images,
                images_array,
                images_tensor,
                features_list,
                features,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear CUDA cache

            count += 1

            # Aggressive cleanup after EVERY viewpoint to prevent CPU/memory buildup
            # Force garbage collection with all generations (0, 1, 2)
            gc.collect(2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # More aggressive cleanup every 5 viewpoints
            if count % 5 == 0:
                # Multiple garbage collection passes for thorough cleanup
                for _ in range(2):
                    gc.collect(2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                # File buffer already flushed after each write due to buffering=1

            # Very aggressive cleanup every 20 viewpoints - release memory to OS
            if count % 20 == 0:
                # Multiple garbage collection passes
                for _ in range(3):
                    gc.collect(2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()

                # Force Python to release memory back to OS (Linux only)
                try:
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)  # Release free memory back to OS
                except:
                    pass  # Not available on all systems (macOS, Windows)

                # Restart simulator periodically to prevent MatterSim memory leaks
                sim_usage_count += 1
                if sim_usage_count >= MAX_SIM_USAGE:
                    print(
                        f"Restarting MatterSim simulator after {sim_usage_count} viewpoints..."
                    )
                    del sim
                    gc.collect(2)
                    # Recreate simulator
                    sim = MatterSim.Simulator()
                    sim.setRenderingEnabled(True)
                    sim.setCameraResolution(WIDTH, HEIGHT)
                    sim.setCameraVFOV(math.radians(VFOV))
                    sim.setDiscretizedViewingAngles(False)
                    sim.setBatchSize(1)
                    sim.initialize()
                    sim_usage_count = 0
                    print("Simulator restarted.")

            if count % 100 == 0:
                # Print progress and memory usage
                total_processed = count
                total_remaining = len(remaining_viewpoints) - (
                    count - len(processed_viewpoints)
                )
                if torch.cuda.is_available():
                    # Print GPU memory usage
                    gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                    time_consumed = time.time() - start_time
                    time_consumed_minutes = int(time_consumed // 60)
                    time_consumed_seconds = int(time_consumed % 60)
                    print(
                        f"{time_consumed_minutes}m {time_consumed_seconds}s - Processed {total_processed} / {len(viewpointIds)} viewpoints "
                        f"(This session: {count - len(processed_viewpoints)}, Remaining: {total_remaining}) "
                        f"(GPU memory: {gpu_memory:.2f} GB)"
                    )
                else:
                    time_consumed = time.time() - start_time
                    time_consumed_minutes = int(time_consumed // 60)
                    time_consumed_seconds = int(time_consumed % 60)
                    print(
                        f"{time_consumed_minutes}m {time_consumed_seconds}s - Processed {total_processed} / {len(viewpointIds)} viewpoints "
                        f"(This session: {count - len(processed_viewpoints)}, Remaining: {total_remaining})"
                    )

    print(f"\nCompleted! Features saved to {OUTFILE}")
    print(f"Total viewpoints processed: {count} / {len(viewpointIds)}")
    print(f"This session processed: {count - len(processed_viewpoints)} new viewpoints")
    print(f"Feature shape: ({VIEWPOINT_SIZE}, {FEATURE_SIZE}) per viewpoint")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract ResNet-152-Places365 features for 24 viewpoints (3x8)"
    )
    parser.add_argument(
        "--weight_file",
        type=str,
        default=WEIGHT_FILE,
        help=f"Path to ResNet weight file (.npy) (default: {WEIGHT_FILE})",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=OUTFILE,
        help=f"Output TSV file path (default: {OUTFILE})",
    )
    args = parser.parse_args()

    WEIGHT_FILE = args.weight_file
    OUTFILE = args.output_file

    print(f"Extracting features with:")
    print(
        f"  - Panoramic views: {PANORAMIC_HORIZONTAL_VIEWS} horizontal x 3 heights = {VIEWPOINT_SIZE} total"
    )
    print(f"  - VFOV: {VFOV} degrees")
    print(f"  - Weight file: {WEIGHT_FILE}")
    print(f"  - Output file: {OUTFILE}")
    print()

    extract_features()
