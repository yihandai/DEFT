#!/usr/bin/env python3

"""Script to precompute image features using a Caffe ResNet CNN, using discretized views
at each viewpoint, and the provided camera WIDTH, HEIGHT and VFOV parameters."""

import numpy as np
import cv2
import json
import math
import base64
import sys
import os
import argparse

# Caffe and MatterSim need to be on the Python path
import MatterSim

caffe_root = "../"  # your caffe build

from timer import Timer


TSV_FIELDNAMES = ["scanId", "viewpointId", "image_w", "image_h", "vfov", "features"]
FEATURE_SIZE = 2048
BATCH_SIZE = 4  # Some fraction of viewpoint size - batch size 4 equals 11GB memory
GPU_ID = 0
PROTO = "models/ResNet-152-deploy.prototxt"
MODEL = "models/ResNet-152-model.caffemodel"  # You need to download this, see README.md
# MODEL = 'models/resnet152_places365.caffemodel'
OUTFILE = "img_features/ResNet-152-imagenet.tsv"
GRAPHS = "connectivity/"

# Simulator image parameters
WIDTH = 640
HEIGHT = 480
# VFOV will be set via command line argument or default to 60


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


def build_training_set(panoramic_horizontal_views=12, vfov=60):
    """
    Build training set with configurable panoramic views.
    Args:
        panoramic_horizontal_views: Number of horizontal views (default: 12 for 3x12, use 8 for 3x8)
        vfov: Vertical field of view in degrees (default: 60, use 45 for NavGPT surrogate model)
    """
    VIEWPOINT_SIZE = 3 * panoramic_horizontal_views  # 3 heights * horizontal views
    angle_increment = 360.0 / panoramic_horizontal_views

    # Set up the simulator
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(vfov))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

    count = 0
    t_render = Timer()
    t_net = Timer()

    # Loop all the viewpoints in the simulator
    viewpointIds = load_viewpointids()
    for scanId, viewpointId in viewpointIds:
        t_render.tic()
        # Loop all discretized views from this location
        features = np.empty([VIEWPOINT_SIZE, FEATURE_SIZE], dtype=np.float32)
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
            elif ix % panoramic_horizontal_views == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])

            state = sim.getState()[0]
            assert state.viewIndex == ix
            # save state.rgb under folder `RGB_train/{scanId}/{viewpointID}`
            save_dir = os.path.join("RGB_train", scanId, viewpointId)
            os.makedirs(save_dir, exist_ok=True)
            image_path = os.path.join(save_dir, f"{ix:02d}.jpg")
            cv2.imwrite(image_path, cv2.cvtColor(state.rgb, cv2.COLOR_RGB2BGR))

        t_render.toc()
        t_net.tic()
        # Run as many forward passes as necessary
        ix = 0
        count += 1
        t_net.toc()
        if count % 100 == 0:
            print(
                "Processed %d / %d viewpoints, %.1fs avg render time, %.1fs avg net time, projected %.1f hours"
                % (
                    count,
                    len(viewpointIds),
                    t_render.average_time,
                    t_net.average_time,
                    (t_render.average_time + t_net.average_time)
                    * len(viewpointIds)
                    / 3600,
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract RGB images for panoramic views"
    )
    parser.add_argument(
        "--panoramic_horizontal_views",
        type=int,
        default=12,
        help="Number of horizontal views in panoramic image (default: 12 for 3x12, use 8 for 3x8)",
    )
    parser.add_argument(
        "--vfov",
        type=float,
        default=60,
        help="Vertical field of view in degrees (default: 60, use 45 for NavGPT surrogate model)",
    )
    args = parser.parse_args()

    print(
        f"Using {args.panoramic_horizontal_views} horizontal views (total: {3 * args.panoramic_horizontal_views} views)"
    )
    print(f"Using VFOV: {args.vfov} degrees")
    build_training_set(args.panoramic_horizontal_views, args.vfov)
    print("Completed!")
