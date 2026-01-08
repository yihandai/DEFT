"""Batched REVERIE navigation environment with RGB image support"""

import json
import os
import gc
import numpy as np
import math
import random
import networkx as nx
from collections import defaultdict
import copy
import torch
import torchvision.transforms as transforms
from PIL import Image

import MatterSim
import sys
from pathlib import Path

# Add parent directory to sys.path for standalone NavGPT_2 execution
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from utils.data import load_nav_graphs, new_simulator
    from utils.data import angle_feature, get_all_point_angle_feature
except ImportError:
    from NavGPT_2.map_nav_src.utils.data import load_nav_graphs, new_simulator
    from NavGPT_2.map_nav_src.utils.data import (
        angle_feature,
        get_all_point_angle_feature,
    )

try:
    from r2r.eval_utils import cal_dtw, cal_cls
except ImportError:
    from NavGPT_2.map_nav_src.r2r.eval_utils import cal_dtw, cal_cls

ERROR_MARGIN = 3.0


class EnvBatchRGB(object):
    """A simple wrapper for a batch of MatterSim environments,
    using RGB images instead of pretrained features"""

    def __init__(
        self, connectivity_dir, scan_data_dir=None, visual_encoder=None, batch_size=100
    ):
        """
        1. Load RGB images from MatterSim
        2. Init the Simulator.
        :param visual_encoder: Visual encoder model to encode RGB images to features
        :param batch_size: Used to create the simulator list.
        """
        self.visual_encoder = visual_encoder
        self.image_w = 640
        self.image_h = 480
        self.vfov = 60
        self.panoramic_horizontal_views = 12  # Default: 12 horizontal views
        self.viewpoint_size = 36  # 3 elevations * 12 horizontal views

        # Image preprocessing for visual encoder
        # This should match the preprocessing used during training
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            if scan_data_dir:
                sim.setDatasetPath(scan_data_dir)
            sim.setNavGraphPath(connectivity_dir)
            sim.setRenderingEnabled(True)  # Enable rendering to get RGB images
            sim.setDiscretizedViewingAngles(True)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.setBatchSize(1)
            sim.initialize()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + "_" + viewpointId

    def _get_all_views_rgb(self, sim, scanId, viewpointId):
        """
        Get all 36 RGB images from a viewpoint (3 elevations * 12 horizontal views).
        Returns: numpy array of shape (36, 480, 640, 3) - RGB images
        """
        images = []

        for ix in range(self.viewpoint_size):
            if ix == 0:
                sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
            elif ix % self.panoramic_horizontal_views == 0:
                sim.makeAction([0], [1.0], [1.0])  # Move up
            else:
                sim.makeAction([0], [1.0], [0])  # Rotate horizontally

            state = sim.getState()[0]
            # state.rgb is (480, 640, 3) numpy array
            rgb_image = np.array(state.rgb, copy=True, dtype=np.uint8)
            images.append(rgb_image)

        return np.stack(images, axis=0)  # (36, 480, 640, 3)

    def _encode_rgb_to_features(self, rgb_images):
        """
        Encode RGB images to features using visual encoder.

        Args:
            rgb_images: numpy array of shape (36, 480, 640, 3)

        Returns:
            features: numpy array of shape (36, feature_dim)
        """
        if self.visual_encoder is None:
            raise ValueError(
                "Visual encoder is not provided. Cannot encode RGB images."
            )

        # Convert to PIL Images and apply transforms
        # Single viewpoint: (36, 480, 640, 3)
        batch_images = []
        for img in rgb_images:
            pil_img = Image.fromarray(img)
            tensor_img = self.transform(pil_img)
            batch_images.append(tensor_img)
        batch_tensor = torch.stack(batch_images, dim=0)  # (36, 3, 224, 224)

        # Move to device
        device = next(self.visual_encoder.parameters()).device
        batch_tensor = batch_tensor.to(device)

        # Encode with visual encoder
        with torch.no_grad():
            # Visual encoder expects (batch, 3, 224, 224)
            # For BLIP-2, visual_encoder returns (batch, num_patches+1, feature_dim) where +1 is CLS token
            # We need to use ln_vision as well
            if hasattr(self.visual_encoder, "ln_vision"):
                # BLIP-2 style: visual_encoder + ln_vision
                image_embeds = self.visual_encoder(batch_tensor)
                if hasattr(self.visual_encoder, "ln_vision"):
                    image_embeds = self.visual_encoder.ln_vision(image_embeds)
                # Extract CLS token (first token) or use all tokens
                # For compatibility, we'll use the CLS token if available
                if image_embeds.dim() == 3:
                    # Use CLS token (first token)
                    features = image_embeds[:, 0, :]  # (36, feature_dim)
                else:
                    features = image_embeds
            else:
                # Simple visual encoder
                features = self.visual_encoder(batch_tensor)
                if features.dim() == 3:
                    # Use CLS token or average pool
                    if hasattr(self.visual_encoder, "get_cls_token"):
                        features = self.visual_encoder.get_cls_token(features)
                    else:
                        # Use first token as CLS token
                        features = features[:, 0, :]  # (36, feature_dim)

        return features.cpu().numpy()

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(
            zip(scanIds, viewpointIds, headings)
        ):
            self.sims[i].newEpisode([scanId], [viewpointId], [heading], [0])

    def getStates(self):
        """
        Get list of states with RGB images encoded to features.
        :return: [ (feature, state) ] * batch_size
        where feature is (36, feature_dim) numpy array
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()[0]

            # Get all RGB images for this viewpoint
            rgb_images = self._get_all_views_rgb(
                sim, state.scanId, state.location.viewpointId
            )

            # Encode RGB images to features
            features = self._encode_rgb_to_features(rgb_images)

            feature_states.append((features, state))
        return feature_states

    def makeActions(self, actions):
        """Take an action using the full state dependent action interface (with batched input).
        Every action element should be an (index, heading, elevation) tuple."""
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction([index], [heading], [elevation])


# Import the original R2RNavBatch to inherit from it
try:
    from r2r.env import R2RNavBatch
except ImportError:
    from NavGPT_2.map_nav_src.r2r.env import R2RNavBatch


class R2RNavBatchRGB(R2RNavBatch):
    """Extends R2RNavBatch to support RGB images instead of pretrained features"""

    def __init__(
        self,
        view_db,  # Can be None or visual_encoder
        instr_data,
        connectivity_dir,
        candidate_file_dir,
        batch_size=64,
        angle_feat_size=4,
        seed=0,
        name=None,
        sel_data_idxs=None,
        visual_encoder=None,
    ):
        # If view_db is a visual_encoder, use it
        if visual_encoder is None and view_db is not None:
            # Check if view_db is actually a visual encoder (has forward method)
            if hasattr(view_db, "forward") or hasattr(view_db, "__call__"):
                visual_encoder = view_db
                view_db = None

        # Initialize environment with RGB support
        self.env = EnvBatchRGB(
            connectivity_dir,
            scan_data_dir=None,
            visual_encoder=visual_encoder,
            batch_size=batch_size,
        )

        # Call parent class initialization (but we'll override env)
        self.data = instr_data
        self.scans = set([x["scan"] for x in self.data])
        self.connectivity_dir = connectivity_dir
        self.batch_size = batch_size
        self.angle_feat_size = angle_feat_size
        self.name = name

        self.gt_trajs = self._get_gt_trajs(self.data)

        if sel_data_idxs is not None:
            t_split, n_splits = sel_data_idxs
            ndata_per_split = len(self.data) // n_splits
            start_idx = ndata_per_split * t_split
            if t_split == n_splits - 1:
                end_idx = None
            else:
                end_idx = start_idx + ndata_per_split
            self.data = self.data[start_idx:end_idx]

        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self._load_nav_graphs()

        self.candidates_dict = json.load(open(candidate_file_dir, "r"))

        print(
            "%s loaded with %d instructions, using splits: %s (RGB mode)"
            % (self.__class__.__name__, len(self.data), self.name)
        )

        self.data_dict = {x["instr_id"]: x for x in self.data}
