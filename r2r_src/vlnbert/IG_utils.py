from typing import Any
import numpy as np
import sys
import os
import math
from PIL import Image
from vlnbert.caffe_resnet import CNN

# Import FG_CAM_resnet from FG_CAM or FG-CAM directory
# Handle both directory name formats (hyphen vs underscore)
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try direct import first, fallback to importlib if directory has hyphen
from CAM.FG_CAM_resnet import FG_CAM_resnet
import cv2
import torch
import torch.nn.functional as F

# Caffe and MatterSim need to be on the Python path
import MatterSim

import vln_utils
from param import args

from vlnbert.guided_ig.guided_ig import GuidedIG
from vlnbert.guided_ig.base import INPUT_OUTPUT_GRADIENTS

# Note: NavGPT2_genAction_v2 is passed as a parameter to avoid circular import


def get_grad(input_, output_):
    grad_output = torch.autograd.grad(
        output_,
        input_,
        grad_outputs=torch.ones_like(output_),
        retain_graph=True,
    )[0]
    return grad_output


class Exp:
    def __init__(self, bert, critical_head):
        # Simulator image parameters
        self.WIDTH = 640
        self.HEIGHT = 480
        self.VFOV = args.vfov
        # Calculate total views: 3 heights * horizontal_views
        self.VIEWPOINT_SIZE = (
            3 * args.panoramic_horizontal_views
        )  # Number of discretized views from one viewpoint
        self.FEATURE_SIZE = 2048
        self.BATCH_SIZE = (
            4  # Some fraction of viewpoint size - batch size 4 equals 11GB memory
        )
        self.GPU_ID = 0

        # model weight
        self.wt_path = "./feat_checkpoints/CNN/30913b5b6a4c411bb1b6020f492e5862.npy"

        # inference model
        self.bert = bert
        self.critical_head = critical_head
        self.feature_model = self.load_feature_model()
        self.feature_size = 2048

        # load simulator
        self.sim = self.load_sim()

    def transform_img(self, im):
        """Prep opencv 3 channel image for the network"""
        im = np.array(im, copy=True)
        im_orig = im.astype(np.float32, copy=True)
        im_orig = cv2.resize(im_orig, (224, 224))
        # im_orig = im.astype(np.float32, copy=True)
        im_orig -= np.array([[[103.1, 115.9, 123.2]]])  # BGR pixel mean
        # blob = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.float32)
        # blob[:, :, :] = im_orig
        blob = im_orig.transpose((2, 0, 1))
        # (3, 224, 224)
        return blob

    def reverse_transforms(self, ori_image):
        # (C, H, W)
        # denorm_tensor = (tensor * 0.5) + 0.5
        if isinstance(ori_image, torch.Tensor):
            ori_image = ori_image.clone()
            image = ori_image.permute(1, 2, 0).detach().cpu().numpy()  # (H, W, C)
        elif isinstance(ori_image, np.ndarray):
            ori_image = ori_image.copy()
            image = ori_image.transpose(1, 2, 0)
        else:
            print("type of image should be tensor or ndarray")
            exit(0)
        image = image + np.array([[[103.1, 115.9, 123.2]]])  # BGR pixel mean
        # image = (image * 255).clip(0, 255).astype(np.uint8)
        image = image.clip(0, 255).astype(np.uint8)

        return image

    def ZeroCenter(self, path, size, BGRTranspose=False):
        img = Image.open(path)
        if isinstance(size, tuple):
            h, w = size[0], size[1]
        else:
            h, w = size, size
        img = img.resize((h, w))
        x = np.array(img, dtype=np.float32)

        # Reference: 1) Keras image preprocess: https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
        #            2) tensorflow github issue: https://github.com/tensorflow/models/issues/517
        # R-G-B for Imagenet === [123.68, 116.78, 103.94]

        # x[..., 0] -= 123.68
        # x[..., 1] -= 116.779
        # x[..., 2] -= 103.939
        # [103.1, 115.9, 123.2]
        x[..., 0] -= 123.2
        x[..., 1] -= 115.9
        x[..., 2] -= 103.1
        # im_orig -= np.array([[[103.1, 115.9, 123.2]]]) # BGR pixel mean
        if BGRTranspose == True:
            x = x[..., ::-1]

        return x

    def load_sim(self):
        # Set up the simulator
        sim = MatterSim.Simulator()
        sim.setRenderingEnabled(True)
        sim.setCameraResolution(self.WIDTH, self.HEIGHT)
        sim.setCameraVFOV(math.radians(self.VFOV))
        # Use non-discretized mode when panoramic_horizontal_views != 12 to support custom angle increments
        use_discretized = args.panoramic_horizontal_views == 12
        sim.setDiscretizedViewingAngles(use_discretized)
        sim.setBatchSize(1)
        sim.initialize()
        return sim

    def load_feature_model(self):
        """
        the ResNet-152 model needs images in BGR format
        """
        model = CNN(weight_file=self.wt_path).cuda()
        model.eval()
        return model

    def get_vp_feature(self, model, blobs) -> torch.tensor:
        """
        blobs: torch.Tensor [bs, vp, C, H, W] where vp = VIEWPOINT_SIZE
        returns: torch.Tensor [bs, vp, FEATURE_SIZE]
        keeps gradient flow from `features` back into `blobs`
        """
        # print("SIZE OF BLOBS: ", blobs.shape)
        bs, vp, dim, _, _ = blobs.shape
        # if hasattr(self, "VIEWPOINT_SIZE"):
        #     assert (
        #         vp == self.VIEWPOINT_SIZE
        #     ), f"vp={vp} != VIEWPOINT_SIZE={self.VIEWPOINT_SIZE}"

        B = getattr(self, "BATCH_SIZE", vp)  # chunk size over vp dimension

        out_per_batch = []
        for b in range(bs):
            vp_chunks = []
            for start in range(0, vp, B):
                end = min(start + B, vp)
                # shape: [chunk, C, H, W]
                batch_blobs = blobs[b, start:end, :]
                # forward keeps autograd graph (no .detach(), no .numpy())
                feat_chunk, _ = model(batch_blobs)  # [chunk, FEATURE_SIZE]
                feat_chunk = feat_chunk[:, :, 0, 0]  # [chunk, FEATURE_SIZE]
                vp_chunks.append(feat_chunk)
            # [vp, FEATURE_SIZE]
            vp_feats = torch.cat(vp_chunks, dim=0)
            out_per_batch.append(vp_feats.unsqueeze(0))  # [1, vp, FEATURE_SIZE]
            # out_per_batch.append(vp_feats)  # [1, vp, FEATURE_SIZE]

        # [bs, vp, FEATURE_SIZE]
        features = torch.cat(out_per_batch, dim=0)
        return features

    def get_vp_feature_and_logits(self, model, blobs) -> torch.tensor:
        """
        blobs: torch.Tensor [bs, vp, C, H, W] where vp = VIEWPOINT_SIZE
        returns: features [bs, vp, FEATURE_SIZE], logits [bs, vp, num_logits]
        keeps gradient flow from `features` back into `blobs`
        """
        bs, vp, dim, _, _ = blobs.shape
        # B = getattr(self, "BATCH_SIZE", vp)  # chunk size over vp dimension
        B = 32

        features_per_batch = []
        logits_per_batch = []
        for b in range(bs):
            vp_feat_chunks = []
            vp_logits_chunks = []
            for start in range(0, vp, B):
                end = min(start + B, vp)
                batch_blobs = blobs[b, start:end, :]
                feat_chunk, logits_chunk = model(batch_blobs)
                # Both feat_chunk, logits_chunk shape: [chunk, CLASS or FEAT, 1, 1]
                feat_chunk = feat_chunk[:, :, 0, 0]  # [chunk, FEATURE_SIZE]
                logits_chunk = logits_chunk  # [chunk, num_logits]
                vp_feat_chunks.append(feat_chunk)
                vp_logits_chunks.append(logits_chunk)
            vp_feats = torch.cat(vp_feat_chunks, dim=0)  # [vp, FEATURE_SIZE]
            vp_logits = torch.cat(vp_logits_chunks, dim=0)  # [vp, num_logits]
            features_per_batch.append(vp_feats.unsqueeze(0))  # [1, vp, FEATURE_SIZE]
            logits_per_batch.append(vp_logits.unsqueeze(0))  # [1, vp, num_logits]
        features = torch.cat(features_per_batch, dim=0)  # [bs, vp, FEATURE_SIZE]
        logits = torch.cat(logits_per_batch, dim=0)  # [bs, vp, num_logits]
        return features, logits

    def get_vp_images(self, sim, scanId, viewpointId) -> np.array:
        blobs = []
        num_horizontal_views = args.panoramic_horizontal_views
        angle_increment_rad = math.radians(360.0 / num_horizontal_views)
        use_discretized = args.panoramic_horizontal_views == 12

        for ix in range(self.VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
            elif ix % num_horizontal_views == 0:
                # Move up one elevation level (30 degrees)
                if use_discretized:
                    sim.makeAction([0], [1.0], [1.0])
                else:
                    # Non-discretized mode: use radians (30 degrees = π/6)
                    sim.makeAction([0], [0.0], [math.radians(30)])
            else:
                # Rotate horizontally
                if use_discretized:
                    sim.makeAction([0], [1.0], [0])
                else:
                    # Non-discretized mode: rotate by angle_increment_rad (e.g., 45 degrees = π/4)
                    sim.makeAction([0], [angle_increment_rad], [0])

            state = sim.getState()[0]
            # Note: When panoramic_horizontal_views != 12, we use non-discretized mode
            # so viewIndex may not match our expected index. Only assert for 12 views.
            if use_discretized:
                assert state.viewIndex == ix

            # Transform and save generated image
            blobs.append(self.transform_img(state.rgb))
        blobs = np.array(blobs)  # [vp, C, H, W]
        return blobs

    def compute_integrated_gradients(
        self,
        obs,
        # gmaps,
        t,
        h_t_input,
        language_features,
        language_inputs,
        language_attention_mask,
        token_type_ids,
        steps=50,
        mode="IG",
    ):
        """
        Args:
            mode:   the saliency map be choosen to use. options ["IG", "temporal", "IG_temporal"]
        Returns:
            images: list of images in shape of [vp, 3, 224, 224]
            heatmaps: list of attribution map in shape of [vp, 224, 224]
        """
        bs = len(obs)
        steps = 5
        alphas = [alpha for alpha in np.linspace(0, 1, steps)]
        # grads = []
        # grads_temporal = []

        # get panorama images and transform them -> np
        images_numpys = []
        for i in range(bs):
            scanId = obs[i]["scan"]
            viewpointId = obs[i]["viewpoint"]
            images_numpy = self.get_vp_images(self.sim, scanId, viewpointId)
            images_numpys.append(images_numpy)
        images_numpys = np.stack(images_numpys)  # [bs, vp, C, H, W]

        # to tensor
        images = torch.autograd.Variable(
            torch.from_numpy(images_numpys), requires_grad=False
        ).cuda()
        # get feature through ResNet-152-InPlace365
        with torch.no_grad():
            feature_tensor = self.get_vp_feature(self.feature_model, images)
        # original action index
        target_nav_logits, target_action, target_h_t, candidata_list = self.do_forward(
            self.bert,
            obs,
            t,
            h_t_input,
            language_features,
            language_inputs,
            language_attention_mask,
            token_type_ids,
            feature_tensor,
        )
        target_critical_logits = self.critical_head(target_h_t).unsqueeze(0)
        _, critical = target_critical_logits.max(1)

        total_grad = total_grad_temporal = 0
        # baseline = torch.zeros_like(images).cpu()

        baseline = torch.zeros_like(torch.from_numpy(images_numpys))
        baseline_img = np.zeros_like(images_numpys)

        B, V, C, H, W = images.shape
        assert B == 1, "This version assumes batch size = 1"

        # candidate_list: [1, vp] -> [vp]
        cand_idx = candidata_list[0]  # [vp]
        all_idx = [x for x in range(V)]
        non_cand_idx = [i for i in all_idx if i not in cand_idx]

        for alpha in alphas:
            images = baseline_img + (images_numpys - baseline_img) * alpha

            # images: [1, V, C, H, W] where V = VIEWPOINT_SIZE
            images = torch.from_numpy(images).cuda()

            # Candidate images [1, vp, C, H, W]
            cand_images = images[:, cand_idx, :, :, :].requires_grad_(True)
            feat_cand = self.get_vp_feature(
                self.feature_model, cand_images
            )  # [1, vp, D]

            # Non-candidate images [1, V-vp, C, H, W]
            non_cand_images = images[:, non_cand_idx, :, :, :].requires_grad_(False)
            with torch.no_grad():
                feat_non_cand = self.get_vp_feature(
                    self.feature_model, non_cand_images
                )  # [1, V-vp, D]

            # Reconstruct [1, V, D] feature tensor in the correct order
            D = feat_cand.size(-1)
            feat_full = torch.empty((B, V, D), device=images.device)
            feat_full[:, cand_idx] = feat_cand
            feat_full[:, non_cand_idx] = feat_non_cand

            nav_logits, a_t, h_t, _ = self.do_forward(
                self.bert,
                obs,
                t,
                h_t_input,
                language_features,
                language_inputs,
                language_attention_mask,
                token_type_ids,
                feat_full,
            )
            nav_probs = torch.softmax(nav_logits, 1)
            critical_logits = self.critical_head(h_t).unsqueeze(0)
            target_prob = nav_probs[:, target_action]

            # get the gradient
            grad_output = get_grad(cand_images, target_prob)
            grad_output_temporal = get_grad(
                cand_images, target_critical_logits - critical_logits
            )
            # grad_output = torch.zeros_like(images)
            # grad_output_temporal = torch.zeros_like(images)

            total_grad += grad_output.detach().cpu()
            total_grad_temporal += grad_output_temporal.detach().cpu()
            # grads.append(grad_output.detach().cpu())
            # grads_temporal.append(grad_output_temporal.detach().cpu())

        # grads [steps, bs, vp, 3, 224, 224]
        # 近似积分
        # avg_grad = torch.mean(torch.stack(grads), dim=0)
        # avg_grad_temporal = torch.mean(torch.stack(grads_temporal), dim=0)
        avg_grad = total_grad / steps
        avg_grad_temporal = total_grad_temporal / steps

        # avg_grad  [bs, vp, 3, 224, 224]
        # images    [bs, vp, 3, 224, 224]

        heatmaps = []
        print("Critical? ", critical)

        images_return = np.array(
            [
                [self.reverse_transforms(image) for image in images]
                for images in images_numpys
            ]
        )  # [B, V, H, W, C] where V = VIEWPOINT_SIZE
        # images = images.detach().cpu()
        images = torch.from_numpy(images_numpys)
        for i in range(bs):
            ig = (
                images[i][cand_idx, :, :, :] - baseline[i][cand_idx, :, :, :]
            ) * avg_grad[i]
            ig_temporal = (
                images[i][cand_idx, :, :, :] - baseline[i][cand_idx, :, :, :]
            ) * avg_grad_temporal[i]
            # ig    [vp, 3, 224, 244]
            heatmap = self.gen_heatmap(ig)  # [vp, 224, 224]
            heatmap_temporal = self.gen_heatmap(ig_temporal)

            heatmap_all = np.zeros((self.VIEWPOINT_SIZE, H, W), dtype=np.uint8)
            heatmap_all[cand_idx] = heatmap
            heatmap_temporal_all = np.zeros((self.VIEWPOINT_SIZE, H, W), dtype=np.uint8)
            heatmap_temporal_all[cand_idx] = heatmap_temporal

            self.draw_heatmaps2(
                images_return[i],
                heatmap_all,
                obs[i],
                "heatmaps/heatmap",
                candidata_list[i],
            )
            self.draw_heatmaps2(
                images_return[i],
                heatmap_temporal_all,
                obs[i],
                "heatmaps/temporal",
                candidata_list[i],
            )

            heatmap = self.integrate(
                heatmap_all, heatmap_temporal_all, mode=mode, c=critical
            )
            heatmaps.append(heatmap)
        # return images, torch.stack(igs, dim=0)
        return images_return, heatmaps, candidata_list

    def compute_integrated_gradients_guided_ig(
        self,
        obs,
        # gmaps,
        t,
        h_t_input,
        language_features,
        language_inputs,
        language_attention_mask,
        token_type_ids,
        steps=50,
        mode="IG",
    ):
        """
        Args:
            mode:   the saliency map be choosen to use. options ["IG", "temporal", "IG_temporal"]
        Returns:
            images: list of images in shape of [vp, 3, 224, 224]
            heatmaps: list of attribution map in shape of [vp, 224, 224]
        """
        bs = len(obs)
        steps = 5
        alphas = [alpha for alpha in np.linspace(0, 1, steps)]
        # grads = []
        # grads_temporal = []

        # get panorama images and transform them -> np
        images_numpys = []
        for i in range(bs):
            scanId = obs[i]["scan"]
            viewpointId = obs[i]["viewpoint"]
            images_numpy = self.get_vp_images(self.sim, scanId, viewpointId)
            images_numpys.append(images_numpy)
        images_numpys = np.stack(images_numpys)  # [bs, vp, C, H, W]

        # to tensor
        images = torch.autograd.Variable(
            torch.from_numpy(images_numpys), requires_grad=False
        ).cuda()
        # get feature through ResNet-152-InPlace365
        with torch.no_grad():
            feature_tensor = self.get_vp_feature(self.feature_model, images)
        # original action index
        target_nav_logits, target_action, target_h_t, candidata_list = self.do_forward(
            self.bert,
            obs,
            t,
            h_t_input,
            language_features,
            language_inputs,
            language_attention_mask,
            token_type_ids,
            feature_tensor,
        )
        target_critical_logits = self.critical_head(target_h_t).unsqueeze(0)
        _, critical = target_critical_logits.max(1)

        total_grad = total_grad_temporal = 0
        # baseline = torch.zeros_like(images).cpu()

        baseline = torch.zeros_like(torch.from_numpy(images_numpys))
        baseline_img = np.zeros_like(images_numpys)

        B, V, C, H, W = images.shape
        assert B == 1, "This version assumes batch size = 1"

        # candidate_list: [1, vp] -> [vp]
        cand_idx = candidata_list[0]  # [vp]
        all_idx = [x for x in range(V)]
        non_cand_idx = [i for i in all_idx if i not in cand_idx]

        for alpha in alphas:
            images = baseline_img + (images_numpys - baseline_img) * alpha

            # images: [1, V, C, H, W] where V = VIEWPOINT_SIZE
            images = torch.from_numpy(images).cuda()

            # Candidate images [1, vp, C, H, W]
            cand_images = images[:, cand_idx, :, :, :].requires_grad_(True)
            feat_cand = self.get_vp_feature(
                self.feature_model, cand_images
            )  # [1, vp, D]

            # Non-candidate images [1, V-vp, C, H, W]
            non_cand_images = images[:, non_cand_idx, :, :, :].requires_grad_(False)
            with torch.no_grad():
                feat_non_cand = self.get_vp_feature(
                    self.feature_model, non_cand_images
                )  # [1, V-vp, D]

            # Reconstruct [1, V, D] feature tensor in the correct order
            D = feat_cand.size(-1)
            feat_full = torch.empty((B, V, D), device=images.device)
            feat_full[:, cand_idx] = feat_cand
            feat_full[:, non_cand_idx] = feat_non_cand

            nav_logits, a_t, h_t, _ = self.do_forward(
                self.bert,
                obs,
                t,
                h_t_input,
                language_features,
                language_inputs,
                language_attention_mask,
                token_type_ids,
                feat_full,
            )
            nav_probs = torch.softmax(nav_logits, 1)
            critical_logits = self.critical_head(h_t).unsqueeze(0)
            target_prob = nav_probs[:, target_action]

            # get the gradient
            grad_output = get_grad(cand_images, target_prob)
            grad_output_temporal = get_grad(
                cand_images, target_critical_logits - critical_logits
            )
            # grad_output = torch.zeros_like(images)
            # grad_output_temporal = torch.zeros_like(images)

            total_grad += grad_output.detach().cpu()
            total_grad_temporal += grad_output_temporal.detach().cpu()
            # grads.append(grad_output.detach().cpu())
            # grads_temporal.append(grad_output_temporal.detach().cpu())

        # grads [steps, bs, vp, 3, 224, 224]
        # 近似积分
        # avg_grad = torch.mean(torch.stack(grads), dim=0)
        # avg_grad_temporal = torch.mean(torch.stack(grads_temporal), dim=0)
        avg_grad = total_grad / steps
        avg_grad_temporal = total_grad_temporal / steps

        # avg_grad  [bs, vp, 3, 224, 224]
        # images    [bs, vp, 3, 224, 224]

        heatmaps = []
        print("Critical? ", critical)

        images_return = np.array(
            [
                [self.reverse_transforms(image) for image in images]
                for images in images_numpys
            ]
        )  # [B, V, H, W, C] where V = VIEWPOINT_SIZE
        # images = images.detach().cpu()
        images = torch.from_numpy(images_numpys)
        for i in range(bs):
            ig = (
                images[i][cand_idx, :, :, :] - baseline[i][cand_idx, :, :, :]
            ) * avg_grad[i]
            ig_temporal = (
                images[i][cand_idx, :, :, :] - baseline[i][cand_idx, :, :, :]
            ) * avg_grad_temporal[i]
            # ig    [vp, 3, 224, 244]
            heatmap = self.gen_heatmap(ig)  # [vp, 224, 224]
            heatmap_temporal = self.gen_heatmap(ig_temporal)

            heatmap_all = np.zeros((self.VIEWPOINT_SIZE, H, W), dtype=np.uint8)
            heatmap_all[cand_idx] = heatmap
            heatmap_temporal_all = np.zeros((self.VIEWPOINT_SIZE, H, W), dtype=np.uint8)
            heatmap_temporal_all[cand_idx] = heatmap_temporal

            self.draw_heatmaps2(
                images_return[i],
                heatmap_all,
                obs[i],
                "heatmaps/heatmap",
                candidata_list[i],
            )
            self.draw_heatmaps2(
                images_return[i],
                heatmap_temporal_all,
                obs[i],
                "heatmaps/temporal",
                candidata_list[i],
            )

            heatmap = self.integrate(
                heatmap_all, heatmap_temporal_all, mode=mode, c=critical
            )
            heatmaps.append(heatmap)
        # return images, torch.stack(igs, dim=0)
        return images_return, heatmaps, candidata_list

    def compute_FG_CAM(
        self,
        obs,
        # gmaps,
        t,
        h_t_input,
        language_features,
        language_inputs,
        language_attention_mask,
        token_type_ids,
        steps=50,
        mode="IG",
        target_layer=-1,
        denoising=False,
    ):
        """
        Compute FG-CAM explanations for panorama images

        Args:
            images: torch.Tensor [bs, vp, C, H, W] - panorama images
            target_layer: int - target layer for FG-CAM (-1 for input layer)
            denoising: bool - whether to use SVD denoising

        Returns:
            images_return: np.array [B, V, H, W, C] - original images where V = VIEWPOINT_SIZE
            heatmaps: list of np.array [V, H, W] - FG-CAM heatmaps where V = VIEWPOINT_SIZE
            candidata_list: list - candidate viewpoint indices
        """
        bs = len(obs)

        # get panorama images and transform them -> np
        images_numpys = []
        for i in range(bs):
            scanId = obs[i]["scan"]
            viewpointId = obs[i]["viewpoint"]
            images_numpy = self.get_vp_images(self.sim, scanId, viewpointId)
            images_numpys.append(images_numpy)
        images_numpys = np.stack(images_numpys)  # [bs, vp, C, H, W]

        # to tensor
        images = torch.autograd.Variable(
            torch.from_numpy(images_numpys), requires_grad=True
        ).cuda()
        fg_cam = FG_CAM_resnet(self.feature_model, "grad_cam")

        B, V, C, H, W = images.shape
        assert B == 1, "This version assumes batch size = 1"

        # candidate_list: [1, vp] -> [vp]
        candidata_list = self.get_only_can_list(obs)
        cand_idx = candidata_list[0]  # [vp]

        # Prepare images for return (reverse transform)
        images_return = np.array(
            [
                [self.reverse_transforms(image) for image in images]
                for images in images_numpys
            ]
        )  # [B, V, H, W, C] where V = VIEWPOINT_SIZE

        # Process each batch
        heatmaps = []
        for i in range(bs):
            # Initialize heatmap arrays for all viewpoints
            heatmap_all = np.zeros((self.VIEWPOINT_SIZE, H, W), dtype=np.uint8)

            # Process each candidate viewpoint through FG-CAM
            for vp_idx in cand_idx:
                # Extract single viewpoint image: [vp, C, H, W] -> [C, H, W] -> [1, C, H, W]
                single_image = images[i, vp_idx : vp_idx + 1, :, :, :]  # [1, C, H, W]
                # print("single_image", single_image.shape)
                # single_image.requires_grad = True

                # Generate FG-CAM explanation for this viewpoint
                try:
                    explanation, predicted_class = fg_cam(
                        single_image,
                        denoising=denoising,
                        target_layer=target_layer,
                        target_class=None,  # Use predicted class
                    )

                    # Convert explanation to heatmap format
                    # explanation shape: [1, H, W] or [H, W] (after torch.sum in FG_CAM)
                    if isinstance(explanation, torch.Tensor):
                        explanation_np = explanation.detach().cpu().numpy()
                    else:
                        explanation_np = explanation
                    # print("explanation_np", explanation_np.shape)
                    # print("explanation_np", explanation_np)

                    def visual_explanation(heatmap):
                        """Normalize and resize heatmap for visualization"""
                        # Handle different input shapes
                        if len(heatmap.shape) == 3:
                            # [batch, H, W] -> [H, W]
                            heatmap = heatmap[0]
                        elif len(heatmap.shape) > 3:
                            # [batch, C, H, W] or similar -> take first element
                            heatmap = heatmap[0]

                        # Ensure 2D
                        if len(heatmap.shape) != 2:
                            raise ValueError(
                                f"Expected 2D heatmap, got shape {heatmap.shape}"
                            )

                        # Normalize
                        heatmap = (heatmap - heatmap.min()) / (
                            heatmap.max() - heatmap.min() + 1e-9
                        )

                        # Resize to target size (cv2.resize expects (width, height))
                        heatmap = cv2.resize(heatmap, (W, H))
                        return heatmap

                    # Process explanation: normalize and resize
                    explanation_np = visual_explanation(explanation_np)
                    # print("explanation_np", explanation_np)
                    # Verify shape matches expected dimensions
                    assert explanation_np.shape == (
                        H,
                        W,
                    ), f"Shape mismatch. Got {explanation_np.shape} instead of (H, W)"

                    # Convert to heatmap format (convert normalized float to uint8)
                    # explanation_np is already normalized [0, 1] from visual_explanation
                    heatmap = (explanation_np * 255).clip(0, 255).astype(np.uint8)
                    # visualize heatmap
                    # cv2.imwrite("./fg_cam_heatmap.png", heatmap)
                    # Store in heatmap array
                    heatmap_all[vp_idx] = heatmap

                except Exception as e:
                    print(
                        f"Warning: Failed to generate FG-CAM for viewpoint {vp_idx}: {e}"
                    )
                    # Leave as zeros if failed
                    continue

            # # Draw heatmaps
            # self.draw_heatmaps2(
            #     images_return[i],
            #     heatmap_all,
            #     obs[i],
            #     "heatmaps/fg_cam",
            #     candidata_list[i],
            # )

            # heatmap_temporal_all = heatmap_all.copy()

            # self.draw_heatmaps2(
            #     images_return[i],
            #     heatmap_temporal_all,
            #     obs[i],
            #     "heatmaps/fg_cam_temporal",
            #     candidata_list[i],
            # )

            heatmaps.append(heatmap_all)

        return images_return, heatmaps, candidata_list

    def compute_random_salency(
        self,
        obs,
        # gmaps,
        t,
        h_t_input,
        language_features,
        language_inputs,
        language_attention_mask,
        token_type_ids,
        # steps=50,
        # mode="IG",
    ):

        bs = len(obs)
        # get panorama images and transform them -> np
        images_numpys = []
        for i in range(bs):
            scanId = obs[i]["scan"]
            viewpointId = obs[i]["viewpoint"]
            images_numpy = self.get_vp_images(self.sim, scanId, viewpointId)
            images_numpys.append(images_numpy)
        images_numpys = np.stack(images_numpys)  # [bs, vp, C, H, W]

        images_return = np.array(
            [
                [self.reverse_transforms(image) for image in images]
                for images in images_numpys
            ]
        )  # [B, V, H, W, C] where V = VIEWPOINT_SIZE
        B, V, C, H, W = images_numpys.shape
        candidata_list = self.get_only_can_list(obs)

        # gen a random salency map [bs, vp, 224, 224]
        # Generate spatially coherent random saliency using random Gaussian blobs
        # This creates more natural-looking clusters compared to simple downsampling
        heatmaps = []
        entry = 2
        for i in range(bs):
            heatmap = np.zeros((self.VIEWPOINT_SIZE, 3, H, W), dtype=np.float32)
            if entry == 0:
                for vp in range(self.VIEWPOINT_SIZE):
                    # Create coordinate grids once per viewpoint
                    y_coords, x_coords = np.mgrid[0:H, 0:W].astype(np.float32)

                    # Generate blobs for all 3 channels together
                    blob_map_all_channels = np.zeros((3, H, W), dtype=np.float32)

                    # Generate random Gaussian blobs (clusters)
                    num_blobs = np.random.randint(3, 8)  # Random number of clusters

                    for _ in range(num_blobs):
                        # Random center position
                        center_y = np.random.uniform(0, H)
                        center_x = np.random.uniform(0, W)

                        # Random blob size (sigma for Gaussian)
                        sigma_y = np.random.uniform(H * 0.1, H * 0.3)
                        sigma_x = np.random.uniform(W * 0.1, W * 0.3)

                        # Random intensity for each of the 3 channels
                        intensities = np.random.uniform(50, 255, size=3)

                        # Create Gaussian blob (same shape for all channels)
                        gaussian = np.exp(
                            -(
                                (x_coords - center_x) ** 2 / (2 * sigma_x**2)
                                + (y_coords - center_y) ** 2 / (2 * sigma_y**2)
                            )
                        )

                        # Apply different intensities to each channel
                        blob_map_all_channels += (
                            gaussian[np.newaxis, :, :]
                            * intensities[:, np.newaxis, np.newaxis]
                        )

                    heatmap[vp] = blob_map_all_channels
            elif entry == 1:
                # 预先计算全景图的总宽度
                total_width = W * self.VIEWPOINT_SIZE
                # Get candidate viewpoints for this batch item
                candidate_vps = candidata_list[i]

                y_coords, x_coords = np.mgrid[0:H, 0:total_width].astype(np.float32)

                # 初始化全景 heatmap [3, H, Total_W]
                panorama_blob_map = np.zeros((3, H, total_width), dtype=np.float32)

                # Generate random Gaussian blobs (clusters) for each viewpoint
                num_blobs = np.random.randint(3, 8)  # Random number of clusters

                for _ in range(num_blobs):
                    # Random center position for each viewpoint: [vp,]
                    center_y = np.random.uniform(0, H)
                    center_x = np.random.uniform(0, total_width)

                    # Random blob size (sigma for Gaussian) for each viewpoint: [vp,]
                    sigma_y = np.random.uniform(H * 0.1, H * 0.4)
                    sigma_x = np.random.uniform(W * 0.2, W * 1.0)

                    # Random intensity for each of the 3 channels
                    intensities = np.random.uniform(50, 255, size=3)

                    # 计算 X 轴距离，需要处理全景图首尾相接 (Wrap-around)
                    # 距离是环形距离：min(|x-c|, Total - |x-c|)
                    dist_x = np.abs(x_coords - center_x)
                    dist_x = np.minimum(dist_x, total_width - dist_x)

                    dist_y = y_coords - center_y

                    # 生成 Gaussian
                    gaussian = np.exp(
                        -(dist_x**2 / (2 * sigma_x**2) + dist_y**2 / (2 * sigma_y**2))
                    )
                    # 叠加到全景图上
                    panorama_blob_map += (
                        gaussian[np.newaxis, :, :]
                        * intensities[:, np.newaxis, np.newaxis]
                    )
                # 2. 将全景图大切片切分为 Viewpoints [V, 3, H, W] where V = VIEWPOINT_SIZE
                # panorama_blob_map: [3, H, W*V] -> reshape -> [3, H, V, W] -> transpose -> [V, 3, H, W]
                panorama_blob_map = panorama_blob_map.reshape(
                    3, H, self.VIEWPOINT_SIZE, W
                )
                panorama_blob_map = panorama_blob_map.transpose(2, 0, 1, 3)

                # 3. 仅提取 candidate viewpoints 的部分并赋值
                for vp in candidate_vps:
                    heatmap[vp] = panorama_blob_map[vp]

            elif entry == 2:
                # random assign importance score to each pixel
                heatmap = np.random.rand(self.VIEWPOINT_SIZE, 3, H, W)

            heatmap = self.gen_heatmap(heatmap)
            heatmaps.append(heatmap)

        return images_return, heatmaps, candidata_list

    def get_only_can_list(self, obs):
        canditate_list = []
        for i, ob in enumerate(obs):
            candidate_idxs = []
            for j, cc in enumerate[Any](ob["candidate"]):
                candidate_idx = cc["pointId"]
                candidate_idxs.append(candidate_idx)
            canditate_list.append(candidate_idxs)
        return canditate_list

    def get_images_and_candidata_list(self, obs):
        bs = len(obs)
        images_numpys = []
        for i in range(bs):
            scanId = obs[i]["scan"]
            viewpointId = obs[i]["viewpoint"]
            images_numpy = self.get_vp_images(self.sim, scanId, viewpointId)
            images_numpys.append(images_numpy)
        images_numpys = np.stack(images_numpys)  # [bs, vp, C, H, W]

        images_return = np.array(
            [
                [self.reverse_transforms(image) for image in images]
                for images in images_numpys
            ]
        )  # [B, V, H, W, C] where V = VIEWPOINT_SIZE

        canditate_list = []
        for i, ob in enumerate(obs):
            candidate_idxs = []
            for j, cc in enumerate(ob["candidate"]):
                candidate_idx = cc["pointId"]
                candidate_idxs.append(candidate_idx)
            canditate_list.append(candidate_idxs)

        return images_return, canditate_list

    def generate_pseudo_action(self, logit, candidate_mask, mode="sample"):
        # h_t, logit = self.vln_bert_noneupdate(**visual_inputs)
        # h_t, logit = self.vln_bert(**visual_inputs)
        logit.masked_fill_(candidate_mask, -float("inf"))
        if mode == "sample":
            probs = F.softmax(logit, 1)  # sampling an action from model
            c = torch.distributions.Categorical(probs)
            a_t = c.sample().detach()
        elif mode == "argmax":
            _, a_t = logit.max(1)  # student forcing - argmax
            a_t = a_t.detach()
        return logit, a_t

    def _candidate_variable(self, obs, feature_tensors):
        candidate_leng = [len(ob["candidate"]) + 1 for ob in obs]  # +1 is for the end
        candidate_feat = torch.zeros(
            (len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size),
            dtype=torch.float32,
        ).cuda()
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        canditate_list = []
        for i, ob in enumerate(obs):
            candidate_idxs = []
            for j, cc in enumerate(ob["candidate"]):
                candidate_idx = cc["pointId"]
                candidate_idxs.append(candidate_idx)
                loc_heading = cc["heading"]
                loc_elevation = cc["elevation"]
                # candidate_feat[i, j, :] = cc["feature"]
                visual_feat = feature_tensors[i, candidate_idx]
                angle_feat = vln_utils.angle_feature(loc_heading, loc_elevation)
                angle_feat = torch.from_numpy(angle_feat).cuda()
                candidate_feat[i, j, :] = torch.cat((visual_feat, angle_feat), -1)
            canditate_list.append(candidate_idxs)
        return candidate_feat, candidate_leng, canditate_list

    def get_input_feat(self, obs, feature_tensors):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = vln_utils.angle_feature(ob["heading"], ob["elevation"])
        input_a_t = torch.from_numpy(input_a_t).cuda()
        # f_t = self._feature_variable(obs)      # Pano image features from obs
        candidate_feat, candidate_leng, candidata_list = self._candidate_variable(
            obs, feature_tensors
        )

        return input_a_t, candidate_feat, candidate_leng, candidata_list

    def do_forward(
        self,
        vln_bert,
        perm_obs,
        t,
        h_t,
        language_features,
        language_inputs,
        language_attention_mask,
        token_type_ids,
        feature_tensors,
    ):
        batch_size = len(perm_obs)
        # Language input
        if t < 1:
            h_t, language_features = vln_bert(**language_inputs)
        # the first [CLS] token, initialized by the language BERT, serves
        # as the agent's state passing through time steps
        elif t >= 1:
            language_features = torch.cat(
                (h_t.unsqueeze(1), language_features[:, 1:, :]), dim=1
            )

        input_a_t, candidate_feat, candidate_leng, candidata_list = self.get_input_feat(
            perm_obs, feature_tensors
        )

        visual_temp_mask = (vln_utils.length2mask(candidate_leng) == 0).long()
        visual_attention_mask = torch.cat(
            (language_attention_mask, visual_temp_mask), dim=-1
        )

        vln_bert.vln_bert.config.directions = max(candidate_leng)
        """ Visual BERT """
        visual_inputs = {
            "mode": "visual",
            "sentence": language_features,
            "attention_mask": visual_attention_mask,
            "lang_mask": language_attention_mask,
            "vis_mask": visual_temp_mask,
            "token_type_ids": token_type_ids,
            "action_feats": input_a_t,
            # 'pano_feats':         f_t,
            "cand_feats": candidate_feat,
        }
        h_t, logit = vln_bert(**visual_inputs)
        # hidden_states.append(h_t)

        candidate_mask = vln_utils.length2mask(candidate_leng)
        logit, action = self.generate_pseudo_action(
            logit, candidate_mask, mode="argmax"
        )
        return logit, action, h_t, candidata_list

    def gen_heatmap(self, img):
        # img: [V, 3, 224, 224] where V = VIEWPOINT_SIZE
        # print("img", img.shape)
        if isinstance(img, torch.Tensor):
            heatmap = img.abs().sum(dim=1).detach().cpu().numpy()
        elif isinstance(img, np.ndarray):
            heatmap = np.abs(img).sum(axis=1)
        else:
            raise ValueError("Invalid input type")
        # normalization
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = (heatmap * 255).clip(0, 255).astype(np.uint8)
        return heatmap  # [V, 224, 224] where V = VIEWPOINT_SIZE

    def draw_heatmaps2(self, imgs, heatmap, ob, path="./heatmaps", list_=None):
        bs = self.VIEWPOINT_SIZE
        scanId = ob["scan"]
        viewpointId = ob["viewpoint"]
        if list_ is None:
            list_ = range(self.VIEWPOINT_SIZE)

        for ix in range(bs):
            if ix in list_:
                self.draw_heatmap2(imgs[ix], heatmap[ix], scanId, viewpointId, ix, path)

    def draw_heatmap2(
        self, img, heatmap, scanId, viewpointId, idx, root_path="./heatmaps"
    ):
        # img: [224, 224]
        target_path = os.path.join(root_path, scanId, viewpointId)
        # print(target_path)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        # img = self.reverse_transforms(img)
        hm_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(img, 0.5, hm_color, 0.5, 0)
        # superimposed = hm_color
        # superimposed = hm_color
        cv2.imwrite(os.path.join(target_path, str(idx)) + ".png", superimposed)

    def integrate(self, map1, map2, mode="IG", c=0, l=0.5):
        """
        Returns:
            map: [vp, 224, 224]
        """
        if mode == "IG":
            return map1
        elif mode == "temporal":
            return map2
        # elif mode == "IG_temporal":
        #     if c == 1:  # critical
        #         new_map = map1 + l * map2
        #     elif c == 0:
        #         new_map = map1 - l * map2
        #     # normalize
        #     # Normalize new_map to [0, 1]
        #     new_map = (new_map - new_map.min()) / (new_map.max() - new_map.min() + 1e-8)
        #     # Scale to [0, 255]
        #     new_map = new_map * 255
        #     new_map = new_map.clip(0, 255).astype(np.uint8)
        #     return new_map

        # elif mode == "IG_temporal":
        #     new_map = map1 + l * map2
        #     # normalize
        #     # Normalize new_map to [0, 1]
        #     new_map = (new_map - new_map.min()) / (new_map.max() - new_map.min() + 1e-8)
        #     # Scale to [0, 255]
        #     new_map = new_map * 255
        #     new_map = new_map.clip(0, 255).astype(np.uint8)
        # return new_map
        elif mode == "IG_temporal":
            new_map = map1 * map2
            # new_map = l * np.minimum(map1, map2) + (1 - l) * np.maximum(map1, map2)
            # normalize
            # Normalize new_map to [0, 1]
            new_map = (new_map - new_map.min()) / (new_map.max() - new_map.min() + 1e-8)
            # Scale to [0, 255]
            new_map = new_map * 255
            new_map = new_map.clip(0, 255).astype(np.uint8)
            return new_map
        # elif mode == "IG_temporal":
        #     m1 = map1.astype(np.float32) / 255.0
        #     m2 = map2.astype(np.float32) / 255.0
        #     if c == 0:
        #         m2 = 1.0 - m2
        #     # alpha in [0,1], controls how much temporal contributes
        #     alpha = np.clip(l * m2, 0.0, 1.0)
        #     fused = (1.0 - alpha) * m1 + alpha * m2
        #     fused = np.clip(fused, 0.0, 1.0)
        #     return (fused * 255.0).astype(np.uint8)
        else:
            print("integrate mode error")
            exit(0)

    def compute_gradient(self, images, call_model_args, expected_keys=None):
        # images np.numpy
        (
            obs,
            t,
            h_t_input,
            language_features,
            language_inputs,
            language_attention_mask,
            token_type_ids,
            feature_tensors,
            target_action,
            # non_cand_idx,
            candidata_list,
        ) = call_model_args

        images = torch.from_numpy(images).cuda()
        B, V, C, H, W = images.shape
        cand_idx = candidata_list[0]  # [vp]
        all_idx = [x for x in range(V)]
        non_cand_idx = [i for i in all_idx if i not in cand_idx]

        cand_images = images[:, cand_idx, :, :, :].requires_grad_(True)
        feat_cand = self.get_vp_feature(self.feature_model, cand_images)  # [1, vp, D]

        # Non-candidate images [1, V-vp, C, H, W]
        non_cand_images = images[:, non_cand_idx, :, :, :].requires_grad_(False)
        with torch.no_grad():
            feat_non_cand = self.get_vp_feature(
                self.feature_model, non_cand_images
            )  # [1, V-vp, D]

        # Reconstruct [1, V, D] feature tensor in the correct order
        D = feat_cand.size(-1)
        feat_full = torch.empty((B, V, D), device=images.device)
        feat_full[:, cand_idx] = feat_cand
        feat_full[:, non_cand_idx] = feat_non_cand

        nav_logits, a_t, h_t, _ = self.do_forward(
            self.bert,
            obs,
            t,
            h_t_input,
            language_features,
            language_inputs,
            language_attention_mask,
            token_type_ids,
            feat_full,
        )

        nav_probs = torch.softmax(nav_logits, 1)
        critical_logits = self.critical_head(h_t).unsqueeze(0)
        target_prob = nav_probs[:, target_action]

        # get the gradient
        grad_output = get_grad(cand_images, target_prob)

        # return grad_output
        return {INPUT_OUTPUT_GRADIENTS: grad_output}

    def get_guided_ig(
        self,
        obs,
        # gmaps,
        t,
        h_t_input,
        language_features,
        language_inputs,
        language_attention_mask,
        token_type_ids,
    ):
        # images np.numpy

        images_numpys = []
        for i, ob in enumerate(obs):
            scanId = ob["scan"]
            viewpointId = ob["viewpoint"]
            images_numpy = self.get_vp_images(self.sim, scanId, viewpointId)
            images_numpys.append(images_numpy)
        images_numpys = np.stack(images_numpys)  # [bs, vp, C, H, W]

        # to tensor
        images = torch.autograd.Variable(
            torch.from_numpy(images_numpys), requires_grad=False
        ).cuda()
        # get feature through ResNet-152-InPlace365
        with torch.no_grad():
            feature_tensor = self.get_vp_feature(self.feature_model, images)
        B, V, C, H, W = images.shape

        # original action index
        target_nav_logits, target_action, target_h_t, candidata_list = self.do_forward(
            self.bert,
            obs,
            t,
            h_t_input,
            language_features,
            language_inputs,
            language_attention_mask,
            token_type_ids,
            feature_tensor,
        )

        call_model_args = (
            obs,
            t,
            h_t_input,
            language_features,
            language_inputs,
            language_attention_mask,
            token_type_ids,
            feature_tensor,
            target_action,
            candidata_list,
        )

        guided_ig = GuidedIG()
        grad_output = guided_ig.GetMask(
            images_numpys,
            self.compute_gradient,
            call_model_args,
            x_steps=10,
            # x_baseline=baseline,
            max_dist=1.0,
            fraction=0.5,
        )
        # exit(0)
        print("grad_output", grad_output.shape)
        # return images_numpys, grad_output, candidata_list

        images_return = np.array(
            [
                [self.reverse_transforms(image) for image in images]
                for images in images_numpys
            ]
        )  # [B, V, H, W, C] where V = VIEWPOINT_SIZE
        bs = images_return.shape[0]
        heatmaps = []
        for i in range(bs):
            ig = grad_output[i][candidata_list[i], :, :, :]
            heatmap = self.gen_heatmap(ig)

            heatmap_all = np.zeros((self.VIEWPOINT_SIZE, H, W), dtype=np.uint8)
            heatmap_all[candidata_list[i]] = heatmap

            self.draw_heatmaps2(
                images_return[i],
                heatmap_all,
                obs[i],
                path="./heatmaps/guided_ig_heatmaps",
                list_=candidata_list[i],
            )
            heatmaps.append(heatmap_all)
        return images_return, heatmaps, candidata_list

    def compute_hsic_attribution(
        self,
        obs,
        t,
        h_t_input,
        language_features,
        language_inputs,
        language_attention_mask,
        token_type_ids,
        grid_size=8,
        nb_design=500,
        perturbation_function="inpainting",
        batch_size=32,
    ):
        """
        Compute HSIC (Hilbert-Schmidt Independence Criterion) Attribution Method for NavGPT2.

        Based on: Novello, Fel, Vigouroux, "Making Sense of Dependance: Efficient Black-box
        Explanations Using Dependence Measure", https://arxiv.org/abs/2206.06219

        Args:
            obs: List of observations
            t: Time step
            h_t_input: Hidden state input
            language_features: Language features
            language_inputs: Language inputs dict
            language_attention_mask: Language attention mask
            token_type_ids: Token type IDs
            grid_size: Size of the grid (grid_size x grid_size) to estimate an index per cell
            nb_design: Number of design for the sampler (Monte Carlo samples)
            perturbation_function: Function to apply perturbation ('inpainting', 'blurring', or 'amplitude')
            batch_size: Batch size for forward passes

        Returns:
            images_return: np.array [B, V, H, W, C] where V = VIEWPOINT_SIZE
            heatmaps: list of np.array [V, H, W] - HSIC attribution maps where V = VIEWPOINT_SIZE
            candidata_list: list of candidate viewpoint indices
        """
        bs = len(obs)

        # Get panorama images and transform them -> np
        images_numpys = []
        for i in range(bs):
            scanId = obs[i]["scan"]
            viewpointId = obs[i]["viewpoint"]
            images_numpy = self.get_vp_images(self.sim, scanId, viewpointId)
            images_numpys.append(images_numpy)
        images_numpys = np.stack(images_numpys)  # [bs, vp, C, H, W]

        # Get original features and action
        images = torch.autograd.Variable(
            torch.from_numpy(images_numpys), requires_grad=False
        ).cuda()
        with torch.no_grad():
            feature_tensor = self.get_vp_feature(self.feature_model, images)

        # Get original action index
        target_nav_logits, target_action, target_h_t, candidata_list = self.do_forward(
            self.bert,
            obs,
            t,
            h_t_input,
            language_features,
            language_inputs,
            language_attention_mask,
            token_type_ids,
            feature_tensor,
        )

        B, V, C, H, W = images.shape
        assert B == 1, "This version assumes batch size = 1"

        # candidate_list: [1, vp] -> [vp]
        cand_idx = candidata_list[0]  # [vp]
        all_idx = [x for x in range(V)]
        non_cand_idx = [i for i in all_idx if i not in cand_idx]

        # Determine vp: use candidate_len if available, otherwise use len(cand_idx)
        vp = len(cand_idx)

        # Prepare images for return
        images_return = np.array(
            [
                [self.reverse_transforms(image) for image in images]
                for images in images_numpys
            ]
        )  # [B, V, H, W, C] where V = VIEWPOINT_SIZE

        # Initialize heatmaps
        heatmaps = []

        # Process each batch item
        for i in range(bs):
            # Initialize heatmap for all viewpoints
            heatmap_all = np.zeros((self.VIEWPOINT_SIZE, H, W), dtype=np.float32)

            # Process each candidate viewpoint
            for vp_idx in cand_idx:
                # Get single viewpoint image: [C, H, W]
                single_image = images_numpys[i, vp_idx]  # [C, H, W]

                # Compute HSIC attribution for this viewpoint
                hsic_scores = self._compute_hsic_for_image(
                    single_image,
                    obs[i],
                    t,
                    h_t_input,
                    language_features,
                    language_inputs,
                    language_attention_mask,
                    token_type_ids,
                    feature_tensor[i],
                    target_action,
                    cand_idx,
                    non_cand_idx,
                    vp_idx,  # Pass the current viewpoint index
                    grid_size,
                    nb_design,
                    perturbation_function,
                    batch_size,
                )

                # Resize HSIC scores from grid_size to H, W
                hsic_map = cv2.resize(
                    hsic_scores, (W, H), interpolation=cv2.INTER_CUBIC
                )

                # Normalize and convert to uint8
                hsic_map = (hsic_map - hsic_map.min()) / (
                    hsic_map.max() - hsic_map.min() + 1e-9
                )
                hsic_map = (hsic_map * 255).clip(0, 255).astype(np.uint8)

                heatmap_all[vp_idx] = hsic_map

            heatmaps.append(heatmap_all)

        return images_return, heatmaps, candidata_list

    def _compute_hsic_for_image(
        self,
        image,  # [C, H, W]
        ob,
        t,
        h_t_input,
        language_features,
        language_inputs,
        language_attention_mask,
        token_type_ids,
        original_features,  # [V, D]
        target_action,
        cand_idx,
        non_cand_idx,
        vp_idx,  # Current viewpoint index being processed
        grid_size,
        nb_design,
        perturbation_function,
        batch_size,
    ):
        """
        Compute HSIC scores for a single image using grid-based approach.

        Returns:
            hsic_scores: np.array [grid_size, grid_size] - HSIC scores per grid cell
        """
        C, H, W = image.shape

        # Calculate cell size
        cell_h = H // grid_size
        cell_w = W // grid_size

        # Initialize HSIC scores
        hsic_scores = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Generate binary masks for each grid cell (using Sobol sequence for better coverage)
        # For simplicity, we'll use random binary masks
        np.random.seed(42)  # For reproducibility

        # Generate nb_design binary masks
        # Each mask is a binary grid indicating which cells are perturbed
        binary_masks = np.random.binomial(
            1, 0.5, size=(nb_design, grid_size, grid_size)
        )

        # Get baseline (original) output
        with torch.no_grad():
            baseline_output = self._get_model_output(
                image,
                ob,
                t,
                h_t_input,
                language_features,
                language_inputs,
                language_attention_mask,
                token_type_ids,
                original_features,
                target_action,
                cand_idx,
                non_cand_idx,
            )

        # For each grid cell, compute HSIC
        for i in range(grid_size):
            for j in range(grid_size):
                # Create masks that vary only this cell
                # We'll use the binary masks and compute correlation
                cell_outputs = []

                for mask_idx in range(min(nb_design, len(binary_masks))):
                    # Create perturbation mask for this sample
                    pert_mask = binary_masks[mask_idx].copy()

                    # Apply perturbation to image based on mask
                    perturbed_image = self._apply_perturbation(
                        image.copy(),
                        pert_mask,
                        grid_size,
                        cell_h,
                        cell_w,
                        perturbation_function,
                    )

                    # Get model output with perturbed image
                    with torch.no_grad():
                        output = self._get_model_output(
                            perturbed_image,
                            ob,
                            t,
                            h_t_input,
                            language_features,
                            language_inputs,
                            language_attention_mask,
                            token_type_ids,
                            original_features,
                            target_action,
                            cand_idx,
                            non_cand_idx,
                            vp_idx=vp_idx,  # Use the current viewpoint index
                            perturbed_image_tensor=perturbed_image,
                        )
                        cell_outputs.append(output.item())

                # Compute HSIC between cell mask and outputs
                # Simplified HSIC: correlation between cell presence and output change
                cell_presence = binary_masks[: len(cell_outputs), i, j]
                outputs = np.array(cell_outputs)

                # Normalize
                cell_presence = (cell_presence - cell_presence.mean()) / (
                    cell_presence.std() + 1e-9
                )
                outputs = (outputs - outputs.mean()) / (outputs.std() + 1e-9)

                # HSIC approximation: correlation
                hsic_score = np.abs(np.mean(cell_presence * outputs))
                hsic_scores[i, j] = hsic_score

        return hsic_scores

    def _apply_perturbation(
        self,
        image,  # [C, H, W]
        pert_mask,  # [grid_size, grid_size]
        grid_size,
        cell_h,
        cell_w,
        perturbation_function,
    ):
        """
        Apply perturbation to image based on grid mask.

        Args:
            image: Input image [C, H, W]
            pert_mask: Binary mask [grid_size, grid_size] indicating which cells to perturb
            grid_size: Size of grid
            cell_h, cell_w: Height and width of each cell
            perturbation_function: Type of perturbation ('inpainting', 'blurring', 'amplitude')

        Returns:
            perturbed_image: [C, H, W]
        """
        C, H, W = image.shape
        perturbed_image = image.copy()

        if perturbation_function == "inpainting":
            # Replace with zeros (black)
            for i in range(grid_size):
                for j in range(grid_size):
                    if pert_mask[i, j] == 1:
                        h_start = i * cell_h
                        h_end = min((i + 1) * cell_h, H)
                        w_start = j * cell_w
                        w_end = min((j + 1) * cell_w, W)
                        perturbed_image[:, h_start:h_end, w_start:w_end] = 0.0

        elif perturbation_function == "blurring":
            # Apply Gaussian blur (simplified: use mean)
            for i in range(grid_size):
                for j in range(grid_size):
                    if pert_mask[i, j] == 1:
                        h_start = i * cell_h
                        h_end = min((i + 1) * cell_h, H)
                        w_start = j * cell_w
                        w_end = min((j + 1) * cell_w, W)
                        # Simple blur: replace with mean
                        mean_val = image[:, h_start:h_end, w_start:w_end].mean(
                            axis=(1, 2), keepdims=True
                        )
                        perturbed_image[:, h_start:h_end, w_start:w_end] = mean_val

        elif perturbation_function == "amplitude":
            # Reduce amplitude (multiply by small factor)
            for i in range(grid_size):
                for j in range(grid_size):
                    if pert_mask[i, j] == 1:
                        h_start = i * cell_h
                        h_end = min((i + 1) * cell_h, H)
                        w_start = j * cell_w
                        w_end = min((j + 1) * cell_w, W)
                        perturbed_image[:, h_start:h_end, w_start:w_end] *= 0.1

        return perturbed_image

    def _get_model_output(
        self,
        image,  # [C, H, W] numpy array
        ob,
        t,
        h_t_input,
        language_features,
        language_inputs,
        language_attention_mask,
        token_type_ids,
        original_features,  # [V, D]
        target_action,
        cand_idx,
        non_cand_idx,
        vp_idx=None,
        perturbed_image_tensor=None,
    ):
        """
        Get model output for a perturbed image.

        Returns:
            output: Scalar output (probability of target action)
        """
        # Convert image to tensor if needed
        if perturbed_image_tensor is None:
            image_tensor = torch.from_numpy(image).unsqueeze(0).cuda()  # [1, C, H, W]
        else:
            image_tensor = torch.from_numpy(perturbed_image_tensor).unsqueeze(0).cuda()

        # Get feature for this viewpoint
        with torch.no_grad():
            feat_vp, _ = self.feature_model(image_tensor)
            feat_vp = feat_vp[:, :, 0, 0]  # [1, D]

        # Reconstruct full feature tensor
        # original_features is [V, D], we need to add batch dimension
        if len(original_features.shape) == 2:
            feat_full = original_features.unsqueeze(0).clone()  # [1, V, D]
        else:
            feat_full = original_features.clone()

        B, V, D = feat_full.shape
        if vp_idx is not None and vp_idx < V:
            feat_full[0, vp_idx] = feat_vp[0]

        # Forward through model
        with torch.no_grad():
            nav_logits, a_t, h_t, _ = self.do_forward(
                self.bert,
                [ob],
                t,
                h_t_input,
                language_features,
                language_inputs,
                language_attention_mask,
                token_type_ids,
                feat_full,
            )
            nav_probs = torch.softmax(nav_logits, 1)
            target_prob = nav_probs[0, target_action]

        return target_prob

    def compute_hsic_attribution_navgpt2(
        self,
        target_obs,  # Changed from obs to target_obs - this should be NavGPT2 obs, not VLN-BERT obs
        t,
        target_agent,
        gmaps,
        instructions,
        navgpt2_gen_action_fn,  # NavGPT2_genAction_v2 function passed as parameter
        grid_size=8,
        nb_design=500,
        perturbation_function="inpainting",
        batch_size=32,
    ):
        """
        Compute HSIC Attribution Method for NavGPT2 (not VLN-BERT).

        This version uses NavGPT2's visual encoder and action prediction,
        instead of VLN-BERT as a surrogate model.

        Args:
            target_obs: List of target observations (NavGPT2 format, NOT VLN-BERT obs)
            t: Time step
            target_agent: NavGPT2 agent (GMapNavAgent instance)
            gmaps: List of GraphMap objects for NavGPT2
            instructions: List of instruction strings
            navgpt2_gen_action_fn: Function to call NavGPT2 action generation (NavGPT2_genAction_v2)
            grid_size: Size of the grid (grid_size x grid_size)
            nb_design: Number of Monte Carlo samples
            perturbation_function: Type of perturbation
            batch_size: Batch size (not directly used, reserved for future)

        Returns:
            images_return: np.array [B, V, H, W, C] where V = VIEWPOINT_SIZE
            heatmaps: list of np.array [V, H, W] - HSIC attribution maps
            candidata_list: list of candidate viewpoint indices
        """
        import copy
        from PIL import Image
        from torchvision import transforms

        bs = len(target_obs)

        # Get panorama images
        images_numpys = []
        for i in range(bs):
            scanId = target_obs[i]["scan"]
            viewpointId = target_obs[i]["viewpoint"]
            images_numpy = self.get_vp_images(self.sim, scanId, viewpointId)
            images_numpys.append(images_numpy)
        images_numpys = np.stack(images_numpys)  # [bs, vp, C, H, W]

        # Debug: Check original target_obs feature format before calling NavGPT2
        print("=" * 80)
        print("DEBUG: Checking original target_obs feature format")
        for i, ob in enumerate(target_obs):
            print(f"  target_obs[{i}] has {len(ob.get('candidate', []))} candidates")
            for j, cc in enumerate(
                ob.get("candidate", [])[:3]
            ):  # Check first 3 candidates
                if "feature" in cc:
                    feat = cc["feature"]
                    if isinstance(feat, np.ndarray):
                        print(
                            f"    candidate[{j}] feature shape: {feat.shape}, dtype: {feat.dtype}"
                        )
                    else:
                        print(f"    candidate[{j}] feature type: {type(feat)}")
        print("=" * 80)

        # Get original NavGPT2 action
        # Note: Now using target_obs (NavGPT2 format) instead of obs (VLN-BERT format)
        with torch.no_grad():
            a_t_original, nav_vpids_list, nav_inputs_dict = navgpt2_gen_action_fn(
                target_agent,
                target_obs,  # Use target_obs (NavGPT2 format) instead of obs
                gmaps,
                instructions,
                t,
                ended=None,
                feedback="argmax",
            )
        target_action = a_t_original[0]  # Original action index

        B, V, C, H, W = images_numpys.shape
        assert B == 1, "This version assumes batch size = 1"

        # Get candidate viewpoints from NavGPT2
        # Extract from target_obs structure
        candidata_list = []
        for i, ob in enumerate(target_obs):
            candidate_idxs = []
            for j, cc in enumerate(ob["candidate"]):
                candidate_idx = cc["pointId"]
                candidate_idxs.append(candidate_idx)
            candidata_list.append(candidate_idxs)

        cand_idx = candidata_list[0]  # [vp]

        # Prepare images for return
        images_return = np.array(
            [
                [self.reverse_transforms(image) for image in images]
                for images in images_numpys
            ]
        )  # [B, V, H, W, C]

        # Get NavGPT2's visual encoder
        # Check if visual_encoder exists (it's deleted when load_patch_feature=True)
        if not hasattr(target_agent.NavGPT.llm.Blip2InstructNav, "visual_encoder"):
            # Check load_patch_feature setting
            load_patch_feature = getattr(
                target_agent.NavGPT.llm, "load_patch_feature", None
            )
            if load_patch_feature:
                raise ValueError(
                    "visual_encoder not found because load_patch_feature=True. "
                    "HSIC attribution requires visual_encoder to encode perturbed images. "
                    "Please set args_target.load_patch_feature=False when initializing the NavGPT2 agent."
                )
            else:
                raise ValueError(
                    "visual_encoder not found. Please check NavGPT2 model initialization. "
                    "HSIC attribution requires visual_encoder to encode perturbed images."
                )

        visual_encoder = target_agent.NavGPT.llm.Blip2InstructNav.visual_encoder
        ln_vision = target_agent.NavGPT.llm.Blip2InstructNav.ln_vision

        # Prepare image transform (same as NavGPT-2 uses)
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Initialize heatmaps
        heatmaps = []

        # Process each batch item
        for i in range(bs):
            # Initialize heatmap for all viewpoints
            heatmap_all = np.zeros((self.VIEWPOINT_SIZE, H, W), dtype=np.float32)

            # Process each candidate viewpoint
            for vp_idx in cand_idx:
                # Get single viewpoint image: [C, H, W]
                single_image = images_numpys[i, vp_idx]  # [C, H, W]

                # Compute HSIC attribution for this viewpoint
                hsic_scores = self._compute_hsic_for_image_navgpt2(
                    single_image,
                    target_obs[i],  # Use target_obs instead of obs
                    t,
                    target_agent,
                    gmaps[i],
                    instructions[i],
                    vp_idx,
                    cand_idx,
                    visual_encoder,
                    ln_vision,
                    transform,
                    target_action,
                    navgpt2_gen_action_fn,  # Pass the function
                    grid_size,
                    nb_design,
                    perturbation_function,
                )

                # Resize HSIC scores from grid_size to H, W
                hsic_map = cv2.resize(
                    hsic_scores, (W, H), interpolation=cv2.INTER_CUBIC
                )

                # Normalize and convert to uint8
                hsic_map = (hsic_map - hsic_map.min()) / (
                    hsic_map.max() - hsic_map.min() + 1e-9
                )
                hsic_map = (hsic_map * 255).clip(0, 255).astype(np.uint8)

                heatmap_all[vp_idx] = hsic_map

            heatmaps.append(heatmap_all)

        return images_return, heatmaps, candidata_list

    def _compute_hsic_for_image_navgpt2(
        self,
        image,  # [C, H, W]
        ob,
        t,
        target_agent,
        gmap,
        instruction,
        vp_idx,
        cand_idx,
        visual_encoder,
        ln_vision,
        transform,
        target_action,
        navgpt2_gen_action_fn,  # NavGPT2_genAction_v2 function
        grid_size,
        nb_design,
        perturbation_function,
    ):
        """
        Compute HSIC scores for a single image using NavGPT2.

        Returns:
            hsic_scores: np.array [grid_size, grid_size] - HSIC scores per grid cell
        """
        import copy
        from PIL import Image

        C, H, W = image.shape

        # Calculate cell size
        cell_h = H // grid_size
        cell_w = W // grid_size

        # Initialize HSIC scores
        hsic_scores = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Generate binary masks
        # np.random.seed(42)  # For reproducibility
        binary_masks = np.random.binomial(
            1, 0.5, size=(nb_design, grid_size, grid_size)
        )

        # Get baseline (original) output
        baseline_output = self._get_model_output_navgpt2(
            image,
            ob,
            t,
            target_agent,
            gmap,
            instruction,
            vp_idx,
            visual_encoder,
            ln_vision,
            transform,
            target_action,
            navgpt2_gen_action_fn,
        )

        # For each grid cell, compute HSIC
        for i in range(grid_size):
            for j in range(grid_size):
                cell_outputs = []

                for mask_idx in range(min(nb_design, len(binary_masks))):
                    # Create perturbation mask for this sample
                    pert_mask = binary_masks[mask_idx].copy()

                    # Apply perturbation to image based on mask
                    perturbed_image = self._apply_perturbation(
                        image.copy(),
                        pert_mask,
                        grid_size,
                        cell_h,
                        cell_w,
                        perturbation_function,
                    )

                    # Get model output with perturbed image
                    output = self._get_model_output_navgpt2(
                        perturbed_image,
                        ob,
                        t,
                        target_agent,
                        gmap,
                        instruction,
                        vp_idx,
                        visual_encoder,
                        ln_vision,
                        transform,
                        target_action,
                        navgpt2_gen_action_fn,
                    )
                    cell_outputs.append(output)

                # Compute HSIC between cell mask and outputs
                cell_presence = binary_masks[: len(cell_outputs), i, j]
                outputs = np.array(cell_outputs)

                # Normalize
                cell_presence = (cell_presence - cell_presence.mean()) / (
                    cell_presence.std() + 1e-9
                )
                outputs = (outputs - outputs.mean()) / (outputs.std() + 1e-9)

                # HSIC approximation: correlation
                hsic_score = np.abs(np.mean(cell_presence * outputs))
                hsic_scores[i, j] = hsic_score

        return hsic_scores

    def _get_model_output_navgpt2(
        self,
        image,  # [C, H, W] numpy array
        ob,
        t,
        target_agent,
        gmap,
        instruction,
        vp_idx,
        visual_encoder,
        ln_vision,
        transform,
        target_action,
        navgpt2_gen_action_fn,  # NavGPT2_genAction_v2 function
    ):
        """
        Get NavGPT2 model output for a perturbed image.

        Returns:
            output: Scalar output (probability or action index matching target_action)
        """
        import copy
        from PIL import Image

        # Convert image from [C, H, W] (BGR, mean-subtracted) to [H, W, C] (RGB, uint8) for PIL
        # Note: get_vp_images returns BGR format with mean subtracted (via transform_img)
        # transform_img subtracts BGR mean [103.1, 115.9, 123.2] and returns [C, H, W]
        # We need to: 1) Add mean back, 2) Convert BGR to RGB, 3) Convert to uint8
        if image.shape[0] == 3:  # C, H, W format
            # Add BGR mean back (reverse the subtraction in transform_img)
            # Mean shape needs to be broadcastable: [3, 1, 1] for [C, H, W]
            bgr_mean = np.array([103.1, 115.9, 123.2]).reshape(3, 1, 1)
            image_denorm = image + bgr_mean  # [C, H, W]
            # Transpose to [H, W, C]
            image_pil = image_denorm.transpose(1, 2, 0)  # H, W, C
            # Convert BGR to RGB (reverse channel order)
            image_pil = image_pil[:, :, ::-1]  # Reverse channel order
        else:
            image_pil = image

        # Clip to valid range and convert to uint8
        image_pil = image_pil.clip(0, 255).astype(np.uint8)

        # Convert to PIL Image and apply transform
        img_pil = Image.fromarray(image_pil)
        tensor_img = transform(img_pil)

        # Encode with visual encoder
        device = next(visual_encoder.parameters()).device
        batch_tensor = tensor_img.unsqueeze(0).to(device)  # [1, 3, 224, 224]

        print(
            f"DEBUG _get_model_output_navgpt2: batch_tensor shape: {batch_tensor.shape}"
        )
        with torch.no_grad():
            # visual_encoder outputs [1, num_patches+1, embed_dim]
            # ln_vision expects [*, normalized_shape] where normalized_shape=1408
            # According to NavGPT_model.py line 475-476, the correct way is:
            # image_embeds = ln_vision(visual_encoder(images))
            image_embeds_raw = visual_encoder(
                batch_tensor
            )  # [1, num_patches+1, embed_dim]
            print(
                f"DEBUG _get_model_output_navgpt2: visual_encoder output shape: {image_embeds_raw.shape}"
            )
            print(
                f"DEBUG _get_model_output_navgpt2: ln_vision normalized_shape: {ln_vision.normalized_shape}"
            )
            # Check if ln_vision expects a different input shape
            # The error suggests ln_vision expects [*, 1408] but got [3, 2176]
            # This means visual_encoder outputs [1, 257, 2176] but ln_vision expects [*, 1408]
            # The issue might be that visual_encoder output needs reshaping or ln_vision is called incorrectly
            # Try calling ln_vision directly on visual_encoder output (as in NavGPT_model.py)
            try:
                image_embeds = ln_vision(image_embeds_raw)  # [1, num_patches+1, 1408]
                print(
                    f"DEBUG _get_model_output_navgpt2: ln_vision output shape: {image_embeds.shape}"
                )
            except RuntimeError as e:
                # If dimension mismatch, try to understand the actual shapes
                print(f"ERROR in ln_vision: {e}")
                print(f"  visual_encoder output shape: {image_embeds_raw.shape}")
                print(f"  ln_vision normalized_shape: {ln_vision.normalized_shape}")
                raise
            # Extract full sequence for NavGPT2 (not just CLS token)
            # NavGPT2 expects features in format (257, 1024) or (257, 1408)
            # We need to construct the full feature format: (257, feature_dim)
            if image_embeds.dim() == 3:
                # Keep full sequence (257, feature_dim) for NavGPT2
                features_np = image_embeds.cpu().numpy()[
                    0
                ]  # [257, feature_dim] = [257, 1408]
            else:
                features_np = image_embeds.cpu().numpy()[0]
            print(
                f"DEBUG _get_model_output_navgpt2: features_np shape (full sequence): {features_np.shape}"
            )

        # Create a deep copy of obs and replace feature for the specific candidate
        # Use custom deep copy that handles non-picklable objects (like MatterSim.ViewPoint)
        def safe_deepcopy(obj, memo=None):
            """Deep copy that skips non-picklable objects (keeps reference instead)."""
            if memo is None:
                memo = {}

            # Check if already copied
            obj_id = id(obj)
            if obj_id in memo:
                return memo[obj_id]

            # Handle basic types
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, np.ndarray):
                # Deep copy numpy arrays
                result = obj.copy()
                memo[obj_id] = result
                return result
            elif isinstance(obj, dict):
                # Deep copy dictionaries
                result = {}
                memo[obj_id] = result
                for key, value in obj.items():
                    try:
                        result[safe_deepcopy(key, memo)] = safe_deepcopy(value, memo)
                    except (TypeError, AttributeError):
                        # If key or value cannot be deep copied, keep reference
                        result[key] = value
                return result
            elif isinstance(obj, (list, tuple)):
                # Deep copy lists and tuples
                result = []
                memo[obj_id] = result
                for item in obj:
                    try:
                        result.append(safe_deepcopy(item, memo))
                    except (TypeError, AttributeError):
                        # If item cannot be deep copied, keep reference
                        result.append(item)
                return tuple(result) if isinstance(obj, tuple) else result
            else:
                # For other types (like MatterSim.ViewPoint), try to deep copy
                # If it fails, keep reference
                try:
                    return copy.deepcopy(obj, memo)
                except (TypeError, AttributeError):
                    # Keep reference for non-picklable objects
                    memo[obj_id] = obj
                    return obj

        ob_copy = safe_deepcopy(ob)
        candidates = ob_copy["candidate"]

        # Find the candidate index corresponding to vp_idx
        cand_list_idx = None
        for j, cc in enumerate(candidates):
            if cc["pointId"] == vp_idx:
                cand_list_idx = j
                break

        if cand_list_idx is not None and cand_list_idx < len(candidates):
            # Replace the feature
            # NavGPT2 expects features in format (257, 1024) or (257, 1408) depending on model config
            # From _local_feature_variable: view_img_fts.append(cc["feature"])  # (257, 1024) or (3, 224, 224)
            # CRITICAL: We must preserve the old_feature's shape exactly to avoid dimension mismatch errors
            # The error shows ln_vision expects [*, 1408] but got [3, 2176], which suggests
            # the feature format in obs might be different from what we're generating
            old_feature = candidates[cand_list_idx]["feature"]
            if isinstance(old_feature, np.ndarray):
                # Debug: print shapes to understand the mismatch
                print(
                    f"DEBUG: old_feature shape: {old_feature.shape}, features_np shape: {features_np.shape}"
                )

                # Handle different feature formats
                if len(old_feature.shape) == 2 and old_feature.shape[0] > 1:
                    # Old feature is 2D (257, 1024) or (257, 1408)
                    # Replace with new full sequence if dimensions match
                    if old_feature.shape == features_np.shape:
                        # Same shape, direct replacement
                        candidates[cand_list_idx]["feature"] = features_np.copy()
                        print(
                            f"DEBUG: Replaced 2D feature with matching shape: {features_np.shape}"
                        )
                    elif old_feature.shape[0] == features_np.shape[0]:
                        # Same sequence length, replace entire sequence
                        candidates[cand_list_idx]["feature"] = features_np.copy()
                        print(
                            f"DEBUG: Replaced 2D feature (same seq len). Old: {old_feature.shape}, New: {features_np.shape}"
                        )
                    else:
                        # Different sequence length, only replace CLS token
                        new_feature = old_feature.copy()
                        if (
                            features_np.shape[0] > 0
                            and old_feature.shape[1] == features_np.shape[1]
                        ):
                            new_feature[0] = features_np[0]  # Replace CLS token only
                        candidates[cand_list_idx]["feature"] = new_feature
                        print(
                            f"DEBUG: Replaced CLS token only. Old: {old_feature.shape}, New CLS: {features_np[0].shape}"
                        )
                elif len(old_feature.shape) == 1:
                    # Old feature is 1D (2176,) - need to convert to 2D format
                    # If features_np is 2D (257, 1408), use it directly
                    if len(features_np.shape) == 2:
                        candidates[cand_list_idx]["feature"] = features_np.copy()
                        print(
                            f"DEBUG: Converted 1D to 2D. Old: {old_feature.shape}, New: {features_np.shape}"
                        )
                    else:
                        # features_np is also 1D, keep it as is (but this might cause issues)
                        candidates[cand_list_idx]["feature"] = features_np.copy()
                        print(
                            f"DEBUG: Replaced 1D feature directly. Old: {old_feature.shape}, New: {features_np.shape}"
                        )
                else:
                    # For other formats, try to replace with new features
                    candidates[cand_list_idx]["feature"] = features_np.copy()
                    print(
                        f"DEBUG: Replaced feature in other format. Old: {old_feature.shape}, New: {features_np.shape}"
                    )

        # Call NavGPT2_genAction_v2 with perturbed obs using the passed function
        with torch.no_grad():
            a_t, nav_vpids_list, nav_inputs_dict = navgpt2_gen_action_fn(
                target_agent,
                [ob_copy],
                [gmap],
                [instruction],
                t,
                ended=None,
                feedback="argmax",
            )

        # Return 1 if action matches target_action, 0 otherwise
        # This gives us a binary output for HSIC computation
        return 1.0 if a_t[0] == target_action else 0.0
