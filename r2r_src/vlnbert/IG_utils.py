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
        # return new_map
        elif mode == "IG_temporal":
            new_map = map1 * map2
            # normalize
            # Normalize new_map to [0, 1]
            new_map = (new_map - new_map.min()) / (new_map.max() - new_map.min() + 1e-8)
            # Scale to [0, 255]
            new_map = new_map * 255
            new_map = new_map.clip(0, 255).astype(np.uint8)
            return new_map
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
