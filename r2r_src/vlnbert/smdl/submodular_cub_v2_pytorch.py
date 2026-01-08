# -*- coding: utf-8 -*-

"""
PyTorch version of CubSubModularExplanationV2
Translated from TensorFlow/Keras to PyTorch
"""

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
import cv2
from PIL import Image

from collections import OrderedDict

import time

import torchvision.transforms as transforms

from tqdm import tqdm

from vlnbert.IG_utils import Exp


class CubSubModularExplanationV2(Exp):
    def __init__(
        self,
        bert,
        critical_head,
        k=40,
        lambda1=1.0,
        lambda2=1.0,
        lambda3=1.0,
        lambda4=1.0,
        max_batch_size=32,
    ):
        super(CubSubModularExplanationV2, self).__init__(bert, critical_head)

        # Parameters of the submodular / submodular的超参数
        self.k = k

        # Parameter of the LtLG algorithm / LtLG贪婪算法的参数
        self.ltl_log_ep = 5

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4

        # Maximum batch size for do_forward to avoid OOM and slowdown
        # Larger batches will be split into chunks and processed sequentially
        self.max_batch_size = max_batch_size

        # PyTorch softmax (replaces tf.keras.layers.Softmax)
        self.softmax = nn.Softmax(dim=-1)

    def compute_uncertainty(self, logits):
        """
        Compute uncertainty using entropy of recognition model predictions.

        Uncertainty is calculated as normalized entropy:
        u = -sum(P(y_i|x) * log(P(y_i|x))) / log(C)

        where:
        - P(y_i|x) is the softmax probability of class i given input x
        - C is the number of classes
        - The entropy is normalized by log(C) to get a value between 0 and 1

        Args:
            logits: numpy array (batch, num_classes) or torch tensor

        Returns:
            uncertainty: numpy array of shape (batch,) with normalized entropy values
        """

        # Apply softmax to get probabilities
        if isinstance(logits, torch.Tensor):
            probs = self.softmax(logits)
        else:
            probs = torch.from_numpy(logits).to("cuda")
            probs = self.softmax(probs)

        # Compute entropy: -sum(P(y_i|x) * log(P(y_i|x)))
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        log_probs = torch.log(probs + eps)
        entropy = -torch.sum(probs * log_probs, dim=1)

        # Normalize by log(C) to get uncertainty in [0, 1]
        # log_C = math.log(self.num_classes)
        log_C = math.log(probs.shape[1])
        uncertainty = entropy / log_C

        # Check if requires_grad before .detach().cpu().numpy()
        if isinstance(uncertainty, torch.Tensor):
            if uncertainty.requires_grad:
                return uncertainty.detach().cpu().numpy()
            else:
                return uncertainty.cpu().numpy()
        return uncertainty

    def compute_effectiveness_score(self, features):
        """
        Computes Effectiveness Score: The point should be distant from all the other elements in the subset.
        features: torch.Tensor of shape (batch, d)
        Return:
            shape()(0-dimensional tensor)

        """
        # PyTorch equivalent of tf.nn.l2_normalize
        norm_feature = F.normalize(features, p=2, dim=1)

        # Cosine Similarity: Use torch.einsum to avoid CUBLAS_STATUS_EXECUTION_FAILED
        cosine_similarity = torch.einsum("ik,jk->ij", norm_feature, norm_feature)

        # PyTorch equivalent of tf.clip_by_value
        cosine_similarity = torch.clamp(cosine_similarity, -1, 1)

        # Normalize 0-1: PyTorch equivalent of tf.acos
        cosine_dist = torch.acos(cosine_similarity) / math.pi

        if cosine_dist.shape[0] == 1:
            # PyTorch equivalent of tf.eye
            eye = 1 - torch.eye(norm_feature.shape[0], device="cuda")
            masked_dist = cosine_dist * eye
            # PyTorch equivalent of tf.reduce_sum(tf.reduce_min(...))
            e_score = torch.sum(torch.min(masked_dist, dim=1)[0])
        else:
            # PyTorch equivalent of tf.eye and operations
            eye = torch.eye(norm_feature.shape[0], device="cuda")
            adjusted_cosine_dist = cosine_dist + eye
            # PyTorch equivalent of tf.reduce_sum(tf.reduce_min(...))
            e_score = torch.sum(torch.min(adjusted_cosine_dist, dim=1)[0])
        return e_score

    def proccess_compute_effectiveness_score(
        self, components_image_feature, combination_list
    ):
        """
        Args:
            Compute each S's effectiveness score
            components_image_feature: torch.Tensor
            combination_list: list of index arrays
        Return:
            1D numpy array with shape (len(combination_list),)
        """
        e_scores = []
        for sub_index in combination_list:
            # PyTorch equivalent of tf.gather
            sub_feature_set = components_image_feature[sub_index]  # shape=(batch, 1024)

            e_score = self.compute_effectiveness_score(sub_feature_set)
            e_scores.append(e_score.item())

        return np.array(e_scores)

    def merge_image(self, images, sub_index_set, partition_mask_set):
        """
        merge image
        """
        # images: [vp, C, H, W]
        sub_mask_set_ = np.array(partition_mask_set)[sub_index_set]

        # Partition_mask_set is expected to have shape [num_superpixels, vp, H, W]
        # images is [vp, C, H, W]

        # Merge superpixel masks for the chosen subset
        # Resulting mask: [vp, H, W]
        mask = sub_mask_set_.sum(0)
        mask = np.clip(mask, 0, 1)  # Ensure mask is 0/1 even if overlaps

        # Apply mask to all channels of each viewpoint
        # images: [vp, C, H, W], mask: [vp, H, W]
        # Expand mask to shape [vp, 1, H, W]
        mask = mask[:, None, :, :]
        images_merged = (images * mask).astype(np.float32)

        return images_merged

    def evaluation_maximun_sample(
        self,
        images,
        main_set,
        candidate_set,
        partition_mask_set,
        call_model_args,
        monotonically_increasing,
    ):
        """
        Given a subset, return a best sample index
        """
        sub_index_sets = []
        for candidate_ in candidate_set:
            sub_index_sets.append(
                np.concatenate((main_set, np.array([candidate_]))).astype(np.int16)
            )  # shape [len(main_set) + 1]

        # Compute uncertainty using entropy of recognition model / 使用识别模型的熵计算不确定性
        # merge images / 组合图像
        start_time = time.perf_counter()
        batch_input_images_u = np.array(
            [
                self.merge_image(images, sub_index_set, partition_mask_set)
                for sub_index_set in sub_index_sets
            ]
        )  # 准备由子集生成的输入图像

        logits_subset, features_subset = self.get_logits_and_feature(
            batch_input_images_u, call_model_args
        )
        # ----------------confidence----------------
        score_confidence = 1 - self.compute_uncertainty(logits_subset)
        confidence_time = time.perf_counter() - start_time
        # print(f"[Score Timing] confidence: {confidence_time:.4f}s")

        # --------------effectiveness----------------
        # Compute Effectiveness Score / 计算有效性分数
        start_time = time.perf_counter()
        batch_partition_image = np.array(
            [
                # self.convert_prepare_image(partition_image)
                images * partition_mask[:, None, :, :]
                for partition_mask in partition_mask_set
            ]
        )

        # Convert to torch tensor and process
        batch_partition_image_tensor = torch.from_numpy(batch_partition_image).to(
            "cuda"
        )

        # Extract features and predictions
        logtis_all, features_all = self.get_logits_and_feature(
            batch_partition_image_tensor, call_model_args
        )

        if features_all.requires_grad:
            features_all = features_all.detach()

        if logtis_all.requires_grad:
            logtis_all = logtis_all.detach()

        score_effectiveness = self.proccess_compute_effectiveness_score(
            # logtis_all, sub_index_sets
            features_all,
            sub_index_sets,
        )
        effectiveness_time = time.perf_counter() - start_time
        # print(f"[Score Timing] effectiveness: {effectiveness_time:.4f}s")

        # ---------------consistency----------------
        # using fully connected layer of the classifier for a specified class
        score_consistency = logits_subset

        # Apply softmax and extract target label
        start_time = time.perf_counter()
        if isinstance(score_consistency, torch.Tensor):
            score_consistency = self.softmax(score_consistency)
            score_consistency = (
                score_consistency[:, self.target_label].squeeze().cpu().numpy()
            )
        else:
            score_consistency = score_consistency.numpy()[
                :, self.target_label
            ].squeeze()
        consistency_time = time.perf_counter() - start_time
        # print(f"[Score Timing] consistency: {consistency_time:.4f}s")

        # ---------------collaboration----------------
        # using fully connected layer of the classifier for a specified class
        start_time = time.perf_counter()
        batch_input_images_reverse = np.array(
            [
                self.merge_image(images, sub_index_set, 1 - partition_mask_set)
                for sub_index_set in sub_index_sets
            ]
        )

        # Convert to torch tensor and process
        batch_input_images_reverse_tensor = torch.from_numpy(
            batch_input_images_reverse
        ).to("cuda")
        logits_reverse, features_reverse = self.get_logits_and_feature(
            batch_input_images_reverse_tensor, call_model_args
        )

        # Apply softmax and extract target label
        if isinstance(logits_reverse, torch.Tensor):
            score_collaboration = self.softmax(logits_reverse)
            score_collaboration = (
                1 - score_collaboration[:, self.target_label].squeeze().cpu().numpy()
            )
        else:
            score_collaboration = (
                1 - logits_reverse.numpy()[:, self.target_label].squeeze()
            )
        collaboration_time = time.perf_counter() - start_time
        # print(f"[Score Timing] collaboration: {collaboration_time:.4f}s")

        # Clear pre-computed features
        del features_all
        torch.cuda.empty_cache()

        # submodular score
        smdl_score = (
            self.lambda1 * score_confidence
            + self.lambda2 * score_effectiveness
            + self.lambda3 * score_consistency
            + self.lambda4 * score_collaboration
        )

        arg_max_index = (
            smdl_score.argmax().item()
            if isinstance(smdl_score, np.ndarray)
            else smdl_score.argmax()
        )

        return sub_index_sets[
            arg_max_index
        ]  # sub_index_sets is [main_set, new_candidate]
        # shape of sub_index_sets[arg_max_index] increase by 1

    def get_merge_set(
        self, images, partition, call_model_args, monotonically_increasing=False
    ):
        """ """
        Subset = np.array([])

        # NOTE: delete later
        indexes = np.arange(len(partition))

        self.smdl_score_best = 0

        # for j in tqdm(range(self.k)):
        for j in tqdm(range(len(indexes))):
            # Sample a subsize of size s.
            diff = np.setdiff1d(
                indexes, np.array(Subset)
            )  # in indexes but not in Subset

            sub_candidate_indexes = diff

            Subset = self.evaluation_maximun_sample(
                images,
                Subset,
                sub_candidate_indexes,
                partition,
                call_model_args,
                monotonically_increasing,
            )

        return Subset

    def SubRegionDivision(
        self,
        images,
        candidate_list,
        segmentation_method="grid",
        grid_rows=2,
        grid_cols=2,
        felzenszwalb_scale=1000,
        felzenszwalb_min_size=500,
    ):
        """
        Divide each candidate viewpoint's image into regions using either grid-based or felzenszwalb segmentation.

        Args:
            images: torch.Tensor or np.ndarray of shape [vp, C, H, W]
            candidate_list: list of vp indices to consider
            segmentation_method: str, either 'grid' or 'felzenszwalb' (default='grid')
                - 'grid': Simple grid-based segmentation, produces exactly grid_rows * grid_cols segments per image
                - 'felzenszwalb': Uses felzenszwalb algorithm, produces variable number of segments
            grid_rows: int, number of rows in grid segmentation (default=4)
            grid_cols: int, number of columns in grid segmentation (default=4)
            felzenszwalb_scale: float, larger values will yield fewer, larger segments (default=1000)
                Only used when segmentation_method='felzenszwalb'
            felzenszwalb_min_size: int, minimum segment size (default=500)
                Only used when segmentation_method='felzenszwalb'

        Return:
            mask tensor of shape [total_superpixels, vp, H, W], where each mask is 1 for pixels in the region, 0 elsewhere.
            Only masks for vps in candidate_list are computed and included.
        """
        import numpy as np

        # Convert to numpy if it's a torch.Tensor
        if hasattr(images, "detach"):
            if images.requires_grad:
                images_np = images.detach().cpu().numpy()
            else:
                images_np = images.cpu().numpy()
        else:
            images_np = images

        vp_num, C, H, W = images_np.shape

        region_labels_per_vp = {}
        num_regions_per_vp = {}

        candidate_list = list(candidate_list)

        if segmentation_method == "grid":
            # Grid-based segmentation - faster and produces exactly grid_rows * grid_cols segments
            for v in candidate_list:
                label_map = np.zeros((H, W), dtype=np.int32)
                region_idx = 0

                # Calculate step sizes
                row_step = H // grid_rows
                col_step = W // grid_cols

                # Create grid regions
                for row in range(grid_rows):
                    row_start = row * row_step
                    row_end = (row + 1) * row_step if row < grid_rows - 1 else H

                    for col in range(grid_cols):
                        col_start = col * col_step
                        col_end = (col + 1) * col_step if col < grid_cols - 1 else W

                        # Assign region label
                        label_map[row_start:row_end, col_start:col_end] = region_idx
                        region_idx += 1

                region_labels_per_vp[v] = label_map
                num_regions_per_vp[v] = grid_rows * grid_cols

        elif segmentation_method == "felzenszwalb":
            # Felzenszwalb segmentation - produces variable number of segments
            from skimage.segmentation import felzenszwalb
            from skimage.util import img_as_ubyte

            for v in candidate_list:
                img_c = images_np[v]  # [C, H, W]
                img_c_np = img_c.transpose(1, 2, 0)
                # Ensure 3 channels
                if img_c_np.shape[2] > 3:
                    img_c_np = img_c_np[:, :, :3]
                elif img_c_np.shape[2] == 1:
                    img_c_np = np.repeat(img_c_np, 3, axis=2)
                # Convert to uint8 if not already for felzenszwalb stability
                if img_c_np.dtype != np.uint8:
                    img_c_np = img_as_ubyte(img_c_np / np.max(img_c_np))
                # Use felzenszwalb segmentation with larger scale/min_size for fewer regions
                label_map = felzenszwalb(
                    img_c_np,
                    scale=felzenszwalb_scale,
                    sigma=0.8,
                    min_size=felzenszwalb_min_size,
                )
                region_labels_per_vp[v] = label_map
                num_regions_per_vp[v] = label_map.max() + 1
        else:
            raise ValueError(
                f"Unknown segmentation_method: {segmentation_method}. Must be 'grid' or 'felzenszwalb'"
            )

        total_regions = sum(num_regions_per_vp[v] for v in candidate_list)
        masks = np.zeros((total_regions, vp_num, H, W), dtype=np.uint8)

        region_global_idx = 0
        for v in candidate_list:
            label_map = region_labels_per_vp[v]
            num_regions = num_regions_per_vp[v]
            for i in range(num_regions):
                mask = (label_map == i).astype(np.uint8)
                masks[region_global_idx, v, :, :] = mask
                region_global_idx += 1

        print(f"mask shape: {masks.shape}, method: {segmentation_method}")
        if segmentation_method == "grid":
            print(
                f"  Grid: {grid_rows}x{grid_cols} = {grid_rows * grid_cols} segments per image"
            )
        else:
            for v in candidate_list:
                print(f"  VP {v}: {num_regions_per_vp[v]} segments")
        return masks  # [total_regions, vp, H, W]

    # def exp(self, image_set, id=None):
    def exp(
        self,
        obs,
        t,
        h_t_input,
        language_features,
        language_inputs,
        language_attention_mask,
        token_type_ids,
    ):
        """
        Compute Source Face Submodular Score
            @image_set: [mask_image 1, ..., mask_image m] (cv2 format)
        """
        images_numpys = []
        for i, ob in enumerate(obs):
            scanId = ob["scan"]
            viewpointId = ob["viewpoint"]
            images_numpy = self.get_vp_images(self.sim, scanId, viewpointId)
            images_numpys.append(images_numpy)
        images_numpys = np.stack(images_numpys, dtype=np.float32)  # [bs, vp, C, H, W]

        # to tensor
        images = torch.autograd.Variable(
            torch.from_numpy(images_numpys), requires_grad=False
        ).cuda()
        # get feature through ResNet-152-InPlace365
        with torch.no_grad():
            feature_tensor = self.get_vp_feature(self.feature_model, images)

        # original action index
        target_nav_logits, target_action, target_h_t, candidate_list = self.do_forward(
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
        self.source_feature, self.target_label = target_nav_logits, target_action

        # Convert to numpy if needed
        if isinstance(self.target_label, torch.Tensor):
            if self.target_label.requires_grad:
                self.target_label = self.target_label.detach().cpu().numpy()
            else:
                self.target_label = self.target_label.cpu().numpy()
        if isinstance(self.source_feature, torch.Tensor):
            if self.source_feature.requires_grad:
                self.source_feature = self.source_feature.detach().cpu().numpy()
            else:
                self.source_feature = self.source_feature.cpu().numpy()

        Subset_merge = self.SubRegionDivision(
            images_numpys[0], candidate_list[0]
        )  # mask [total_superpixels, vp, H, W] (0/1 mask)

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
            candidate_list,
        )

        Submodular_Subset = self.get_merge_set(  # array([30, 31,  1, ...])
            images_numpys[0],
            Subset_merge,
            call_model_args,
            monotonically_increasing=True,
        )

        submodular_image_set = Subset_merge[Submodular_Subset]  # sub_k x (vp, H, W)

        # Generate significance: smaller index = higher importance
        # Score: max_importance for index 0, min_importance for index N-1
        sub_k = len(Submodular_Subset)
        if sub_k == 0:
            # Avoid division by zero. Just fill zeros.
            heatmap = np.zeros(
                (self.VIEWPOINT_SIZE, images_numpys.shape[-2], images_numpys.shape[-1]),
                dtype=np.float32,
            )
        else:
            # Decreasing linear importance: Score = (sub_k - i) for i in 0..sub_k-1
            importance_scores = np.linspace(
                sub_k, 1, sub_k
            )  # Shape: [sub_k,], highest score first
            masks = submodular_image_set.astype(np.float32)  # [sub_k, vp, H, W]
            # Weighted sum of masks: accumulate importance for each pixel
            weighted_masks = (
                importance_scores[:, None, None, None] * masks
            )  # [sub_k, vp, H, W]
            heatmap = weighted_masks.sum(axis=0)  # [vp, H, W]
            # Normalize heatmap to [0, 1] then scale to [0, 255]
            min_val = heatmap.min()
            max_val = heatmap.max()
            if max_val > min_val:
                heatmap = (heatmap - min_val) / (max_val - min_val)
            else:
                heatmap = np.zeros_like(heatmap)
            heatmap = (heatmap * 255).clip(0, 255).astype(np.uint8)  # [vp, H, W]
        heatmap = heatmap.reshape(1, self.VIEWPOINT_SIZE, 224, 224)  # [1, vp, H, W]

        images_return = np.array(
            [
                [self.reverse_transforms(image) for image in images]
                for images in images_numpys
            ]
        )  # [B, vp, H, W, C]

        # Return heatmap for the batch (assuming bs == images_numpys.shape[0])
        # You can return heatmap along with images_return and submodular_image_set if needed.

        return images_return, heatmap, candidate_list

    def get_logits_and_feature(self, input_images, call_model_args):
        """
        Get logits and feature from recognition model.

        Args:
            input_images: numpy array (batch, 3, w, h) or torch tensor

        Returns:
            logits: numpy array of shape (batch, num_classes)
            feature: numpy array of shape (batch, feature_dim)
        """
        with torch.no_grad():
            # Convert to torch tensor if needed
            if isinstance(input_images, np.ndarray):
                input_images = torch.from_numpy(input_images)
            input_images = input_images.to("cuda")
            B, V, C, H, W = input_images.shape

            # # Get predictions from recognition model
            # _, predictions = self.recognition_model(input_images)
            (
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
            ) = call_model_args

            # compute masked feature tensor
            # masked_feature_tensor = self.get_vp_feature(
            #     self.feature_model, input_images
            # )
            masked_feature_tensor, masked_logits = self.get_vp_feature_and_logits(
                self.feature_model, input_images
            )  # [B, vp, FEATURE_SIZE], [B, vp, num_logits]
            masked_feature_tensor = masked_feature_tensor.mean(
                dim=1
            )  # [B, FEATURE_SIZE]
            masked_logits = masked_logits.mean(dim=1)  # [B, num_logits]
            return masked_logits, masked_feature_tensor

            # Helper function to repeat tensor along batch dimension
            def repeat_tensor(tensor, batch_size):
                """Repeat tensor along first dimension to match batch_size"""
                if tensor.dim() == 0:  # scalar
                    return tensor.unsqueeze(0).repeat(batch_size)
                elif tensor.dim() == 1:
                    # Add batch dim and repeat: (dim,) -> (1, dim) -> (B, dim)
                    return tensor.unsqueeze(0).repeat(batch_size, 1)
                else:
                    # Repeat along first dimension: (1, ...) or (dim1, ...) -> (B, ...)
                    repeat_dims = [batch_size] + [1] * (tensor.dim() - 1)
                    return tensor.repeat(*repeat_dims)

            # Helper function to prepare batched args for a given chunk size
            def prepare_batch_args(chunk_size, start_idx=0):
                """Prepare batched arguments for a chunk of size chunk_size starting at start_idx"""
                end_idx = start_idx + chunk_size

                # Repeat obs chunk_size times
                if isinstance(obs, np.ndarray):
                    chunk_obs = np.stack([obs[0] for _ in range(chunk_size)], axis=0)
                else:
                    chunk_obs = [obs] * chunk_size

                # Stack/repeat tensors along batch dimension
                chunk_h_t_input = repeat_tensor(h_t_input, chunk_size)
                chunk_language_features = repeat_tensor(language_features, chunk_size)

                # Handle language_inputs dict - repeat each tensor value
                chunk_language_inputs = {}
                for key, value in language_inputs.items():
                    if isinstance(value, torch.Tensor):
                        chunk_language_inputs[key] = repeat_tensor(value, chunk_size)
                    else:
                        chunk_language_inputs[key] = value

                chunk_language_attention_mask = repeat_tensor(
                    language_attention_mask, chunk_size
                )
                chunk_token_type_ids = repeat_tensor(token_type_ids, chunk_size)

                # Repeat candidate_list
                if isinstance(candidata_list, list):
                    chunk_candidata_list = candidata_list * chunk_size
                else:
                    chunk_candidata_list = [candidata_list] * chunk_size

                # Get corresponding chunk of masked_feature_tensor
                chunk_masked_feature_tensor = masked_feature_tensor[start_idx:end_idx]

                return (
                    chunk_obs,
                    chunk_h_t_input,
                    chunk_language_features,
                    chunk_language_inputs,
                    chunk_language_attention_mask,
                    chunk_token_type_ids,
                    chunk_candidata_list,
                    chunk_masked_feature_tensor,
                )

            # Process in chunks if batch size exceeds max_batch_size to avoid OOM and slowdown
            if B > 1 and B > self.max_batch_size:
                # Split into chunks and process sequentially
                all_logits = []
                all_target_h_t = []

                for start_idx in range(0, B, self.max_batch_size):
                    chunk_size = min(self.max_batch_size, B - start_idx)

                    (
                        chunk_obs,
                        chunk_h_t_input,
                        chunk_language_features,
                        chunk_language_inputs,
                        chunk_language_attention_mask,
                        chunk_token_type_ids,
                        chunk_candidata_list,
                        chunk_masked_feature_tensor,
                    ) = prepare_batch_args(chunk_size, start_idx)

                    # Process chunk
                    (
                        chunk_target_nav_logits,
                        chunk_target_action,
                        chunk_target_h_t,
                        chunk_candidata_list,
                    ) = self.do_forward(
                        self.bert,
                        chunk_obs,
                        t,
                        chunk_h_t_input,
                        chunk_language_features,
                        chunk_language_inputs,
                        chunk_language_attention_mask,
                        chunk_token_type_ids,
                        chunk_masked_feature_tensor,
                    )

                    # Collect results
                    if isinstance(chunk_target_nav_logits, torch.Tensor):
                        all_logits.append(chunk_target_nav_logits)
                    else:
                        all_logits.append(
                            torch.from_numpy(chunk_target_nav_logits).to("cuda")
                        )

                    all_target_h_t.append(chunk_target_h_t)

                # Concatenate all chunks
                logits = torch.cat(all_logits, dim=0)
                target_h_t = torch.cat(all_target_h_t, dim=0)
            elif B > 1:
                # B > 1 but within max_batch_size, process normally
                (
                    batch_obs,
                    batch_h_t_input,
                    batch_language_features,
                    batch_language_inputs,
                    batch_language_attention_mask,
                    batch_token_type_ids,
                    batch_candidata_list,
                    _,
                ) = prepare_batch_args(B)

                # original action index
                target_nav_logits, target_action, target_h_t, candidata_list = (
                    self.do_forward(
                        self.bert,
                        batch_obs,
                        t,
                        batch_h_t_input,
                        batch_language_features,
                        batch_language_inputs,
                        batch_language_attention_mask,
                        batch_token_type_ids,
                        masked_feature_tensor,
                    )
                )

                # Apply softmax to get probabilities
                if isinstance(target_nav_logits, torch.Tensor):
                    logits = target_nav_logits
                else:
                    logits = torch.from_numpy(target_nav_logits).to("cuda")

                target_h_t = target_h_t
            else:
                # B == 1, use original args
                target_nav_logits, target_action, target_h_t, candidata_list = (
                    self.do_forward(
                        self.bert,
                        obs,
                        t,
                        h_t_input,
                        language_features,
                        language_inputs,
                        language_attention_mask,
                        token_type_ids,
                        masked_feature_tensor,
                    )
                )

                # Apply softmax to get probabilities
                if isinstance(target_nav_logits, torch.Tensor):
                    logits = target_nav_logits
                else:
                    logits = torch.from_numpy(target_nav_logits).to("cuda")

                target_h_t = target_h_t

            if logits.requires_grad:
                logits = logits.detach()

            if target_h_t.requires_grad:
                target_h_t = target_h_t.detach()

            return logits, target_h_t
