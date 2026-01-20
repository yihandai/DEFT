import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import cv2
from skimage.transform import resize
from kornia.filters.gaussian import gaussian_blur2d
import json
import copy

try:
    from openai import OpenAI

    OPENAI_NEW_API = True
except ImportError:
    import openai

    OPENAI_NEW_API = False
# 在文件顶部添加
OPENAI_API_KEY = "YOUR_API_KEY"
OPENAI_BASE_URL = "https://api.chatanywhere.tech/v1"
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from param import args
from vlnbert.feature_level_eval_utils import process_perturbed_images
from collections import defaultdict

# HW = 224 * 224


# Plots image from tensor
def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class NpImage(object):
    def __init__(self):
        self.version = "v3"
        self.root_dir = "./tmp_img"
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        self.root = os.path.join(self.root_dir, args.name + "_" + self.version)
        # self.root = "tmp_img_test_og_v3"
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        # Calculate total views: 3 heights * horizontal_views
        self.VIEWPOINT_SIZE = 3 * args.panoramic_horizontal_views

    def reverse_transforms(self, ori_image):
        # (C, H, W)
        if isinstance(ori_image, torch.Tensor):
            image = ori_image.permute(1, 2, 0).detach().cpu().numpy()  # (H, W, C)
        elif isinstance(ori_image, np.ndarray):
            image = ori_image.transpose(1, 2, 0)
        else:
            print("type of image should be tensor or ndarray")
            exit(0)
        image = image + np.array([[[103.1, 115.9, 123.2]]])  # BGR pixel mean
        image = image.clip(0, 255).astype(np.uint8)

        return image

    def save_np2file(
        self, imgs, ob, list_=None, instr_id=None, t=None, perc=None, mode=None
    ):
        # bs = VIEWPOINT_SIZE  # 36
        bs = len(imgs)
        scanId = ob["scan"]
        viewpointId = ob["viewpoint"]

        if list_ is None:
            list_ = range(self.VIEWPOINT_SIZE)

        image_list = []
        for ix in range(bs):
            image_list.append(
                self.save_(
                    imgs[ix], scanId, viewpointId, list_[ix], instr_id, t, perc, mode
                )
            )

        return image_list

    def save_(self, img, scanId, viewpointId, idx, instr_id, t, perc, mode):
        # img: [224, 224]
        # target_path = os.path.join(self.root, scanId, viewpointId)
        target_path = os.path.join(self.root, instr_id, str(t), str(perc), mode)
        # print(target_path)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        # img = self.reverse_transforms(img)

        img_path = os.path.join(target_path, f"{idx}_{scanId}_{viewpointId}.jpg")
        cv2.imwrite(img_path, img)

        return img_path

    def delete_images(self, image_list=None):
        """
        Delete images by index list for a given observation.
        Args:
            ob: dict with keys "scan" and "viewpoint"
            list_: list of indices to delete (default: delete all viewpoints)
        """
        for img_path in image_list:
            if os.path.exists(img_path):
                os.remove(img_path)


class CausalMetric(object):
    def __init__(
        self,
        call_fn,
        substrate_fn,
        W,
        H,
        target="MapGPT",
        openai_api_key=None,
        openai_base_url=None,
    ):
        """Create deletion/insertion metric instance.
        Args:
            model(nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
            openai_api_key: str, OpenAI API key (default: from environment variable OPENAI_API_KEY)
            openai_base_url: str, OpenAI API base URL for third-party proxy (default: from environment variable OPENAI_BASE_URL)
        """
        # self.model = model.eval().cuda()
        self.substrate_fn = substrate_fn
        self.call_fn = call_fn
        self.W = W
        self.H = H
        self.target = target

        # OpenAI API configuration
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.openai_base_url = openai_base_url or OPENAI_BASE_URL

        self.insertion_curr = {
            0.0: {"num": 0, "curr": 0, "rate": 0},
            0.25: {"num": 0, "curr": 0, "rate": 0},
            0.5: {"num": 0, "curr": 0, "rate": 0},
            0.75: {"num": 0, "curr": 0, "rate": 0},
            1.0: {"num": 0, "curr": 0, "rate": 0},
        }
        self.deletion_curr = {
            0.0: {"num": 0, "curr": 0, "rate": 0},
            0.25: {"num": 0, "curr": 0, "rate": 0},
            0.5: {"num": 0, "curr": 0, "rate": 0},
            0.75: {"num": 0, "curr": 0, "rate": 0},
            1.0: {"num": 0, "curr": 0, "rate": 0},
        }

        # self.insertion_curr = {
        #     0.0: {"num": 0, "curr": 0, "rate": 0},
        #     0.25: {"num": 146, "curr": 95, "rate": 0.6507},
        #     0.5: {"num": 146, "curr": 90, "rate": 0.6164},
        #     0.75: {"num": 146, "curr": 102, "rate": 0.6986},
        #     1.0: {"num": 0, "curr": 0, "rate": 0},
        # }
        # self.deletion_curr = {
        #     0.0: {"num": 0, "curr": 0, "rate": 0},
        #     0.25: {"num": 146, "curr": 95, "rate": 0.6507},
        #     0.5: {"num": 146, "curr": 102, "rate": 0.6986},
        #     0.75: {"num": 146, "curr": 89, "rate": 0.6096},
        #     1.0: {"num": 0, "curr": 0, "rate": 0},
        # }

        # self.insertion_curr = {"num": 310, "curr": 136, "rate": 0.43870967741935485}
        # self.deletion_curr = {"num": 310, "curr": 129, "rate": 0.4161290322580645}

    def upsample_numpy(self, x, new_H, new_W):
        """
        Upsample a numpy array by a scale factor.
        Works for [B, H, W, C] and [B, H, W].
        """
        if x.ndim == 4:  # [B, H, W, C]
            B, H, W, C = x.shape
            out = np.zeros((B, new_H, new_W, C), dtype=x.dtype)
            for i in range(B):
                out[i] = resize(
                    x[i],
                    (new_H, new_W, C),
                    order=1,
                    preserve_range=True,
                    anti_aliasing=False,
                )
        elif x.ndim == 3:  # [B, H, W]
            B, H, W = x.shape
            out = np.zeros((B, new_H, new_W), dtype=x.dtype)
            for i in range(B):
                out[i] = resize(
                    x[i],
                    (new_H, new_W),
                    order=1,
                    preserve_range=True,
                    anti_aliasing=False,
                )
        else:
            raise ValueError("Input must be [B,H,W,C] or [B,H,W]")

        return out

    def average_drop(
        self,
        img,  # [V, H, W, C] where V = VIEWPOINT_SIZE
        mask,  # [B, H, W]
        params,
        cls_idx=None,
        verbose=0,
        save_to=None,
        mode="del",
        mask_perc=0.5,
        candidate_idx=None,
    ):
        # `img` is V images (V = VIEWPOINT_SIZE), and mask has the len of len(candidata)
        assert mode in ["del", "ins"]
        if mode == "ins":
            self.substrate_fn = np.zeros_like

        elif mode == "del":
            # Function that blurs input image
            # Make sure the input dimension is (H, W, C), and GaussianBlur expects (H, W, C)
            # so blur will take a single image array of (H, W, C)
            blur = lambda x: cv2.GaussianBlur(x, (51, 51), 50.0)
            # Optionally, you may want to check dimensions here:
            # if x.ndim != 3 or x.shape[2] != 3:
            #     raise ValueError("Input to blur should be (H, W, C)")
            self.substrate_fn = blur

        NUM_PANO, H, W, C = img.shape

        mask = mask.detach().cpu().numpy()  # torch.Tensor --> np.ndarray

        # upsample images and mask if needed
        if H != self.H or W != self.W:
            img = self.upsample_numpy(img, new_H=self.H, new_W=self.W)
            mask = self.upsample_numpy(mask, new_H=self.H, new_W=self.W)

        # num_candidate = len(img)
        img_candidate = img[candidate_idx]
        num_candidate = len(img_candidate)

        if cls_idx is None:
            cls_idx = self.call_fn(*params)

        # num_pixels = int(mask_perc * VIEWPOINT_SIZE * HW)
        # only count the pixels in the first batch
        num_pixels = int(mask_perc * len(candidate_idx) * self.H * self.W)
        if mode == "ins":
            start = []
            for img_i in img_candidate:
                start.append(self.substrate_fn(img_i))
            start = np.stack(start)
            finish = img_candidate.copy()

        if mode == "del":
            start = img_candidate.copy()
            finish = []
            for img_i in img_candidate:
                finish.append(self.substrate_fn(img_i))
            finish = np.stack(finish)

        # 输入形状 (36, H, W)
        salient_order = np.argsort(mask.reshape(-1))[::-1]  # 输出形状 (B*H*W,)
        print(salient_order.shape)

        coords = salient_order[:num_pixels]
        # 1. [B, H, W, 3] --> [B, 3, W, H]
        # 2. 展平为 (B, 3, HW)，HW = H * W
        start_flat = start.transpose(0, 3, 1, 2).reshape(
            num_candidate, 3, -1
        )  # -1 自动计算HW
        finish_flat = finish.transpose(0, 3, 1, 2).reshape(num_candidate, 3, -1)

        # 将全局坐标 coords 分解为子图编号和子图内坐标
        subimage_ids, pixel_indices = (
            coords // (self.H * self.W),
            coords % (self.H * self.W),
        )

        # 批量赋值（无需循环，直接向量化操作）
        start_flat[subimage_ids, :, pixel_indices] = finish_flat[
            subimage_ids, :, pixel_indices
        ]

        # _, nav_logits, _ = self.call_fn(*params, processed_imgs=start)
        # nav_probs = torch.softmax(nav_logits, 1)
        # target_prob = nav_probs[:, cls_idx]

        # c = torch.distributions.Categorical(nav_probs)
        # # print("orign",c.probs)
        # sample_mode = "max"
        # if sample_mode == "sample":
        #     a_t = c.sample()
        # elif sample_mode == "max":
        #     a_t = torch.argmax(c.probs, dim=-1)

        # cls_idx_new = a_t.item()
        cls_idx_new = self.call_fn(
            *params,
            new_imgs=[start],
            candidata_list=[candidate_idx],
        )

        if mode == "ins":
            self.insertion_curr["num"] += 1
            if cls_idx_new[0] == cls_idx:
                curr = 1
            else:
                curr = 0
            self.insertion_curr["curr"] += curr
            self.insertion_curr["rate"] = (
                self.insertion_curr["curr"] / self.insertion_curr["num"]
            )
            print(
                "Insertion sample: {}. Over-all: {}".format(
                    curr, self.insertion_curr["rate"]
                )
            )
        else:
            self.deletion_curr["num"] += 1
            if cls_idx_new[0] == cls_idx:
                curr = 1
            else:
                curr = 0
            self.deletion_curr["curr"] += curr
            self.deletion_curr["rate"] = (
                self.deletion_curr["curr"] / self.deletion_curr["num"]
            )
            print(
                "Deletion sample: {}. Over-all: {}".format(
                    curr, self.deletion_curr["rate"]
                )
            )
        # return scores

    def average_drop2(
        self,
        img,  # [V, H, W, C] where V = VIEWPOINT_SIZE
        mask_rank,  # [valid_pano, H, W]
        mask,  # [valid_pano, H, W]
        params,
        cls_idx=None,
        verbose=0,
        save_to=None,
        mode="del",
        mask_perc=None,
        topK=None,
        candidate_idx=None,
        causal_metric_dir=None,
    ):
        # # test
        # # visualize mask_rank and mask
        # # scale rank to 0~255
        # # normalize mask to 0~255
        # mask_rank = (
        #     (mask_rank - mask_rank.min())
        #     / (mask_rank.max() - mask_rank.min() + 1e-9)
        #     * 255
        # )
        # mask_rank = mask_rank.astype(np.uint8)
        # cv2.imwrite("./mask_rank.png", mask_rank[0])
        # # cv2.imwrite("./mask.png", mask)
        # print("mask_rank", mask_rank[0])
        # # exit(0)
        # `img` is V images (V = VIEWPOINT_SIZE), and mask has the len of len(candidata)
        assert mode in ["del", "ins"]

        if mode == "ins":
            self.substrate_fn = np.zeros_like

        elif mode == "del":
            # Function that blurs input image
            # Make sure the input dimension is (H, W, C), and GaussianBlur expects (H, W, C)
            # so blur will take a single image array of (H, W, C)
            blur = lambda x: cv2.GaussianBlur(x, (51, 51), 50.0)
            # Optionally, you may want to check dimensions here:
            # if x.ndim != 3 or x.shape[2] != 3:
            #     raise ValueError("Input to blur should be (H, W, C)")
            self.substrate_fn = blur

        NUM_PANO, H, W, C = img.shape

        if type(mask_rank) == torch.Tensor:
            mask_rank = mask_rank.detach().cpu().numpy()  # torch.Tensor --> np.ndarray
            mask = mask.detach().cpu().numpy()  # torch.Tensor --> np.ndarray
        elif type(mask_rank) == np.ndarray:
            mask_rank = mask_rank.copy()
            mask = mask.copy()
        else:
            raise ValueError(f"Invalid mask_rank type: {type(mask_rank)}")

        # upsample images and mask if needed
        if H != self.H or W != self.W:
            img = self.upsample_numpy(img, new_H=self.H, new_W=self.W)
            mask_rank = self.upsample_numpy(mask_rank, new_H=self.H, new_W=self.W)
            mask = self.upsample_numpy(mask, new_H=self.H, new_W=self.W)
        # num_candidate = len(img)
        img_candidate = img[candidate_idx]
        num_candidate = len(img_candidate)

        if cls_idx is None:
            cls_idx = self.call_fn(*params)

        # num_pixels = int(mask_perc * VIEWPOINT_SIZE * HW)
        # only count the pixels in the first batch
        # num_pixels = int(mask_perc * len(candidate_idx) * self.H * self.W)

        if mode == "ins":
            start = []
            for img_i in img_candidate:
                start.append(self.substrate_fn(img_i))
            start = np.stack(start)
            finish = img_candidate.copy()

        elif mode == "del":
            start = img_candidate.copy()
            finish = []
            for img_i in img_candidate:
                finish.append(self.substrate_fn(img_i))
            finish = np.stack(finish)

        # 输入形状 (V, H, W) where V = VIEWPOINT_SIZE
        # 输入形状 (V, H, W) where V = VIEWPOINT_SIZE
        if mask_perc is not None:
            topK = int(mask_rank.max() * mask_perc)
        flat_mask_rank = mask_rank.reshape(-1)  # 展平
        flat_mask = mask.reshape(-1)  # 展平
        valid_idx = np.where((flat_mask_rank >= 0) & (flat_mask_rank < topK + 1))[
            0
        ]  # 过滤值在 [1,6) 范围的像素索引
        non_valid_idx = np.where((flat_mask_rank < 0) | (flat_mask_rank >= topK + 1))[0]

        # 只在这些索引里做排序
        coords = valid_idx[np.argsort(flat_mask_rank[valid_idx])[::-1]]
        # salient_order 依然是一维索引，代表满足条件且排序后的像素位置

        # salient_order = np.argsort(mask.reshape(-1))[::-1]  # 输出形状 (B*H*W,)
        # print(salient_order.shape)

        # coords = salient_order[:num_pixels]

        # 1. [B, H, W, 3] --> [B, 3, W, H]
        # 2. 展平为 (B, 3, HW)，HW = H * W
        start_flat = start.transpose(0, 3, 1, 2).reshape(
            num_candidate, 3, -1
        )  # -1 自动计算HW
        finish_flat = finish.transpose(0, 3, 1, 2).reshape(num_candidate, 3, -1)

        # 将全局坐标 coords 分解为子图编号和子图内坐标
        subimage_ids, pixel_indices = (
            coords // (self.H * self.W),
            coords % (self.H * self.W),
        )

        # 批量赋值（无需循环，直接向量化操作）
        start_flat[subimage_ids, :, pixel_indices] = finish_flat[
            subimage_ids, :, pixel_indices
        ]
        obs = params[1]
        ob = obs[0]
        instr_id = ob["instr_id"]
        # print("vp: ", ob["viewpoint"])
        # for _ in range(5):
        prediction = self.call_fn(
            *params,
            new_imgs=[start],
            candidata_list=[candidate_idx],
            instr_id=instr_id,
            perc=mask_perc,
            mode=mode,
        )

        if self.target == "MapGPT":
            print("gt{}\tprediction{}".format(cls_idx, prediction[0]))
            cls_idx_new = prediction[0]
        elif self.target == "NavGPT2":
            print("gt{}\tprediction{}".format(cls_idx, prediction[0]))
            cls_idx_new = prediction[0]

        if mode == "ins":
            self.insertion_curr[mask_perc]["num"] += 1
            if cls_idx_new[0] == cls_idx:
                curr = 1
            else:
                curr = 0
            self.insertion_curr[mask_perc]["curr"] += curr
            self.insertion_curr[mask_perc]["rate"] = (
                self.insertion_curr[mask_perc]["curr"]
                / self.insertion_curr[mask_perc]["num"]
            )
            print(
                "Insertion sample: {}. Over-all: {} for mask percentage: {}".format(
                    curr, self.insertion_curr[mask_perc]["rate"], mask_perc
                )
            )
        else:
            self.deletion_curr[mask_perc]["num"] += 1
            if cls_idx_new[0] == cls_idx:
                curr = 1
            else:
                curr = 0
            self.deletion_curr[mask_perc]["curr"] += curr
            self.deletion_curr[mask_perc]["rate"] = (
                self.deletion_curr[mask_perc]["curr"]
                / self.deletion_curr[mask_perc]["num"]
            )
            print(
                "Deletion sample: {}. Over-all: {} for mask percentage: {}".format(
                    curr, self.deletion_curr[mask_perc]["rate"], mask_perc
                )
            )
        # NOTE: ?
        if args.feature_level_baseline == "smdl":
            flat_mask = flat_mask[::-1]
        collect_consistency_importance_score = True
        if collect_consistency_importance_score:
            if mode == "ins":
                self.collect_consistency_importance_score(
                    params[1],
                    params[2],
                    curr,
                    flat_mask[non_valid_idx],
                    causal_metric_dir,
                    mask_perc=mask_perc,
                    mode="Insertion",
                )
            else:
                self.collect_consistency_importance_score(
                    params[1],
                    params[2],
                    curr,
                    flat_mask[valid_idx],
                    causal_metric_dir,
                    mask_perc=mask_perc,
                    mode="Deletion",
                )
            # save temporary results in a better structured format
            temp_results = {
                "mode": mode,
                "mask_percentage": mask_perc,
                "stats": (
                    self.insertion_curr[mask_perc]
                    if mode == "ins"
                    else self.deletion_curr[mask_perc] if mode == "del" else None
                ),
            }

            causal_metric_dir = os.path.join("snap", args.name, "causal_metric")
            if temp_results["stats"] is None:
                raise ValueError(f"Invalid mode: {mode}")
            # NOTE: delete later
            # with open(os.path.join(causal_metric_dir, "temporary_results.json"), "a+") as f:
            #     json.dump(temp_results, f, indent=4)

    def average_drop_navgpt2(
        self,
        img,  # [V, H, W, C] where V = VIEWPOINT_SIZE
        mask_rank,  # [valid_pano, H, W]
        mask,  # [valid_pano, H, W]
        params,
        cls_idx=None,
        verbose=0,
        save_to=None,
        mode="del",
        mask_perc=None,
        topK=None,
        candidate_idx=None,
        causal_metric_dir=None,
    ):
        """
        Average drop metric for NavGPT-2 with RGB image perturbation.

        This function handles RGB image perturbation by:
        1. Applying perturbation to RGB images
        2. Encoding perturbed images to features using visual_encoder
        3. Replacing features in obs for candidate_idx
        4. Calling NavGPT2_genAction_v2 for inference
        """
        assert mode in ["del", "ins"]

        # Import here to avoid circular imports
        from PIL import Image
        import torchvision.transforms as transforms

        if mode == "ins":
            self.substrate_fn = np.zeros_like
        elif mode == "del":
            blur = lambda x: cv2.GaussianBlur(x, (51, 51), 50.0)
            self.substrate_fn = blur

        NUM_PANO, H, W, C = img.shape

        if type(mask_rank) == torch.Tensor:
            mask_rank = mask_rank.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()
        elif type(mask_rank) == np.ndarray:
            mask_rank = mask_rank.copy()
            mask = mask.copy()
        else:
            raise ValueError(f"Invalid mask_rank type: {type(mask_rank)}")

        # Upsample images and mask if needed
        if H != self.H or W != self.W:
            img = self.upsample_numpy(img, new_H=self.H, new_W=self.W)
            mask_rank = self.upsample_numpy(mask_rank, new_H=self.H, new_W=self.W)
            mask = self.upsample_numpy(mask, new_H=self.H, new_W=self.W)

        img_candidate = img[candidate_idx]
        num_candidate = len(img_candidate)

        # Get original prediction
        if cls_idx is None:
            cls_idx = self.call_fn(*params)

        # Prepare start and finish images based on mode
        if mode == "ins":
            start = []
            for img_i in img_candidate:
                start.append(self.substrate_fn(img_i))
            start = np.stack(start)
            finish = img_candidate.copy()
        elif mode == "del":
            start = img_candidate.copy()
            finish = []
            for img_i in img_candidate:
                finish.append(self.substrate_fn(img_i))
            finish = np.stack(finish)

        # Apply perturbation based on mask_rank
        if mask_perc is not None:
            topK = int(mask_rank.max() * mask_perc)
        flat_mask_rank = mask_rank.reshape(-1)
        flat_mask = mask.reshape(-1)
        valid_idx = np.where((flat_mask_rank >= 0) & (flat_mask_rank < topK + 1))[0]
        non_valid_idx = np.where((flat_mask_rank < 0) | (flat_mask_rank >= topK + 1))[0]

        # Sort valid indices by mask_rank
        coords = valid_idx[np.argsort(flat_mask_rank[valid_idx])[::-1]]

        # Reshape images for pixel-level manipulation
        start_flat = start.transpose(0, 3, 1, 2).reshape(num_candidate, 3, -1)
        finish_flat = finish.transpose(0, 3, 1, 2).reshape(num_candidate, 3, -1)

        # Decompose global coordinates into subimage IDs and pixel indices
        subimage_ids, pixel_indices = (
            coords // (self.H * self.W),
            coords % (self.H * self.W),
        )

        # Apply perturbation
        start_flat[subimage_ids, :, pixel_indices] = finish_flat[
            subimage_ids, :, pixel_indices
        ]

        # # Reshape back to (num_candidate, H, W, C)
        # start = start_flat.reshape(num_candidate, 3, self.H, self.W).transpose(
        #     0, 2, 3, 1
        # )

        # Get agent and obs from params
        agent = params[0]
        obs = params[1]
        gmaps = params[2]
        instructions = params[3]
        t = params[4]
        ended = params[5]

        # Get visual encoder from agent
        if not hasattr(agent.NavGPT.llm.Blip2InstructNav, "visual_encoder"):
            raise ValueError(
                "visual_encoder not found. Please set args.load_patch_feature=False "
                "to prevent visual_encoder from being deleted."
            )

        visual_encoder = agent.NavGPT.llm.Blip2InstructNav.visual_encoder
        ln_vision = agent.NavGPT.llm.Blip2InstructNav.ln_vision

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

        # Encode perturbed RGB images to features
        device = next(visual_encoder.parameters()).device
        batch_images = []
        for img_i in start:
            # Convert numpy array (H, W, C) to PIL Image
            img_pil = Image.fromarray(img_i.astype(np.uint8))
            tensor_img = transform(img_pil)
            batch_images.append(tensor_img)
        batch_tensor = torch.stack(batch_images, dim=0).to(
            device
        )  # (num_candidate, 3, 224, 224)

        # Encode with visual encoder
        with torch.no_grad():
            image_embeds = visual_encoder(batch_tensor)
            image_embeds = ln_vision(image_embeds)
            if image_embeds.dim() == 3:
                # Use CLS token (first token) or all tokens
                features = image_embeds[:, 0, :]  # (num_candidate, feature_dim)
            else:
                features = image_embeds

        features_np = features.cpu().numpy()  # (num_candidate, feature_dim)

        # Create a deep copy of obs and replace features for candidate_idx
        # Use deepcopy to ensure nested structures (candidate list, feature arrays) are independent

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

        obs_perturbed = []
        for i, ob in enumerate(obs):
            # Deep copy to ensure all nested structures are independent
            ob_copy = safe_deepcopy(ob)
            print("i: ", i)
            if i == 0:  # Only process first batch item
                # Replace features for candidates in candidate_idx
                # candidate_idx contains viewpoint IDs (pointId) from VLN-BERT env
                # We need to find corresponding candidates in target_obs (NavGPT-2 env) by viewpoint ID
                candidates = ob_copy["candidate"]

                # Build a mapping from viewpoint ID (pointId) to candidate index in target_obs
                target_candidate_id_to_idx = {}
                for idx, cand in enumerate(candidates):
                    if "pointId" in cand:
                        target_candidate_id_to_idx[cand["pointId"]] = idx
                    elif "viewpointId" in cand:
                        # Some formats use viewpointId instead of pointId
                        target_candidate_id_to_idx[cand["viewpointId"]] = idx

                print("candidate_idx (viewpoint IDs from VLN-BERT): ", candidate_idx)
                print("target candidate IDs: ", list(target_candidate_id_to_idx.keys()))
                print("len(features_np): ", len(features_np))
                print("len(candidates): ", len(candidates))

                for j, cand_viewpoint_id in enumerate(candidate_idx):
                    print("j: ", j, "cand_viewpoint_id: ", cand_viewpoint_id)
                    # Find the corresponding candidate index in target_obs by viewpoint ID
                    if cand_viewpoint_id in target_candidate_id_to_idx:
                        target_cand_idx = target_candidate_id_to_idx[cand_viewpoint_id]
                        if j < len(features_np) and target_cand_idx < len(candidates):
                            # Replace the feature
                            # NavGPT-2 expects features in format (257, 1024) or similar
                            # We need to construct the full feature format
                            old_feature = candidates[target_cand_idx]["feature"]
                            print("old_feature: ", type(old_feature))
                            if isinstance(old_feature, np.ndarray):
                                print("old_feature.shape: ", old_feature.shape)
                                # If old feature is (257, 1024), we need to replace CLS token
                                if (
                                    len(old_feature.shape) == 2
                                    and old_feature.shape[0] > 1
                                ):
                                    # Replace CLS token (first row) with new feature
                                    new_feature = old_feature.copy()
                                    new_feature[0] = features_np[j]
                                    candidates[target_cand_idx]["feature"] = new_feature
                                elif len(old_feature.shape) == 1:
                                    # If old feature is 1D, replace it directly
                                    candidates[target_cand_idx]["feature"] = (
                                        features_np[j].copy()
                                    )
                                else:
                                    # For other formats, try to replace CLS token
                                    new_feature = old_feature.copy()
                                    if new_feature.shape[0] > 0:
                                        new_feature[0] = features_np[j]
                                    candidates[target_cand_idx]["feature"] = new_feature
                        else:
                            print(
                                f"Warning: target_cand_idx {target_cand_idx} out of range for features_np[{j}] or candidates"
                            )
                    else:
                        print(
                            f"Warning: viewpoint ID {cand_viewpoint_id} not found in target_obs candidates"
                        )
            obs_perturbed.append(ob_copy)

        # Call NavGPT2_genAction_v2 with perturbed obs
        ob = obs[0]
        instr_id = ob["instr_id"]

        # Call NavGPT2_genAction_v2 with perturbed obs
        # Note: NavGPT2_genAction_v2 returns (a_t, nav_vpids_list, nav_inputs_dict)
        # where a_t is a numpy array of action indices
        a_t, nav_vpids_list, nav_inputs_dict = self.call_fn(
            agent,
            obs_perturbed,
            gmaps,
            instructions,
            t,
            ended=ended,
            feedback="argmax",
            new_imgs=[start],
            candidata_list=[candidate_idx],
            instr_id=instr_id,
            perc=mask_perc,
            mode=mode,
        )

        # Extract action index for the first sample
        if self.target == "NavGPT2":
            cls_idx_new = np.array([a_t[0]])  # Convert to array format for consistency
            print("gt{}\tprediction{}".format(cls_idx, cls_idx_new[0]))
        else:
            cls_idx_new = np.array([a_t[0]])

        # Update statistics
        if mode == "ins":
            self.insertion_curr[mask_perc]["num"] += 1
            if cls_idx_new[0] == cls_idx:
                curr = 1
            else:
                curr = 0
            self.insertion_curr[mask_perc]["curr"] += curr
            self.insertion_curr[mask_perc]["rate"] = (
                self.insertion_curr[mask_perc]["curr"]
                / self.insertion_curr[mask_perc]["num"]
            )
            print(
                "Insertion sample: {}. Over-all: {} for mask percentage: {}".format(
                    curr, self.insertion_curr[mask_perc]["rate"], mask_perc
                )
            )
        else:
            self.deletion_curr[mask_perc]["num"] += 1
            if cls_idx_new[0] == cls_idx:
                curr = 1
            else:
                curr = 0
            self.deletion_curr[mask_perc]["curr"] += curr
            self.deletion_curr[mask_perc]["rate"] = (
                self.deletion_curr[mask_perc]["curr"]
                / self.deletion_curr[mask_perc]["num"]
            )
            print(
                "Deletion sample: {}. Over-all: {} for mask percentage: {}".format(
                    curr, self.deletion_curr[mask_perc]["rate"], mask_perc
                )
            )

        # Collect consistency importance score
        if args.feature_level_baseline == "smdl":
            flat_mask = flat_mask[::-1]
        collect_consistency_importance_score = True
        if collect_consistency_importance_score:
            if mode == "ins":
                self.collect_consistency_importance_score(
                    params[1],
                    params[4],
                    curr,
                    flat_mask[non_valid_idx],
                    causal_metric_dir,
                    mask_perc=mask_perc,
                    mode="Insertion",
                )
            else:
                self.collect_consistency_importance_score(
                    params[1],
                    params[4],
                    curr,
                    flat_mask[valid_idx],
                    causal_metric_dir,
                    mask_perc=mask_perc,
                    mode="Deletion",
                )
            # Save temporary results
            temp_results = {
                "mode": mode,
                "mask_percentage": mask_perc,
                "stats": (
                    self.insertion_curr[mask_perc]
                    if mode == "ins"
                    else self.deletion_curr[mask_perc] if mode == "del" else None
                ),
            }

            causal_metric_dir = os.path.join("snap", args.name, "causal_metric")
            if temp_results["stats"] is None:
                raise ValueError(f"Invalid mode: {mode}")

    def average_drop_navgpt_gentext(
        self,
        img,  # [V, H, W, C] where V = VIEWPOINT_SIZE
        mask_rank,  # [valid_pano, H, W]
        mask,  # [valid_pano, H, W]
        params,
        cls_idx=None,
        verbose=0,
        save_to=None,
        mode="del",
        mask_perc=None,
        topK=None,
        candidate_idx=None,
        causal_metric_dir=None,
    ):
        assert mode in ["del", "ins"]

        if mode == "ins":
            self.substrate_fn = np.zeros_like

        elif mode == "del":
            blur = lambda x: cv2.GaussianBlur(x, (51, 51), 50.0)
            self.substrate_fn = blur

        NUM_PANO, H, W, C = img.shape

        if type(mask_rank) == torch.Tensor:
            mask_rank = mask_rank.detach().cpu().numpy()  # torch.Tensor --> np.ndarray
            mask = mask.detach().cpu().numpy()  # torch.Tensor --> np.ndarray
        elif type(mask_rank) == np.ndarray:
            mask_rank = mask_rank.copy()
            mask = mask.copy()
        else:
            raise ValueError(f"Invalid mask_rank type: {type(mask_rank)}")

        # upsample images and mask if needed
        if H != self.H or W != self.W:
            img = self.upsample_numpy(img, new_H=self.H, new_W=self.W)
            mask_rank = self.upsample_numpy(mask_rank, new_H=self.H, new_W=self.W)
            mask = self.upsample_numpy(mask, new_H=self.H, new_W=self.W)
        # num_candidate = len(img)
        img_candidate = img[candidate_idx]
        num_candidate = len(img_candidate)

        # if cls_idx is None:
        #     cls_idx = self.call_fn(*params)

        if mode == "ins":
            start = []
            for img_i in img_candidate:
                start.append(self.substrate_fn(img_i))
            start = np.stack(start)
            finish = img_candidate.copy()

        elif mode == "del":
            start = img_candidate.copy()
            finish = []
            for img_i in img_candidate:
                finish.append(self.substrate_fn(img_i))
            finish = np.stack(finish)

        # 输入形状 (V, H, W) where V = VIEWPOINT_SIZE
        # 输入形状 (V, H, W) where V = VIEWPOINT_SIZE
        if mask_perc is not None:
            topK = int(mask_rank.max() * mask_perc)
        flat_mask_rank = mask_rank.reshape(-1)  # 展平
        flat_mask = mask.reshape(-1)  # 展平
        valid_idx = np.where((flat_mask_rank >= 0) & (flat_mask_rank < topK + 1))[
            0
        ]  # 过滤值在 [1,6) 范围的像素索引
        non_valid_idx = np.where((flat_mask_rank < 0) | (flat_mask_rank >= topK + 1))[0]

        # 只在这些索引里做排序
        coords = valid_idx[np.argsort(flat_mask_rank[valid_idx])[::-1]]
        # salient_order 依然是一维索引，代表满足条件且排序后的像素位置

        # 1. [B, H, W, 3] --> [B, 3, W, H]
        # 2. 展平为 (B, 3, HW)，HW = H * W
        start_flat = start.transpose(0, 3, 1, 2).reshape(
            num_candidate, 3, -1
        )  # -1 自动计算HW
        finish_flat = finish.transpose(0, 3, 1, 2).reshape(num_candidate, 3, -1)

        # 将全局坐标 coords 分解为子图编号和子图内坐标
        subimage_ids, pixel_indices = (
            coords // (self.H * self.W),
            coords % (self.H * self.W),
        )

        # 批量赋值（无需循环，直接向量化操作）
        start_flat[subimage_ids, :, pixel_indices] = finish_flat[
            subimage_ids, :, pixel_indices
        ]

        # 将 start 从 (num_candidate, 3, H*W) 恢复为 (num_candidate, H, W, 3)
        start = start_flat.reshape(num_candidate, 3, self.H, self.W).transpose(
            0, 2, 3, 1
        )

        # 确保图片值在 [0, 255] 范围内，并转换为 uint8
        start = np.clip(start, 0, 255).astype(np.uint8)

        # 使用 BLIP-2 和 Fast-RCNN 处理扰动后的图片
        results = process_perturbed_images(start, blip_prompt="This is a scene of ")

        # 使用 blip_descriptions 更新原始描述
        obs = params[6]  # perm_obs
        ob = obs[0]
        target_obs = params[1]  # target_perm_obs
        target_ob = target_obs[0]

        map_cand_idx_to_viewpointId = self.mapping_cand_idx_to_viewpointId(
            candidate_idx, ob
        )
        map_viewpointId_to_target_cand_idx = (
            self.mapping_viewpointId_to_target_cand_idx(target_ob)
        )

        to_update_dict = defaultdict(list)
        for idx, cand_idx in enumerate(candidate_idx):
            if cand_idx < 8:
                level = "down"
            elif cand_idx < 16:
                level = "middle"
            else:
                level = "top"
            viewpointId = map_cand_idx_to_viewpointId[cand_idx]
            target_cand_idx = map_viewpointId_to_target_cand_idx[viewpointId]

            to_update_dict[target_cand_idx].append(
                {
                    "level": level,
                    "blip_description": results["blip_descriptions"][idx],
                    "detected_objects": results["detected_objects"][idx],
                }
            )
        return_dict = defaultdict(dict)
        for target_cand_idx, updates in to_update_dict.items():
            # get original description and new description
            ori_desc = self.get_original_low_mid_high_descriptions(
                target_cand_idx, target_ob
            )
            new_desc = {}
            for update in updates:
                new_desc[update["level"]] = update["blip_description"]

            # get original objects and new objects
            ori_objects = self.get_original_object_descriptions(
                target_cand_idx, target_ob
            )
            new_objects = []
            for update in updates:
                # 检查 detected_objects 是否为空（某些图片可能没有检测到对象）
                if update["detected_objects"] and len(update["detected_objects"]) > 0:
                    new_objects.append(update["detected_objects"][0]["class_name"])
                # 如果 detected_objects 为空，跳过（不添加任何对象）

            # merge description and objects
            merged_desc = self.merge_high_mid_low_descriptions(ori_desc, new_desc)
            merged_objects = self.merge_object_descriptions(ori_objects, new_objects)
            return_dict[target_cand_idx] = {
                "description": merged_desc,
                "objects": merged_objects,
            }
        return return_dict

    def mapping_cand_idx_to_viewpointId(self, candidate_idx, ob):
        candidate_list = [x["viewpointId"] for x in ob.get("candidate", [])]
        # mappingdict
        mapping = {
            candidate_idx[i]: candidate_list[i] for i in range(len(candidate_idx))
        }
        return mapping

    def mapping_viewpointId_to_target_cand_idx(self, target_ob):
        mapping = {}
        candidate_dict = target_ob.get("candidate", {})
        for vp_id, vp_data in candidate_dict.items():
            vp_heading = np.rad2deg(vp_data["heading"])
            vp_range_idx = int((vp_heading - 22.5) // 45) + 1
            vp_range_idx = vp_range_idx % 8
            mapping[vp_id] = vp_range_idx
        return mapping

    def get_original_low_mid_high_descriptions(self, target_candi_idx, target_ob):
        obs_list = target_ob.get("obs_list", [])  # 8 observations
        # example:
        # "down: a red pig statue is on the stairs\nmiddle: a red pig statue is sitting in front of a large window\ntop: a room with a wooden ceiling and a window",
        # split by `down: `, `\nmiddle: `, `\ntop: `
        description = obs_list[target_candi_idx]
        down_desc = description.split("down: ")[1].split("\nmiddle: ")[0]
        middle_desc = description.split("middle: ")[1].split("\ntop: ")[0]
        top_desc = description.split("top: ")[1]
        return {
            "down": down_desc,
            "middle": middle_desc,
            "top": top_desc,
        }

    def merge_high_mid_low_descriptions(self, ori_desc, new_desc):
        """
        Merge the descriptions of the three levels into a single description.
        Using gpt3.5 api to merge the descriptions.
        The prompt is:
            "Here is a single scene view from top, down and middle:\n{description}\nSummarize the scene in one sentence:"

        where the "{description}" is replaced with the generated text of top, middle and down images

        Args:
            ori_desc: dict, format like {"down": "...", "middle": "...", "top": "..."}
            new_desc: dict, format like {"down": "...", "middle": "...", "top": "..."}

        Returns:
            str: Merged description from GPT-3.5
        """
        # Validate inputs
        if not isinstance(ori_desc, dict):
            raise ValueError(f"ori_desc must be a dict, got {type(ori_desc)}")
        if not isinstance(new_desc, dict):
            raise ValueError(f"new_desc must be a dict, got {type(new_desc)}")

        # Merge descriptions: use new_desc if available, otherwise use ori_desc
        levels = ["down", "middle", "top"]
        merged_descriptions = []

        for level in levels:
            # Prefer new_desc over ori_desc
            if level in new_desc and new_desc[level]:
                merged_descriptions.append(f"{level}: {new_desc[level]}")
            elif level in ori_desc and ori_desc[level]:
                merged_descriptions.append(f"{level}: {ori_desc[level]}")

        # Combine descriptions
        combined_desc = "\n".join(merged_descriptions)
        # Call GPT-3.5 API to merge descriptions with retry mechanism
        try:
            response = self._call_gpt_with_retry(combined_desc)
            return response
        except Exception as e:
            print(f"Error calling GPT-3.5 API after retries: {e}")
            # Return combined description as fallback
            return combined_desc

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_random_exponential(min=1, max=60),
    )
    def _call_gpt_with_retry(self, combined_desc):
        """
        Call GPT-3.5 API with retry mechanism using exponential backoff.

        Args:
            combined_desc: str, combined description string

        Returns:
            str: Merged description from GPT-3.5
        """
        prompt = f"Here is a single scene view from top, down and middle:\n{combined_desc}\nSummarize the scene in one sentence:"

        if OPENAI_NEW_API:
            # New API (openai >= 1.0.0)
            # Build client kwargs
            client_kwargs = {}
            if self.openai_api_key:
                client_kwargs["api_key"] = self.openai_api_key
            if self.openai_base_url:
                client_kwargs["base_url"] = self.openai_base_url

            llm = OpenAI(**client_kwargs)
            response = llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            return response.choices[0].message.content
        else:
            # Old API (openai < 1.0.0)
            # Set API key if provided
            if self.openai_api_key:
                openai.api_key = self.openai_api_key
            # Note: Old API may not support base_url, but we can try
            # Some third-party proxies might work with api_base
            api_kwargs = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 100,
            }
            # Try to set base_url if provided (may not work with all old versions)
            if self.openai_base_url:
                try:
                    openai.api_base = self.openai_base_url
                except AttributeError:
                    # Old version doesn't support api_base, ignore
                    pass

            response = openai.ChatCompletion.create(**api_kwargs)
            return response.choices[0].message.content

    def get_original_object_descriptions(self, target_cand_idx, target_ob):
        obs_list = target_ob.get("objects", [])  # 8 views
        # example:
        # "7b99fad7a2b243dea6b50dc65d03fbc7": [
        #     {
        #         "bottle": {
        #             "heading": -3.56,
        #             "distance": 0.55
        #         }
        #     },
        #     {},
        #     {
        #         "towel": {
        #             "heading": 103.41,
        #             "distance": 1.15
        #         }
        #     },
        #     {
        #         "mirror": {
        #             "heading": 137.8,
        #             "distance": 1.86
        #         },
        #         "towel": {
        #             "heading": 131.02,
        #             "distance": 1.15
        #         }
        #     },
        #     {
        #         "mirror": {
        #             "heading": 166.78,
        #             "distance": 1.86
        #         }
        #     },
        #     {},
        #     {},
        #     {
        #         "bottle": {
        #             "heading": 324.28,
        #             "distance": 0.55
        #         }
        #     }
        # ],
        objects = obs_list[target_cand_idx]
        return objects

    def merge_object_descriptions(self, ori_objects, new_objects):
        """
        Merge original objects and new objects.
        If original objects set has object not in new objects, then delete it.

        Args:
            ori_objects: dict, original objects
                example:
                    {
                        "bottle": {
                            "heading": -3.56,
                            "distance": 0.55
                        },
                        "towel": {
                            "heading": 103.41,
                            "distance": 1.15
                        }
                    }
            new_objects: list, new objects (list of object class names)
                example: ["bottle", "towel"]

        Returns:
            dict: Merged objects (only objects that exist in new_objects are kept)
        """
        # Validate inputs
        if not isinstance(ori_objects, dict):
            raise ValueError(f"ori_objects must be a dict, got {type(ori_objects)}")
        if not isinstance(new_objects, list):
            raise ValueError(f"new_objects must be a list, got {type(new_objects)}")

        # Convert new_objects to set for faster lookup
        new_objects_set = set(new_objects)

        # Filter ori_objects: only keep objects that exist in new_objects
        merged_objects = {}
        for obj_name, obj_data in ori_objects.items():
            if obj_name in new_objects_set:
                merged_objects[obj_name] = obj_data
        return merged_objects

    def average_drop_navgpt_inference(
        self,
        img,  # [V, H, W, C] where V = VIEWPOINT_SIZE
        mask_rank,  # [valid_pano, H, W]
        mask,  # [valid_pano, H, W]
        params,
        cls_idx=None,
        verbose=0,
        save_to=None,
        mode="del",
        mask_perc=None,
        topK=None,
        candidate_idx=None,
        causal_metric_dir=None,
        description_update_dir=None,
    ):
        assert mode in ["del", "ins"]

        # NOTE: for sum(importance_score) = sum(mask)
        if mode == "ins":
            self.substrate_fn = np.zeros_like

        elif mode == "del":
            blur = lambda x: cv2.GaussianBlur(x, (51, 51), 50.0)
            self.substrate_fn = blur

        NUM_PANO, H, W, C = img.shape

        if type(mask_rank) == torch.Tensor:
            mask_rank = mask_rank.detach().cpu().numpy()  # torch.Tensor --> np.ndarray
            mask = mask.detach().cpu().numpy()  # torch.Tensor --> np.ndarray
        elif type(mask_rank) == np.ndarray:
            mask_rank = mask_rank.copy()
            mask = mask.copy()
        else:
            raise ValueError(f"Invalid mask_rank type: {type(mask_rank)}")

        # upsample images and mask if needed
        if H != self.H or W != self.W:
            img = self.upsample_numpy(img, new_H=self.H, new_W=self.W)
            mask_rank = self.upsample_numpy(mask_rank, new_H=self.H, new_W=self.W)
            mask = self.upsample_numpy(mask, new_H=self.H, new_W=self.W)
        # num_candidate = len(img)
        img_candidate = img[candidate_idx]
        num_candidate = len(img_candidate)

        # if cls_idx is None:
        #     cls_idx = self.call_fn(*params)

        if mode == "ins":
            start = []
            for img_i in img_candidate:
                start.append(self.substrate_fn(img_i))
            start = np.stack(start)
            finish = img_candidate.copy()

        elif mode == "del":
            start = img_candidate.copy()
            finish = []
            for img_i in img_candidate:
                finish.append(self.substrate_fn(img_i))
            finish = np.stack(finish)

        # 输入形状 (V, H, W) where V = VIEWPOINT_SIZE
        # 输入形状 (V, H, W) where V = VIEWPOINT_SIZE
        if mask_perc is not None:
            topK = int(mask_rank.max() * mask_perc)
        flat_mask_rank = mask_rank.reshape(-1)  # 展平
        flat_mask = mask.reshape(-1)  # 展平
        valid_idx = np.where((flat_mask_rank >= 0) & (flat_mask_rank < topK + 1))[
            0
        ]  # 过滤值在 [1,6) 范围的像素索引
        non_valid_idx = np.where((flat_mask_rank < 0) | (flat_mask_rank >= topK + 1))[0]

        # 只在这些索引里做排序
        coords = valid_idx[np.argsort(flat_mask_rank[valid_idx])[::-1]]
        # salient_order 依然是一维索引，代表满足条件且排序后的像素位置

        # 1. [B, H, W, 3] --> [B, 3, W, H]
        # 2. 展平为 (B, 3, HW)，HW = H * W
        start_flat = start.transpose(0, 3, 1, 2).reshape(
            num_candidate, 3, -1
        )  # -1 自动计算HW
        finish_flat = finish.transpose(0, 3, 1, 2).reshape(num_candidate, 3, -1)

        # 将全局坐标 coords 分解为子图编号和子图内坐标
        subimage_ids, pixel_indices = (
            coords // (self.H * self.W),
            coords % (self.H * self.W),
        )
        # NOTE: end

        # 批量赋值（无需循环，直接向量化操作）
        start_flat[subimage_ids, :, pixel_indices] = finish_flat[
            subimage_ids, :, pixel_indices
        ]
        # load file
        t = params[2]
        obs = params[6]  # perm_obs
        ob = obs[0]
        target_obs = params[1]  # target_perm_obs
        target_ob = target_obs[0]
        instr_id = ob["instr_id"]
        description_update_dir = os.path.join(
            description_update_dir,
            f"{instr_id}",
            f"{t}",
            f"{mask_perc}",
            mode,
        )
        if args.bagging_agents is not None:
            description_file_name = (
                "description_update" + f"_{args.bagging_agents}" + ".json"
            )
        else:
            description_file_name = "description_update.json"
        with open(
            os.path.join(description_update_dir, description_file_name), "r"
        ) as f:
            description_update = json.load(f)
        # format of description_update:
        # {
        #     target_cand_idx_1: {
        #         "description": merged_desc,
        #         "objects": merged_objects,
        #     },
        #     target_cand_idx_2: {
        #         "description": merged_desc,
        #         "objects": merged_objects,
        #     },
        #     ...
        # }
        # merge description and objects

        # NavGPT_genAction_v2 只接受 6 个位置参数，params 包含 7 个元素
        # 只展开前 6 个参数（agent, obs, t, previous_angle, do_inference, ended）
        # description_update 作为关键字参数传递
        prediction = self.call_fn(
            params[0],  # agent
            params[1],  # obs (perm_obs, 当前观察，可能包含扰动后的图像)
            params[2],  # t
            params[3],  # previous_angle
            params[4],  # do_inference
            params[5],  # ended
            description_update=description_update,
        )

        if self.target == "MapGPT":
            print("gt: {}\tprediction: {}".format(cls_idx, prediction[0]))
            cls_idx_new = prediction[0]
        if self.target == "NavGPT":
            print("gt: {}\tprediction: {}".format(cls_idx, prediction[-1]))
            cls_idx_new = prediction[-1]
        if mode == "ins":
            self.insertion_curr[mask_perc]["num"] += 1
            if cls_idx_new[0] == cls_idx:
                curr = 1
            else:
                curr = 0
            self.insertion_curr[mask_perc]["curr"] += curr
            self.insertion_curr[mask_perc]["rate"] = (
                self.insertion_curr[mask_perc]["curr"]
                / self.insertion_curr[mask_perc]["num"]
            )
            print(
                "Insertion sample: {}. Over-all: {} for mask percentage: {}".format(
                    curr, self.insertion_curr[mask_perc]["rate"], mask_perc
                )
            )
        else:
            self.deletion_curr[mask_perc]["num"] += 1
            if cls_idx_new[0] == cls_idx:
                curr = 1
            else:
                curr = 0
            self.deletion_curr[mask_perc]["curr"] += curr
            self.deletion_curr[mask_perc]["rate"] = (
                self.deletion_curr[mask_perc]["curr"]
                / self.deletion_curr[mask_perc]["num"]
            )
            print(
                "Deletion sample: {}. Over-all: {} for mask percentage: {}".format(
                    curr, self.deletion_curr[mask_perc]["rate"], mask_perc
                )
            )
        # NOTE: ?
        if args.feature_level_baseline == "smdl":
            flat_mask = flat_mask[::-1]
        collect_consistency_importance_score = True
        if collect_consistency_importance_score:
            if mode == "ins":
                self.collect_consistency_importance_score(
                    params[1],
                    params[2],
                    curr,
                    flat_mask[non_valid_idx],
                    causal_metric_dir,
                    mask_perc=mask_perc,
                    mode="Insertion",
                )
            else:
                self.collect_consistency_importance_score(
                    params[1],
                    params[2],
                    curr,
                    flat_mask[valid_idx],
                    causal_metric_dir,
                    mask_perc=mask_perc,
                    mode="Deletion",
                )
            # save temporary results in a better structured format
            temp_results = {
                "mode": mode,
                "mask_percentage": mask_perc,
                "stats": (
                    self.insertion_curr[mask_perc]
                    if mode == "ins"
                    else self.deletion_curr[mask_perc] if mode == "del" else None
                ),
            }

            causal_metric_dir = os.path.join("snap", args.name, "causal_metric")
            if temp_results["stats"] is None:
                raise ValueError(f"Invalid mode: {mode}")
            # NOTE: delete later
            # with open(os.path.join(causal_metric_dir, "temporary_results.json"), "a+") as f:
            #     json.dump(temp_results, f, indent=4)

    def collect_consistency_importance_score(
        self,
        obs,
        t,
        consistency_score,
        mask,
        causal_metric_dir,
        mask_perc=None,
        mode=None,
    ):
        ob = obs[0]
        instr_id = ob["instr_id"]
        importance_score = mask.sum()
        mode = "ins" if mode == "Insertion" else "del"
        # out_file = f"scripts/{args.feature_level_baseline}.json"
        # out_file = "scripts/ensemble_v3_2025_12_15_phase3.json"
        # out_file = "scripts/ig_v3_2025_12_15_phase3.json"
        # out_file = "scripts/random_v3_2025_12_15_phase23_r.json"
        # if os.path.exists(out_file):
        #     with open(out_file, "r") as f:
        #         results = json.load(f)
        #     if instr_id in results:
        #         if str(mask_perc) in results[instr_id]["mask"]:
        #             if mode in results[instr_id]["mask"][str(mask_perc)]:
        #                 consistency_score = results[instr_id]["mask"][str(mask_perc)][
        #                     mode
        #                 ][str(t)]
        #                 print("load consistency score from file successfully")

        save_tuple = np.array([consistency_score, importance_score])
        if not os.path.exists(
            os.path.join(
                causal_metric_dir,
                "consistency_importance_score",
                instr_id,
                str(t),
                str(mask_perc),
                mode,
            )
        ):
            os.makedirs(
                os.path.join(
                    causal_metric_dir,
                    "consistency_importance_score",
                    instr_id,
                    str(t),
                    str(mask_perc),
                    mode,
                )
            )
        np.save(
            os.path.join(
                causal_metric_dir,
                "consistency_importance_score",
                instr_id,
                str(t),
                str(mask_perc),
                mode,
                "score.npy",
            ),
            save_tuple,
        )

    def compute_muFidelity(self, causal_metric_dir):
        consistency_score = []
        importance_score = []
        for instr_id in os.listdir(
            os.path.join(causal_metric_dir, "consistency_importance_score")
        ):
            for t in os.listdir(
                os.path.join(
                    causal_metric_dir, "consistency_importance_score", instr_id
                )
            ):
                for mask_perc in os.listdir(
                    os.path.join(
                        causal_metric_dir,
                        "consistency_importance_score",
                        instr_id,
                        str(t),
                    )
                ):
                    if mask_perc != "score.npy":
                        for mode in ["ins", "del"]:
                            score = np.load(
                                os.path.join(
                                    causal_metric_dir,
                                    "consistency_importance_score",
                                    instr_id,
                                    str(t),
                                    str(mask_perc),
                                    mode,
                                    "score.npy",
                                )
                            )
                            consistency_score.append(score[0])
                            importance_score.append(score[1])
        return self.muFidelity(consistency_score, importance_score)

    def muFidelity(self, consistency_score, importance_score):
        """
        Compute the μ-fidelity correlation between attribution consistency scores
        and the corresponding model predictions.

        Args:
            consistency_score: Sequence[float],
                Model outputs f(x) associated with each subset evaluation. We use
                the convention f(x) = 1, so the impact term
                becomes 1 - f(x_{[x_i = x_0 | i ∈ S]}).
            importance_score: Sequence[float],
                Attribution scores ∑_{i∈S} g(f, x)_i. Can be a flat list (one score per
                evaluation) or a nested list/array where each inner sequence
                represents the scores of the features included in a subset S.

        Returns:
            float: Pearson correlation coefficient between ∑_{i∈S} g(f, x)_i and
                   1 - f(x_{[x_i = x_0 | i ∈ S]}). Returns 0.0 when correlation is undefined.
        """

        consistency_arr = np.asarray(consistency_score, dtype=np.float64)
        importance_arr = np.asarray(importance_score, dtype=np.float64)

        if consistency_arr.ndim != 1:
            raise ValueError(
                "importance_score must be a 1-D sequence of model outputs."
            )

        if importance_arr.ndim == 1:
            summed_importance = importance_arr
        elif importance_arr.ndim == 2:
            summed_importance = importance_arr.sum(axis=1)
        else:
            raise ValueError(
                "importance_score must be a 1-D or 2-D sequence of attributions."
            )

        if summed_importance.shape[0] != importance_arr.shape[0]:
            raise ValueError(
                "importance_score and importance_score must describe the same number "
                "of subset evaluations."
            )

        # Apply the specified baseline f(x_{[x_i = x_0 | i ∈ S]}) = 1
        impact = 1.0 - consistency_arr

        std_importance = np.std(summed_importance)
        std_impact = np.std(impact)

        if std_importance == 0 or std_impact == 0:
            return 0.0

        corr_matrix = np.corrcoef(summed_importance, impact)
        corr = corr_matrix[0, 1]

        if np.isnan(corr):
            return 0.0

        return float(corr)

    def overlay_heatmap_on_panoramic_images(
        self,
        images,  # [B, V, H, W, C] where B=1, V=VIEWPOINT_SIZE
        attr_map,  # [valid_pano, H, W] or [H, W]
        candidate_idx,  # list of candidate indices
        instr_id,
        t,
        output_dir,
        alpha=0.5,
        colormap=cv2.COLORMAP_JET,
    ):
        """
        Overlay heatmap on panoramic images and save as a 6x6 grid of all 36 views.

        Args:
            images: numpy array of shape [B, V, H, W, C] where B=1, V=VIEWPOINT_SIZE (36)
            attr_map: numpy array of shape [valid_pano, H, W] or [H, W], saliency map
            candidate_idx: list of candidate viewpoint indices
            instr_id: instruction ID
            t: time step
            output_dir: directory to save overlaid images
            alpha: transparency factor for heatmap overlay (0.0 to 1.0)
            colormap: OpenCV colormap to use for heatmap visualization

        Returns:
            list: paths to saved overlaid images (single 6x6 grid image)
        """
        # Ensure output directory exists
        save_dir = os.path.join(output_dir, instr_id, str(t))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Get images for first batch item
        images_original = images[0]  # [V, H, W, C] where V=36

        # Convert to BGR format for OpenCV (if needed)
        if images_original.shape[-1] == 3:
            # Assume RGB, convert to BGR for OpenCV
            images_bgr = images_original[..., ::-1].copy()
        else:
            images_bgr = images_original.copy()

        # Get image dimensions
        num_views = len(images_bgr)  # Should be 36
        if num_views != 36:
            print(f"Warning: Expected 36 views, got {num_views}")

        img_h, img_w = images_bgr[0].shape[:2]

        # Create a mapping from candidate_idx to heatmap and collect all candidate heatmaps
        cand_to_heatmap = {}
        candidate_heatmaps = []

        if len(attr_map.shape) == 2:
            # Single heatmap [H, W], apply to all candidate views
            for i, cand_idx in enumerate(candidate_idx):
                if cand_idx < num_views:
                    cand_to_heatmap[cand_idx] = attr_map
                    candidate_heatmaps.append(attr_map)
        elif len(attr_map.shape) == 3:
            # Multiple heatmaps [valid_pano, H, W]
            for i, cand_idx in enumerate(candidate_idx):
                if i < len(attr_map) and cand_idx < num_views:
                    cand_to_heatmap[cand_idx] = attr_map[i]
                    candidate_heatmaps.append(attr_map[i])
        else:
            raise ValueError(
                f"Unsupported attr_map shape: {attr_map.shape}. "
                "Expected [H, W] or [valid_pano, H, W]"
            )

        # Calculate minimum value from candidate heatmaps for non-candidate views
        if len(candidate_heatmaps) > 0:
            # Find minimum value across all candidate heatmaps
            all_cand_values = np.concatenate(
                [hm.flatten() for hm in candidate_heatmaps]
            )
            cand_global_min = all_cand_values.min()
        else:
            # Fallback if no candidates
            cand_global_min = 0.0

        # Process all 36 views
        processed_images = []
        for vp_idx in range(num_views):
            img = images_bgr[vp_idx].astype(np.uint8)

            # Get heatmap for this view
            if vp_idx in cand_to_heatmap:
                # Use actual heatmap for candidate views
                heatmap = cand_to_heatmap[vp_idx]
            else:
                # Use minimum value from candidate heatmaps as baseline for non-candidate views
                heatmap = np.full((img_h, img_w), cand_global_min, dtype=attr_map.dtype)

            # Resize heatmap to match image if needed
            if heatmap.shape != img.shape[:2]:
                heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            else:
                heatmap_resized = heatmap

            # Normalize heatmap to [0, 255] using its own min/max (independent normalization)
            # This ensures candidate views look exactly the same as when processed individually
            if heatmap_resized.max() > 255 or heatmap_resized.dtype != np.uint8:
                heatmap_min = heatmap_resized.min()
                heatmap_max = heatmap_resized.max()
                if heatmap_max - heatmap_min > 1e-8:
                    heatmap_norm = (
                        (heatmap_resized - heatmap_min)
                        / (heatmap_max - heatmap_min + 1e-8)
                        * 255
                    ).astype(np.uint8)
                    # Clip to [0, 255] range
                    heatmap_norm = np.clip(heatmap_norm, 0, 255)
                else:
                    # All values are the same, use zeros
                    heatmap_norm = np.zeros_like(heatmap_resized, dtype=np.uint8)
            else:
                heatmap_norm = heatmap_resized.astype(np.uint8)

            # Apply colormap
            hm_color = cv2.applyColorMap(heatmap_norm, colormap)

            # Overlay heatmap on image
            superimposed = cv2.addWeighted(img, 1.0 - alpha, hm_color, alpha, 0)
            processed_images.append(superimposed)

        # Arrange images in 6x6 grid
        # Layout: 3 heights × 12 horizontal views
        # Each height level has 12 views, arranged as 2 rows × 6 columns per height
        # Total: 6 rows × 6 columns
        grid_rows = 6
        grid_cols = 6

        # Create grid image
        grid_image = np.zeros((grid_rows * img_h, grid_cols * img_w, 3), dtype=np.uint8)

        # Fill grid: arrange by height levels
        # Height 0 (down): views 0-11 -> rows 0-1
        # Height 1 (middle): views 12-23 -> rows 2-3
        # Height 2 (top): views 24-35 -> rows 4-5
        for vp_idx in range(num_views):
            height_level = vp_idx // 12  # 0, 1, or 2
            horiz_idx = vp_idx % 12  # 0-11

            # Calculate grid position
            # Each height level takes 2 rows
            grid_row = height_level * 2 + (
                horiz_idx // 6
            )  # 0-1 for first 6, 2-3 for next 6, etc.
            grid_col = horiz_idx % 6  # 0-5

            # Place image in grid
            row_start = grid_row * img_h
            row_end = row_start + img_h
            col_start = grid_col * img_w
            col_end = col_start + img_w

            grid_image[row_start:row_end, col_start:col_end] = processed_images[vp_idx]

        # Save grid image
        save_path = os.path.join(save_dir, "panorama_grid_6x6.png")
        cv2.imwrite(save_path, grid_image)

        return [save_path]

    # def compute_AUC(self, causal_metric_dir):
    #     consistency_score = []
    #     importance_score = []
    #     for instr_id in os.listdir(
    #         os.path.join(causal_metric_dir, "consistency_importance_score")
    #     ):
    #         for t in os.listdir(
    #             os.path.join(causal_metric_dir, "consistency_importance_score", instr_id)
    #         ):
    #             score = np.load(
    #                 os.path.join(
    #                     causal_metric_dir,
    #                     "consistency_importance_score",
    #                     instr_id,
    #                     str(t),
    #                     "score.npy",
    #                 )
    #             )
    #             consistency_score.append(score[0])
    #             importance_score.append(score[1])
    #     return self.AUC(consistency_score, importance_score)

    # def AUC(self, consistency_score, importance_score):
    #     return roc_auc_score(consistency_score, importance_score)


def main():
    pass
