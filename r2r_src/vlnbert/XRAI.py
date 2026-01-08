# from IG_utils import compute_integrated_gradients
import numpy as np
from skimage import segmentation
from skimage.morphology import dilation
from skimage.morphology import disk
from skimage.transform import resize
import logging

from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights

import torch
import cv2
import os
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks
from pathlib import Path
from torch.autograd import grad
from torchvision.transforms import Resize
from torch.utils.checkpoint import checkpoint

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

np.set_printoptions(threshold=np.inf)

_logger = logging.getLogger(__name__)
# _FELZENSZWALB_SCALE_VALUES = [50, 100, 150, 250, 500, 1200]
# _FELZENSZWALB_SCALE_VALUES = [50, 100, 150]
_FELZENSZWALB_SCALE_VALUES = [50]
_FELZENSZWALB_SIGMA_VALUES = [0.8]
_FELZENSZWALB_IM_RESIZE = (224, 224)
_FELZENSZWALB_IM_VALUE_RANGE = [-1.0, 1.0]
_FELZENSZWALB_MIN_SEGMENT_SIZE = 150

# Import args to get panoramic_horizontal_views
try:
    from param import args

    VIEWPOINT_SIZE = 3 * args.panoramic_horizontal_views
except ImportError:
    # Fallback if args not available (e.g., when used as standalone script)
    # Default to 36 (3 heights * 12 horizontal views)
    VIEWPOINT_SIZE = 36


def extract_object_masks_yolo(ims, resize_image=True, scale_range=None, dilation_rad=5):
    # load model
    image_processor = AutoImageProcessor.from_pretrained(
        "./feat_checkpoints/mask2former-swin-base-ade-semantic"
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "./feat_checkpoints/mask2former-swin-base-ade-semantic"
    )

    segs = []
    seg_pano = []
    start_seg_idx = 0
    for i in range(len(ims)):

        inputs = image_processor(ims[i], return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # Perform post-processing to get instance segmentation map
        pred_instance_map = image_processor.post_process_instance_segmentation(
            outputs, target_sizes=[(224, 224)]
        )[0]
        seg = pred_instance_map["segmentation"]
        seg = seg.type(torch.int32)
        # print("seg", seg.shape)
        seg = seg + start_seg_idx
        # print(i, seg)
        # print(seg)
        seg_pano.append(seg)
        start_seg_idx = seg.max() + 2
    seg_pano = np.stack(seg_pano)
    segs.append(seg_pano)
    segs = np.stack(segs)
    masks = _unpack_segs_to_masks(segs)

    masks = [
        torch.stack(
            # [torch.from_numpy(mask).to("cuda") for mask in masks_i], dim=0
            [torch.from_numpy(mask) for mask in masks_i],
            dim=0,
        )  # 内层堆叠
        for masks_i in masks
    ]

    return masks


def _attr_aggregation_max(attr, axis=-1):
    return attr.max(axis=axis)


def _gain_density(masks, attr, den_mode="mean"):
    # Compute the attr density over mask1. If mask2 is specified, compute density
    # for mask1 \ mask2
    N, V, H, W = masks.shape
    added_masks = masks
    # Collapse viewpoints into panorama
    masks_flat = masks.view(N, -1)  # [N, V*H*W]
    attr_flat = attr.reshape(-1)  # [V*H*W]

    # # Pixel counts
    # counts = added_masks.sum(dim=(1, 2))  # [N]
    # Pixel counts
    counts = masks_flat.sum(dim=1)  # [N]

    # Handle empty masks (avoid division by zero)
    valid = counts > 0
    if not valid.any():
        return torch.full((masks.size(0),), -torch.inf, device=masks.device)

    # Numerator: sum of attr values inside added_masks
    # Expand attr to [N, H, W] and mask
    # sums = (added_masks * attr).sum(dim=(1, 2))  # [N]
    # Attribution sums inside masks
    sums = (masks_flat * attr_flat).sum(dim=1)  # [N]

    gains = torch.full((masks.size(0),), -torch.inf, device=masks.device)

    if den_mode == "mean":  # Mean density
        gains[valid] = sums[valid] / counts[valid].float()
    elif den_mode == "sum":  # Sum density
        gains[valid] = sums[valid].float()

    return gains


def _get_diff_mask(add_mask, base_mask):
    # return np.logical_and(add_mask, np.logical_not(base_mask))
    return torch.logical_and(add_mask, torch.logical_not(base_mask))


def _get_diff_cnt(add_mask, base_mask):
    # return np.sum(_get_diff_mask(add_mask, base_mask))
    return torch.sum(_get_diff_mask(add_mask, base_mask))


def _unpack_segs_to_masks(segs):
    masks = []
    for seg in segs:
        for l in range(seg.min(), seg.max() + 1):
            masks.append(seg == l)
    return masks


def extract_object_masks(imgs, dilation_rad=5):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    transforms = weights.transforms()
    # proba_threshold = 0.1
    proba_threshold = 0
    model = maskrcnn_resnet50_fpn_v2(weights=weights, progress=False)
    model.cuda()
    model = model.eval()

    for i in range(len(imgs)):
        imgs[i] = Resize((224, 224))(imgs[i])
    batch = torch.stack([transforms(d) for d in imgs]).cuda()
    # out = []
    # # mini_batch?
    mini_bs = 2
    num_images = len(imgs)
    segs = []
    i = 0
    for start_idx in range(0, num_images, mini_bs):
        end_idx = min(start_idx + mini_bs, num_images)
        mini_bs_images = batch[start_idx:end_idx]
        # mini_bs_out = model(mini_bs_images)
        mini_bs_out = checkpoint(model, mini_bs_images, use_reentrant=False)
        for out_i in mini_bs_out:
            seg = out_i["masks"] > proba_threshold
            seg = seg.squeeze(1)
            # print(seg.shape)
            for j, seg_i in enumerate(seg):
                new_mask = torch.zeros(num_images, 224, 224, dtype=torch.bool).to(
                    "cuda"
                )
                new_mask[i] = seg_i
                segs.append(new_mask)
            i += 1
        # out.append(mini_bs_out)

    # out = torch.stack(out)
    segs = torch.stack(segs, axis=0)
    # segs = np.stack(segs, 0)
    print(segs.shape)
    return segs


class XRAIParameters(object):
    """Dictionary of parameters to specify how to XRAI and return outputs."""

    def __init__(
        self,
        steps=100,
        area_threshold=1.0,
        return_baseline_predictions=False,
        # return_ig_attributions=False,
        return_ig_attributions=True,
        # return_xrai_segments=False,
        return_xrai_segments=True,
        flatten_xrai_segments=True,
        algorithm="full",
    ):
        # TODO(tolgab) add return_ig_for_every_step functionality

        # Number of steps to use for calculating the Integrated Gradients
        # attribution. The higher the number of steps the higher is the precision
        # but lower the performance. (see also XRAIOutput.error).
        self.steps = steps
        # The fraction of the image area that XRAI should calculate the segments
        # for. All segments that exceed that threshold will be merged into a single
        # segment. The parameter is used to accelerate the XRAI computation if the
        # caller is only interested in the top fraction of segments, e.g. 20%. The
        # value should be in the [0.0, 1.0] range, where 1.0 means that all segments
        # should be returned (slowest). Fast algorithm ignores this setting.
        self.area_threshold = area_threshold
        # TODO(tolgab) Enable return_baseline_predictions
        # If set to True returns predictions for the baselines as float32 [B] array,
        # where B is the number of baselines. (see XraiOutput.baseline_predictions).
        # self.return_baseline_predictions = return_baseline_predictions
        # If set to True, the XRAI output returns Integrated Gradients attributions
        # for every baseline. (see XraiOutput.ig_attribution)
        self.return_ig_attributions = return_ig_attributions
        # If set to True the XRAI output returns XRAI segments in the order of their
        # importance. This parameter works in conjunction with the
        # flatten_xrai_sements parameter. (see also XraiOutput.segments)
        self.return_xrai_segments = return_xrai_segments
        # If set to True, the XRAI segments are returned as an integer array with
        # the same dimensions as the input (excluding color channels). The elements
        # of the array are set to values from the [1,N] range, where 1 is the most
        # important segment and N is the least important segment. If
        # flatten_xrai_sements is set to False, the segments are returned as a
        # boolean array, where the first dimension has size N. The [0, ...] mask is
        # the most important and the [N-1, ...] mask is the least important. This
        # parameter has an effect only if return_xrai_segments is set to True.
        self.flatten_xrai_segments = flatten_xrai_segments
        # Specifies a flavor of the XRAI algorithm. full - executes slower but more
        # precise XRAI algorithm. fast - executes faster but less precise XRAI
        # algorithm.
        self.algorithm = algorithm
        # EXPERIMENTAL - Contains experimental parameters that may change in future.
        self.experimental_params = {"min_pixel_diff": 50}


class XRAI(object):
    def __init__(self):
        pass

    def GetMaskWithDetails(
        self,
        x_value,
        segments=None,
        base_attribution=None,
        batch_size=1,
        extra_parameters=None,
        candidata_idx=None,
        obs=None,
    ):
        """
        Args:
            x_value:    images
            segments:   segments arrording to sme instance segmentation model
            base_atrribution:   heatmap
            batch_size:
            extra_parameters:
            candidata_idx:
        Returns:
            attr_map: saliency map in shape of [B, H, W]
            attr_data: torch.Tensor that has the size shape with attr_map, but save the importance order
        """
        if extra_parameters is None:
            extra_parameters = XRAIParameters()

        # Check the shape of base_attribution.
        if base_attribution is not None:
            if not isinstance(base_attribution, np.ndarray):
                base_attribution = np.array(base_attribution)

        attr = base_attribution

        # Merge attribution channels for XRAI input
        if len(attr.shape) > 3:
            attr = _attr_aggregation_max(attr)

        _logger.info("Done with IG. Computing XRAI...")
        if segments is not None:
            segs = segments
        # else:
        #     segs = _get_segments_felzenszwalb(x_value)
        else:
            print("Arg seg cannot be None")

        if extra_parameters.algorithm == "full":
            attr_map, attr_data = self._xrai(
                attr=attr,
                segs=segs,
                area_perc_th=extra_parameters.area_threshold,
                min_pixel_diff=extra_parameters.experimental_params["min_pixel_diff"],
                gain_fun=_gain_density,
                integer_segments=extra_parameters.flatten_xrai_segments,
            )
        else:
            raise ValueError(
                "Unknown algorithm type: {}".format(extra_parameters.algorithm)
            )
        # print("attr_map", attr_map)
        # print("attr_ranks", attr_data)

        heatmap = gen_heatmap(attr_map)
        draw_heatmaps2(x_value, heatmap, obs, "heatmaps/object", candidata_idx)
        return attr_map, attr_data

    def getMaskPixel(
        self,
        x_value,
        base_attribution=None,
        batch_size=1,
        extra_parameters=None,
        candidata_idx=None,
        obs=None,
    ):
        """
        Compute attribution map and ordering directly at pixel level without
        relying on pre-computed segments.

        Args:
            x_value:            Images tensor/list used only for visualization.
            base_attribution:   Attribution heatmap (numpy array or tensor).
            batch_size:         Unused, kept for interface compatibility.
            extra_parameters:   Optional XRAIParameters instance.
            candidata_idx:      View indices for visualization.
            obs:                Observation metadata for saving heatmaps.

        Returns:
            attr_map: torch.Tensor with attribution scores (same shape as input).
            attr_data: torch.Tensor containing pixel-wise importance ordering.
        """
        if extra_parameters is None:
            extra_parameters = XRAIParameters()

        if base_attribution is None:
            raise ValueError("base_attribution must be provided for getMaskPixel.")

        if not isinstance(base_attribution, np.ndarray):
            base_attribution = np.array(base_attribution)

        attr = base_attribution
        if attr.ndim > 3:
            attr = _attr_aggregation_max(attr)

        attr_map = torch.from_numpy(attr.copy())

        flat_attr = attr_map.view(attr_map.size(0), -1)
        sorted_vals, sorted_idx = torch.sort(flat_attr, dim=1, descending=True)

        ranks = torch.zeros_like(flat_attr, dtype=torch.int)
        order = torch.arange(1, flat_attr.size(1) + 1, dtype=torch.int)
        ranks.scatter_(1, sorted_idx, order.unsqueeze(0).expand_as(ranks))

        attr_data = ranks.view_as(attr_map)

        heatmap = gen_heatmap(attr_map)
        draw_heatmaps2(x_value, heatmap, obs, "heatmaps/object", candidata_idx)

        return attr_map, attr_data

    @staticmethod
    def _xrai(
        attr,
        segs,
        gain_fun=_gain_density,
        area_perc_th=1.0,
        min_pixel_diff=50,
        integer_segments=True,
    ):
        # attr = torch.from_numpy(attr).to("cuda")  # [N, H, W]
        attr = torch.from_numpy(attr)  # [N, H, W]
        # output_attr = -torch.inf * torch.ones(size=attr.shape, dtype=torch.float32).to(
        #     "cuda"
        # )
        output_attr = -torch.inf * torch.ones(size=attr.shape, dtype=torch.float32)
        # masks_tensor = torch.stack(segs, dim=0).to("cuda")  # [N, V, H, W]
        masks_tensor = torch.stack(segs, dim=0)  # [N, V, H, W]

        masks_trace = []

        # Compute gains in parallel
        gains = gain_fun(masks_tensor, attr, den_mode="mean")  # should handle batch
        print(gains.shape)
        # gains[~remaining] = -torch.inf

        # Sort descending
        sorted_gains, sorted_idx = torch.sort(gains, descending=True)  # [N], [N]

        masks_trace = [
            (masks_tensor[sorted_idx[i]], sorted_gains[i])
            for i in range(len(sorted_gains))
        ]

        for i in range(len(sorted_gains)):
            output_attr[masks_tensor[sorted_idx[i]]] = sorted_gains[i]

        # uncomputed_mask = output_attr == -torch.inf
        # # Assign the uncomputed areas a value such that sum is same as ig
        # output_attr[uncomputed_mask] = gain_fun(uncomputed_mask, attr)

        masks_trace = [v[0] for v in sorted(masks_trace, key=lambda x: -x[1])]
        if integer_segments:
            attr_ranks = torch.zeros(size=attr.shape, dtype=torch.int)
            for i, mask in enumerate(masks_trace):
                attr_ranks[mask] = i + 1
            return output_attr, attr_ranks
        else:
            return output_attr, masks_trace


def gen_heatmap(img):
    # img: [36, 3, 224, 224]
    # print("img", img.shape)
    # heatmap = img.abs().sum(dim=1).detach().cpu().numpy()
    heatmap = img.detach().cpu().numpy()
    # normalization
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = (heatmap * 255).clip(0, 255).astype(np.uint8)
    return heatmap  # [36, 224, 224]


def reverse_transforms(ori_image):
    # (C, H, W)
    # denorm_tensor = (tensor * 0.5) + 0.5
    if isinstance(ori_image, torch.Tensor):
        image = ori_image.permute(1, 2, 0).detach().cpu().numpy()  # (H, W, C)
    elif isinstance(ori_image, np.ndarray):
        image = ori_image.transpose(1, 2, 0)
    else:
        print("type of image should be tensor or ndarray")
        exit(0)
    image = image + np.array([[[103.1, 115.9, 123.2]]])  # BGR pixel mean
    # image = (image * 255).clip(0, 255).astype(np.uint8)
    image = image.clip(0, 255).astype(np.uint8)

    return image


def draw_heatmaps2(imgs, heatmap, ob, path="./heatmaps", list_=None):
    # bs = VIEWPOINT_SIZE  # 36
    bs = len(heatmap)
    scanId = ob["scan"]
    viewpointId = ob["viewpoint"]

    if list_ is None:
        list_ = range(VIEWPOINT_SIZE)

    for ix in range(bs):
        draw_heatmap2(imgs[ix], heatmap[ix], scanId, viewpointId, list_[ix], path)


def draw_heatmap2(img, heatmap, scanId, viewpointId, idx, root_path="./heatmaps"):
    # img: [224, 224]
    target_path = os.path.join(root_path, scanId, viewpointId)
    # print(target_path)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    # img = reverse_transforms(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imread() and cv2.imwrite() load images as BGR
    hm_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 0.5, hm_color, 0.5, 0)
    # superimposed = hm_color
    cv2.imwrite(os.path.join(target_path, str(idx)) + ".png", superimposed)


if __name__ == "__main__":
    XRAI_test = XRAI()
    # --------- compute panorama's IG ------------
    # compute_integrated_gradients(
    #     args,
    #     vit,
    #     vln_bert,
    #     obs,
    #     gmaps,
    #     txt_embeds,
    #     language_inputs,
    #     img_transforms,
    #     steps=steps,
    # )
    # img = decode_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")
    img = decode_image("/Users/ian/Desktop/car.jpg")
    # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)
    print(batch.shape)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")

    steps = 50
    alphas = [alpha for alpha in torch.linspace(0, 1, steps)]
    grads = []
    baseline = torch.zeros_like(batch)
    for alpha in alphas:
        input_img = baseline + (batch - baseline) * alpha
        input_img.requires_grad_(True)
        prediction = model(input_img).squeeze(0).softmax(0)
        # score = prediction[class_id].item()
        score = prediction[class_id]

        grad_output = grad(score, input_img, retain_graph=False)[0]
        grads.append(grad_output.detach())

    # 近似积分
    avg_grad = torch.mean(torch.stack(grads), dim=0)
    print("gard", avg_grad.shape)
    print("batch", batch.shape)

    ig = (batch - torch.zeros_like(batch)) * avg_grad
    heatmap = gen_heatmap(ig)
    print(heatmap)
    # draw_heatmap2(batch[0], heatmap[0])
    attribution = ig.detach().cpu().numpy().transpose(0, 2, 3, 1)
    x = batch.detach().cpu().numpy().transpose(0, 2, 3, 1)
    use_object_seg = True
    object_seg = None
    if use_object_seg:
        object_seg = extract_object_masks([img])
    XRAI_test.GetMaskWithDetails(
        x,
        None,
        None,
        None,
        object_seg,
        attribution,
    )
