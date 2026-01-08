# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Implements Guided Integrated Gradients method.

This implementation of the method allows setting the maximum distance that
the Guided IG path can deviate from the straight-line path.

https://arxiv.org/abs/TBD
"""


from .base import CoreSaliency
from .base import INPUT_OUTPUT_GRADIENTS
import math
import numpy as np

# A very small number for comparing floating point values.
EPSILON = 1e-9


def l1_distance(x1, x2):
    """Returns L1 distance between two points."""
    return np.abs(x1 - x2).sum()


def translate_x_to_alpha(x, x_input, x_baseline):
    """Translates a point on straight-line path to its corresponding alpha value.

    Args:
      x: the point on the straight-line path.
      x_input: the end point of the straight-line path.
      x_baseline: the start point of the straight-line path.

    Returns:
      The alpha value in range [0, 1] that shows the relative location of the
      point between x_baseline and x_input.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            x_input - x_baseline != 0, (x - x_baseline) / (x_input - x_baseline), np.nan
        )


def translate_alpha_to_x(alpha, x_input, x_baseline):
    """Translates alpha to the point coordinates within straight-line interval.

     Args:
      alpha: the relative location of the point between x_baseline and x_input.
      x_input: the end point of the straight-line path.
      x_baseline: the start point of the straight-line path.

    Returns:
      The coordinates of the point within [x_baseline, x_input] interval
      that correspond to the given value of alpha.
    """
    assert 0 <= alpha <= 1.0
    return x_baseline + (x_input - x_baseline) * alpha


def guided_ig_impl(
    x_input,
    x_baseline,
    grad_func,
    steps=200,
    fraction=0.25,
    max_dist=0.02,
    cand_list=None,  # [B, vp] vp <= VIEWPOINT_SIZE
):
    """Calculates and returns Guided IG attribution.

    Args:
        x_input: model input that should be explained.
        x_baseline: chosen baseline for the input explanation.
        grad_func: gradient function that accepts a model input and returns
            the corresponding output gradients. In case of many class model, it is
            responsibility of the implementer of the function to return gradients
            for the specific class of interest.
        steps: the number of Riemann sum steps for path integral approximation.
        fraction: the fraction of features [0, 1] that should be selected and
            changed at every approximation step. E.g., value `0.25` means that 25% of
            the input features with smallest gradients are selected and changed at
            every step.
        max_dist: the relative maximum L1 distance [0, 1] that any feature can
            deviate from the straight line path. Value `0` allows no deviation and,
            therefore, corresponds to the Integrated Gradients method that is
            calculated on the straight-line path. Value `1` corresponds to the
            unbounded Guided IG method, where the path can go through any point within
            the baseline-input hyper-rectangular.
    """

    x_input = np.asarray(x_input, dtype=np.float32)
    x_baseline = np.asarray(x_baseline, dtype=np.float32)
    x = x_baseline.copy()
    attr = np.zeros_like(x_input, dtype=np.float32)

    # Create mask for candidate viewpoints
    def create_cand_mask(x_shape, cand_list):
        """Create a boolean mask for candidate viewpoints."""
        B, V = x_shape[0], x_shape[1]
        mask = np.zeros((B, V), dtype=bool)
        for b in range(B):
            for vp in cand_list[b]:
                vp_idx = int(vp)
                mask[b, vp_idx] = True
        # Expand mask to match x_shape: [B, V] -> [B, V, 3, 224, 224]
        # Use broadcasting: add dimensions for [3, 224, 224]
        mask_expanded = mask[:, :, np.newaxis, np.newaxis, np.newaxis]
        mask_expanded = np.broadcast_to(mask_expanded, x_shape)
        return mask_expanded

    # Compute L1 distance only over candidate viewpoints (per batch)
    def l1_distance_masked(x1, x2, mask):
        """Returns L1 distance per batch element, only over masked regions.

        Returns:
            Array of shape [B] with L1 distance for each batch element.
        """
        # Compute L1 distance per batch: sum over [V, 3, 224, 224] for each batch element
        diff = np.abs(x1 - x2) * mask  # Only consider masked (candidate) viewpoints
        # Sum over all dimensions except batch dimension
        l1_per_batch = diff.reshape(diff.shape[0], -1).sum(axis=1)
        return l1_per_batch

    cand_mask = create_cand_mask(x.shape, cand_list)
    l1_total = l1_distance_masked(x_input, x_baseline, cand_mask)  # Shape: [B]

    # If all batch elements have zero L1 distance, return zero attribution
    if np.all(l1_total == 0):
        return attr

    # For candidate index broadcasting
    def broadcast_grad_to_full(grad_actual, x_shape, cand_list=None):
        """
        Broadcast grad_actual of shape [B, vp, 3, 224, 224] to shape [B,V,3,224,224] matching x_input
        If cand_list is None, simply tile grad_actual to full shape (for compatibility)
        """
        assert cand_list is not None, "cand_list is required"
        # cand_list: [B, vp]
        B, V = x_shape[0], x_shape[1]
        # Use inf for non-candidate viewpoints to exclude them from selection
        # (consistent with how features that reached x_max are handled)
        full_grad = np.full(x_shape, np.inf, dtype=grad_actual.dtype)
        for b in range(B):
            for idx_in_cand, vp in enumerate(cand_list[b]):
                # cand_list elements may be np.int64, ensure index is used as integer
                vp_idx = int(vp)
                full_grad[b, vp_idx] = grad_actual[b, idx_in_cand]
        return full_grad

    # Iterate through every step.
    for step in range(steps):
        # print(f"step {step}")
        # Calculate gradients and make a copy.
        grad_actual = grad_func(x)
        grad_actual_broadcasted = broadcast_grad_to_full(
            grad_actual, x.shape, cand_list
        )
        grad = grad_actual_broadcasted.copy()

        # Calculate current step alpha and the ranges of allowed values for this step.
        alpha = (step + 1.0) / steps
        alpha_min = max(alpha - max_dist, 0.0)
        alpha_max = min(alpha + max_dist, 1.0)
        x_min = translate_alpha_to_x(alpha_min, x_input, x_baseline)
        x_max = translate_alpha_to_x(alpha_max, x_input, x_baseline)
        # The goal of every step is to reduce L1 distance to the input.
        # `l1_target` is the desired L1 distance after completion of this step.
        l1_target = l1_total * (1 - (step + 1) / steps)

        # Iterate until the desired L1 distance has been reached.
        gamma = np.inf
        while gamma > 1.0:
            # print("while loop (inner iteration)")
            x_old = x.copy()
            x_alpha = translate_x_to_alpha(x, x_input, x_baseline)
            x_alpha[np.isnan(x_alpha)] = alpha_max
            # All features that fell behind the [alpha_min, alpha_max] interval in
            # terms of alpha, should be assigned the x_min values.
            x[x_alpha < alpha_min] = x_min[x_alpha < alpha_min]

            # Calculate current L1 distance from the input (only over candidate viewpoints).
            l1_current = l1_distance_masked(x, x_input, cand_mask)
            # If the current L1 distance is close enough to the desired one then
            # update the attribution and proceed to the next step.
            if math.isclose(l1_target, l1_current, rel_tol=EPSILON, abs_tol=EPSILON):
                attr += (x - x_old) * grad_actual_broadcasted
                break

            # Features that reached `x_max` should not be included in the selection.
            # Assign very high gradients to them so they are excluded.
            grad[x == x_max] = np.inf

            # Select features with the lowest absolute gradient.
            threshold = np.quantile(np.abs(grad), fraction, interpolation="lower")
            s = np.logical_and(np.abs(grad) <= threshold, grad != np.inf)

            # Find by how much the L1 distance can be reduced by changing only the
            # selected features.
            l1_s = (np.abs(x - x_max) * s).sum()

            # Calculate ratio `gamma` that show how much the selected features should
            # be changed toward `x_max` to close the gap between current L1 and target
            # L1.
            if l1_s > 0:
                gamma = (l1_current - l1_target) / l1_s
            else:
                gamma = np.inf

            if gamma > 1.0:
                # Gamma higher than 1.0 means that changing selected features is not
                # enough to close the gap. Therefore change them as much as possible to
                # stay in the valid range.
                x[s] = x_max[s]
            else:
                assert gamma > 0, gamma
                x[s] = translate_alpha_to_x(gamma, x_max, x)[s]
            # Update attribution to reflect changes in `x`.
            attr += (x - x_old) * grad_actual_broadcasted
    return attr


class GuidedIG(CoreSaliency):
    """Implements ML framework independent version of Guided IG."""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(
        self,
        x_value,
        call_model_function,
        call_model_args=None,
        x_baseline=None,
        x_steps=200,
        fraction=0.25,
        max_dist=0.02,
    ):
        """Computes and returns the Guided IG attribution.

        Args:
          x_value: an input (ndarray) for which the attribution should be computed.
          call_model_function: A function that interfaces with a model to return
            specific data in a dictionary when given an input and other arguments.
            Expected function signature:
            - call_model_function(x_value_batch,
                                  call_model_args=None,
                                  expected_keys=None):
              x_value_batch - Input for the model, given as a batch (i.e. dimension
                0 is the batch dimension, dimensions 1 through n represent a single
                input).
              call_model_args - user defined arguments. The value of this argument
                is the value of `call_model_args` argument of the nesting method.
              expected_keys - List of keys that are expected in the output. For this
                method (Guided IG), the expected keys are
                INPUT_OUTPUT_GRADIENTS - Gradients of the output being
                  explained (the logit/softmax value) with respect to the input.
                  Shape should be the same shape as x_value_batch.
          call_model_args: The arguments that will be passed to the call model
            function, for every call of the model.
          x_baseline: Baseline value used in integration. Defaults to 0.
          x_steps: Number of integrated steps between baseline and x.
          fraction: the fraction of features [0, 1] that should be selected and
            changed at every approximation step. E.g., value `0.25` means that 25%
            of the input features with smallest gradients are selected and changed
            at every step.
          max_dist: the relative maximum L1 distance [0, 1] that any feature can
            deviate from the straight line path. Value `0` allows no deviation and;
            therefore, corresponds to the Integrated Gradients method that is
            calculated on the straight-line path. Value `1` corresponds to the
            unbounded Guided IG method, where the path can go through any point
            within the baseline-input hyper-rectangular.
        """

        # x_value = np.asarray(x_value)
        x_value = np.array(x_value)  # [bs, vp, D]
        if x_baseline is None:
            x_baseline = np.zeros_like(x_value)  # [bs, vp, D]
        else:
            x_baseline = np.array(x_baseline)  # [bs, vp, D]

        assert x_baseline.shape == x_value.shape

        return guided_ig_impl(
            x_input=x_value,
            x_baseline=x_baseline,
            grad_func=self._get_grad_func(call_model_function, call_model_args),
            steps=x_steps,
            fraction=fraction,
            max_dist=max_dist,
            cand_list=call_model_args[9],  # candidata_list
        )

    def _get_grad_func(self, call_model_function, call_model_args):
        def _grad_func(x_value):
            call_model_output = call_model_function(
                # np.expand_dims(x_value, axis=0),
                x_value,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys,
            )
            # return np.asarray(call_model_output[INPUT_OUTPUT_GRADIENTS][0])
            return np.asarray(call_model_output[INPUT_OUTPUT_GRADIENTS].cpu())

        return _grad_func
