import torch

# fix the import error
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from r2r_src.vlnbert.caffe_resnet import CNN
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from PIL import Image


class FG_CAM_resnet:
    def __init__(self, model, base_cam):
        self.model = model
        self.base_cam = base_cam

    def svd(self, I):
        """
        SVD denoising: remove the noise in the explanation component
        """
        I = torch.nan_to_num(I[0])
        reshaped_I = (I).reshape(I.shape[0], -1)
        reshaped_I = reshaped_I - reshaped_I.mean(dim=1)[:, None]
        U, S, VT = torch.linalg.svd(reshaped_I, full_matrices=True)
        d = int(S.shape[0] * 0.1)
        s = torch.diag(S[:d], 0)
        new_I = U[:, :d].mm(s).mm(VT[:d, :])
        new_I = new_I.reshape(I.size())
        return new_I

    def find_last_layer(self):
        """
        Find the last convolutional layer before pooling/fc
        For ResNet, we look for the last conv layer in res5c or res4b35
        """
        # print("find_last_layer", self.model)
        # if isinstance(self.model, CNN):
        # print("model is CNN")
        # Find the last conv layer - typically res5c_branch2c or similar
        # We'll use res5c_branch2c as the target layer
        for name, module in reversed(list(self.model.named_modules())):
            if "res5c_branch2c" in name and hasattr(module, "weight"):
                return module
        # Fallback: return the last conv layer we can find
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
        return None

    def get_weight_by_grad_cam(self, input, target_class, layer):
        value = dict()

        def backward_hook(module, grad_input, grad_output):
            value["gradients"] = grad_output[0]

        def forward_hook(module, input, output):
            value["activations"] = output

        try:
            h1 = layer.register_forward_hook(forward_hook)
            h2 = layer.register_backward_hook(backward_hook)
            output = self.model(input)
        except Exception as e:
            print(f"Error registering hooks: {e}")
            print(f"Layer: {layer}")
            print(f"Input: {input}")
            print(f"Output: {output}")
            return None, None
        # For ResNet, output is (pool5, prob), we need prob for target_class
        if isinstance(output, tuple):
            prob = output[1]
        else:
            prob = output
        prob[0][target_class].backward()
        h1.remove()
        h2.remove()
        return value["gradients"], value["activations"]

    def get_weight_by_score_cam(self, input, target_class, layer):
        value = dict()

        def forward_hook(module, input, output):
            value["activations"] = output

        h = layer.register_forward_hook(forward_hook)

        with torch.no_grad():
            output = self.model(input)
            h.remove()
            activations = value["activations"]
            weight = None
            batch = 8
            saliency_map = F.interpolate(
                activations, size=(224, 224), mode="bilinear", align_corners=False
            )
            saliency_map = torch.nan_to_num(saliency_map)
            maxs = saliency_map.view(
                saliency_map.size(0), saliency_map.size(1), -1
            ).max(dim=-1)[0]
            mins = saliency_map.view(
                saliency_map.size(0), saliency_map.size(1), -1
            ).min(dim=-1)[0]
            eps = torch.where(maxs == 0, 1e-9, 0.0)
            saliency_map = (saliency_map - mins[:, :, None, None]) / (
                maxs[:, :, None, None] - mins[:, :, None, None] + eps[:, :, None, None]
            )
            saliency_map = saliency_map[0]

            for i in range(0, saliency_map.size(0), batch):
                x = input * saliency_map[i : i + batch, None, :, :]
                output = self.model(x)
                if isinstance(output, tuple):
                    prob = output[1]
                else:
                    prob = output
                prob = torch.softmax(prob, dim=1)
                y = prob[:, target_class]
                if i == 0:
                    weight = y.clone()
                else:
                    weight = torch.cat([weight, y])
            return weight, activations

    def get_explanation_component(self, input, target_class, layer=None):
        if layer is None:
            layer = self.find_last_layer()
            print("find_last_layer", layer)
        if self.base_cam.lower() == "grad_cam":
            weight, activation = self.get_weight_by_grad_cam(input, target_class, layer)
            I = torch.mean(weight, dim=(2, 3), keepdim=True) * activation
        if self.base_cam.lower() == "score_cam":
            weight, activation = self.get_weight_by_score_cam(
                input, target_class, layer
            )
            I = weight[None, :, None, None] * activation

        return I

    def forward(self, input, denoising, target_layer, target_class):
        if target_class is None:
            output = self.model(input)
            if isinstance(output, tuple):
                prob = output[1]
            else:
                prob = output
            target_class = prob.argmax(dim=-1)[0].item()

        # get the explanation component
        I = self.get_explanation_component(input, target_class)

        self.model.register_hook()
        self.model(input)
        self.model.remove_hook()
        if denoising:
            I = self.svd(I)
        I = self.model.improve_resolution(I, target_layer)
        I = torch.sum(I, dim=1)
        return I, target_class

    def __call__(self, input, denoising, target_layer, target_class=None):
        return self.forward(input, denoising, target_layer, target_class)


def ZeroCenter(path, size, BGRTranspose=False):
    """Preprocess image for ResNet (Caffe-style normalization)"""
    img = Image.open(path)
    if isinstance(size, tuple):
        h, w = size[0], size[1]
    else:
        h, w = size, size
    img = img.resize((h, w))
    x = np.array(img, dtype=np.float32)

    # Caffe-style normalization: subtract [103.1, 115.9, 123.2]
    x[..., 0] -= 123.2
    x[..., 1] -= 115.9
    x[..., 2] -= 103.1
    if BGRTranspose == True:
        x = x[..., ::-1]

    return x


def visual_explanation(heatmap):
    """Normalize and resize heatmap for visualization"""
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-9)
    heatmap = heatmap.detach().cpu().numpy()
    if len(heatmap.shape) == 2:
        return cv2.resize(heatmap, (224, 224))
    elif len(heatmap.shape) == 3:
        return cv2.resize(np.transpose(heatmap, (1, 2, 0)), (224, 224))
    else:
        return cv2.resize(heatmap[0], (224, 224))


def get_target_layer_index(model, layer_name=None):
    """Get the index of a target layer in model.features"""
    if layer_name is None:
        # Return index of last conv layer (res5c_branch2c)
        for i, module in enumerate(reversed(model.features)):
            if hasattr(module, "weight") and len(module.weight.shape) == 4:
                return len(model.features) - 1 - i
        return len(model.features) - 1

    # Find by name
    for i, module in enumerate(model.features):
        if hasattr(module, "__class__"):
            if layer_name in module.__class__.__name__.lower():
                return i
    return len(model.features) - 1


def get_argparser():
    parser = argparse.ArgumentParser(description="Test FG-CAM for ResNet CNN")

    parser.add_argument(
        "--weight_file",
        type=str,
        default="/Users/ian/Project/VLN/R2R/30913b5b6a4c411bb1b6020f492e5862.npy",
        help="Path to ResNet weight file (.npy)",
    )
    parser.add_argument(
        "--class_file",
        type=str,
        default="/Users/ian/Project/VLN/R2R/checkpoints/categories_places365.txt",
        help="Path to class names file",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to input image(s) or directory containing images",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing images (alternative to --image_path)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing multiple images",
    )
    parser.add_argument(
        "--base_cam",
        type=str,
        default="grad_cam",
        choices=["grad_cam", "score_cam"],
        help="Base CAM method: grad_cam or score_cam",
    )
    parser.add_argument(
        "--denoising", action="store_true", help="Whether to use SVD denoising"
    )
    parser.add_argument(
        "--target_layer",
        type=int,
        default=-1,
        help="Target layer index in features list (-1 for input layer, or specific index)",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=None,
        help="Target class index (None for predicted class)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save visualization (None to display)",
    )

    return parser


def get_image_paths(opts):
    """Get list of image paths from arguments"""
    image_paths = []

    # If image_dir is specified, get all images from directory
    if opts.image_dir:
        if not os.path.isdir(opts.image_dir):
            raise ValueError(f"Image directory not found: {opts.image_dir}")
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        for filename in os.listdir(opts.image_dir):
            if any(filename.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(opts.image_dir, filename))
        image_paths.sort()

    # Add paths from --image_path argument
    if opts.image_path:
        for path in opts.image_path:
            if os.path.isdir(path):
                # If it's a directory, add all images in it
                extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
                for filename in os.listdir(path):
                    if any(filename.lower().endswith(ext) for ext in extensions):
                        image_paths.append(os.path.join(path, filename))
            elif os.path.isfile(path):
                image_paths.append(path)
            else:
                print(f"Warning: Path not found: {path}")

    if not image_paths:
        raise ValueError("No valid image paths found!")

    return image_paths


def load_batch_images(image_paths, device):
    """Load and preprocess a batch of images"""
    batch_images = []
    batch_paths = []

    for path in image_paths:
        try:
            img = ZeroCenter(path, 224, BGRTranspose=True)
            batch_images.append(img)
            batch_paths.append(path)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            continue

    if not batch_images:
        raise ValueError("No images could be loaded!")

    # Stack into batch
    batch_array = np.stack(batch_images, axis=0)  # (N, H, W, C)
    input_data = torch.from_numpy(batch_array)
    input_data = input_data.permute(0, 3, 1, 2)  # (N, C, H, W)
    input_data = input_data.to(device)
    input_data.requires_grad = True

    return input_data, batch_paths


def main(opts):
    # Load class names
    classes = []
    try:
        with open(opts.class_file) as class_file:
            for line in class_file:
                classes.append(line.strip().split(" ")[0][3:])
    except FileNotFoundError:
        print(f"Warning: Class file not found at {opts.class_file}")
        print("Using placeholder class names")
        classes = [f"class_{i}" for i in range(365)]
    classes = tuple(classes)

    # Load model
    print(f"Loading model from {opts.weight_file}")
    model = CNN(weight_file=opts.weight_file)
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Get image paths
    image_paths = get_image_paths(opts)
    print(f"\nFound {len(image_paths)} image(s) to process")

    # Process images in batches
    batch_size = opts.batch_size
    all_results = []

    for batch_idx in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_idx : batch_idx + batch_size]
        print(
            f"\nProcessing batch {batch_idx // batch_size + 1}/{(len(image_paths) + batch_size - 1) // batch_size}"
        )
        print(f"  Images: {[os.path.basename(p) for p in batch_paths]}")

        # Load batch
        input_data, valid_paths = load_batch_images(batch_paths, device)
        actual_batch_size = input_data.shape[0]

        # # Get model predictions
        # print("  Running forward pass...")
        # with torch.no_grad():
        #     _, logit = model(input_data)
        #     h_x = F.softmax(logit, 1).data  # (batch_size, num_classes)
        #     probs, idx = h_x.sort(1, descending=True)

        #     # Print top predictions for each image
        #     for i in range(actual_batch_size):
        #         print(f"\n  Image {i+1} ({os.path.basename(valid_paths[i])}):")
        #         print(f"    Top 3 predictions:")
        #         for j in range(min(3, len(classes))):
        #             print(f"      {probs[i][j]:.3f} -> {classes[idx[i][j]]}")

        # Create FG-CAM instance
        fg_cam = FG_CAM_resnet(model, opts.base_cam)

        # Determine target layer index
        if opts.target_layer == -1:
            target_layer = -1
        else:
            target_layer = opts.target_layer
            if target_layer >= len(model.features):
                print(
                    f"Warning: target_layer {target_layer} >= features length {len(model.features)}"
                )
                target_layer = len(model.features) - 1

        # Process each image in the batch
        batch_results = []
        for i in range(actual_batch_size):
            single_input = input_data[i : i + 1]  # Keep batch dimension
            print("single_input", single_input.shape)

            # Generate explanation
            explanation, predicted_class = fg_cam(
                single_input,
                denoising=opts.denoising,
                target_layer=target_layer,
                target_class=opts.target_class,
            )

            # Process explanation
            print("explanation", explanation.shape)
            explanation = torch.relu(explanation)
            print("explanation", explanation.shape)
            explanation_vis = visual_explanation(explanation[0])  # Remove batch dim

            # Load original image for visualization
            original_img = Image.open(valid_paths[i]).convert("RGB")
            original_img = np.array(original_img)
            original_img = cv2.resize(original_img, (224, 224))

            # Get base CAM for comparison (if target_layer is not -1)
            base_cam_vis = None
            if opts.target_layer != -1:
                layer = fg_cam.find_last_layer()
                base_cam_explanation = fg_cam.get_explanation_component(
                    single_input, predicted_class, layer
                )
                base_cam_explanation = torch.sum(base_cam_explanation, dim=1)
                base_cam_explanation = torch.relu(base_cam_explanation)
                base_cam_vis = visual_explanation(base_cam_explanation[0])

            batch_results.append(
                {
                    "path": valid_paths[i],
                    "original_img": original_img,
                    "explanation_vis": explanation_vis,
                    "base_cam_vis": base_cam_vis,
                    "predicted_class": predicted_class,
                    "class_name": classes[predicted_class],
                }
            )

        all_results.extend(batch_results)

    # Visualize all results in a grid
    num_images = len(all_results)
    num_cols = 3 if opts.target_layer != -1 else 2
    num_rows = num_images

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    if num_images == 1:
        axes = axes.reshape(1, -1)

    for idx, result in enumerate(all_results):
        row = idx

        # Original image
        axes[row, 0].imshow(result["original_img"])
        axes[row, 0].set_title(
            f"{os.path.basename(result['path'])}\nPredicted: {result['class_name']}",
            fontsize=10,
        )
        axes[row, 0].axis("off")

        # FG-CAM explanation
        im = axes[row, 1].imshow(result["explanation_vis"], cmap="jet")
        axes[row, 1].set_title(f"FG-CAM ({opts.base_cam})", fontsize=10)
        axes[row, 1].axis("off")

        # Base CAM (if applicable)
        if num_cols == 3:
            if result["base_cam_vis"] is not None:
                axes[row, 2].imshow(result["base_cam_vis"], cmap="jet")
            axes[row, 2].set_title(f"Base {opts.base_cam}", fontsize=10)
            axes[row, 2].axis("off")

    plt.tight_layout()

    if opts.output_path:
        plt.savefig(opts.output_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved visualization to {opts.output_path}")
    else:
        plt.show()

    plt.close()
    print(f"\nDone! Processed {num_images} image(s).")


if __name__ == "__main__":
    opts = get_argparser().parse_args()
    main(opts)
