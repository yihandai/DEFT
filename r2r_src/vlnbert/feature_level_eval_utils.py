"""
Utility functions for feature-level evaluation, including BLIP-2 image captioning
and Fast-RCNN object detection.
"""

import numpy as np
import os
import torch
from PIL import Image

# Optional imports with helpful error messages
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
except ImportError:
    Blip2Processor = None
    Blip2ForConditionalGeneration = None
    print(
        "Warning: transformers library not found. "
        "Please install it with: pip install transformers"
    )

try:
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    )
except ImportError:
    fasterrcnn_resnet50_fpn = None
    FasterRCNN_ResNet50_FPN_Weights = None
    print(
        "Warning: torchvision.models.detection not found. "
        "Please ensure torchvision is properly installed."
    )

# 默认模型权重目录（相对于项目根目录）
DEFAULT_MODEL_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    # "../",
    "model_checkpoints_2025_12_26",
    "model_checkpoints",
)
print(f"DEFAULT_MODEL_CACHE_DIR: {DEFAULT_MODEL_CACHE_DIR}")
# COCO 类别名称将从 torchvision weights 中获取，确保准确性
# 这里提供一个备用映射（COCO ID -> 类别名称）
# COCO 类别 ID 范围是 1-90，但不连续（缺失的 ID: 12, 26, 29, 30, 45, 66, 68, 69, 71, 83）
COCO_ID_TO_NAME = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}


class BLIP2Captioner:
    """BLIP-2 image captioning model wrapper."""

    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl", cache_dir=None):
        """Initialize BLIP-2 model.

        Args:
            model_name: HuggingFace model name for BLIP-2, or local path to model
            cache_dir: Directory to cache/download model weights. If None, uses DEFAULT_MODEL_CACHE_DIR
        """
        self.model_name = model_name
        self.cache_dir = cache_dir if cache_dir else DEFAULT_MODEL_CACHE_DIR
        # 如果 model_name 是本地路径，使用它；否则在 cache_dir 下创建子目录
        if os.path.isdir(model_name):
            self.model_path = model_name
        else:
            # 在 cache_dir 下创建 blip2 子目录
            self.model_path = os.path.join(self.cache_dir, "blip2-flan-t5-xl")
        self.processor = None
        self.model = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of BLIP-2 model."""
        if self._initialized:
            return

        if Blip2Processor is None or Blip2ForConditionalGeneration is None:
            raise ImportError(
                "transformers library is required for BLIP-2. "
                "Please install it with: pip install transformers"
            )

        print(f"Loading BLIP-2 model from {self.model_name}...")
        print(f"Cache directory: {self.model_path}")

        # 如果本地路径存在，使用本地路径；否则从 HuggingFace 下载
        local_model_exists = os.path.isdir(self.model_path) and os.path.exists(
            os.path.join(self.model_path, "config.json")
        )

        if local_model_exists:
            print(f"Loading from local path: {self.model_path}")
            try:
                self.processor = Blip2Processor.from_pretrained(self.model_path)
                # 先加载模型，然后转换 dtype（兼容不同版本的 transformers）
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_path,
                )
                # 转换 dtype 并移动到 GPU（如果需要）
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda").to(torch.float16)
                else:
                    self.model = self.model.to(torch.float32)
            except Exception as e:
                # 如果加载失败（可能是 tokenizer 版本不兼容或缓存损坏），清除 tokenizer 相关文件并重试
                error_msg = str(e)
                if (
                    "PyPreTokenizerTypeWrapper" in error_msg
                    or "tokenizer" in error_msg.lower()
                ):
                    print(
                        f"⚠ Tokenizer loading failed (possibly version incompatibility): {e}"
                    )
                    print(
                        "  Attempting to fix by clearing tokenizer cache and re-downloading..."
                    )

                    # 清除可能损坏的 tokenizer 文件
                    tokenizer_files = [
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "spiece.model",
                        "vocab.txt",
                        "merges.txt",
                    ]
                    for tokenizer_file in tokenizer_files:
                        tokenizer_path = os.path.join(self.model_path, tokenizer_file)
                        if os.path.exists(tokenizer_path):
                            try:
                                os.remove(tokenizer_path)
                                print(f"  Removed: {tokenizer_file}")
                            except Exception as rm_e:
                                print(
                                    f"  Warning: Could not remove {tokenizer_file}: {rm_e}"
                                )

                    # 尝试从 HuggingFace 重新下载（使用 local_files_only=False 强制重新下载 tokenizer）
                    print("  Re-downloading tokenizer from HuggingFace...")
                    try:
                        self.processor = Blip2Processor.from_pretrained(
                            self.model_name,
                            cache_dir=self.model_path,
                            local_files_only=False,  # 强制重新下载
                        )
                        # 如果 processor 加载成功，尝试加载模型（可能使用本地权重）
                        self.model = Blip2ForConditionalGeneration.from_pretrained(
                            self.model_path,
                            local_files_only=False,  # 如果模型文件也有问题，会重新下载
                        )
                        # 转换 dtype 并移动到 GPU（如果需要）
                        if torch.cuda.is_available():
                            self.model = self.model.to("cuda").to(torch.float16)
                        else:
                            self.model = self.model.to(torch.float32)
                        print("  ✓ Successfully re-downloaded and loaded tokenizer!")
                    except Exception as retry_e:
                        print(f"  ✗ Re-download failed: {retry_e}")
                        raise RuntimeError(
                            f"Failed to load BLIP-2 model after retry. "
                            f"Original error: {e}. Retry error: {retry_e}. "
                            f"This might be due to incompatible tokenizers library version. "
                            f"Try: pip install --upgrade tokenizers transformers"
                        ) from retry_e
                else:
                    # 其他类型的错误，直接抛出
                    raise
        else:
            print(f"Downloading model to: {self.model_path}")
            os.makedirs(self.model_path, exist_ok=True)
            self.processor = Blip2Processor.from_pretrained(
                self.model_name, cache_dir=self.model_path
            )
            # 先加载模型，然后转换 dtype（兼容不同版本的 transformers）
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=self.model_path,
            )
            # 转换 dtype 并移动到 GPU（如果需要）
            if torch.cuda.is_available():
                self.model = self.model.to("cuda").to(torch.float16)
            else:
                self.model = self.model.to(torch.float32)

        self.model.eval()
        self._initialized = True

    def caption_images(self, images, prompt="This is a scene of ", max_length=50):
        """Generate captions for a batch of images.

        Args:
            images: numpy array of shape (N, H, W, 3) with uint8 values in [0, 255]
            prompt: text prompt for captioning
            max_length: maximum length of generated caption

        Returns:
            List of caption strings, one per image
        """
        self._initialize()

        descriptions = []
        for img in images:
            img_pil = Image.fromarray(img)
            inputs = self.processor(images=img_pil, text=prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_length=max_length)

            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            descriptions.append(generated_text)

        return descriptions


class FastRCNNDetector:
    """Fast-RCNN object detection model wrapper."""

    def __init__(self, score_threshold=0.5, cache_dir=None):
        """Initialize Fast-RCNN model.

        Args:
            score_threshold: minimum confidence score for detected objects
            cache_dir: Directory to cache model weights. If None, uses DEFAULT_MODEL_CACHE_DIR
        """
        self.score_threshold = score_threshold
        self.cache_dir = cache_dir if cache_dir else DEFAULT_MODEL_CACHE_DIR
        self.model = None
        self.weights = None
        self.category_names = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of Fast-RCNN model."""
        if self._initialized:
            return

        if fasterrcnn_resnet50_fpn is None or FasterRCNN_ResNet50_FPN_Weights is None:
            raise ImportError(
                "torchvision.models.detection is required for Fast-RCNN. "
                "Please ensure torchvision is properly installed."
            )

        print("Loading Fast-RCNN model...")
        # torchvision 会自动使用 torch.hub 的缓存目录
        # 我们可以设置 TORCH_HOME 环境变量来指定缓存位置
        torch_home = os.path.join(self.cache_dir, "torchvision")
        os.makedirs(torch_home, exist_ok=True)
        original_torch_home = os.environ.get("TORCH_HOME", None)
        os.environ["TORCH_HOME"] = torch_home

        try:
            # 尝试使用新版本的 weights API
            self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn(weights=self.weights)
            # 从 weights 中获取类别名称（最可靠的方法）
            if hasattr(self.weights, "meta") and "categories" in self.weights.meta:
                categories = self.weights.meta["categories"]
                # 将类别列表转换为字典（COCO ID -> 类别名称）
                # torchvision 的 categories 列表：索引 0 是 '__background__'，索引 1 开始对应 COCO ID 1
                if isinstance(categories, list):
                    # 创建从 COCO ID 到类别名称的映射
                    # 注意：torchvision 的 categories 列表索引 0 是 '__background__'，
                    # 索引 1 开始对应 COCO ID 1（person），索引 2 对应 COCO ID 2（bicycle），等等
                    self.category_names = {}
                    for idx, name in enumerate(categories):
                        # 跳过 background 类别（索引 0）
                        if idx == 0 and name == "__background__":
                            continue
                        # categories[1] -> COCO ID 1, categories[2] -> COCO ID 2, etc.
                        coco_id = idx
                        self.category_names[coco_id] = name
                    # 验证关键类别的映射（用于调试）
                    if 2 in self.category_names and 3 in self.category_names:
                        assert (
                            self.category_names[2] == "bicycle"
                        ), f"Expected bicycle, got {self.category_names[2]}"
                        assert (
                            self.category_names[3] == "car"
                        ), f"Expected car, got {self.category_names[3]}"
                elif isinstance(categories, dict):
                    self.category_names = categories
                else:
                    self.category_names = COCO_ID_TO_NAME
            else:
                # 备用方案：使用预定义的映射
                self.category_names = COCO_ID_TO_NAME
        except:
            # 回退到旧版本的 pretrained 参数
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)
            # 旧版本没有 weights，使用备用映射
            self.category_names = COCO_ID_TO_NAME
        finally:
            # 恢复原始的 TORCH_HOME
            if original_torch_home is not None:
                os.environ["TORCH_HOME"] = original_torch_home
            elif "TORCH_HOME" in os.environ:
                del os.environ["TORCH_HOME"]

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self._initialized = True

    def detect_objects(self, images):
        """Detect objects in a batch of images.

        Args:
            images: numpy array of shape (N, H, W, 3) with uint8 values in [0, 255]

        Returns:
            List of lists, where each inner list contains detected objects for one image.
            Each object is a dict with keys: 'class_name', 'class_id', 'score', 'bbox'
        """
        self._initialize()

        detected_objects = []
        for img in images:
            # Convert numpy array to tensor: (H, W, 3) -> (3, H, W) and normalize to [0, 1]
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()

            with torch.no_grad():
                predictions = self.model([img_tensor])

            # Extract detected objects
            pred = predictions[0]
            objects = []
            for i in range(len(pred["labels"])):
                # print(pred["scores"][i])
                if pred["scores"][i] > self.score_threshold:
                    class_id = pred["labels"][i].item()
                    # 获取类别名称：使用字典映射（COCO ID -> 类别名称）
                    class_name = self.category_names.get(class_id, f"class_{class_id}")
                    score = pred["scores"][i].item()
                    bbox = pred["boxes"][i].cpu().numpy().tolist()
                    objects.append(
                        {
                            "class_name": class_name,
                            "class_id": class_id,
                            "score": score,
                            "bbox": bbox,
                        }
                    )
            detected_objects.append(objects)

        return detected_objects


def process_perturbed_images(images, blip_prompt="This is a scene of ", cache_dir=None):
    """Process perturbed images with BLIP-2 captioning and Fast-RCNN detection.

    Args:
        images: numpy array of shape (N, H, W, 3) with uint8 values in [0, 255]
        blip_prompt: prompt for BLIP-2 captioning
        cache_dir: Directory to cache model weights

    Returns:
        Dictionary with keys:
            - 'blip_descriptions': List of caption strings
            - 'detected_objects': List of lists of detected objects
    """
    # Initialize models (singleton pattern via class attributes)
    cache_key = cache_dir if cache_dir else "default"
    if not hasattr(process_perturbed_images, "_blip_captioners"):
        process_perturbed_images._blip_captioners = {}
    if not hasattr(process_perturbed_images, "_faster_rcnn_detectors"):
        process_perturbed_images._faster_rcnn_detectors = {}

    if cache_key not in process_perturbed_images._blip_captioners:
        process_perturbed_images._blip_captioners[cache_key] = BLIP2Captioner(
            cache_dir=cache_dir
        )
    if cache_key not in process_perturbed_images._faster_rcnn_detectors:
        process_perturbed_images._faster_rcnn_detectors[cache_key] = FastRCNNDetector(
            cache_dir=cache_dir
        )

    blip_captioner = process_perturbed_images._blip_captioners[cache_key]
    faster_rcnn_detector = process_perturbed_images._faster_rcnn_detectors[cache_key]

    # Generate captions
    blip_descriptions = blip_captioner.caption_images(images, prompt=blip_prompt)

    # Detect objects
    detected_objects = faster_rcnn_detector.detect_objects(images)

    return {
        "blip_descriptions": blip_descriptions,
        "detected_objects": detected_objects,
    }


def main():
    """Test function for BLIP-2 and Fast-RCNN models."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Test BLIP-2 and Fast-RCNN models")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to test image (if not provided, will use a test image from CAM/images/)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache model weights (default: model_checkpoints/)",
    )
    args = parser.parse_args()

    # 确定测试图片路径
    if args.image_path:
        test_image_path = args.image_path
    else:
        # 使用项目中的测试图片
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        test_image_path = os.path.join("CAM", "images", "ILSVRC2012_val_00000001.JPEG")
        print(f"Test image path: {test_image_path}")
        if not os.path.exists(test_image_path):
            # 尝试其他图片
            test_image_path = os.path.join(project_root, "CAM", "images", "pic1.jpg")
            if not os.path.exists(test_image_path):
                print(f"Error: Test image not found. Please provide --image_path")
                sys.exit(1)

    print(f"Using test image: {test_image_path}")

    # 加载测试图片
    try:
        img = Image.open(test_image_path).convert("RGB")
        # 调整大小到 224x224（Fast-RCNN 的输入尺寸）
        img = img.resize((224, 224))
        img_array = np.array(img)
        print(f"Image shape: {img_array.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # 准备图片数组 (N, H, W, 3)
    images = np.expand_dims(img_array, axis=0)

    print("\n" + "=" * 60)
    print("Testing BLIP-2 and Fast-RCNN models")
    print("=" * 60)

    # 测试 BLIP-2
    print("\n[1/2] Testing BLIP-2 captioning...")
    try:
        blip_captioner = BLIP2Captioner(cache_dir=args.cache_dir)
        descriptions = blip_captioner.caption_images(
            images, prompt="This is a scene of "
        )
        print(f"✓ BLIP-2 caption: {descriptions[0]}")
    except Exception as e:
        print(f"✗ BLIP-2 failed: {e}")
        import traceback

        traceback.print_exc()

    # 测试 Fast-RCNN
    print("\n[2/2] Testing Fast-RCNN object detection...")
    try:
        faster_rcnn_detector = FastRCNNDetector(cache_dir=args.cache_dir)
        detected_objects = faster_rcnn_detector.detect_objects(images)
        objects = detected_objects[0]
        print(f"✓ Detected {len(objects)} objects:")
        for i, obj in enumerate(objects[:10]):  # 只显示前10个
            print(
                f"  {i+1}. {obj['class_name']} (ID: {obj['class_id']}, "
                f"score: {obj['score']:.3f}, bbox: {obj['bbox']})"
            )
        if len(objects) > 10:
            print(f"  ... and {len(objects) - 10} more objects")
    except Exception as e:
        print(f"✗ Fast-RCNN failed: {e}")
        import traceback

        traceback.print_exc()

    # 测试完整流程
    print("\n[3/3] Testing complete pipeline...")
    try:
        results = process_perturbed_images(images, cache_dir=args.cache_dir)
        print(f"✓ Pipeline completed successfully!")
        print(f"  - BLIP-2 descriptions: {len(results['blip_descriptions'])}")
        print(
            f"  - Detected objects: {sum(len(objs) for objs in results['detected_objects'])}"
        )
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    print(f"\nModel weights are cached in: {args.cache_dir or DEFAULT_MODEL_CACHE_DIR}")


if __name__ == "__main__":
    main()
