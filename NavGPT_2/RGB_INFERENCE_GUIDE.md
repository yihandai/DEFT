# NavGPT-2 RGB图像直接推理指南

本指南说明如何修改NavGPT-2，使其能够从RGB图像直接推理，而不使用本地预存的图像特征。

## 概述

NavGPT-2默认使用预存的图像特征（存储在LMDB数据库中）。要从RGB图像直接推理，需要：

1. 启用MatterSim的渲染功能以获取RGB图像
2. 使用视觉编码器（visual_encoder）将RGB图像编码为特征
3. 修改环境代码以支持RGB图像处理

## 修改步骤

### 1. 修改环境初始化

在初始化NavGPT-2环境时，需要：

- 传入视觉编码器（visual_encoder）而不是特征数据库（ImageFeaturesDB）
- 使用`R2RNavBatchRGB`替代`R2RNavBatch`

### 2. 获取视觉编码器

视觉编码器可以从NavGPT-2模型中获取：

```python
from NavGPT_2.map_nav_src.r2r.agent import GMapNavAgent

# 初始化agent
agent = GMapNavAgent(args, env, rank=0)

# 获取视觉编码器
# 注意：需要确保模型没有删除visual_encoder
visual_encoder = agent.NavGPT.llm.Blip2InstructNav.visual_encoder
ln_vision = agent.NavGPT.llm.Blip2InstructNav.ln_vision

# 创建一个包装器，包含visual_encoder和ln_vision
class VisualEncoderWrapper:
    def __init__(self, visual_encoder, ln_vision):
        self.visual_encoder = visual_encoder
        self.ln_vision = ln_vision
    
    def __call__(self, x):
        # 先通过visual_encoder，再通过ln_vision
        features = self.visual_encoder(x)
        features = self.ln_vision(features)
        return features

visual_encoder_wrapper = VisualEncoderWrapper(visual_encoder, ln_vision)
```

### 3. 修改模型配置

在`NavGPT_model.py`中，确保`load_patch_feature`配置为`False`，这样visual_encoder不会被删除：

```python
# 在配置文件中设置
config.load_patch_feature = False
```

或者在初始化时：

```python
args.load_patch_feature = False
```

### 4. 使用RGB环境

修改环境初始化代码：

```python
from NavGPT_2.map_nav_src.r2r.env_rgb import R2RNavBatchRGB

# 原来的代码（使用预存特征）
# feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
# navgpt2_env = R2RNavBatch(
#     feat_db,
#     None,
#     args.connectivity_dir,
#     args.candidate_file_dir,
#     batch_size=args.batchSize,
#     ...
# )

# 新的代码（使用RGB图像）
navgpt2_env = R2RNavBatchRGB(
    visual_encoder_wrapper,  # 传入视觉编码器
    None,  # instr_data，稍后设置
    args.connectivity_dir,
    args.candidate_file_dir,
    batch_size=args.batchSize,
    angle_feat_size=args.angle_feat_size,
    seed=args.seed + rank,
    sel_data_idxs=None,
    name="train",
    visual_encoder=visual_encoder_wrapper,
)
```

### 5. 完整示例

以下是一个完整的修改示例：

```python
import sys
import os
sys.path.insert(0, 'NavGPT_2/map_nav_src')

from NavGPT_2.map_nav_src.r2r.agent import GMapNavAgent
from NavGPT_2.map_nav_src.r2r.env_rgb import R2RNavBatchRGB
from NavGPT_2.map_nav_src.utils.data import ImageFeaturesDB

# 1. 初始化agent（临时环境，仅用于获取visual_encoder）
# 注意：这里仍然需要传入一个环境，但我们可以稍后替换它
temp_feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
temp_env = R2RNavBatch(
    temp_feat_db,
    None,
    args.connectivity_dir,
    args.candidate_file_dir,
    batch_size=1,
    angle_feat_size=args.angle_feat_size,
    seed=args.seed,
    sel_data_idxs=None,
    name="temp",
)

# 2. 初始化agent并加载checkpoint
agent = GMapNavAgent(args, temp_env, rank=0)
if hasattr(args, "resume_file") and args.resume_file is not None:
    agent.load(args.resume_file)

# 3. 获取视觉编码器
visual_encoder = agent.NavGPT.llm.Blip2InstructNav.visual_encoder
ln_vision = agent.NavGPT.llm.Blip2InstructNav.ln_vision

# 4. 创建视觉编码器包装器
class VisualEncoderWrapper:
    def __init__(self, visual_encoder, ln_vision):
        self.visual_encoder = visual_encoder
        self.ln_vision = ln_vision
        self.eval()  # 设置为评估模式
    
    def eval(self):
        self.visual_encoder.eval()
        if self.ln_vision is not None:
            self.ln_vision.eval()
    
    def __call__(self, x):
        with torch.no_grad():
            features = self.visual_encoder(x)
            if self.ln_vision is not None:
                features = self.ln_vision(features)
            return features
    
    def parameters(self):
        # 用于获取device
        return self.visual_encoder.parameters()

visual_encoder_wrapper = VisualEncoderWrapper(visual_encoder, ln_vision)

# 5. 创建RGB环境
rgb_env = R2RNavBatchRGB(
    visual_encoder_wrapper,
    instr_data,
    args.connectivity_dir,
    args.candidate_file_dir,
    batch_size=args.batchSize,
    angle_feat_size=args.angle_feat_size,
    seed=args.seed + rank,
    sel_data_idxs=None,
    name="train",
    visual_encoder=visual_encoder_wrapper,
)

# 6. 重新初始化agent，使用RGB环境
agent = GMapNavAgent(args, rgb_env, rank=0)
if hasattr(args, "resume_file") and args.resume_file is not None:
    agent.load(args.resume_file)
```

## 注意事项

1. **性能影响**：从RGB图像直接推理会比使用预存特征慢，因为需要实时编码图像。

2. **内存使用**：视觉编码器会占用额外的GPU内存。

3. **模型配置**：确保`load_patch_feature=False`，否则visual_encoder会被删除。

4. **图像预处理**：RGB图像会被预处理为224x224大小，使用ImageNet的均值和标准差归一化。

5. **特征格式**：编码后的特征应该与预存特征的格式兼容。如果格式不匹配，可能需要调整`_encode_rgb_to_features`方法。

## 故障排除

### 问题1：visual_encoder不存在

**原因**：`load_patch_feature=True`导致visual_encoder被删除。

**解决**：设置`args.load_patch_feature = False`。

### 问题2：特征维度不匹配

**原因**：编码后的特征维度与预存特征不同。

**解决**：检查`_encode_rgb_to_features`方法，确保输出特征维度正确。

### 问题3：推理速度慢

**原因**：实时编码图像比加载预存特征慢。

**解决**：这是预期的行为。可以考虑：
- 使用更小的batch size
- 使用混合精度推理（FP16/BF16）
- 缓存已编码的特征

## 相关文件

- `NavGPT_2/map_nav_src/r2r/env_rgb.py` - RGB环境实现
- `NavGPT_2/map_nav_src/r2r/env.py` - 原始环境实现
- `NavGPT_2/map_nav_src/models/NavGPT_model.py` - NavGPT模型定义
- `NavGPT_2/map_nav_src/models/lavis/models/blip2_models/blip2_t5_instruct_nav.py` - BLIP-2模型定义

