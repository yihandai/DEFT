# HSIC Attribution Method 详细讲解

## 目录
1. [概述](#概述)
2. [核心原理](#核心原理)
3. [代码结构](#代码结构)
4. [详细实现解析](#详细实现解析)
5. [工作流程](#工作流程)
6. [关键参数说明](#关键参数说明)

---

## 概述

**HSIC (Hilbert-Schmidt Independence Criterion) Attribution Method** 是一种基于依赖度量的黑盒解释方法，用于生成特征级别的归因显著性图。

### 论文参考
- **论文**: "Making Sense of Dependance: Efficient Black-box Explanations Using Dependence Measure"
- **作者**: Novello, Fel, Vigouroux
- **链接**: https://arxiv.org/abs/2206.06219

### 核心思想
通过计算输入图像的每个区域（网格单元）与模型输出之间的**依赖关系**（HSIC分数），来确定哪些区域对模型决策最重要。

---

## 核心原理

### HSIC 数学原理

HSIC 衡量两个随机变量之间的独立性：
- **输入**: 图像网格单元的扰动掩码（二进制）
- **输出**: 模型预测概率的变化
- **HSIC分数**: 衡量两者之间的依赖程度

**简化公式**（本实现使用）:
```
HSIC(X, Y) ≈ |E[(X - μ_X)(Y - μ_Y)]|
```
其中：
- `X`: 网格单元的扰动状态（0或1）
- `Y`: 模型输出的变化
- 值越大，表示该区域对输出的影响越大

---

## 代码结构

### 主要方法层次

**NavGPT2版本**（当前实现）:
```
compute_hsic_attribution_navgpt2()          # 主入口函数（NavGPT2专用）
├── _compute_hsic_for_image_navgpt2()      # 为单个图像计算HSIC（NavGPT2）
│   ├── _apply_perturbation()              # 应用扰动
│   └── _get_model_output_navgpt2()        # 获取NavGPT2模型输出
```

**VLN-BERT版本**（旧版本，已废弃）:
```
compute_hsic_attribution()          # 主入口函数（VLN-BERT）
├── _compute_hsic_for_image()       # 为单个图像计算HSIC
│   ├── _apply_perturbation()       # 应用扰动
│   └── _get_model_output()         # 获取VLN-BERT模型输出
```

**注意**: 当前实现专门针对NavGPT2，直接在NavGPT2模型上计算HSIC，而不是使用VLN-BERT作为替代模型。

---

## 详细实现解析

### 1. `compute_hsic_attribution_navgpt2()` - 主函数（NavGPT2版本）

#### 功能
为 NavGPT2 的每个导航步骤生成 HSIC 归因显著性图。**直接在NavGPT2模型上计算**，不使用替代模型。

#### 输入参数

```python
obs: List[dict]                    # 观察列表（包含scan、viewpoint等信息）
t: int                             # 当前时间步
target_agent: GMapNavAgent         # NavGPT2 agent实例
gmaps: List[GraphMap]              # NavGPT2的图映射对象列表
instructions: List[str]            # 指令字符串列表
grid_size: int = 8                 # 网格大小（8x8 = 64个单元）
nb_design: int = 500               # 蒙特卡洛采样数量
perturbation_function: str = "inpainting"  # 扰动类型
batch_size: int = 32               # 批处理大小（预留，当前未直接使用）
```

#### 执行流程

```python
# 步骤1: 获取全景图像
images_numpys = []  # 存储所有视角的图像
for i in range(bs):
    scanId = obs[i]["scan"]
    viewpointId = obs[i]["viewpoint"]
    images_numpy = self.get_vp_images(self.sim, scanId, viewpointId)
    images_numpys.append(images_numpy)
images_numpys = np.stack(images_numpys)  # [bs, vp, C, H, W]
```

**说明**:
- 从模拟器中获取当前视角的全景图像
- 形状: `[batch_size, VIEWPOINT_SIZE, Channels, Height, Width]`
- 通常 `VIEWPOINT_SIZE = 36` (3个高度 × 12个水平视角)

```python
# 步骤2: 获取原始NavGPT2动作
with torch.no_grad():
    a_t_original, nav_vpids_list, nav_inputs_dict = NavGPT2_genAction_v2(
        target_agent,
        obs,
        gmaps,
        instructions,
        t,
        ended=None,
        feedback="argmax",
    )
target_action = a_t_original[0]  # 原始动作索引
```

**说明**:
- **直接调用NavGPT2**获取原始动作预测
- `target_action`是NavGPT2预测的动作索引（0=停止，1+=导航动作）
- 后续HSIC计算将比较扰动后的动作是否与`target_action`匹配

```python
# 步骤3: 获取NavGPT2的visual encoder
visual_encoder = target_agent.NavGPT.llm.Blip2InstructNav.visual_encoder
ln_vision = target_agent.NavGPT.llm.Blip2InstructNav.ln_vision

# 步骤4: 处理每个候选视角
for vp_idx in cand_idx:
    single_image = images_numpys[i, vp_idx]  # [C, H, W]
    hsic_scores = self._compute_hsic_for_image_navgpt2(
        single_image,
        obs[i],
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
        grid_size,
        nb_design,
        perturbation_function,
    )  # [grid_size, grid_size]
    
    # 将网格分数上采样到原始图像尺寸
    hsic_map = cv2.resize(hsic_scores, (W, H), interpolation=cv2.INTER_CUBIC)
    
    # 归一化到 [0, 255]
    hsic_map = (hsic_map - hsic_map.min()) / (hsic_map.max() - hsic_map.min() + 1e-9)
    hsic_map = (hsic_map * 255).clip(0, 255).astype(np.uint8)
    
    heatmap_all[vp_idx] = hsic_map
```

**说明**:
- 获取NavGPT2的**BLIP2 visual encoder**用于特征提取
- 只处理**候选视角**（可导航的视角）
- 为每个视角生成 `[grid_size, grid_size]` 的HSIC分数矩阵
- 通过双三次插值上采样到原始图像尺寸 `[H, W]`
- 归一化并转换为uint8格式用于可视化

#### 输出

```python
return images_return, heatmaps, candidata_list
```

- `images_return`: `[B, V, H, W, C]` - 原始图像（用于可视化）
- `heatmaps`: `List[np.array[V, H, W]]` - HSIC显著性图
- `candidata_list`: 候选视角索引列表

---

### 2. `_compute_hsic_for_image_navgpt2()` - 核心计算函数（NavGPT2版本）

#### 功能
为单个图像计算每个网格单元的HSIC分数，使用NavGPT2模型。

#### 关键步骤

##### 步骤1: 网格划分

```python
C, H, W = image.shape  # 例如: [3, 224, 224]
cell_h = H // grid_size  # 224 // 8 = 28
cell_w = W // grid_size  # 224 // 8 = 28
hsic_scores = np.zeros((grid_size, grid_size), dtype=np.float32)
```

**说明**:
- 将 `224×224` 的图像划分为 `8×8 = 64` 个网格单元
- 每个单元大小为 `28×28` 像素
- 初始化HSIC分数矩阵

##### 步骤2: 生成扰动掩码

```python
np.random.seed(42)  # 固定随机种子，保证可复现
binary_masks = np.random.binomial(1, 0.5, size=(nb_design, grid_size, grid_size))
```

**说明**:
- 生成 `nb_design` 个二进制掩码（默认500个）
- 每个掩码是 `[grid_size, grid_size]` 的二进制矩阵
- 每个位置有50%概率为1（扰动）或0（保留）
- **作用**: 用于蒙特卡洛采样，探索不同扰动组合对输出的影响

**示例掩码**:
```
掩码1: [[1, 0, 1, ...],    # 扰动第1,3,...个单元
        [0, 1, 0, ...],    # 保留第2,4,...个单元
        ...]
掩码2: [[0, 1, 0, ...],    # 不同的扰动组合
        [1, 0, 1, ...],
        ...]
```

##### 步骤3: 获取基线输出

```python
baseline_output = self._get_model_output_navgpt2(
    image,  # 原始图像
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
)
```

**说明**:
- 使用**原始图像**（无扰动）获取NavGPT2模型输出
- 返回二进制值：1.0（动作匹配target_action）或0.0（不匹配）
- 作为后续比较的基准（虽然当前实现中未直接使用，但可用于未来优化）

##### 步骤4: 计算每个网格单元的HSIC分数

```python
for i in range(grid_size):      # 遍历每个网格行
    for j in range(grid_size):  # 遍历每个网格列
        cell_outputs = []
        
        # 对每个扰动掩码
        for mask_idx in range(nb_design):
            pert_mask = binary_masks[mask_idx].copy()
            
            # 应用扰动
            perturbed_image = self._apply_perturbation(
                image.copy(),
                pert_mask,
                grid_size,
                cell_h,
                cell_w,
                perturbation_function,
            )
            
            # 获取扰动后的NavGPT2模型输出
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
            )
            cell_outputs.append(output)  # output是float (1.0或0.0)
        
        # 计算HSIC分数
        cell_presence = binary_masks[:len(cell_outputs), i, j]  # 该单元在所有掩码中的状态
        outputs = np.array(cell_outputs)  # 对应的模型输出
        
        # 标准化
        cell_presence = (cell_presence - cell_presence.mean()) / (cell_presence.std() + 1e-9)
        outputs = (outputs - outputs.mean()) / (outputs.std() + 1e-9)
        
        # HSIC近似: 相关性
        hsic_score = np.abs(np.mean(cell_presence * outputs))
        hsic_scores[i, j] = hsic_score
```

**详细解释**:

1. **双重循环**: 遍历每个网格单元 `(i, j)`

2. **蒙特卡洛采样**: 
   - 对每个扰动掩码，应用扰动并获取模型输出
   - 收集 `nb_design` 个输出值

3. **HSIC计算**:
   - `cell_presence`: 该单元在所有掩码中的扰动状态（0或1的数组）
   - `outputs`: 对应的模型输出数组
   - **标准化**: 减去均值，除以标准差（Z-score标准化）
   - **相关性计算**: `E[(X - μ_X)(Y - μ_Y)]`，即 `mean(cell_presence * outputs)`
   - **取绝对值**: 只关心依赖强度，不关心方向

4. **物理意义**:
   - 如果该单元对输出影响大：
     - 当单元被扰动（1）时，输出变化大
     - 当单元保留（0）时，输出变化小
     - 两者相关性高 → HSIC分数高
   - 如果该单元对输出影响小：
     - 无论是否扰动，输出变化都不大
     - 相关性低 → HSIC分数低

---

### 3. `_apply_perturbation()` - 扰动函数

#### 功能
根据二进制掩码对图像应用扰动。

#### 支持的扰动类型

##### 3.1 Inpainting（修复/遮挡）

```python
if perturbation_function == "inpainting":
    for i in range(grid_size):
        for j in range(grid_size):
            if pert_mask[i, j] == 1:  # 需要扰动的单元
                h_start = i * cell_h
                h_end = min((i + 1) * cell_h, H)
                w_start = j * cell_w
                w_end = min((j + 1) * cell_w, W)
                perturbed_image[:, h_start:h_end, w_start:w_end] = 0.0
```

**效果**:
- 将被标记的网格单元**置零**（黑色遮挡）
- 模拟该区域信息被移除

**可视化**:
```
原始图像:         扰动后（掩码=1的单元被遮挡）:
[区域1][区域2]    [黑色][区域2]
[区域3][区域4]    [区域3][黑色]
```

##### 3.2 Blurring（模糊）

```python
elif perturbation_function == "blurring":
    for i in range(grid_size):
        for j in range(grid_size):
            if pert_mask[i, j] == 1:
                h_start = i * cell_h
                h_end = min((i + 1) * cell_h, H)
                w_start = j * cell_w
                w_end = min((j + 1) * cell_w, W)
                # 用该区域的均值替换（简化版模糊）
                mean_val = image[:, h_start:h_end, w_start:w_end].mean(
                    axis=(1, 2), keepdims=True
                )
                perturbed_image[:, h_start:h_end, w_start:w_end] = mean_val
```

**效果**:
- 将被标记的网格单元替换为**该区域的均值**
- 模拟该区域信息被模糊化

##### 3.3 Amplitude（幅度衰减）

```python
elif perturbation_function == "amplitude":
    for i in range(grid_size):
        for j in range(grid_size):
            if pert_mask[i, j] == 1:
                h_start = i * cell_h
                h_end = min((i + 1) * cell_h, H)
                w_start = j * cell_w
                w_end = min((j + 1) * cell_w, W)
                perturbed_image[:, h_start:h_end, w_start:w_end] *= 0.1
```

**效果**:
- 将被标记的网格单元的像素值**乘以0.1**（降低到10%）
- 模拟该区域信息被衰减

---

### 4. `_get_model_output_navgpt2()` - NavGPT2模型输出获取

#### 功能
获取给定图像的NavGPT2模型动作预测（二进制输出：匹配/不匹配）。

#### 执行流程

```python
# 步骤1: 图像格式转换
# get_vp_images返回的是[C, H, W]格式，BGR，且已减去均值
# 需要：1) 加回均值 2) 转换为RGB 3) 转换为uint8
if image.shape[0] == 3:  # C, H, W format
    # 加回BGR均值 [103.1, 115.9, 123.2]
    bgr_mean = np.array([103.1, 115.9, 123.2]).reshape(3, 1, 1)
    image_denorm = image + bgr_mean  # [C, H, W]
    # 转换为[H, W, C]
    image_pil = image_denorm.transpose(1, 2, 0)  # H, W, C
    # BGR转RGB
    image_pil = image_pil[:, :, ::-1]  # Reverse channel order
image_pil = image_pil.clip(0, 255).astype(np.uint8)

# 步骤2: 转换为PIL Image并应用NavGPT2的transform
img_pil = Image.fromarray(image_pil)
tensor_img = transform(img_pil)  # NavGPT2的transform（归一化到ImageNet标准）

# 步骤3: 使用NavGPT2的visual encoder提取特征
device = next(visual_encoder.parameters()).device
batch_tensor = tensor_img.unsqueeze(0).to(device)  # [1, 3, 224, 224]

with torch.no_grad():
    image_embeds = visual_encoder(batch_tensor)  # BLIP2 visual encoder
    image_embeds = ln_vision(image_embeds)  # Layer normalization
    if image_embeds.dim() == 3:
        features = image_embeds[:, 0, :]  # [1, feature_dim] - CLS token
    else:
        features = image_embeds

features_np = features.cpu().numpy()[0]  # [feature_dim]

# 步骤4: 替换obs中对应候选的特征
ob_copy = copy.deepcopy(ob)
candidates = ob_copy["candidate"]

# 找到vp_idx对应的候选索引
cand_list_idx = None
for j, cc in enumerate(candidates):
    if cc["pointId"] == vp_idx:
        cand_list_idx = j
        break

# 替换特征（保持NavGPT2的特征格式）
if cand_list_idx is not None:
    old_feature = candidates[cand_list_idx]["feature"]
    if isinstance(old_feature, np.ndarray):
        if len(old_feature.shape) == 2 and old_feature.shape[0] > 1:
            # NavGPT2特征格式: (257, 1024)，替换CLS token（第一行）
            new_feature = old_feature.copy()
            new_feature[0] = features_np
            candidates[cand_list_idx]["feature"] = new_feature
        elif len(old_feature.shape) == 1:
            # 1D特征，直接替换
            candidates[cand_list_idx]["feature"] = features_np.copy()

# 步骤5: 调用NavGPT2获取动作预测
with torch.no_grad():
    a_t, nav_vpids_list, nav_inputs_dict = NavGPT2_genAction_v2(
        target_agent,
        [ob_copy],  # 使用扰动后的obs
        [gmap],
        [instruction],
        t,
        ended=None,
        feedback="argmax",
    )

# 步骤6: 返回二进制输出
return 1.0 if a_t[0] == target_action else 0.0
```

**关键点**:
1. **图像格式处理**: 
   - `get_vp_images`返回BGR格式且已减去均值
   - 需要加回均值、转换为RGB、转换为uint8
   - 应用NavGPT2的transform（ImageNet归一化）

2. **特征提取**: 
   - 使用**NavGPT2的BLIP2 visual encoder**（不是ResNet）
   - 提取CLS token作为特征

3. **特征替换**: 
   - 替换`obs["candidate"][idx]["feature"]`中的特征
   - 保持NavGPT2的特征格式（通常是`(257, 1024)`，替换第一行CLS token）

4. **模型推理**: 
   - 直接调用`NavGPT2_genAction_v2`获取动作预测
   - **不使用VLN-BERT作为替代模型**

5. **二进制输出**: 
   - 返回1.0（动作匹配target_action）或0.0（不匹配）
   - 用于HSIC计算中的相关性分析

---

## 工作流程

### 完整流程图

```
开始
  │
  ├─> 获取全景图像 [bs, vp, C, H, W]
  │
  ├─> 提取特征并获取目标动作
  │
  ├─> 对每个候选视角 (vp_idx):
  │   │
  │   ├─> 获取单视角图像 [C, H, W]
  │   │
  │   ├─> 生成 nb_design 个扰动掩码
  │   │
  │   ├─> 对每个网格单元 (i, j):
  │   │   │
  │   │   ├─> 对每个扰动掩码:
  │   │   │   ├─> 应用扰动
  │   │   │   ├─> 提取特征
  │   │   │   └─> 获取模型输出
  │   │   │
  │   │   └─> 计算HSIC分数（相关性）
  │   │
  │   ├─> 上采样到 [H, W]
  │   │
  │   └─> 归一化并存储
  │
  └─> 返回显著性图 [V, H, W]
```

### 示例计算

假设 `grid_size=8`, `nb_design=500`:

1. **图像划分**: `224×224` → `8×8` 网格，每个单元 `28×28`

2. **生成掩码**: 500个随机二进制掩码

3. **计算单元(0,0)的HSIC**:
   ```
   掩码1: pert_mask[0,0]=1 → 输出=0.85
   掩码2: pert_mask[0,0]=0 → 输出=0.92
   掩码3: pert_mask[0,0]=1 → 输出=0.83
   ...
   掩码500: pert_mask[0,0]=0 → 输出=0.91
   
   cell_presence = [1, 0, 1, ..., 0]  # 500个值
   outputs = [0.85, 0.92, 0.83, ..., 0.91]  # 500个值
   
   标准化后计算相关性 → HSIC分数
   ```

4. **重复**: 对所有64个单元计算

5. **结果**: `[8, 8]` 的HSIC分数矩阵 → 上采样到 `[224, 224]`

---

## 关键参数说明

### `grid_size` (默认: 8)

- **含义**: 将图像划分为 `grid_size × grid_size` 个单元
- **影响**:
  - 值越大：更细粒度，但计算量更大
  - 值越小：更粗粒度，但计算更快
- **推荐**: 8（平衡精度和效率）

### `nb_design` (默认: 500)

- **含义**: 蒙特卡洛采样数量
- **影响**:
  - 值越大：HSIC估计更准确，但计算时间更长
  - 值越小：计算更快，但可能不够准确
- **推荐**: 500-1000

### `perturbation_function` (默认: "inpainting")

- **选项**:
  - `"inpainting"`: 遮挡（置零）- 最常用
  - `"blurring"`: 模糊（均值替换）
  - `"amplitude"`: 幅度衰减（×0.1）
- **选择**: 根据任务特性选择，inpainting通常效果最好

### `batch_size` (默认: 32)

- **含义**: 批处理大小（当前实现中未直接使用，预留）
- **用途**: 未来可优化为批量处理多个扰动图像

---

## 性能考虑

### 计算复杂度

- **时间复杂度**: `O(grid_size² × nb_design × 模型前向时间)`
- **示例**: `8×8 × 500 × ~10ms = ~3.2秒` 每个视角

### 优化建议

1. **减少 `nb_design`**: 如果速度优先，可降至200-300
2. **减少 `grid_size`**: 降至4或6，但会损失空间精度
3. **并行化**: 可并行处理多个扰动掩码（需要修改代码）

---

## 与其他方法的对比

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **IG** | 梯度积分 | 快速、可微 | 需要梯度，可能饱和 |
| **HSIC** | 依赖度量 | 黑盒、无梯度需求 | 计算量大 |
| **FG-CAM** | 特征图加权 | 快速 | 需要特征图访问 |
| **Random** | 随机 | 极快 | 无意义 |

---

## 使用示例

### 在配置文件中启用

```yaml
feature_level_baseline: "hsic"
hsic_grid_size: 8
hsic_nb_design: 500
hsic_perturbation: "inpainting"
```

### 在代码中调用（NavGPT2版本）

```python
# 在 FeatureAgent_NavGPT2.rollout_mask_test_navgpt2_feature_phase2() 中
images, attribution, candidata_list = (
    self.exp.compute_hsic_attribution_navgpt2(
        perm_obs,
        t,
        target_agent=self.target_agent,      # NavGPT2 agent
        gmaps=target_gmaps,                   # NavGPT2 graph maps
        instructions=target_instructions,      # Instructions
        grid_size=getattr(args, "hsic_grid_size", 8),
        nb_design=getattr(args, "hsic_nb_design", 500),
        perturbation_function=getattr(args, "hsic_perturbation", "inpainting"),
        batch_size=getattr(args, "hsic_batch_size", 32),
    )
)
```

**重要**: 
- 必须传入NavGPT2相关的参数（`target_agent`, `gmaps`, `instructions`）
- 该方法**直接在NavGPT2上计算**，不使用VLN-BERT

---

## 总结

HSIC Attribution Method (NavGPT2版本) 通过以下方式工作：

1. **网格化**: 将图像划分为多个单元（`grid_size × grid_size`）
2. **扰动采样**: 生成大量随机扰动组合（`nb_design`个二进制掩码）
3. **特征提取**: 使用NavGPT2的BLIP2 visual encoder提取扰动图像的特征
4. **模型推理**: 直接调用NavGPT2获取动作预测（不使用替代模型）
5. **依赖计算**: 计算每个单元与动作匹配度的依赖关系（HSIC分数）
6. **显著性生成**: 将依赖分数映射为显著性图 `[vp, H, W]`

**核心优势**: 
- **直接在NavGPT2上计算**，不使用替代模型，结果更准确
- 无需梯度信息（黑盒方法）
- 基于统计依赖，理论上更稳健
- 适用于任何可调用的模型

**与VLN-BERT版本的区别**:
- ✅ 使用NavGPT2的BLIP2 visual encoder（而非ResNet-152）
- ✅ 直接调用`NavGPT2_genAction_v2`（而非`do_forward`）
- ✅ 输出是二进制动作匹配度（而非概率值）
- ✅ 特征格式符合NavGPT2要求（`(257, 1024)`格式）

**适用场景**:
- 需要理解NavGPT2模型的决策依据
- 需要特征级别的解释（像素级显著性图）
- 模型不可微或梯度不可用
- 需要黑盒解释方法

