# Action 和 Candidate 逻辑详解

本文档详细解释 `env.py` 和 `agent.py` 中关于 action 和 candidate 的核心逻辑，特别是离散模式（12 views, 30°）和连续模式（8 views, 45°）的区别。

## 目录

1. [核心概念](#核心概念)
2. [env.py: make_candidate 函数](#envpy-make_candidate-函数)
3. [env.py: _get_obs 函数](#envpy-_get_obs-函数)
4. [agent.py: make_equiv_action 函数](#agentpy-make_equiv_action-函数)
5. [关键设计决策](#关键设计决策)
6. [一致性保证](#一致性保证)

---

## 核心概念

### 1. MatterSim 的两种模式

**离散模式（Discretized Mode）**：
- `panoramic_horizontal_views == 12`
- MatterSim 使用固定的 30° 增量
- `viewIndex` 直接对应 30° 的倍数（0-35）
- `viewIndex` 可以直接用于导航

**连续模式（Non-discretized Mode）**：
- `panoramic_horizontal_views != 12`（例如 8）
- MatterSim 使用自定义角度增量（例如 45°）
- `viewIndex` 仍然基于 30° 增量，**不能直接使用**
- 需要从 `heading` 和 `elevation` 计算自定义的 `pointId`

### 2. 关键术语

- **viewIndex**: MatterSim 的内部索引，基于 30° 增量（0-35）
- **pointId**: 自定义索引，基于 `panoramic_horizontal_views`（例如 8 views: 0-23）
- **base_heading**: 基准 heading，用于计算相对 heading
- **relative_heading**: 相对于 base_heading 的 heading

---

## env.py: make_candidate 函数

### 功能
生成从当前 viewpoint 可导航的所有候选视图点（candidates），每个 candidate 包含：
- `pointId`: 视图点的索引
- `heading`: 相对 heading（相对于 base_heading）
- `elevation`: elevation 值
- `viewpointId`: 目标 viewpoint 的 ID

### 关键逻辑

#### 1. base_heading 的计算（第 323-335 行）

```python
if args.panoramic_horizontal_views == 12:
    base_heading = (viewId % num_horizontal_views) * angle_increment_rad
else:
    # For non-discretized mode, viewId is the pointId
    horiz_idx = viewId % num_horizontal_views
    base_heading = horiz_idx * angle_increment_rad
```

**设计依据**：
- `viewId` 是调用 `make_candidate` 时传入的当前视图 ID
- 在离散模式下，`viewId` 是 MatterSim 的 `viewIndex`（0-35）
- 在连续模式下，`viewId` 是从 `_get_obs` 计算的 `pointId`（0-23 for 8 views）
- `base_heading` 表示当前视图的水平方向偏移量

**示例**：
- 离散模式，`viewId=5`：`base_heading = 5 * 30° = 150°`
- 连续模式（8 views），`viewId=5`：`base_heading = 5 * 45° = 225°`

#### 2. 遍历所有视图点（第 343-362 行）

```python
for ix in range(num_total_views):  # num_total_views = 3 * num_horizontal_views
    if ix == 0:
        sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
    elif ix % num_horizontal_views == 0:
        # Move up one elevation level
        if use_discretized:
            sim.makeAction([0], [1.0], [1.0])  # Rotate + up
        else:
            sim.makeAction([0], [0.0], [math.radians(30)])  # Pure up
    else:
        # Rotate horizontally
        if use_discretized:
            sim.makeAction([0], [1.0], [0])  # 30° rotation
        else:
            sim.makeAction([0], [angle_increment_rad], [0])  # 45° rotation
```

**设计依据**：
- `ix` 从 0 开始，遍历所有可能的视图点
- `ix % num_horizontal_views == 0` 表示移动到下一个 elevation level
- 离散模式：`[1.0]` 表示 30° 增量
- 连续模式：使用弧度值（例如 `math.radians(45)`）

**视图点布局**：
```
Level 0 (elevation ≈ -30°): ix = 0, 1, 2, ..., num_horizontal_views-1
Level 1 (elevation ≈ 0°):   ix = num_horizontal_views, ..., 2*num_horizontal_views-1
Level 2 (elevation ≈ 30°):  ix = 2*num_horizontal_views, ..., 3*num_horizontal_views-1
```

#### 3. 相对 heading 的计算（第 375 行）

```python
heading = (state.heading - base_heading) % (2 * math.pi)
```

**设计依据**：
- MatterSim 的 `state.heading` 是绝对 heading（相对于初始方向）
- 减去 `base_heading` 得到相对 heading（相对于当前视图的水平方向）
- 归一化到 `[0, 2π)` 范围

**为什么需要相对 heading？**：
- 候选视图点的特征需要相对于当前视图的角度
- 这样模型可以学习"向右转 45°"而不是"转到 225°"

#### 4. pointId 的存储（第 400 行）

```python
"pointId": ix,
```

**设计依据**：
- `pointId` 直接等于循环索引 `ix`
- 这确保了 `pointId` 与视图点的布局一致
- `pointId` 用于后续的导航逻辑

---

## env.py: _get_obs 函数

### 功能
获取当前环境的观察（observation），包括：
- 当前视图的特征
- 可导航的候选视图点列表
- 当前视图的 `viewId`

### 关键逻辑（第 447-475 行）

#### 离散模式（第 447-449 行）

```python
if args.panoramic_horizontal_views == 12:
    base_view_id = state.viewIndex
    viewId = state.viewIndex
```

**设计依据**：
- MatterSim 的 `viewIndex` 直接对应 30° 增量
- 可以直接使用 `viewIndex` 作为 `viewId`

#### 连续模式（第 450-475 行）

```python
else:
    # Calculate pointId from current state
    num_horizontal_views = args.panoramic_horizontal_views
    angle_increment_rad = math.radians(360.0 / num_horizontal_views)
    
    # Determine elevation level
    if state.elevation < -0.2:
        elev_level = 0
    elif state.elevation > 0.2:
        elev_level = 2
    else:
        elev_level = 1
    
    # Calculate horizontal index from heading
    heading_normalized = state.heading % (2 * math.pi)
    horiz_idx = (
        int(round(heading_normalized / angle_increment_rad))
        % num_horizontal_views
    )
    
    # Calculate pointId
    point_id = elev_level * num_horizontal_views + horiz_idx
    base_view_id = point_id
    viewId = point_id
```

**设计依据**：
1. **Elevation Level 判断**：
   - `-0.2` 和 `0.2` 是阈值，用于区分三个 elevation level
   - 初始 elevation = -30° ≈ -0.524 rad
   - 第一个 level: elevation < -0.2
   - 第二个 level: -0.2 ≤ elevation ≤ 0.2
   - 第三个 level: elevation > 0.2

2. **Horizontal Index 计算**：
   - 从绝对 heading 计算水平索引
   - `heading_normalized / angle_increment_rad` 得到索引
   - 取模确保在 `[0, num_horizontal_views)` 范围内

3. **pointId 计算**：
   - `point_id = elev_level * num_horizontal_views + horiz_idx`
   - 这与 `make_candidate` 中的 `ix` 计算方式一致

**为什么需要这个计算？**：
- MatterSim 的 `viewIndex` 在连续模式下不准确（仍然基于 30°）
- 需要从实际的 `heading` 和 `elevation` 计算正确的 `pointId`
- 这确保了 `viewId` 与 `make_candidate` 中生成的 `pointId` 一致

---

## agent.py: make_equiv_action 函数

### 功能
将全景视图的动作（选择 candidate）转换为 MatterSim 的自我中心视图动作序列：
1. UP/DOWN：调整 elevation level
2. LEFT/RIGHT：旋转到正确的水平方向
3. FORWARD：移动到目标 viewpoint

### 关键逻辑

#### 1. take_action 函数（第 413-450 行）

```python
def take_action(i, idx, name):
    if type(name) is int:  # Go to the next view
        self.env.env.sims[idx].makeAction([name], [0], [0])
    else:  # Adjust
        use_discretized = args.panoramic_horizontal_views == 12
        if use_discretized:
            self.env.env.sims[idx].makeAction(*self.env_actions[name])
        else:
            angle_increment_rad = math.radians(360.0 / args.panoramic_horizontal_views)
            if name == "up":
                self.env.env.sims[idx].makeAction([0], [0.0], [math.radians(30)])
            elif name == "down":
                self.env.env.sims[idx].makeAction([0], [0.0], [math.radians(-30)])
            elif name == "right":
                self.env.env.sims[idx].makeAction([0], [angle_increment_rad], [0])
            elif name == "left":
                self.env.env.sims[idx].makeAction([0], [-angle_increment_rad], [0])
```

**设计依据**：
- **离散模式**：使用预定义的 `env_actions`（例如 `[0], [0], [1]` 表示 up）
- **连续模式**：使用弧度值
  - UP/DOWN：固定 30°（`math.radians(30)`）
  - LEFT/RIGHT：使用 `angle_increment_rad`（例如 45°）

**为什么 UP/DOWN 固定为 30°？**：
- MatterSim 的 elevation 变化始终是 30° 的倍数
- 无论水平视图数量如何，elevation 的增量都是 30°

#### 2. Elevation Level 调整（第 462-471 行）

```python
src_level = src_point // num_horizontal_views
trg_level = trg_point // num_horizontal_views
while src_level < trg_level:  # Tune up
    take_action(i, idx, "up")
    src_level += 1
while src_level > trg_level:  # Tune down
    take_action(i, idx, "down")
    src_level -= 1
```

**设计依据**：
- `pointId // num_horizontal_views` 得到 elevation level（0, 1, 2）
- 先调整 elevation，再调整水平方向
- 这确保了导航的正确顺序

#### 3. 水平方向调整：离散模式（第 475-494 行）

```python
if args.panoramic_horizontal_views == 12:
    while self.env.env.sims[idx].getState()[0].viewIndex != trg_point:
        take_action(i, idx, "right")
        # ... safety checks ...
```

**设计依据**：
- 直接比较 `viewIndex` 和 `trg_point`
- 因为 `viewIndex` 和 `pointId` 在离散模式下一致

#### 4. 水平方向调整：连续模式（第 495-567 行）

```python
else:
    def get_current_point_id(state, num_horizontal_views, base_view_index):
        # Calculate base_heading
        angle_increment_rad = math.radians(360.0 / num_horizontal_views)
        base_heading = (base_view_index % num_horizontal_views) * angle_increment_rad
        
        # Calculate relative heading
        relative_heading = (state.heading - base_heading) % (2 * math.pi)
        
        # Determine elevation level
        if elevation < -0.2:
            elev_level = 0
        elif elevation > 0.2:
            elev_level = 2
        else:
            elev_level = 1
        
        # Calculate horizontal index
        horiz_idx = int(round(relative_heading / angle_increment_rad)) % num_horizontal_views
        
        # Calculate pointId
        point_id = elev_level * num_horizontal_views + horiz_idx
        return point_id
    
    base_view_index = perm_obs[i]["viewIndex"]
    current_state = self.env.env.sims[idx].getState()[0]
    current_point_id = get_current_point_id(current_state, num_horizontal_views, base_view_index)
    
    while current_point_id != trg_point:
        take_action(i, idx, "right")
        new_state = self.env.env.sims[idx].getState()[0]
        new_point_id = get_current_point_id(new_state, num_horizontal_views, base_view_index)
        # ... safety checks ...
```

**设计依据**：

1. **get_current_point_id 函数**：
   - 从当前 `state` 计算 `pointId`
   - 使用 `base_view_index` 计算 `base_heading`
   - 计算相对 heading，然后计算 `pointId`
   - **这与 `env.py` 中的逻辑完全一致**

2. **base_view_index 的来源**：
   - `base_view_index = perm_obs[i]["viewIndex"]`
   - 这是初始的 `viewId`（从 `_get_obs` 获取）
   - 用于计算 `base_heading`

3. **为什么需要 base_view_index？**：
   - MatterSim 的 `state.heading` 是绝对 heading
   - 需要知道初始方向才能计算相对 heading
   - `base_view_index` 提供了这个初始方向信息

4. **循环条件**：
   - 比较 `current_point_id` 和 `trg_point`
   - 而不是比较 `viewIndex`（因为 `viewIndex` 在连续模式下不准确）

---

## 关键设计决策

### 1. 为什么需要相对 heading？

**问题**：为什么 `make_candidate` 中存储的是相对 heading 而不是绝对 heading？

**答案**：
- 模型需要学习"相对于当前视图的角度"
- 例如："向右转 45°" 比 "转到 225°" 更容易学习
- 相对 heading 使得特征具有旋转不变性

### 2. 为什么 base_heading 的计算方式不同？

**离散模式**：
```python
base_heading = (viewId % num_horizontal_views) * angle_increment_rad
```

**连续模式**：
```python
horiz_idx = viewId % num_horizontal_views
base_heading = horiz_idx * angle_increment_rad
```

**答案**：
- 实际上两种模式的计算方式相同
- `viewId % num_horizontal_views` 得到水平索引
- 乘以 `angle_increment_rad` 得到 `base_heading`
- 区别在于 `viewId` 的含义：
  - 离散模式：`viewId` 是 MatterSim 的 `viewIndex`
  - 连续模式：`viewId` 是计算的 `pointId`

### 3. 为什么 elevation level 的判断使用 -0.2 和 0.2？

**答案**：
- 初始 elevation = -30° ≈ -0.524 rad
- 第一个 level: elevation < -0.2（约 -11.5°）
- 第二个 level: -0.2 ≤ elevation ≤ 0.2（约 -11.5° 到 11.5°）
- 第三个 level: elevation > 0.2（约 > 11.5°）
- 这些阈值提供了足够的容差，避免浮点数精度问题

### 4. 为什么在连续模式下不能直接使用 viewIndex？

**答案**：
- MatterSim 的 `viewIndex` 始终基于 30° 增量
- 即使设置了 `setDiscretizedViewingAngles(False)`，`viewIndex` 仍然基于 30°
- 例如：在 8 views（45°）模式下，`viewIndex` 可能仍然是 0, 1, 2, ...（30° 增量）
- 因此需要从 `heading` 和 `elevation` 计算正确的 `pointId`

---

## 一致性保证

### 1. pointId 的一致性

**env.py make_candidate**：
```python
"pointId": ix,  # ix 是循环索引
```

**env.py _get_obs**：
```python
point_id = elev_level * num_horizontal_views + horiz_idx
```

**agent.py get_current_point_id**：
```python
point_id = elev_level * num_horizontal_views + horiz_idx
```

**保证**：
- 三种计算方式都使用相同的公式
- `elev_level` 和 `horiz_idx` 的计算方式一致
- 确保了 `pointId` 的一致性

### 2. base_heading 的一致性

**env.py make_candidate**：
```python
base_heading = (viewId % num_horizontal_views) * angle_increment_rad
```

**agent.py get_current_point_id**：
```python
base_heading = (base_view_index % num_horizontal_views) * angle_increment_rad
```

**保证**：
- 两种计算方式相同
- `viewId` 和 `base_view_index` 都来自 `_get_obs`
- 确保了 `base_heading` 的一致性

### 3. 相对 heading 的一致性

**env.py make_candidate**：
```python
heading = (state.heading - base_heading) % (2 * math.pi)
```

**agent.py get_current_point_id**：
```python
relative_heading = (state.heading - base_heading) % (2 * math.pi)
```

**保证**：
- 两种计算方式相同
- 都使用相同的归一化方式
- 确保了相对 heading 的一致性

---

## 总结

### 核心原则

1. **离散模式（12 views）**：
   - 直接使用 MatterSim 的 `viewIndex`
   - 简单直接，性能最优

2. **连续模式（8 views 等）**：
   - 从 `heading` 和 `elevation` 计算 `pointId`
   - 使用相对 heading 进行计算
   - 确保与 `make_candidate` 的一致性

### 关键一致性点

1. **pointId 计算**：`elev_level * num_horizontal_views + horiz_idx`
2. **base_heading 计算**：`(viewId % num_horizontal_views) * angle_increment_rad`
3. **相对 heading 计算**：`(state.heading - base_heading) % (2 * math.pi)`
4. **elevation level 判断**：使用 `-0.2` 和 `0.2` 作为阈值

### 测试验证

所有逻辑都通过了 `test_continuous_vs_discrete.py` 的测试：
- ✓ Elevation 变化正确（30°）
- ✓ Heading 变化正确（离散：30°，连续：45°）
- ✓ pointId 计算正确
- ✓ 导航逻辑一致

