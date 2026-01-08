# 测试脚本说明

这些测试脚本用于验证在连续sim模式（非离散模式）下，`env.py` 和 `agent.py` 中关于 action 和 candidate 的修改是否正确。

## 测试脚本

### 1. `test_continuous_vs_discrete.py`
**主要测试脚本** - 比较离散模式（12 views, 30°）和连续模式（8 views, 45°）的行为

**运行方式：**
```bash
python scripts/test_continuous_vs_discrete.py
```

**测试内容：**
- UP/DOWN 动作的一致性
- RIGHT 旋转动作的正确性
- pointId 计算的准确性
- 导航逻辑的一致性

### 2. `test_continuous_sim.py`
基础测试脚本，测试基本的 makeAction 和 pointId 计算逻辑

**运行方式：**
```bash
python scripts/test_continuous_sim.py
```

### 3. `test_env_agent_continuous.py`
测试 env 和 agent 模块的修改

**运行方式：**
```bash
# 测试离散模式
python scripts/test_env_agent_continuous.py --panoramic_horizontal_views 12

# 测试连续模式
python scripts/test_env_agent_continuous.py --panoramic_horizontal_views 8

# 比较两种模式
python scripts/test_env_agent_continuous.py --compare
```

### 4. `test_env_agent_integration.py`
集成测试脚本，直接测试 env 和 agent 模块的函数

**运行方式：**
```bash
# 测试离散模式
python scripts/test_env_agent_integration.py --mode discrete

# 测试连续模式
python scripts/test_env_agent_integration.py --mode continuous

# 比较两种模式
python scripts/test_env_agent_integration.py --mode compare
```

## 测试重点

### 1. makeAction 行为
- **离散模式（12 views）**：使用 `makeAction([0], [1.0], [0])` 进行30度旋转
- **连续模式（8 views）**：使用 `makeAction([0], [angle_increment_rad], [0])` 进行45度旋转

### 2. pointId 计算
- **离散模式**：直接使用 MatterSim 的 `viewIndex`
- **连续模式**：从 `heading` 和 `elevation` 计算 `pointId`

### 3. base_heading 计算
- 两种模式都使用：`base_heading = (viewId % num_horizontal_views) * angle_increment_rad`

### 4. 相对 heading 计算
- 两种模式都使用：`relative_heading = (state.heading - base_heading) % (2 * math.pi)`

## 验证要点

1. **UP 动作**：两种模式都应该产生30度的 elevation 变化
2. **RIGHT 动作**：
   - 离散模式：30度旋转
   - 连续模式：45度旋转（对于8 views）
3. **pointId 计算**：连续模式下从 heading/elevation 计算的 pointId 应该与预期的 ix 匹配
4. **导航一致性**：从 pointId A 到 pointId B 的导航路径应该正确

## 预期结果

- ✓ UP/DOWN 动作在两种模式下都产生30度的 elevation 变化
- ✓ RIGHT 动作产生正确的 heading 增量（离散：30°，连续：45°）
- ✓ pointId 计算在连续模式下准确匹配预期值
- ✓ 导航逻辑在两种模式下都正确工作

## 注意事项

1. 确保 MatterSim 已正确安装
2. 测试使用的 scanId 和 viewpointId 需要存在于数据集中
3. 如果某些测试失败，检查：
   - `env.py` 中的 `make_candidate` 函数
   - `agent.py` 中的 `make_equiv_action` 函数
   - `get_current_point_id` 函数的计算逻辑

