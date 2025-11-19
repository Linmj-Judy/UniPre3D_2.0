# UniPre3D 改进方案实现文档

## 概述

本文档描述了对UniPre3D模型的系统性改进，主要解决原论文Limitations中提到的"手工选择融合策略"问题。改进方案将固定的规则式融合策略转变为可学习的自适应策略。

## 主要改进

### 1. 可学习门控注意力机制 (Learnable Gating Mechanism)

**动机**: 原始方法在特征融合时简单拼接2D和3D特征，无法根据数据特性自适应调整权重。

**实现**: `fusion/gated_fusion.py::GatedFeatureFusion`

**核心思想**:
- 通过轻量级MLP根据全局统计特征动态生成2D和3D模态的门控权重
- 输入包括:
  - 3D特征的全局平均/最大池化 (捕获整体分布和显著性)
  - 2D特征的全局平均/最大池化
  - 尺度统计特征: log(N), 密度均值/方差/偏度
- 输出: `w_3d`, `w_2d` ∈ [0,1] (经Sigmoid激活)
- 融合公式: `F_fuse = w_3d * F_3d + w_2d * F_2d`

**数学表达**:
```
g_3d = [g_3d_avg; g_3d_max]  # 全局描述符
g_2d = [g_2d_avg; g_2d_max]
s = [log(N); density_mean; density_var; density_skew]  # 尺度统计

h = ReLU(W1 · [g_3d; g_2d; s] + b1)
w_3d = sigmoid(w_3d^T · h + b_3d)
w_2d = sigmoid(w_2d^T · h + b_2d)

F_gate = w_3d * F_3d + w_2d * F_2d
```

### 2. 路由选择机制 (Fusion Routing)

**动机**: 原始方法固定物体级用feature fusion、场景级用point fusion，缺乏灵活性。

**实现**: `fusion/gated_fusion.py::FusionRouter`

**核心思想**:
- 使用Gumbel-Softmax技巧实现可微的离散选择
- 三种路由选项:
  1. 仅3D分支 (无融合)
  2. 特征融合 (解码器后期融合)
  3. 点融合 (编码器早期融合)
- 温度退火: 从1.0线性降至0.1,使选择逐渐确定

**数学表达**:
```
g_e = GlobalAvgPool(P_3D)  # 编码器特征全局池化
z = W_r^(2) · ReLU(W_r^(1) · [g_e; s] + b_r^(1)) + b_r^(2)  # 路由logits

# Gumbel-Softmax
g = -log(-log(u)), u ~ Uniform(0,1)^3
z̃ = softmax((z + g) / τ)

F_route = z̃_1 * F_only + z̃_2 * F_feat + z̃_3 * F_point
```

**稀疏性正则化**:
```
L_sparse = λ_sparse · (1 - max_i(z̃_i))
```
鼓励路由做出确定性选择，避免平均策略。

### 3. 防止2D特征过度依赖

**动机**: 预训练2D模型特征信息丰富，可能导致3D主干过度依赖而自身学习不足。

**实现**: 

#### 3.1 DropPath (Stochastic Depth)
`fusion/gated_fusion.py::DropPath`

随机丢弃整个2D特征路径:
```
F_2d' = { 0,                  with prob p_drop
        { F_2d / (1-p_drop),  otherwise
```

#### 3.2 通道级Dropout
`fusion/gated_fusion.py::FeatureDropout`

随机屏蔽2D特征的某些通道:
```
F_2d^(c) ← F_2d^(c) · m^(c), m^(c) ~ Bernoulli(1-p_feat)
```

#### 3.3 特征一致性损失
`utils/loss_utils.py::feature_consistency_loss`

鼓励3D主干即使无2D辅助也能提取有效特征:
```
L_cons = ||E_3D^with(P) - stopgrad(E_3D^without(P))||_2^2
```

### 4. 总体损失函数

```
L_total = L_render + λ_lpips · L_lpips + λ_sparse · L_sparse + λ_cons · L_cons
```

其中:
- `L_render`: 像素级重建损失 (L1/L2/Focal-L2)
- `L_lpips`: 感知损失 (基于VGG特征)
- `L_sparse`: 路由稀疏性正则化
- `L_cons`: 特征一致性约束

## 配置使用

### 基础配置参数

在 `configs/settings.yaml` 中添加了以下配置项:

```yaml
opt:
  # 启用可学习门控机制
  use_learnable_gating: false
  
  # 启用路由机制
  use_routing: false
  
  # Gumbel-Softmax温度退火
  router_temp_start: 1.0
  router_temp_end: 0.1
  router_temp_anneal_iters: 50000
  
  # 正则化参数
  drop_path_rate: 0.2          # DropPath概率
  feature_dropout_rate: 0.2    # 通道Dropout概率
  
  # 损失权重
  lambda_sparse: 0.01          # 路由稀疏性系数
  lambda_consistency: 0.1      # 特征一致性系数
  
  # 双路前向 (用于一致性损失)
  use_dual_forward: false
```

### 实验配置示例

#### 物体级预训练 (ShapeNet)

使用 `configs/transformer_improved.yaml`:

```bash
python train_network.py --config-name=transformer_improved
```

关键设置:
- `use_learnable_gating: true`  # 启用门控
- `use_routing: false`          # 物体级不需要路由
- `drop_path_rate: 0.2`         # 适度正则化
- `use_dual_forward: true`      # 启用一致性损失

#### 场景级预训练 (ScanNet)

使用 `configs/ptv3_improved.yaml`:

```bash
python train_network.py --config-name=ptv3_improved general.device=[0,1,2,3]
```

关键设置:
- `use_learnable_gating: true`  # 启用门控
- `use_routing: true`           # 场景级启用路由
- `drop_path_rate: 0.3`         # 更强正则化
- `use_dual_forward: true`      # 启用一致性损失

### 消融实验配置

#### 1. 仅原始方法 (Baseline)
```yaml
opt:
  use_learnable_gating: false
  use_routing: false
  use_dual_forward: false
```

#### 2. 仅门控机制
```yaml
opt:
  use_learnable_gating: true
  use_routing: false
  use_dual_forward: false
```

#### 3. 仅路由机制 (场景级)
```yaml
opt:
  use_learnable_gating: false
  use_routing: true
  use_dual_forward: false
```

#### 4. 完整改进
```yaml
opt:
  use_learnable_gating: true
  use_routing: true
  use_dual_forward: true
```

## 评估指标

### 预训练阶段

评估脚本 `eval.py` 已支持以下指标:

1. **PSNR** (Peak Signal-to-Noise Ratio)
   - 衡量像素级重建精度
   - 越高越好

2. **SSIM** (Structural Similarity Index)
   - 评估图像结构保持程度
   - 范围 [0,1], 越高越好

3. **LPIPS** (Learned Perceptual Image Patch Similarity)
   - 基于深层神经网络特征的感知距离
   - 越低越好 (0表示完全相同)

### 下游任务

按照原论文设置,在以下数据集上评估:

#### 物体级任务
- **ScanObjectNN PB_T50_RS**: 分类准确率
- **ShapeNetPart**: 部件分割 mIoU

#### 场景级任务
- **ScanNetV2**: 语义分割 mIoU (20类)
- **ScanNet200**: 语义分割 mIoU (200类, 长尾分布)
- **S3DIS**: 语义分割 mIoU (跨数据集泛化)

## 预期性能提升

根据改进方案设计,预期在关键基准上获得以下提升:

| 数据集 | 指标 | 基线 | 改进 | 提升 |
|--------|------|------|------|------|
| ScanObjectNN PB_T50_RS | Acc | ~91% | - | +0.5~1.0% |
| ScanNet200 | mIoU | - | - | +0.8~1.5% |
| ScanNetV2 | mIoU | - | - | +0.3~0.5% |
| S3DIS | mIoU | - | - | +0.5~0.8% |

预训练质量:
- PSNR: +0.5~1.0 dB
- SSIM: +2~3%
- LPIPS: -5~10% (降低表示改善)

## 实现细节

### 文件结构

```
UniPre3D/
├── fusion/
│   ├── feat_fusion.py              # 原始特征融合
│   ├── point_fusion.py             # 原始点融合
│   ├── gated_fusion.py             # 新增: 门控、路由、正则化模块
│   └── adaptive_feat_fusion.py     # 新增: 自适应特征融合封装
├── configs/
│   ├── settings.yaml               # 更新: 添加改进参数
│   ├── transformer_improved.yaml   # 新增: 物体级改进配置
│   └── ptv3_improved.yaml         # 新增: 场景级改进配置
├── utils/
│   └── loss_utils.py              # 更新: 添加SSIM和新损失项
├── train_network.py               # 更新: 集成新模块和损失
└── IMPROVEMENTS.md                # 本文档
```

### 关键API

#### 1. GatedFeatureFusion
```python
from fusion.gated_fusion import GatedFeatureFusion

gating = GatedFeatureFusion(dim_3d=384, dim_2d=128, hidden_dim=128)
feat_gated, gate_weights = gating(feat_3d, feat_2d, center, point_cloud)
```

#### 2. FusionRouter
```python
from fusion.gated_fusion import FusionRouter

router = FusionRouter(feat_dim=384, hidden_dim=64)
router.set_temperature(0.5)  # 设置Gumbel温度
routing_weights, logits = router(feat_3d, coords, training=True)
```

#### 3. 损失计算
```python
from utils.loss_utils import (
    compute_total_loss,
    feature_consistency_loss,
    routing_sparsity_loss,
)

losses = compute_total_loss(
    rendered_images,
    gt_images,
    cfg,
    device,
    iteration,
    lpips_fn=lpips_fn,
    routing_weights=routing_weights,
    feat_with_2d=feat_with_2d,
    feat_without_2d=feat_without_2d,
)
```

## 训练流程

### 温度退火

路由器温度从 `router_temp_start` 线性退火至 `router_temp_end`:

```python
progress = min(iteration / router_temp_anneal_iters, 1.0)
current_temp = temp_start + (temp_end - temp_start) * progress
router.set_temperature(current_temp)
```

### 双路前向

当 `use_dual_forward=true` 时:
1. 正常前向传播 (包含2D特征) -> 获取 `feat_with_2d`
2. 屏蔽2D特征的前向传播 -> 获取 `feat_without_2d`
3. 计算一致性损失: `L_cons = ||feat_with_2d - stopgrad(feat_without_2d)||^2`

## 调试与监控

### 关键日志

训练时会记录以下额外信息:

```
Iteration 1000:
  l12_loss: 0.0123
  lpips_loss: 0.0456
  sparse_loss: 0.0078      # 路由稀疏性损失
  consistency_loss: 0.0234 # 特征一致性损失
  total_loss: 0.0891
  
  Gate weights: w_3d=0.67, w_2d=0.42
  Routing: [0.15, 0.72, 0.13]  # [only_3d, feat_fusion, point_fusion]
```

### 可视化建议

1. **门控权重变化**: 绘制训练过程中 `w_3d` 和 `w_2d` 的演变
2. **路由分布**: 可视化路由器对不同数据选择的融合策略
3. **温度退火曲线**: 监控Gumbel温度随训练的变化

## 故障排除

### 常见问题

1. **内存不足**
   - 解决: 双路前向会增加内存消耗,可以设置 `use_dual_forward: false`
   - 或降低批大小

2. **路由全部坍缩到一种策略**
   - 检查 `lambda_sparse` 是否过大
   - 尝试延长 `router_temp_anneal_iters`

3. **训练不稳定**
   - 降低 `drop_path_rate` 和 `feature_dropout_rate`
   - 减小 `lambda_consistency`

## 引用

如果使用本改进方案,请引用原始UniPre3D论文,并注明改进内容。

## 许可证

遵循原UniPre3D仓库的许可证。

