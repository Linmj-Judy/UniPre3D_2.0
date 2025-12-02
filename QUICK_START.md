# 快速开始指南

## 🚀 立即运行

### 基线实验(原始模型)

```bash
# 物体级 (ShapeNet)
bash scripts/object_level/original.sh

# 场景级 (ScanNet)  
bash scripts/scene_level/original.sh
```

### 改进实验

```bash
# 物体级 - 仅门控
bash scripts/object_level/only_gating.sh

# 物体级 - 完整改进
bash scripts/object_level/gating_and_consistency_loss.sh

# 场景级 - 仅路由
bash scripts/scene_level/only_routing.sh

# 场景级 - 完整改进
bash scripts/scene_level/full_improvement.sh
```

## ⚙️ 配置参数速查

### 启用/禁用改进功能

```yaml
opt:
  # 基础融合
  use_fusion: true                 # 是否使用2D-3D融合
  
  # 可学习门控
  use_learnable_gating: true       # 启用门控机制
  
  # 路由选择
  use_routing: true                # 启用路由机制(主要用于场景级)
  router_temp_start: 1.0           # Gumbel温度起始值
  router_temp_end: 0.1             # Gumbel温度终止值
  router_temp_anneal_iters: 50000  # 温度退火迭代数
  
  # 正则化
  drop_path_rate: 0.2              # DropPath概率 (物体级: 0.2, 场景级: 0.3)
  feature_dropout_rate: 0.2        # 通道Dropout概率
  
  # 损失权重
  lambda_sparse: 0.01              # 路由稀疏性系数
  lambda_consistency: 0.1          # 特征一致性系数
  
  # 双路前向
  use_dual_forward: true           # 启用一致性损失(需要更多内存)
```

### 物体级推荐配置

```yaml
# 完整改进
use_learnable_gating: true
use_routing: false               # 物体级不需要路由
use_dual_forward: true
drop_path_rate: 0.2
feature_dropout_rate: 0.2
lambda_consistency: 0.1
```

### 场景级推荐配置

```yaml
# 完整改进
use_learnable_gating: true
use_routing: true                # 场景级启用路由
use_dual_forward: true
drop_path_rate: 0.3              # 场景级用更强正则化
feature_dropout_rate: 0.2
lambda_sparse: 0.01
lambda_consistency: 0.1
```

## 📊 监控训练

### 查看日志

```bash
# 训练日志
tail -f logs/training_log.txt

# 验证日志
tail -f logs/validation_log.txt

# SLURM输出
tail -f logs/object_level/only_gating/object_only_gating_*.out
```

### 查看视频

```bash
# 生成的测试视频保存在
ls videos/
```

### W&B同步(如果有网络)

```bash
# 同步离线日志到W&B
wandb sync experiments_out/object_gating_only/wandb/
```

## 🐛 常见问题

### 1. OpenBLAS线程错误

**症状**: `pthread_create failed`

**解决**: 已在脚本中添加线程限制,无需额外操作

### 2. W&B连接挂起

**症状**: `wandb: Network error, entering retry loop`

**解决**: 已自动启用离线模式,无需额外操作

### 3. 内存不足

**解决**:
```yaml
# 方案1: 减小批大小
opt:
  batch_size: 16

# 方案2: 禁用双路前向
opt:
  use_dual_forward: false
```

### 4. CUDA Out of Memory

**解决**:
```yaml
# 减少输入图像数量
data:
  input_images: 1  # 从8降到1

# 或减小分辨率
data:
  training_resolution: 64  # 从128降到64
```

## 📈 评估结果

### 预训练质量

运行训练后查看:
- PSNR: 越高越好 (目标: +0.5~1.0 dB)
- SSIM: 越高越好 (目标: +2~3%)
- LPIPS: 越低越好 (目标: -5~10%)

### 下游任务

参考`IMPROVEMENTS.md`中的评估章节进行下游微调

## 🔍 检查改进是否生效

### 查看训练日志中的新损失项

```bash
grep "Sparse loss" logs/training_log.txt
grep "Consistency loss" logs/training_log.txt
```

应该能看到类似输出:
```
@ Iteration 1000:  Training log10 loss: -2.1234  L12 log10 loss: -2.3456  Sparse loss: 0.0078  Consistency loss: 0.0234
```

### 检查门控权重

在训练过程中,模型会自动学习门控权重。可以通过添加打印语句查看`w_3d`和`w_2d`的值。

## 📂 输出目录结构

```
experiments_out/
├── object_baseline/          # 基线实验
│   ├── wandb/               # W&B日志(离线)
│   ├── logs/                # 文本日志
│   ├── videos/              # 测试视频
│   └── model_best.pth       # 最佳模型
├── object_gating_only/      # 仅门控实验
└── object_full_improved/    # 完整改进实验
```

## 🎯 快速验证

### 1. 确认环境

```bash
conda activate UniPre3D
python -c "import torch; print(torch.cuda.is_available())"  # 应输出True
```

### 2. 快速测试

```bash
# 运行10个迭代测试配置
python train_network.py \
    --config-name=transformer_improved \
    opt.iterations=10 \
    opt.use_learnable_gating=true
```

### 3. 检查输出

确认能看到:
- ✅ "No network connection detected. Running in OFFLINE mode."
- ✅ 训练损失打印
- ✅ 无OpenBLAS错误
- ✅ 无W&B重试循环

## 📚 更多信息

- **详细文档**: `IMPROVEMENTS.md`
- **问题修复**: `FIXES_AND_IMPROVEMENTS_SUMMARY.md`
- **原始README**: `README.md`

## 🤝 贡献

如发现问题或有改进建议,欢迎提交Issue或PR。

---

**最后更新**: 2025-11-19  
**版本**: v1.0


