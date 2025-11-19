#!/bin/bash
# 实验运行脚本
# 用于快速运行不同配置的实验

set -e

# ============================================
# 物体级实验 (ShapeNet)
# ============================================

# Baseline: 原始方法
echo "Running Baseline (Object-level)..."
python train_network.py \
    --config-name=transformer_pretraining \
    opt.use_learnable_gating=false \
    opt.use_routing=false \
    opt.use_dual_forward=false \
    hydra.run.dir=experiments_out/object_baseline

# 仅门控机制
echo "Running Gating Only (Object-level)..."
python train_network.py \
    --config-name=transformer_pretraining \
    opt.use_learnable_gating=true \
    opt.use_routing=false \
    opt.use_dual_forward=false \
    opt.drop_path_rate=0.2 \
    opt.feature_dropout_rate=0.2 \
    hydra.run.dir=experiments_out/object_gating_only

# 门控 + 一致性损失
echo "Running Gating + Consistency (Object-level)..."
python train_network.py \
    --config-name=transformer_pretraining \
    opt.use_learnable_gating=true \
    opt.use_routing=false \
    opt.use_dual_forward=true \
    opt.drop_path_rate=0.2 \
    opt.feature_dropout_rate=0.2 \
    opt.lambda_consistency=0.1 \
    hydra.run.dir=experiments_out/object_full_improved

# ============================================
# 场景级实验 (ScanNet)
# ============================================

# Baseline: 原始方法
echo "Running Baseline (Scene-level)..."
python train_network.py \
    --config-name=ptv3_pretraining \
    opt.use_learnable_gating=false \
    opt.use_routing=false \
    opt.use_dual_forward=false \
    general.device=[0,1,2,3] \
    hydra.run.dir=experiments_out/scene_baseline

# 仅路由机制
echo "Running Routing Only (Scene-level)..."
python train_network.py \
    --config-name=ptv3_pretraining \
    opt.use_learnable_gating=false \
    opt.use_routing=true \
    opt.use_dual_forward=false \
    opt.lambda_sparse=0.01 \
    general.device=[0,1,2,3] \
    hydra.run.dir=experiments_out/scene_routing_only

# 完整改进
echo "Running Full Improvements (Scene-level)..."
python train_network.py \
    --config-name=ptv3_pretraining \
    opt.use_learnable_gating=true \
    opt.use_routing=true \
    opt.use_dual_forward=true \
    opt.drop_path_rate=0.3 \
    opt.feature_dropout_rate=0.2 \
    opt.lambda_sparse=0.01 \
    opt.lambda_consistency=0.1 \
    general.device=[0,1,2,3] \
    hydra.run.dir=experiments_out/scene_full_improved

echo "All experiments completed!"

