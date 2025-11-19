#!/bin/bash
#SBATCH -A a_lingwang
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH -J scene_original
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --time=4320000
#SBATCH --chdir=/public/home/lingwang/lmj/UniPre3D
#SBATCH --output=/public/home/lingwang/lmj/UniPre3D/logs/scene_level/original/%x_%j.out
#SBATCH --error=/public/home/lingwang/lmj/UniPre3D/logs/scene_level/original/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8


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

