#!/bin/bash
#SBATCH -A a_lingwang
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH -J scene_full_improvement
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --time=4320000
#SBATCH --chdir=/public/home/lingwang/lmj/UniPre3D
#SBATCH --output=/public/home/lingwang/lmj/UniPre3D/logs/scene_level/full_improvement/%x_%j.out
#SBATCH --error=/public/home/lingwang/lmj/UniPre3D/logs/scene_level/full_improvement/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

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

