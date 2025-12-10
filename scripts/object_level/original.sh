#!/bin/bash
#SBATCH -A a_lingwang
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH -J object_original
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --time=4320000
#SBATCH --chdir=/public/home/lingwang/lmj/UniPre3D
#SBATCH --output=/public/home/lingwang/lmj/UniPre3D/logs/object_level/original/%x_%j.out
#SBATCH --error=/public/home/lingwang/lmj/UniPre3D/logs/object_level/original/%x_%j.err
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8


source ~/lmj/.bashrc_cu12
conda activate /public/home/lingwang/anaconda3/envs/UniPre3D
cd /public/home/lingwang/lmj/UniPre3D

export PYTHONPATH=/public/home/lingwang/lmj/UniPre3D:$PYTHONPATH

# 限制OpenBLAS线程数以避免资源耗尽
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

export TF_XLA_FLAGS=--tf_xla_enable_xla_devices
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/public/software/cuda/local/cuda-12.1
export CUDA_HOME=/public/software/cuda/local/cuda-12.1
export NVTE_CUDA_INCLUDE_PATH=/public/software/cuda/local/cuda-12.1/include  
export PATH=/public/software/cuda/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/public/software/cuda/local/cuda-12.1/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/cuda/local/cuda-12.1/lib64${LD_LIBRARY_PATH+:${LD_LIBRARY_PATH}}


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
