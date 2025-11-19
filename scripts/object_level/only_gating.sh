#!/bin/bash
#SBATCH -A a_lingwang
#SBATCH --partition=gpuA800
#SBATCH --qos=normal
#SBATCH -J object_only_gating
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --time=4320000
#SBATCH --chdir=/public/home/lingwang/lmj/UniPre3D
#SBATCH --output=/public/home/lingwang/lmj/UniPre3D/logs/object_level/only_gating/%x_%j.out
#SBATCH --error=/public/home/lingwang/lmj/UniPre3D/logs/object_level/only_gating/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8


source ~/lmj/.bashrc_cu12
conda activate /public/home/lingwang/anaconda3/envs/UniPre3D
cd /public/home/lingwang/lmj/UniPre3D

export TF_XLA_FLAGS=--tf_xla_enable_xla_devices
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/public/software/cuda/local/cuda-12.1
export CUDA_HOME=/public/software/cuda/local/cuda-12.1
export NVTE_CUDA_INCLUDE_PATH=/public/software/cuda/local/cuda-12.1/include  
export PATH=/public/software/cuda/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/public/software/cuda/local/cuda-12.1/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/cuda/local/cuda-12.1/lib64${LD_LIBRARY_PATH+:${LD_LIBRARY_PATH}}


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
