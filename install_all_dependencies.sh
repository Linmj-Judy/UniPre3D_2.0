#!/bin/bash
# 完整的依赖安装脚本
# 按照顺序执行以下命令

# ============================================
# 1. PyTorch 和相关核心库
# ============================================
pip install torch==2.2.2 torchvision==0.17.2

# ============================================
# 2. requirements.txt 中的基础依赖
# ============================================
pip install diffusers==0.33.1
pip install transformers==4.52.4
pip install scipy==1.15.3
pip install numpy==1.26.4
pip install pillow==11.2.1
pip install tqdm==4.67.1
pip install einops==0.8.1
pip install h5py==3.13.0
pip install matplotlib==3.10.3
pip install scikit-learn==1.6.1
pip install wandb==0.19.11
pip install hydra-core==1.3.2
pip install omegaconf==2.3.0

# ============================================
# 3. 修复 flash-attn 问题（重新编译安装）
# ============================================
pip uninstall flash-attn -y
pip install flash-attn --no-build-isolation

# ============================================
# 4. PEFT（如果缺失）
# ============================================
pip install peft

# ============================================
# 5. 其他可能需要的依赖
# ============================================
pip install ema-pytorch
pip install safetensors
pip install accelerate
pip install huggingface-hub

# ============================================
# 6. Mamba3D 相关（如果使用 Mamba3D 模型）
# ============================================
pip install causal-conv1d==1.2.2.post1 --no-build-isolation
pip install mamba-ssm==1.2.2 --no-build-isolation

# ============================================
# 7. 验证安装
# ============================================
echo "验证关键包的导入..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from diffusers import AutoencoderKL; print('diffusers 导入成功')"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"



