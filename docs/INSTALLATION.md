# Installation Guide 🔧

1. **Create and activate conda environment**
```bash
conda create -n UniPre3D python=3.11
conda activate UniPre3D
```

2. **Install PyTorch and dependencies**
```bash
# Install PyTorch with CUDA support
pip install torch==2.2.2 torchvision==0.17.2

# Install project dependencies
pip install -r requirements.txt
# Install new gcc
conda install -c conda-forge gcc_linux-64 gxx_linux-64

# Install flash-attn for efficient attention mechanisms
pip install flash-attn --no-build-isolation # show be compiled in GPU node\

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

3. **Install C++ extensions**
```bash
conda install ninja -y

# Install PointNet++ modules
cd openpoints/cpp/pointnet2_batch
python setup.py install
cd ../

# Install Chamfer Distance and emd modules
cd chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user
cd ../../../
```

4. **Install Mamba3D dependencies**
```bash
# Install PointNet2 operations library
# 在线安装（需联网）:
pip install --no-build-isolation "git+ssh://git@github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# 离线安装：
# 1. 先在有网络的电脑上克隆仓库
git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/pointnet2_ops_lib
# 2. 打包项目
python setup.py sdist
# 3. 将 dist/ 下的压缩包拷贝到目标机器
# 4. 在目标机器（离线环境）中安装
pip install dist/*.tar.gz --no-build-isolation

# Install GPU KNN
# 在线安装（需联网）:
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# 离线安装：
# 1. 先在有网络的电脑上下载 wheel 包
git clone git@github.com:unlimblue/KNN_CUDA.git
cd KNN_CUDA
pip install -r requirements.txt
# 或者你也可以直接scp文件到目标机器
# 2. 将下载好的文件拷贝到目标机器
# 3. 在目标机器（离线环境）中安装
pip install . --no-build-isolation


# Install Mamba SSM dependencies
pip install causal-conv1d==1.2.2.post1 --no-build-isolation
pip install mamba-ssm==1.2.2 --no-build-isolation
```

causal-conv1d and mamba-ssm are required for the Mamba3D model, you should select the version that matches your CUDA and pytorch version.

5. **Install Gaussian Splatting Renderer**

The Gaussian Splatting renderer is required for rendering Gaussian Point clouds to images.

```bash
# Clone the repository
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting

# Install the renderer
pip install submodules/diff-gaussian-rasterization --no-build-isolation # show be compiled in GPU node
```

```bash
pip install ema-pytorch
pip install shortuuid
pip install multimethod
pip install easydict
pip install timm
```

6. **Download pre-trained image feature extractor**

Please download the pre-trained image feature extractor `diffusion_pytorch_model.bin` from [here](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main) and put it in the `weights` folder.

### Troubleshooting

- If you encounter issues installing PointNet2 operations, please refer to [this solution](https://github.com/erikwijmans/Pointnet2_PyTorch/issues/174#issuecomment-2232300080) for manual installation steps.
- For Gaussian Splatting, ensure your system meets the [hardware requirements](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/README.md#hardware-requirements).