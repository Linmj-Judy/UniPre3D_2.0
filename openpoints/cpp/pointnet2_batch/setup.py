from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 设置 conda 环境中的 GCC 路径
gcc_path = os.path.expanduser("/public/home/lingwang/anaconda3/envs/UniPre3D/bin/gcc")
gxx_path = os.path.expanduser("/public/home/lingwang/anaconda3/envs/UniPre3D/bin/g++")

if os.path.exists(gcc_path) and os.path.exists(gxx_path):
    os.environ['CC'] = gcc_path
    os.environ['CXX'] = gxx_path
    # 对于 nvcc，使用 -ccbin 标志指定编译器
    print(f"Using GCC: {gcc_path}")
    print(f"Using G++: {gxx_path}")
else:
    print("Warning: Conda GCC not found, using system default")

setup(
    name='pointnet2_cuda',
    ext_modules=[
        CUDAExtension('pointnet2_batch_cuda', [
        'src/pointnet2_api.cpp',
        'src/ball_query.cpp',
        'src/ball_query_gpu.cu',
        'src/group_points.cpp',
        'src/group_points_gpu.cu',
        'src/interpolate.cpp',
        'src/interpolate_gpu.cu',
        'src/sampling.cpp',
        'src/sampling_gpu.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2', 
                                    '--expt-extended-lambda',
                                    '--expt-relaxed-constexpr',
                                     '-gencode', 'arch=compute_80,code=sm_80']})
    ],
    cmdclass={'build_ext': BuildExtension}
)

