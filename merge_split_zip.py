#!/usr/bin/env python3
"""
合并分割的 zip 文件
"""
import os
import sys

def merge_split_zip(output_file, parts):
    """合并分割的 zip 文件部分"""
    print(f"开始合并 {len(parts)} 个文件...")
    with open(output_file, 'wb') as outfile:
        for i, part in enumerate(parts, 1):
            print(f"正在合并第 {i}/{len(parts)} 部分: {part}")
            with open(part, 'rb') as infile:
                outfile.write(infile.read())
    print(f"合并完成: {output_file}")
    print(f"文件大小: {os.path.getsize(output_file) / (1024**3):.2f} GB")

if __name__ == '__main__':
    base_dir = '/public/home/lingwang/lmj/UniPre3D/Shapenet_multiview'
    os.chdir(base_dir)
    
    # 分割文件的正确顺序：z01, z02, ..., z06, zip
    parts = [
        'shapenet_dataset.z01',
        'shapenet_dataset.z02',
        'shapenet_dataset.z03',
        'shapenet_dataset.z04',
        'shapenet_dataset.z05',
        'shapenet_dataset.z06',
        'shapenet_dataset.zip'
    ]
    
    # 检查所有文件是否存在
    missing = [p for p in parts if not os.path.exists(p)]
    if missing:
        print(f"错误: 以下文件不存在: {missing}")
        sys.exit(1)
    
    output_file = 'shapenet_dataset_merged.zip'
    merge_split_zip(output_file, parts)
    
    print("\n现在可以使用以下命令解压:")
    print(f"  unzip {output_file}")



