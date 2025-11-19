#!/usr/bin/env python3
"""
提取分割的 zip 文件的多个方案
"""
import os
import subprocess
import sys

def main():
    base_dir = '/public/home/lingwang/lmj/UniPre3D/Shapenet_multiview'
    os.chdir(base_dir)
    
    print("=" * 60)
    print("方案1: 使用 zip -F (单F) 修复并指定输出文件")
    print("=" * 60)
    print("命令:")
    print("  cat shapenet_dataset.z01 shapenet_dataset.z02 shapenet_dataset.z03 shapenet_dataset.z04 shapenet_dataset.z05 shapenet_dataset.z06 shapenet_dataset.zip > shapenet_dataset_merged.zip")
    print("  zip -F shapenet_dataset_merged.zip --out shapenet_dataset_fixed.zip")
    print("  unzip shapenet_dataset_fixed.zip")
    print()
    
    print("=" * 60)
    print("方案2: 使用 zip -FF (双F) 直接修复原文件")
    print("=" * 60)
    print("命令:")
    print("  cat shapenet_dataset.z01 shapenet_dataset.z02 shapenet_dataset.z03 shapenet_dataset.z04 shapenet_dataset.z05 shapenet_dataset.z06 shapenet_dataset.zip > shapenet_dataset_merged.zip")
    print("  zip -FF shapenet_dataset_merged.zip")
    print("  unzip shapenet_dataset_merged.zip")
    print()
    
    print("=" * 60)
    print("方案3: 直接解压分割文件（如果 unzip 支持多磁盘）")
    print("=" * 60)
    print("命令:")
    print("  unzip shapenet_dataset.zip")
    print("注意: 需要确保所有 .z01-.z06 文件在同一目录")
    print()
    
    print("=" * 60)
    print("方案4: 使用 7z 工具（如果可用）")
    print("=" * 60)
    print("命令:")
    print("  /public/software/ads2024/bin/7za x shapenet_dataset.zip")
    print()
    
    print("=" * 60)
    print("方案5: 手动修复中央目录位置")
    print("=" * 60)
    print("如果以上都不行，可能需要:")
    print("1. 检查合并后的文件大小是否正确（应该是所有部分的总和）")
    print("2. 使用 hexdump 检查文件末尾是否有 PK 05 06 标记（中央目录结束标记）")
    print("3. 可能需要手动调整 zip 文件结构")
    print()
    
    print("=" * 60)
    print("诊断命令:")
    print("=" * 60)
    print("# 检查文件大小")
    print("ls -lh shapenet_dataset.*")
    print()
    print("# 检查合并后文件末尾")
    print("hexdump -C shapenet_dataset_merged.zip | tail -20")
    print()
    print("# 测试 zip 文件")
    print("unzip -t shapenet_dataset_merged.zip")

if __name__ == '__main__':
    main()



