#!/usr/bin/env python3
"""
Debug script to trace where the segfault occurs
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import torch
import traceback
from omegaconf import OmegaConf

print("=" * 80)
print("DEBUG: Testing Dataset Loading")
print("=" * 80)

try:
    print("\n1. Loading configuration...")
    # Use hydra to properly load configs
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    
    config_dir = str(Path.cwd() / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="transformer_pretraining")
    print("✓ Configuration loaded")
    print(f"   Dataset root: {cfg.data.dataset_root}")
    
    print("\n2. Importing dataset class...")
    from dataset.shapenet import ShapeNetDataset
    print("✓ Dataset class imported")
    
    print("\n3. Creating train dataset (without DataLoader)...")
    train_dataset = ShapeNetDataset(cfg, 'train')
    print(f"✓ Train dataset created: {len(train_dataset)} samples")
    
    print("\n4. Testing single sample access...")
    print("   Accessing index 0...")
    try:
        sample = train_dataset[0]
        print(f"✓ Sample 0 loaded successfully")
        print(f"   Keys: {list(sample.keys())}")
        if 'point_cloud' in sample:
            pc = sample['point_cloud']
            print(f"   Point cloud pos shape: {pc['pos'].shape}, device: {pc['pos'].device}")
    except Exception as e:
        print(f"✗ Failed to load sample 0: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\n5. Testing multiple samples...")
    for i in range(min(3, len(train_dataset))):
        try:
            sample = train_dataset[i]
            print(f"✓ Sample {i} loaded")
        except Exception as e:
            print(f"✗ Failed to load sample {i}: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\n6. Creating DataLoader (num_workers=0, no multiprocessing)...")
    from torch.utils.data import DataLoader
    loader_single = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,  # Single process first
        drop_last=False
    )
    print("✓ DataLoader created (single process)")
    
    print("\n7. Testing DataLoader iteration (single process)...")
    try:
        batch = next(iter(loader_single))
        print(f"✓ Batch loaded successfully")
        print(f"   Batch keys: {list(batch.keys())}")
        if 'point_cloud' in batch:
            pc = batch['point_cloud']
            print(f"   Point cloud pos shape: {pc['pos'].shape}, device: {pc['pos'].device}")
    except Exception as e:
        print(f"✗ Failed to load batch: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\n8. Creating DataLoader (num_workers=2, multiprocessing)...")
    loader_multi = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,  # Test multiprocessing
        drop_last=False,
        persistent_workers=False  # Don't keep workers alive
    )
    print("✓ DataLoader created (multiprocessing)")
    
    print("\n9. Testing DataLoader iteration (multiprocessing)...")
    print("   This is where the segfault likely occurs...")
    try:
        batch = next(iter(loader_multi))
        print(f"✓ Batch loaded successfully with multiprocessing")
        print(f"   Batch keys: {list(batch.keys())}")
    except Exception as e:
        print(f"✗ Failed to load batch with multiprocessing: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    
except Exception as e:
    print(f"\n✗ FATAL ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

