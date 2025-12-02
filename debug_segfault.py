#!/usr/bin/env python3
"""
Debug script to find exact location of segfault by running train_network step by step
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import signal
import traceback

def signal_handler(sig, frame):
    print("\n" + "="*80)
    print("SIGNAL RECEIVED:", sig)
    print("="*80)
    traceback.print_stack(frame)
    sys.exit(1)

# Register signal handlers
signal.signal(signal.SIGSEGV, signal_handler)
signal.signal(signal.SIGABRT, signal_handler)

print("=" * 80)
print("DEBUG: Tracing train_network.py execution")
print("=" * 80)

try:
    print("\n[1/10] Importing hydra...")
    import hydra
    from omegaconf import DictConfig
    print("✓ Hydra imported")
    
    print("\n[2/10] Importing torch...")
    import torch
    print("✓ Torch imported")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   CUDA device count: {torch.cuda.device_count()}")
    
    print("\n[3/10] Initializing Hydra...")
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    
    config_dir = str(Path.cwd() / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="transformer_pretraining",
            overrides=[
                "opt.use_learnable_gating=false",
                "opt.use_routing=false", 
                "opt.use_dual_forward=false"
            ]
        )
    print("✓ Config loaded")
    
    print("\n[4/10] Importing train modules...")
    from utils.general_utils import safe_state
    from logger import Logger
    from dataset.dataset_factory import get_dataset
    print("✓ Modules imported")
    
    print("\n[5/10] Calling safe_state...")
    device = safe_state(cfg)
    print(f"✓ Device: {device}")
    
    print("\n[6/10] Creating Logger...")
    vis_dir = os.getcwd()
    logger = Logger(cfg, vis_dir)
    print("✓ Logger created")
    
    print("\n[7/10] Creating Dataset (THIS IS WHERE SEGFAULT LIKELY OCCURS)...")
    print("   Creating train dataset...")
    train_dataset = get_dataset(cfg, "train", device=device)
    print(f"✓ Train dataset created: {len(train_dataset)} samples")
    
    print("\n[8/10] Creating val dataset...")
    val_dataset = get_dataset(cfg, "val", device=device)
    print(f"✓ Val dataset created: {len(val_dataset)} samples")
    
    print("\n[9/10] Creating test dataset...")
    test_dataset = get_dataset(cfg, "test", device=device)
    print(f"✓ Test dataset created: {len(test_dataset)} samples")
    
    print("\n[10/10] Creating model...")
    from model.gaussian_predictor import GaussianSplatPredictor
    model = GaussianSplatPredictor(cfg)
    print("✓ Model created")
    
    print("\n   Moving model to device...")
    model = model.to(device)
    print("✓ Model moved to device")
    
    print("\n" + "=" * 80)
    print("✓ ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
except Exception as e:
    print(f"\n✗ EXCEPTION: {e}")
    traceback.print_exc()
    sys.exit(1)


