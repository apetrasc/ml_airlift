#!/usr/bin/env python3
"""
Create dataset with Channel 3 excluded and save to dropped_data directory.
Loads x_train and t_train, excludes Channel 3 from X, and saves as new files.
"""

import os
import sys
import numpy as np
from omegaconf import OmegaConf


def _load_np_any(path: str, prefer_key: str = None):
    """Load numpy array from .npy or .npz file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found: {path}")
    obj = np.load(path)
    # npz: dict-like, npy: ndarray
    if hasattr(obj, 'keys'):
        keys = list(obj.keys())
        if prefer_key and prefer_key in obj:
            arr = obj[prefer_key]
        else:
            if prefer_key and prefer_key not in obj:
                print(f"[WARN] key '{prefer_key}' not in {path}. Using first key: {keys[0]}")
            arr = obj[keys[0]]
        return arr
    else:
        if prefer_key:
            print(f"[INFO] {path} is an array (npy). Ignoring key '{prefer_key}'.")
        return obj


def main():
    """Main function."""
    # Load config
    config_path = "config/config_real_updated.yaml"
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"[INFO] Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # Get dataset paths
    x_path = cfg.dataset.x_train
    t_path = cfg.dataset.t_train
    x_key = cfg.dataset.get('x_key', None)
    t_key = cfg.dataset.get('t_key', None)
    
    print(f"\n[INFO] X dataset path: {x_path}")
    print(f"[INFO] T dataset path: {t_path}")
    
    # Load data
    print(f"\n[STEP] Loading datasets...")
    try:
        x = _load_np_any(x_path, x_key)
        print(f"[OK] Loaded X: shape={x.shape}, dtype={x.dtype}")
    except Exception as e:
        print(f"[ERROR] Failed to load X dataset: {e}")
        sys.exit(1)
    
    try:
        t = _load_np_any(t_path, t_key)
        print(f"[OK] Loaded T: shape={t.shape}, dtype={t.dtype}")
    except Exception as e:
        print(f"[ERROR] Failed to load T dataset: {e}")
        sys.exit(1)
    
    # Validate sample count match
    if x.shape[0] != t.shape[0]:
        print(f"[ERROR] Sample count mismatch: X={x.shape[0]}, T={t.shape[0]}")
        sys.exit(1)
    
    # Exclude Channel 1 and Channel 3
    print(f"\n[STEP] Excluding Channel 1 and Channel 3...")
    if x.ndim == 4 and x.shape[1] == 4:
        print(f"[INFO] Original X shape: {x.shape}")
        x_dropped = x[:, [0, 2], :, :]  # Keep only channels 0, 2
        print(f"[OK] Excluded Channel 1 and 3. New shape: {x_dropped.shape}")
        print(f"[INFO] Kept channels: 0, 2")
    elif x.ndim == 3 and x.shape[1] == 4:
        print(f"[INFO] Original X shape: {x.shape}")
        x_dropped = x[:, [0, 2], :]  # Keep only channels 0, 2
        print(f"[OK] Excluded Channel 1 and 3. New shape: {x_dropped.shape}")
        print(f"[INFO] Kept channels: 0, 2")
    else:
        print(f"[ERROR] Unexpected X shape: {x.shape}. Expected (N, 4, H, W) or (N, 4, L)")
        sys.exit(1)
    
    # Create output directory
    output_dir = "/mnt/matsubara/signals_exp/dataset/dropped_data"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[INFO] Output directory: {output_dir}")
    
    # Save X dataset
    x_output_path = os.path.join(output_dir, "x_train_dropped.npy")
    print(f"\n[STEP] Saving X dataset (Channel 1 and 3 excluded)...")
    try:
        np.save(x_output_path, x_dropped)
        print(f"[OK] Saved X dataset to: {x_output_path}")
        print(f"      Shape: {x_dropped.shape}, Size: {x_dropped.nbytes / (1024**2):.2f} MB")
    except Exception as e:
        print(f"[ERROR] Failed to save X dataset: {e}")
        sys.exit(1)
    
    # Save T dataset (copy as-is)
    t_output_path = os.path.join(output_dir, "t_train_dropped.npy")
    print(f"\n[STEP] Saving T dataset...")
    try:
        np.save(t_output_path, t)
        print(f"[OK] Saved T dataset to: {t_output_path}")
        print(f"      Shape: {t.shape}, Size: {t.nbytes / (1024**2):.2f} MB")
    except Exception as e:
        print(f"[ERROR] Failed to save T dataset: {e}")
        sys.exit(1)
    
    # Verify saved files
    print(f"\n[STEP] Verifying saved files...")
    try:
        x_loaded = np.load(x_output_path)
        t_loaded = np.load(t_output_path)
        
        if np.array_equal(x_loaded, x_dropped):
            print(f"[OK] X dataset verification passed")
        else:
            print(f"[WARNING] X dataset verification failed - data may differ")
        
        if np.array_equal(t_loaded, t):
            print(f"[OK] T dataset verification passed")
        else:
            print(f"[WARNING] T dataset verification failed - data may differ")
        
        print(f"\n[OK] Verification complete:")
        print(f"      X loaded shape: {x_loaded.shape}")
        print(f"      T loaded shape: {t_loaded.shape}")
        
    except Exception as e:
        print(f"[WARNING] Verification failed: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"[SUCCESS] Dataset with Channel 1 and Channel 3 excluded created successfully!")
    print(f"\nOutput files:")
    print(f"  X: {x_output_path}")
    print(f"  T: {t_output_path}")
    print(f"\nOriginal X shape: {x.shape}")
    print(f"Dropped X shape:  {x_dropped.shape}")
    print(f"T shape:          {t.shape}")
    print(f"\nChannel 1 and Channel 3 have been excluded. Dataset now contains 2 channels (0, 2).")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

