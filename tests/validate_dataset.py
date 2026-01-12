#!/usr/bin/env python3
"""
Validate dataset files specified in config_real_updated.yaml.
Checks for NaN values, data shapes, and other statistics.
Also validates dataset with Channel 3 excluded (for training use).
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from pathlib import Path


def format_bytes(bytes_size):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def validate_channel(arr, channel_idx, channel_name="Channel", show_percentiles=False):
    """
    Validate a single channel and print detailed statistics.
    
    Args:
        arr: Array with shape (N, C, ...) or (N, C, H, W)
        channel_idx: Channel index to validate
        channel_name: Name prefix for the channel
    """
    if arr.ndim == 4:
        ch_data = arr[:, channel_idx, :, :]
    elif arr.ndim == 3:
        ch_data = arr[:, channel_idx, :]
    else:
        print(f"[ERROR] Unsupported array dimensionality for channel validation: {arr.ndim}D")
        return None
    
    print(f"\n{'-'*70}")
    print(f"{channel_name} {channel_idx} - Detailed Statistics")
    print(f"{'-'*70}")
    
    # Basic information
    print(f"Shape: {ch_data.shape}")
    print(f"Dtype: {ch_data.dtype}")
    print(f"Memory size: {format_bytes(ch_data.nbytes)}")
    
    # Check for NaN/Inf values
    nan_count = np.isnan(ch_data).sum()
    nan_percentage = (nan_count / ch_data.size) * 100 if ch_data.size > 0 else 0
    inf_count = np.isinf(ch_data).sum()
    inf_percentage = (inf_count / ch_data.size) * 100 if ch_data.size > 0 else 0
    
    print(f"\nNaN values: {nan_count:,} / {ch_data.size:,} ({nan_percentage:.4f}%)")
    print(f"Inf values: {inf_count:,} / {ch_data.size:,} ({inf_percentage:.4f}%)")
    
    # Statistics (only for valid values)
    valid_mask = np.isfinite(ch_data)
    valid_count = valid_mask.sum()
    
    if valid_count > 0:
        valid_arr = ch_data[valid_mask]
        print(f"\nStatistics (valid values only: {valid_count:,} / {ch_data.size:,}):")
        print(f"  Min: {valid_arr.min():.6f}")
        print(f"  Max: {valid_arr.max():.6f}")
        print(f"  Mean: {valid_arr.mean():.6f}")
        print(f"  Std: {valid_arr.std():.6f}")
        print(f"  Median: {np.median(valid_arr):.6f}")
        
        # Percentiles (optional)
        if show_percentiles:
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            print(f"\nPercentiles:")
            for p in percentiles:
                val = np.percentile(valid_arr, p)
                print(f"  {p:2d}%: {val:12.6f}")
        
        # Check for extreme values
        if np.abs(valid_arr).max() > 1e6:
            print(f"\n[WARNING] Extremely large values detected!")
            print(f"  Max absolute value: {np.abs(valid_arr).max():.6f}")
            print(f"  This may cause numerical instability in training")
        
        # Check value range
        if valid_arr.min() < -1e10 or valid_arr.max() > 1e10:
            print(f"\n[WARNING] Values are extremely large (might cause numerical issues)")
            print(f"  Range: [{valid_arr.min():.6f}, {valid_arr.max():.6f}]")
        
        # Check for constant values
        unique_count = len(np.unique(valid_arr))
        print(f"\nUnique values: {unique_count:,}")
        if unique_count == 1:
            print(f"  [WARNING] Channel contains only one unique value!")
        elif unique_count < 100:
            print(f"  [INFO] Channel has relatively few unique values")
        
        # Sample statistics (first sample)
        if ch_data.shape[0] > 0:
            sample_0 = ch_data[0, :] if ch_data.ndim == 1 else ch_data[0, :, :].flatten()
            if len(sample_0) > 0:
                print(f"\nFirst sample statistics:")
                print(f"  Shape: {ch_data[0].shape}")
                print(f"  Min: {sample_0.min():.6f}, Max: {sample_0.max():.6f}")
                print(f"  Mean: {sample_0.mean():.6f}, Std: {sample_0.std():.6f}")
        
        # Detect extreme values and their locations
        extreme_threshold = 100000  # Values with |value| > this threshold
        extreme_mask = np.abs(ch_data) > extreme_threshold
        extreme_count = extreme_mask.sum()
        
        if extreme_count > 0:
            extreme_percentage = (extreme_count / ch_data.size) * 100
            print(f"\n[WARNING] Extreme values detected (|value| > {extreme_threshold}):")
            print(f"  Count: {extreme_count:,} / {ch_data.size:,} ({extreme_percentage:.4f}%)")
            print(f"  Min extreme value: {ch_data[extreme_mask].min():.6f}")
            print(f"  Max extreme value: {ch_data[extreme_mask].max():.6f}")
            
            # Find locations of extreme values
            if ch_data.ndim == 2:
                # (N, L)
                extreme_indices = np.where(extreme_mask)
                print(f"\n  Extreme value locations (first 20):")
                for i in range(min(20, len(extreme_indices[0]))):
                    sample_idx = extreme_indices[0][i]
                    pos_idx = extreme_indices[1][i]
                    value = ch_data[sample_idx, pos_idx]
                    print(f"    Sample {sample_idx}, Position {pos_idx}: {value:.6f}")
                
                # Count extreme values per sample
                extreme_per_sample = extreme_mask.sum(axis=1)
                samples_with_extreme = (extreme_per_sample > 0).sum()
                print(f"\n  Samples with extreme values: {samples_with_extreme} / {ch_data.shape[0]}")
                if samples_with_extreme > 0:
                    print(f"  Max extreme values in a single sample: {extreme_per_sample.max()}")
                    print(f"  Samples with most extreme values (top 5):")
                    top_samples = np.argsort(extreme_per_sample)[-5:][::-1]
                    for sample_idx in top_samples:
                        if extreme_per_sample[sample_idx] > 0:
                            print(f"    Sample {sample_idx}: {extreme_per_sample[sample_idx]:,} extreme values")
            
            elif ch_data.ndim == 3:
                # (N, H, W)
                extreme_indices = np.where(extreme_mask)
                print(f"\n  Extreme value locations (first 20):")
                for i in range(min(20, len(extreme_indices[0]))):
                    sample_idx = extreme_indices[0][i]
                    h_idx = extreme_indices[1][i]
                    w_idx = extreme_indices[2][i]
                    value = ch_data[sample_idx, h_idx, w_idx]
                    print(f"    Sample {sample_idx}, Position (H={h_idx}, W={w_idx}): {value:.6f}")
                
                # Count extreme values per sample
                extreme_per_sample = extreme_mask.sum(axis=(1, 2))
                samples_with_extreme = (extreme_per_sample > 0).sum()
                print(f"\n  Samples with extreme values: {samples_with_extreme} / {ch_data.shape[0]}")
                if samples_with_extreme > 0:
                    print(f"  Max extreme values in a single sample: {extreme_per_sample.max()}")
                    print(f"  Samples with most extreme values (top 5):")
                    top_samples = np.argsort(extreme_per_sample)[-5:][::-1]
                    for sample_idx in top_samples:
                        if extreme_per_sample[sample_idx] > 0:
                            print(f"    Sample {sample_idx}: {extreme_per_sample[sample_idx]:,} extreme values")
        
        # Check for suspicious patterns (values exactly at -1000000 or 1000000)
        suspicious_values = [-1000000.0, 1000000.0]
        for sus_val in suspicious_values:
            count = (ch_data == sus_val).sum()
            if count > 0:
                percentage = (count / ch_data.size) * 100
                print(f"\n[WARNING] Suspicious exact value detected: {sus_val}")
                print(f"  Count: {count:,} / {ch_data.size:,} ({percentage:.4f}%)")
                print(f"  This might indicate data clipping or placeholder values")
    else:
        print(f"\n[ERROR] No valid (finite) values found in channel!")
    
    return {
        'channel_idx': channel_idx,
        'shape': ch_data.shape,
        'nan_count': nan_count,
        'inf_count': inf_count,
        'valid_count': valid_count,
        'total_count': ch_data.size,
        'has_issues': (nan_count > 0 or inf_count > 0 or valid_count == 0)
    }


def validate_array(arr, name, path, show_percentiles=False):
    """Validate a numpy array and print statistics."""
    print(f"\n{'='*70}")
    print(f"Validating: {name}")
    print(f"Path: {path}")
    print(f"{'='*70}")
    
    # Basic information
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    print(f"Memory size: {format_bytes(arr.nbytes)}")
    
    # Check for NaN values
    nan_count = np.isnan(arr).sum()
    nan_percentage = (nan_count / arr.size) * 100 if arr.size > 0 else 0
    print(f"\nNaN values: {nan_count:,} / {arr.size:,} ({nan_percentage:.4f}%)")
    
    # Check for Inf values
    inf_count = np.isinf(arr).sum()
    inf_percentage = (inf_count / arr.size) * 100 if arr.size > 0 else 0
    print(f"Inf values: {inf_count:,} / {arr.size:,} ({inf_percentage:.4f}%)")
    
    # Find NaN/Inf locations
    if nan_count > 0:
        nan_indices = np.where(np.isnan(arr))
        print(f"\n[WARNING] NaN values found!")
        if arr.ndim <= 3:
            print(f"  First 10 NaN locations:")
            for i, idx in enumerate(zip(*nan_indices)):
                if i >= 10:
                    break
                print(f"    Index {idx}")
    
    if inf_count > 0:
        inf_indices = np.where(np.isinf(arr))
        print(f"\n[WARNING] Inf values found!")
        if arr.ndim <= 3:
            print(f"  First 10 Inf locations:")
            for i, idx in enumerate(zip(*inf_indices)):
                if i >= 10:
                    break
                print(f"    Index {idx}")
    
    # Statistics (only for non-NaN, non-Inf values)
    valid_mask = np.isfinite(arr)
    valid_count = valid_mask.sum()
    
    if valid_count > 0:
        valid_arr = arr[valid_mask]
        print(f"\nStatistics (valid values only: {valid_count:,} / {arr.size:,}):")
        print(f"  Min: {valid_arr.min():.6f}")
        print(f"  Max: {valid_arr.max():.6f}")
        print(f"  Mean: {valid_arr.mean():.6f}")
        print(f"  Std: {valid_arr.std():.6f}")
        print(f"  Median: {np.median(valid_arr):.6f}")
        
        # Percentiles (optional)
        if show_percentiles:
            percentiles = [1, 5, 25, 50, 75, 95, 99]
            print(f"\nPercentiles:")
            for p in percentiles:
                val = np.percentile(valid_arr, p)
                print(f"  {p:2d}%: {val:.6f}")
    else:
        print(f"\n[ERROR] No valid (finite) values found in array!")
    
    # Check for constant values
    if valid_count > 0:
        unique_count = len(np.unique(valid_arr))
        print(f"\nUnique values: {unique_count:,}")
        if unique_count == 1:
            print(f"  [WARNING] Array contains only one unique value!")
    
    # Check data range
    if valid_count > 0:
        if arr.dtype in [np.float32, np.float64]:
            if valid_arr.min() < -1e10 or valid_arr.max() > 1e10:
                print(f"\n[WARNING] Values are extremely large (might cause numerical issues)")
    
    return {
        'shape': arr.shape,
        'dtype': str(arr.dtype),
        'nan_count': nan_count,
        'inf_count': inf_count,
        'valid_count': valid_count,
        'total_count': arr.size,
        'has_issues': (nan_count > 0 or inf_count > 0 or valid_count == 0)
    }


def save_sample_images_per_channel(x, output_root="tests"):
    """
    Save dataset sample images per channel for quick visual inspection.
    
    - Uses the first sample (index 0).
    - Saves one image per channel under tests/dataset_samples.
    """
    if x.ndim < 3:
        print("[INFO] X has less than 3 dimensions, skipping image export.")
        return

    out_dir = Path(output_root) / "dataset_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_samples = x.shape[0]
    if n_samples == 0:
        print("[WARNING] X has zero samples, skipping image export.")
        return

    sample_idx = 0
    sample = x[sample_idx]

    # Determine (C, H, W) or (C, L)
    if sample.ndim == 3:
        # (C, H, W)
        n_channels = sample.shape[0]
        for ch in range(n_channels):
            img = sample[ch]
            plt.figure(figsize=(6, 4))
            plt.imshow(img, aspect="auto", cmap="jet")
            plt.title(f"Sample {sample_idx} - Channel {ch}")
            plt.colorbar()
            save_path = out_dir / f"sample{sample_idx}_ch{ch}.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[INFO] Saved channel image: {save_path}")
    elif sample.ndim == 2:
        # (C, L) -> treat each channel as 2D with shape (1, L)
        n_channels = sample.shape[0]
        for ch in range(n_channels):
            img = sample[ch][None, :]  # (1, L)
            plt.figure(figsize=(6, 3))
            plt.imshow(img, aspect="auto", cmap="jet")
            plt.title(f"Sample {sample_idx} - Channel {ch}")
            plt.colorbar()
            save_path = out_dir / f"sample{sample_idx}_ch{ch}.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[INFO] Saved channel image: {save_path}")
    else:
        print(f"[INFO] Unexpected sample ndim={sample.ndim}, skipping image export.")


def validate_dataset_pair(x_path, t_path, show_percentiles=False, save_samples=True):
    """Validate X and T dataset pair and check consistency."""
    print("\n" + "="*70)
    print("DATASET VALIDATION REPORT")
    print("="*70)
    
    # Check file existence
    if not os.path.exists(x_path):
        print(f"[ERROR] X dataset file not found: {x_path}")
        return False
    
    if not os.path.exists(t_path):
        print(f"[ERROR] T dataset file not found: {t_path}")
        return False
    
    print(f"\n[OK] Both dataset files exist")
    
    # Load data
    try:
        print(f"\n[INFO] Loading X dataset...")
        x = np.load(x_path)
        print(f"[OK] X dataset loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load X dataset: {e}")
        return False
    
    try:
        print(f"\n[INFO] Loading T dataset...")
        t = np.load(t_path)
        print(f"[OK] T dataset loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load T dataset: {e}")
        return False
    
    # Channel-wise detailed analysis FIRST (before overall statistics)
    x_excluded = None
    channel_excluded = False
    
    if x.ndim == 4 and x.shape[1] == 4:
        # First, perform channel-wise analysis
        print(f"\n{'='*70}")
        print("X DATASET - CHANNEL-WISE DETAILED ANALYSIS")
        print(f"{'='*70}")
        print(f"[INFO] Analyzing each channel separately before overall statistics")
        
        # Detailed channel-wise analysis
        channel_results = []
        for ch_idx in range(4):
            ch_result = validate_channel(x, ch_idx, "Channel", show_percentiles=show_percentiles)
            if ch_result:
                channel_results.append(ch_result)
        
        # Summary table of all channels
        print(f"\n{'='*70}")
        print("CHANNEL-WISE SUMMARY TABLE")
        print(f"{'='*70}")
        print(f"{'Channel':<10} {'Min':<15} {'Max':<15} {'Mean':<15} {'Std':<15} {'Extreme':<12} {'NaN':<10} {'Inf':<10}")
        print(f"{'-'*100}")
        for ch_idx in range(4):
            ch_data = x[:, ch_idx, :, :]
            valid_arr = ch_data[np.isfinite(ch_data)]
            extreme_count = (np.abs(ch_data) > 100000).sum()
            if len(valid_arr) > 0:
                extreme_str = f"{extreme_count:,}" if extreme_count > 0 else "0"
                print(f"{'Ch '+str(ch_idx):<10} {valid_arr.min():<15.6f} {valid_arr.max():<15.6f} "
                      f"{valid_arr.mean():<15.6f} {valid_arr.std():<15.6f} "
                      f"{extreme_str:<12} {np.isnan(ch_data).sum():<10,} {np.isinf(ch_data).sum():<10,}")
                if extreme_count > 0:
                    print(f"  {'':<10} {'':<15} {'':<15} {'':<15} {'':<15} "
                          f"{'(!)' if extreme_count > 1000 else '':<12} {'':<10} {'':<10}")
        
        # Then, show overall statistics
        print(f"\n{'='*70}")
        print("X DATASET - OVERALL STATISTICS (All Channels Combined)")
        print(f"{'='*70}")
        x_result_original = validate_array(x, "X (Features) - Original", x_path, show_percentiles=show_percentiles)
        channel_excluded = True
        
        # Validate T dataset
        print(f"\n{'='*70}")
        print("T DATASET - TARGETS VALIDATION")
        print(f"{'='*70}")
        t_result = validate_array(t, "T (Targets)", t_path, show_percentiles=show_percentiles)
        
        print(f"\n{'='*70}")
        print("EXCLUDING CHANNEL 1 AND CHANNEL 3 (Keeping channels 0, 2)")
        print(f"{'='*70}")
        x_excluded = x[:, [0, 2], :, :]
        print(f"[INFO] Original shape: {x.shape}")
        print(f"[INFO] After excluding Channel 1 and 3: {x_excluded.shape}")
        
        # Validate excluded X - overall statistics
        print(f"\n{'='*70}")
        print("EXCLUDED DATASET (After Channel 1 and 3 Exclusion)")
        print(f"{'='*70}")
        x_result = validate_array(x_excluded, "X (Features) - Channel 1 and 3 Excluded", x_path)
        
        # Channel-wise detailed analysis for excluded data (channels 0, 2)
        print(f"\n{'='*70}")
        print("CHANNEL-WISE DETAILED ANALYSIS (After Exclusion - Channels 0, 2)")
        print(f"{'='*70}")
        # Map original channel indices to new indices
        # x_excluded now has shape (N, 2, H, W) where [:, 0, :, :] = original channel 0, [:, 1, :, :] = original channel 2
        for new_ch_idx, orig_ch_idx in enumerate([0, 2]):
            validate_channel(x_excluded, new_ch_idx, f"Channel (original {orig_ch_idx})", show_percentiles=show_percentiles)
        
        # Summary table of remaining channels
        print(f"\n{'='*70}")
        print("CHANNEL-WISE SUMMARY TABLE (After Exclusion - Channels 0, 2)")
        print(f"{'='*70}")
        print(f"{'Orig Ch':<10} {'New Ch':<10} {'Min':<15} {'Max':<15} {'Mean':<15} {'Std':<15} {'Extreme':<12} {'NaN':<10} {'Inf':<10}")
        print(f"{'-'*110}")
        for new_ch_idx, orig_ch_idx in enumerate([0, 2]):
            ch_data = x_excluded[:, new_ch_idx, :, :]
            valid_arr = ch_data[np.isfinite(ch_data)]
            extreme_count = (np.abs(ch_data) > 100000).sum()
            if len(valid_arr) > 0:
                extreme_str = f"{extreme_count:,}" if extreme_count > 0 else "0"
                print(f"{'Ch '+str(orig_ch_idx):<10} {'Ch '+str(new_ch_idx):<10} {valid_arr.min():<15.6f} {valid_arr.max():<15.6f} "
                      f"{valid_arr.mean():<15.6f} {valid_arr.std():<15.6f} "
                      f"{extreme_str:<12} {np.isnan(ch_data).sum():<10,} {np.isinf(ch_data).sum():<10,}")
                if extreme_count > 0:
                    print(f"  {'':<10} {'':<10} {'':<15} {'':<15} {'':<15} {'':<15} "
                          f"{'(!)' if extreme_count > 1000 else '':<12} {'':<10} {'':<10}")
        
        # Compare statistics
        print(f"\n{'='*70}")
        print("COMPARISON: Original vs Excluded")
        print(f"{'='*70}")
        print(f"Original X shape: {x.shape}")
        print(f"Excluded X shape: {x_excluded.shape}")
        print(f"\nOriginal X statistics:")
        valid_original = x[np.isfinite(x)]
        if len(valid_original) > 0:
            print(f"  Min: {valid_original.min():.6f}, Max: {valid_original.max():.6f}")
            print(f"  Mean: {valid_original.mean():.6f}, Std: {valid_original.std():.6f}")
        print(f"\nExcluded X statistics:")
        valid_excluded = x_excluded[np.isfinite(x_excluded)]
        if len(valid_excluded) > 0:
            print(f"  Min: {valid_excluded.min():.6f}, Max: {valid_excluded.max():.6f}")
            print(f"  Mean: {valid_excluded.mean():.6f}, Std: {valid_excluded.std():.6f}")
        
        # Use excluded data for consistency checks
        x = x_excluded
    elif x.ndim == 3 and x.shape[1] == 4:
        # First, perform channel-wise analysis
        print(f"\n{'='*70}")
        print("X DATASET - CHANNEL-WISE DETAILED ANALYSIS")
        print(f"{'='*70}")
        print(f"[INFO] Analyzing each channel separately before overall statistics")
        
        # Detailed channel-wise analysis
        channel_results = []
        for ch_idx in range(4):
            ch_result = validate_channel(x, ch_idx, "Channel")
            if ch_result:
                channel_results.append(ch_result)
        
        # Summary table of all channels
        print(f"\n{'='*70}")
        print("CHANNEL-WISE SUMMARY TABLE")
        print(f"{'='*70}")
        print(f"{'Channel':<10} {'Min':<15} {'Max':<15} {'Mean':<15} {'Std':<15} {'Extreme':<12} {'NaN':<10} {'Inf':<10}")
        print(f"{'-'*100}")
        for ch_idx in range(4):
            ch_data = x[:, ch_idx, :]
            valid_arr = ch_data[np.isfinite(ch_data)]
            extreme_count = (np.abs(ch_data) > 100000).sum()
            if len(valid_arr) > 0:
                extreme_str = f"{extreme_count:,}" if extreme_count > 0 else "0"
                print(f"{'Ch '+str(ch_idx):<10} {valid_arr.min():<15.6f} {valid_arr.max():<15.6f} "
                      f"{valid_arr.mean():<15.6f} {valid_arr.std():<15.6f} "
                      f"{extreme_str:<12} {np.isnan(ch_data).sum():<10,} {np.isinf(ch_data).sum():<10,}")
                if extreme_count > 0:
                    print(f"  {'':<10} {'':<15} {'':<15} {'':<15} {'':<15} "
                          f"{'(!)' if extreme_count > 1000 else '':<12} {'':<10} {'':<10}")
        
        # Then, show overall statistics
        print(f"\n{'='*70}")
        print("X DATASET - OVERALL STATISTICS (All Channels Combined)")
        print(f"{'='*70}")
        x_result_original = validate_array(x, "X (Features) - Original", x_path)
        channel_excluded = True
        
        # Validate T dataset
        print(f"\n{'='*70}")
        print("T DATASET - TARGETS VALIDATION")
        print(f"{'='*70}")
        t_result = validate_array(t, "T (Targets)", t_path)
        
        print(f"\n{'='*70}")
        print("EXCLUDING CHANNEL 1 AND CHANNEL 3 (Keeping channels 0, 2)")
        print(f"{'='*70}")
        x_excluded = x[:, [0, 2], :]
        print(f"[INFO] Original shape: {x.shape}")
        print(f"[INFO] After excluding Channel 1 and 3: {x_excluded.shape}")
        
        # Validate excluded X - overall statistics
        print(f"\n{'='*70}")
        print("EXCLUDED DATASET (After Channel 1 and 3 Exclusion)")
        print(f"{'='*70}")
        x_result = validate_array(x_excluded, "X (Features) - Channel 1 and 3 Excluded", x_path, show_percentiles=show_percentiles)
        
        # Channel-wise detailed analysis for excluded data (channels 0, 2)
        print(f"\n{'='*70}")
        print("CHANNEL-WISE DETAILED ANALYSIS (After Exclusion - Channels 0, 2)")
        print(f"{'='*70}")
        # Map original channel indices to new indices
        for new_ch_idx, orig_ch_idx in enumerate([0, 2]):
            validate_channel(x_excluded, new_ch_idx, f"Channel (original {orig_ch_idx})")
        
        # Summary table of remaining channels
        print(f"\n{'='*70}")
        print("CHANNEL-WISE SUMMARY TABLE (After Exclusion - Channels 0, 2)")
        print(f"{'='*70}")
        print(f"{'Orig Ch':<10} {'New Ch':<10} {'Min':<15} {'Max':<15} {'Mean':<15} {'Std':<15} {'Extreme':<12} {'NaN':<10} {'Inf':<10}")
        print(f"{'-'*110}")
        for new_ch_idx, orig_ch_idx in enumerate([0, 2]):
            ch_data = x_excluded[:, new_ch_idx, :]
            valid_arr = ch_data[np.isfinite(ch_data)]
            extreme_count = (np.abs(ch_data) > 100000).sum()
            if len(valid_arr) > 0:
                extreme_str = f"{extreme_count:,}" if extreme_count > 0 else "0"
                print(f"{'Ch '+str(orig_ch_idx):<10} {'Ch '+str(new_ch_idx):<10} {valid_arr.min():<15.6f} {valid_arr.max():<15.6f} "
                      f"{valid_arr.mean():<15.6f} {valid_arr.std():<15.6f} "
                      f"{extreme_str:<12} {np.isnan(ch_data).sum():<10,} {np.isinf(ch_data).sum():<10,}")
                if extreme_count > 0:
                    print(f"  {'':<10} {'':<10} {'':<15} {'':<15} {'':<15} {'':<15} "
                          f"{'(!)' if extreme_count > 1000 else '':<12} {'':<10} {'':<10}")
        
        # Use excluded data for consistency checks
        x = x_excluded
    else:
        # No channel exclusion needed - validate X and T as-is
        print(f"\n{'='*70}")
        print("X DATASET - VALIDATION")
        print(f"{'='*70}")
        print(f"[INFO] No channel exclusion needed (X has {x.shape[1] if x.ndim > 1 else 1} channels)")
        
        # If X has multiple channels, perform channel-wise analysis
        if x.ndim >= 3 and x.shape[1] > 1:
            print(f"\n{'='*70}")
            print("X DATASET - CHANNEL-WISE DETAILED ANALYSIS")
            print(f"{'='*70}")
            n_channels = x.shape[1]
            channel_results = []
            for ch_idx in range(n_channels):
                ch_result = validate_channel(x, ch_idx, "Channel", show_percentiles=show_percentiles)
                if ch_result:
                    channel_results.append(ch_result)
            
            # Summary table
            print(f"\n{'='*70}")
            print("CHANNEL-WISE SUMMARY TABLE")
            print(f"{'='*70}")
            print(f"{'Channel':<10} {'Min':<15} {'Max':<15} {'Mean':<15} {'Std':<15} {'Extreme':<12} {'NaN':<10} {'Inf':<10}")
            print(f"{'-'*100}")
            for ch_idx in range(n_channels):
                if x.ndim == 4:
                    ch_data = x[:, ch_idx, :, :]
                elif x.ndim == 3:
                    ch_data = x[:, ch_idx, :]
                else:
                    continue
                valid_arr = ch_data[np.isfinite(ch_data)]
                extreme_count = (np.abs(ch_data) > 100000).sum()
                if len(valid_arr) > 0:
                    extreme_str = f"{extreme_count:,}" if extreme_count > 0 else "0"
                    print(f"{'Ch '+str(ch_idx):<10} {valid_arr.min():<15.6f} {valid_arr.max():<15.6f} "
                          f"{valid_arr.mean():<15.6f} {valid_arr.std():<15.6f} "
                          f"{extreme_str:<12} {np.isnan(ch_data).sum():<10,} {np.isinf(ch_data).sum():<10,}")
        
        # Overall statistics
        print(f"\n{'='*70}")
        print("X DATASET - OVERALL STATISTICS")
        print(f"{'='*70}")
        x_result_original = validate_array(x, "X (Features)", x_path, show_percentiles=show_percentiles)
        x_result = x_result_original
        
        # Validate T dataset
        print(f"\n{'='*70}")
        print("T DATASET - TARGETS VALIDATION")
        print(f"{'='*70}")
        t_result = validate_array(t, "T (Targets)", t_path, show_percentiles=show_percentiles)
    
    # Save sample images per channel (using final X)
    if save_samples:
        print(f"\n{'='*70}")
        print("SAVING SAMPLE IMAGES PER CHANNEL")
        print(f"{'='*70}")
        save_sample_images_per_channel(x, output_root="tests")

    # Check consistency
    print(f"\n{'='*70}")
    print("CONSISTENCY CHECKS")
    print(f"{'='*70}")
    
    # Check sample count match
    if x.shape[0] != t.shape[0]:
        print(f"[ERROR] Sample count mismatch!")
        print(f"  X samples: {x.shape[0]}")
        print(f"  T samples: {t.shape[0]}")
        return False
    else:
        print(f"[OK] Sample count matches: {x.shape[0]:,} samples")
    
    # Check target dimensions
    if t.ndim == 1:
        print(f"[INFO] T is 1D array with shape {t.shape}")
    elif t.ndim == 2:
        print(f"[INFO] T is 2D array with shape {t.shape} ({t.shape[1]} targets per sample)")
    else:
        print(f"[WARNING] T has unexpected dimensionality: {t.ndim}D")
    
    # Check X dimensions (after exclusion if applicable)
    if channel_excluded:
        print(f"[INFO] X dimensions checked on Channel 3 excluded data")
    if x.ndim == 2:
        print(f"[INFO] X is 2D array (N, L) - will be reshaped to (N, 1, L) for CNN")
    elif x.ndim == 3:
        print(f"[INFO] X is 3D array (N, C, L) - 1D CNN input with C={x.shape[1]} channels")
    elif x.ndim == 4:
        print(f"[INFO] X is 4D array (N, C, H, W) or (N, H, W, C) - 2D CNN input")
        if x.shape[1] in [1, 2, 3, 4]:
            print(f"  Detected format: (N, C, H, W) with C={x.shape[1]} channels")
            if channel_excluded:
                print(f"  [NOTE] Channel 1 and 3 have been excluded, so C={x.shape[1]} (was 4)")
        else:
            print(f"  Detected format: (N, H, W, C) - will be transposed to (N, C, H, W)")
    else:
        print(f"[WARNING] X has unexpected dimensionality: {x.ndim}D")
    
    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    if channel_excluded:
        print(f"[INFO] Channel 1 and Channel 3 have been excluded from the dataset")
        print(f"       Original shape: {x_result_original['shape']}")
        print(f"       Excluded shape: {x_result['shape']}")
    
    issues = []
    if x_result['has_issues']:
        issues.append("X dataset has issues (NaN/Inf/missing values)")
    if t_result['has_issues']:
        issues.append("T dataset has issues (NaN/Inf/missing values)")
    
    if issues:
        print(f"[WARNING] Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"[OK] All validations passed!")
        if channel_excluded:
            print(f"[OK] Dataset is ready for training with Channel 1 and 3 excluded (2 channels: 0, 2)")
        return True


def main(show_percentiles: bool = False):
    """Main function.

    Args:
        show_percentiles: If True, compute and print percentile statistics.
                          If False, percentile computation is skipped.
    """
    # Load config
    config_path = "config/config_real.yaml"
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"[INFO] Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # Get dataset paths
    x_path = cfg.dataset.x_train
    t_path = cfg.dataset.t_train
    
    print(f"\n[INFO] X dataset path: {x_path}")
    print(f"[INFO] T dataset path: {t_path}")
    
    # Validate
    success = validate_dataset_pair(x_path, t_path, show_percentiles=show_percentiles, save_samples=True)
    
    if success:
        print(f"\n{'='*70}")
        print("[SUCCESS] Dataset validation completed successfully!")
        print(f"{'='*70}")
        sys.exit(0)
    else:
        print(f"\n{'='*70}")
        print("[FAILURE] Dataset validation found issues!")
        print(f"{'='*70}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset and optionally compute percentiles.")
    parser.add_argument(
        "--percentile",
        action="store_true",
        help="If set, compute and print percentile statistics. By default, percentiles are not computed.",
    )
    args = parser.parse_args()

    main(show_percentiles=args.percentile)

