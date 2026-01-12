#!/usr/bin/env python3
"""
Compare datasets: dropped_data vs nowall.
Visualizes the difference between the original dataset and the W-sliced dataset.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.signal import hilbert

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def process_signal(x_numpy, channel_idx=0):
    """
    Process signal using Hilbert transform, normalization, and log1p compression.
    Same processing as in evaluate_gradcam.py
    
    Args:
        x_numpy: Input array of shape [C, H, W]
        channel_idx: Channel index to visualize (default: 0)
    
    Returns:
        Processed image array of shape [H, W]
    """
    # Apply Hilbert transform to input for visualization
    hilbert_images = []
    for c in range(x_numpy.shape[0]):
        hilbert_channel = np.zeros((x_numpy.shape[1], x_numpy.shape[2]))
        for h in range(x_numpy.shape[1]):
            hilbert_signal = hilbert(x_numpy[c, h, :])
            hilbert_channel[h, :] = np.abs(hilbert_signal)
        hilbert_images.append(hilbert_channel)
    
    # Use specified channel for sample image visualization
    # Normalize by max value, then apply log1p compression
    sample_image_raw = hilbert_images[channel_idx]
    max_val = sample_image_raw.max()
    if max_val > 0:
        sample_image_normalized = sample_image_raw / max_val  # Normalize to [0, 1]
    else:
        sample_image_normalized = sample_image_raw
    sample_image = np.log1p(sample_image_normalized)  # Log compression: log(1 + x)
    
    return sample_image


def visualize_comparison(x_dropped, x_nowall, sample_idx, save_path, channel_idx=0):
    """
    Visualize comparison between dropped_data and nowall datasets.
    
    Args:
        x_dropped: Sample from dropped_data, shape [C, H, W=2500]
        x_nowall: Sample from nowall, shape [C, H, W=2000]
        sample_idx: Sample index for title
        save_path: Path to save visualization
        channel_idx: Channel index to visualize (default: 0)
    """
    # First, extract the corresponding region from dropped_data BEFORE processing
    # This ensures we compare the same data with the same processing
    x_dropped_sliced_data = x_dropped[:, :, 500:2500]  # [C, H, W=2000]
    
    # Process all three: full, sliced, and nowall
    # Note: process_signal applies Hilbert transform which depends on signal length,
    # so we need to process them separately
    image_dropped_full = process_signal(x_dropped, channel_idx=channel_idx)
    image_dropped_sliced = process_signal(x_dropped_sliced_data, channel_idx=channel_idx)
    image_nowall = process_signal(x_nowall, channel_idx=channel_idx)
    
    # However, for fair comparison, we should also extract from the processed full image
    # to show what the sliced region looks like when processed with the full context
    image_dropped_full_sliced = image_dropped_full[:, 500:2500]
    
    # Create figure with 3 columns: dropped_data (full), dropped_data (sliced), nowall
    fig = plt.figure(figsize=(20, 6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1], wspace=0.3)
    
    # Column 1: dropped_data (full) with highlighted region
    ax_full = fig.add_subplot(gs[0, 0])
    im_full = ax_full.imshow(image_dropped_full, cmap='jet', aspect='auto', vmin=0.01, vmax=0.5)
    # Draw vertical lines to highlight the sliced region (W=500 and W=2500)
    ax_full.axvline(x=500, color='white', linewidth=2, linestyle='--', alpha=0.8, label='Slice start (W=500)')
    ax_full.axvline(x=2500, color='white', linewidth=2, linestyle='--', alpha=0.8, label='Slice end (W=2500)')
    ax_full.set_title(f'Dropped Data (Full)\nSample {sample_idx} | Channel {channel_idx}\nW shape: {x_dropped.shape[2]}', fontsize=12)
    ax_full.axis('off')
    ax_full.legend(loc='upper right', fontsize=8)
    plt.colorbar(im_full, ax=ax_full, fraction=0.046, pad=0.04)
    
    # Column 2: dropped_data (sliced region processed independently, W=500:2500)
    ax_sliced = fig.add_subplot(gs[0, 1])
    im_sliced = ax_sliced.imshow(image_dropped_sliced, cmap='jet', aspect='auto', vmin=0.01, vmax=0.5)
    ax_sliced.set_title(f'Dropped Data (Sliced, processed separately)\nSample {sample_idx} | Channel {channel_idx}\nW shape: {image_dropped_sliced.shape[1]}', fontsize=12)
    ax_sliced.axis('off')
    plt.colorbar(im_sliced, ax=ax_sliced, fraction=0.046, pad=0.04)
    
    # Column 3: nowall (should match the sliced region when processed separately)
    ax_nowall = fig.add_subplot(gs[0, 2])
    im_nowall = ax_nowall.imshow(image_nowall, cmap='jet', aspect='auto', vmin=0.01, vmax=0.5)
    ax_nowall.set_title(f'Nowall Dataset (processed separately)\nSample {sample_idx} | Channel {channel_idx}\nW shape: {x_nowall.shape[2]}', fontsize=12)
    ax_nowall.axis('off')
    plt.colorbar(im_nowall, ax=ax_nowall, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Main function to compare datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare dropped_data and nowall datasets')
    parser.add_argument('--dropped_path', type=str,
                        default='/home/smatsubara/documents/airlift/data/experiments/dataset/dropped_data/x_train_dropped.npy',
                        help='Path to dropped_data X file')
    parser.add_argument('--nowall_path', type=str,
                        default='/home/smatsubara/documents/airlift/data/experiments/dataset/nowall/x_train_nowall.npy',
                        help='Path to nowall X file')
    parser.add_argument('--output_dir', type=str,
                        default='/home/smatsubara/documents/airlift/test',
                        help='Output directory for comparison visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize (default: 10)')
    parser.add_argument('--channel_idx', type=int, default=0,
                        help='Channel index to visualize (default: 0)')
    args = parser.parse_args()
    
    # Load datasets
    print(f"[INFO] Loading dropped_data from: {args.dropped_path}")
    x_dropped = np.load(args.dropped_path)
    print(f"[INFO] dropped_data shape: {x_dropped.shape}")
    
    print(f"[INFO] Loading nowall from: {args.nowall_path}")
    x_nowall = np.load(args.nowall_path)
    print(f"[INFO] nowall shape: {x_nowall.shape}")
    
    # Validate shapes
    if x_dropped.ndim != 4 or x_nowall.ndim != 4:
        raise ValueError(f"Expected 4D arrays (N, C, H, W), got dropped_data: {x_dropped.ndim}D, nowall: {x_nowall.ndim}D")
    
    if x_dropped.shape[0] != x_nowall.shape[0]:
        raise ValueError(f"Number of samples mismatch: dropped_data={x_dropped.shape[0]}, nowall={x_nowall.shape[0]}")
    
    if x_dropped.shape[1] != x_nowall.shape[1]:
        raise ValueError(f"Number of channels mismatch: dropped_data={x_dropped.shape[1]}, nowall={x_nowall.shape[1]}")
    
    if x_dropped.shape[2] != x_nowall.shape[2]:
        raise ValueError(f"Height mismatch: dropped_data={x_dropped.shape[2]}, nowall={x_nowall.shape[2]}")
    
    # Check if nowall matches the sliced region of dropped_data
    expected_w = x_dropped.shape[3] - 500  # Should be 2000 if dropped_data W=2500
    if x_nowall.shape[3] != expected_w:
        print(f"[WARN] W dimension mismatch: nowall has W={x_nowall.shape[3]}, expected W={expected_w} (dropped_data W={x_dropped.shape[3]} - 500)")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")
    
    # Determine number of samples to process
    num_samples = min(args.num_samples, x_dropped.shape[0], x_nowall.shape[0])
    print(f"[INFO] Processing {num_samples} samples")
    
    # Generate visualizations
    print(f"\n[INFO] Generating comparison visualizations...")
    for sample_idx in range(num_samples):
        x_dropped_sample = x_dropped[sample_idx]  # [C, H, W]
        x_nowall_sample = x_nowall[sample_idx]    # [C, H, W]
        
        save_path = output_dir / f"comparison_sample{sample_idx:04d}_channel{args.channel_idx}.png"
        visualize_comparison(
            x_dropped_sample,
            x_nowall_sample,
            sample_idx,
            save_path,
            channel_idx=args.channel_idx
        )
    
    print(f"\n✅ Comparison visualization complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

