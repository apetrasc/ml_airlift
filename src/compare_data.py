#!/usr/bin/env python3
"""
Compare original and cleaned data side by side.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def compare_data_samples(original_path: str, cleaned_path: str, 
                        sample_idx: int = 0, channel_idx: int = 0,
                        output_dir: str = "comparison_plots") -> None:
    """Compare original and cleaned data samples."""
    
    print(f"🔍 Data Comparison Tool")
    print(f"=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading original data...")
    x_orig = np.load(original_path)
    print(f"Original shape: {x_orig.shape}")
    
    print("Loading cleaned data...")
    x_clean = np.load(cleaned_path)
    print(f"Cleaned shape: {x_clean.shape}")
    
    # Extract samples
    sample_orig = x_orig[sample_idx, channel_idx]  # Shape: (H, W)
    sample_clean = x_clean[sample_idx, channel_idx]  # Shape: (H, W)
    
    print(f"Sample {sample_idx}, Channel {channel_idx}")
    print(f"Original: {sample_orig.shape}, Cleaned: {sample_clean.shape}")
    
    # Statistics comparison
    print("\nStatistics Comparison:")
    print(f"Original - Min: {np.nanmin(sample_orig):.6f}, Max: {np.nanmax(sample_orig):.6f}")
    print(f"Original - Mean: {np.nanmean(sample_orig):.6f}, Std: {np.nanstd(sample_orig):.6f}")
    print(f"Original - NaN: {np.isnan(sample_orig).sum()}, Inf: {np.isinf(sample_orig).sum()}")
    
    print(f"Cleaned  - Min: {np.min(sample_clean):.6f}, Max: {np.max(sample_clean):.6f}")
    print(f"Cleaned  - Mean: {np.mean(sample_clean):.6f}, Std: {np.std(sample_clean):.6f}")
    print(f"Cleaned  - NaN: {np.isnan(sample_clean).sum()}, Inf: {np.isinf(sample_clean).sum()}")
    
    # Create comparison plots
    create_comparison_plots(sample_orig, sample_clean, sample_idx, channel_idx, output_dir)
    
    # Create difference analysis
    create_difference_analysis(sample_orig, sample_clean, sample_idx, channel_idx, output_dir)


def create_comparison_plots(sample_orig: np.ndarray, sample_clean: np.ndarray,
                           sample_idx: int, channel_idx: int, output_dir: str) -> None:
    """Create side-by-side comparison plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original data heatmap
    ax = axes[0, 0]
    finite_mask_orig = np.isfinite(sample_orig)
    if finite_mask_orig.any():
        vis_orig = sample_orig.copy()
        vis_orig[~finite_mask_orig] = 0
        im1 = ax.imshow(vis_orig, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_title(f'Original Data (Sample {sample_idx}, Channel {channel_idx})\n'
                    f'Finite: {finite_mask_orig.sum()}/{sample_orig.size}')
        plt.colorbar(im1, ax=ax, label='Signal Intensity')
    else:
        ax.text(0.5, 0.5, 'No finite values', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Original Data - No Finite Values')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    
    # Cleaned data heatmap
    ax = axes[0, 1]
    im2 = ax.imshow(sample_clean, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_title(f'Cleaned Data (Sample {sample_idx}, Channel {channel_idx})\n'
                f'All values finite')
    plt.colorbar(im2, ax=ax, label='Signal Intensity')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    
    # Difference heatmap
    ax = axes[0, 2]
    if finite_mask_orig.any():
        # Calculate difference only where original is finite
        diff = sample_clean - sample_orig
        diff[~finite_mask_orig] = 0  # Set difference to 0 where original was NaN/Inf
        im3 = ax.imshow(diff, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        ax.set_title(f'Difference (Cleaned - Original)\n'
                    f'Range: [{np.min(diff):.3f}, {np.max(diff):.3f}]')
        plt.colorbar(im3, ax=ax, label='Difference')
    else:
        ax.text(0.5, 0.5, 'No finite original values\nfor comparison', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Difference - No Valid Original Data')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    
    # Intensity profiles comparison
    ax = axes[1, 0]
    if finite_mask_orig.any():
        h_profile_orig = np.nanmean(sample_orig, axis=0)
        h_profile_clean = np.mean(sample_clean, axis=0)
        ax.plot(h_profile_orig, 'b-', alpha=0.7, label='Original')
        ax.plot(h_profile_clean, 'r-', alpha=0.7, label='Cleaned')
        ax.set_title('Horizontal Profile Comparison')
        ax.set_xlabel('Width Position')
        ax.set_ylabel('Mean Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No finite original values', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Horizontal Profile - No Original Data')
    
    # Intensity histograms
    ax = axes[1, 1]
    if finite_mask_orig.any():
        finite_orig = sample_orig[finite_mask_orig]
        ax.hist(finite_orig, bins=50, alpha=0.7, density=True, label='Original (finite)', color='blue')
    ax.hist(sample_clean.flatten(), bins=50, alpha=0.7, density=True, label='Cleaned', color='red')
    ax.set_title('Intensity Distribution Comparison')
    ax.set_xlabel('Signal Intensity')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Data quality metrics
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate metrics
    orig_nan_pct = (np.isnan(sample_orig).sum() / sample_orig.size) * 100
    orig_inf_pct = (np.isinf(sample_orig).sum() / sample_orig.size) * 100
    clean_nan_pct = (np.isnan(sample_clean).sum() / sample_clean.size) * 100
    clean_inf_pct = (np.isinf(sample_clean).sum() / sample_clean.size) * 100
    
    metrics_text = f"""Data Quality Metrics:
    
Original Data:
  NaN: {orig_nan_pct:.2f}%
  Inf: {orig_inf_pct:.2f}%
  Finite: {100-orig_nan_pct-orig_inf_pct:.2f}%
  
Cleaned Data:
  NaN: {clean_nan_pct:.2f}%
  Inf: {clean_inf_pct:.2f}%
  Finite: {100-clean_nan_pct-clean_inf_pct:.2f}%
  
Signal Range:
  Original: [{np.nanmin(sample_orig):.3f}, {np.nanmax(sample_orig):.3f}]
  Cleaned:  [{np.min(sample_clean):.3f}, {np.max(sample_clean):.3f}]
"""
    
    ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_sample_{sample_idx}_channel_{channel_idx}.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison plots saved to {output_dir}/")


def create_difference_analysis(sample_orig: np.ndarray, sample_clean: np.ndarray,
                              sample_idx: int, channel_idx: int, output_dir: str) -> None:
    """Create detailed difference analysis."""
    
    finite_mask_orig = np.isfinite(sample_orig)
    
    if not finite_mask_orig.any():
        print("⚠️  No finite values in original data for difference analysis")
        return
    
    # Calculate differences
    diff = sample_clean - sample_orig
    diff[~finite_mask_orig] = 0  # Set to 0 where original was NaN/Inf
    
    # Create difference analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Difference heatmap
    ax = axes[0, 0]
    im = ax.imshow(diff, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax.set_title(f'Difference Map (Cleaned - Original)\nSample {sample_idx}, Channel {channel_idx}')
    plt.colorbar(im, ax=ax, label='Difference')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    
    # Difference histogram
    ax = axes[0, 1]
    finite_diff = diff[finite_mask_orig]
    ax.hist(finite_diff, bins=50, alpha=0.7, edgecolor='black')
    ax.set_title('Difference Distribution')
    ax.set_xlabel('Difference Value')
    ax.set_ylabel('Count')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Zero difference')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Absolute difference
    ax = axes[1, 0]
    abs_diff = np.abs(diff)
    im2 = ax.imshow(abs_diff, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_title('Absolute Difference Map')
    plt.colorbar(im2, ax=ax, label='|Difference|')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    
    # Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""Difference Statistics:
    
Mean difference: {np.mean(finite_diff):.6f}
Std difference:  {np.std(finite_diff):.6f}
Min difference:  {np.min(finite_diff):.6f}
Max difference:  {np.max(finite_diff):.6f}

Mean |difference|: {np.mean(np.abs(finite_diff)):.6f}
Max |difference|:  {np.max(np.abs(finite_diff)):.6f}

Non-zero differences: {(finite_diff != 0).sum()}
Zero differences:     {(finite_diff == 0).sum()}
"""
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'difference_analysis_sample_{sample_idx}_channel_{channel_idx}.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Difference analysis saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Compare original and cleaned data')
    parser.add_argument('--original', default='/home/smatsubara/documents/airlift/data/sandbox/results/x_train_real.npy',
                       help='Path to original data file')
    parser.add_argument('--cleaned', default='/home/smatsubara/documents/sandbox/ml_airlift/cleaned_data/x_train_real_cleaned.npy',
                       help='Path to cleaned data file')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to compare')
    parser.add_argument('--channel_idx', type=int, default=0,
                       help='Channel index to compare')
    parser.add_argument('--output_dir', default='comparison_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    compare_data_samples(
        original_path=args.original,
        cleaned_path=args.cleaned,
        sample_idx=args.sample_idx,
        channel_idx=args.channel_idx,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()



