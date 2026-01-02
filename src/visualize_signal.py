#!/usr/bin/env python3
"""
Signal visualization tool for N,C,H,W tensor data.
Extracts individual samples and visualizes signal intensity.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from typing import Optional, Tuple


def visualize_signal_sample(x_path: str, sample_idx: int = 0, 
                           channel_idx: Optional[int] = None,
                           output_dir: str = "signal_plots",
                           save_plots: bool = True,
                           show_plots: bool = False) -> None:
    """
    Visualize signal intensity for a single sample from N,C,H,W tensor.
    
    Args:
        x_path: Path to input data file (.npy or .npz)
        sample_idx: Index of sample to visualize (0-based)
        channel_idx: Specific channel to visualize (None for all channels)
        output_dir: Directory to save plots
        save_plots: Whether to save plots to files
        show_plots: Whether to display plots interactively
    """
    print(f"📊 Signal Visualization Tool")
    print(f"=" * 50)
    
    # Create output directory
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {x_path}")
    x = load_data_robust(x_path)
    print(f"Data shape: {x.shape}")
    
    # Validate dimensions
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor (N,C,H,W), got {x.ndim}D")
    
    n_samples, n_channels, height, width = x.shape
    print(f"Dimensions: N={n_samples}, C={n_channels}, H={height}, W={width}")
    
    # Validate sample index
    if sample_idx >= n_samples:
        raise ValueError(f"Sample index {sample_idx} out of range (0-{n_samples-1})")
    
    # Extract sample
    sample = x[sample_idx]  # Shape: (C, H, W)
    print(f"Extracted sample {sample_idx}: shape {sample.shape}")
    
    # Check for problematic values
    nan_count = np.isnan(sample).sum()
    inf_count = np.isinf(sample).sum()
    print(f"Sample contains: {nan_count} NaN, {inf_count} Inf values")
    
    # Determine channels to visualize
    if channel_idx is not None:
        if channel_idx >= n_channels:
            raise ValueError(f"Channel index {channel_idx} out of range (0-{n_channels-1})")
        channels_to_plot = [channel_idx]
    else:
        channels_to_plot = list(range(n_channels))
    
    print(f"Visualizing channels: {channels_to_plot}")
    
    # Create visualizations
    create_signal_visualizations(sample, sample_idx, channels_to_plot, 
                                output_dir, save_plots, show_plots)


def load_data_robust(path: str, key: str = None):
    """Robust data loading supporting both .npy and .npz files."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    obj = np.load(path)
    
    if hasattr(obj, 'keys'):  # .npz file
        keys = list(obj.keys())
        if key and key in obj:
            data = obj[key]
        else:
            if key and key not in obj:
                print(f"   Warning: Key '{key}' not found. Available keys: {keys}")
            data = obj[keys[0]]
            print(f"   Using key: {keys[0]}")
    else:  # .npy file
        data = obj
        if key:
            print(f"   Warning: Key '{key}' ignored for .npy file")
    
    return data


def create_signal_visualizations(sample: np.ndarray, sample_idx: int, 
                                channels_to_plot: list, output_dir: str,
                                save_plots: bool, show_plots: bool) -> None:
    """Create various signal visualizations for the sample."""
    
    n_channels, height, width = sample.shape
    
    # 1. Individual channel heatmaps
    print("Creating channel heatmaps...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, channel_idx in enumerate(channels_to_plot):
        if i >= 4:  # Limit to 4 subplots
            break
            
        ax = axes[i]
        channel_data = sample[channel_idx]
        
        # Handle NaN/Inf values for visualization
        finite_mask = np.isfinite(channel_data)
        if finite_mask.any():
            # Replace NaN/Inf with 0 for visualization
            vis_data = channel_data.copy()
            vis_data[~finite_mask] = 0
            
            im = ax.imshow(vis_data, aspect='auto', cmap='viridis', 
                          interpolation='nearest')
            ax.set_title(f'Channel {channel_idx} (Sample {sample_idx})\n'
                        f'Range: [{np.min(vis_data):.3f}, {np.max(vis_data):.3f}]\n'
                        f'Finite: {finite_mask.sum()}/{channel_data.size}')
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
            plt.colorbar(im, ax=ax, label='Signal Intensity')
        else:
            ax.text(0.5, 0.5, 'No finite values', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(f'Channel {channel_idx} - No Data')
    
    # Hide unused subplots
    for i in range(len(channels_to_plot), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(output_dir, f'sample_{sample_idx}_channels.png'), 
                   dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()
    
    # 2. Signal intensity profiles
    print("Creating signal intensity profiles...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, channel_idx in enumerate(channels_to_plot):
        if i >= 4:
            break
            
        ax = axes[i]
        channel_data = sample[channel_idx]
        
        # Calculate profiles
        finite_mask = np.isfinite(channel_data)
        if finite_mask.any():
            # Horizontal profile (mean along height)
            h_profile = np.nanmean(channel_data, axis=0)
            # Vertical profile (mean along width)
            v_profile = np.nanmean(channel_data, axis=1)
            
            # Plot both profiles
            ax2 = ax.twinx()
            
            line1 = ax.plot(h_profile, 'b-', alpha=0.7, label='Horizontal (H mean)')
            line2 = ax2.plot(v_profile, 'r-', alpha=0.7, label='Vertical (W mean)')
            
            ax.set_xlabel('Position')
            ax.set_ylabel('Horizontal Profile', color='b')
            ax2.set_ylabel('Vertical Profile', color='r')
            ax.set_title(f'Channel {channel_idx} Intensity Profiles')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right')
        else:
            ax.text(0.5, 0.5, 'No finite values', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(f'Channel {channel_idx} - No Data')
    
    # Hide unused subplots
    for i in range(len(channels_to_plot), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(output_dir, f'sample_{sample_idx}_profiles.png'), 
                   dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()
    
    # 3. Signal statistics
    print("Creating signal statistics...")
    create_signal_statistics(sample, sample_idx, channels_to_plot, output_dir)
    
    # 4. 3D surface plot (if data is not too large)
    if height <= 100 and width <= 100:  # Only for reasonably sized data
        print("Creating 3D surface plot...")
        create_3d_surface_plot(sample, sample_idx, channels_to_plot, output_dir, show_plots)
    else:
        print("Skipping 3D plot (data too large)")


def create_signal_statistics(sample: np.ndarray, sample_idx: int, 
                           channels_to_plot: list, output_dir: str) -> None:
    """Create signal statistics visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, channel_idx in enumerate(channels_to_plot):
        if i >= 4:
            break
            
        ax = axes[i]
        channel_data = sample[channel_idx]
        
        # Filter finite values
        finite_data = channel_data[np.isfinite(channel_data)]
        
        if len(finite_data) > 0:
            # Histogram of signal intensities
            ax.hist(finite_data, bins=50, alpha=0.7, density=True, edgecolor='black')
            ax.set_xlabel('Signal Intensity')
            ax.set_ylabel('Density')
            ax.set_title(f'Channel {channel_idx} Intensity Distribution\n'
                        f'Mean: {np.mean(finite_data):.3f}, Std: {np.std(finite_data):.3f}\n'
                        f'Min: {np.min(finite_data):.3f}, Max: {np.max(finite_data):.3f}')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No finite values', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(f'Channel {channel_idx} - No Data')
    
    # Hide unused subplots
    for i in range(len(channels_to_plot), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sample_{sample_idx}_statistics.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()


def create_3d_surface_plot(sample: np.ndarray, sample_idx: int, 
                          channels_to_plot: list, output_dir: str, show_plots: bool) -> None:
    """Create 3D surface plot for signal visualization."""
    from mpl_toolkits.mplot3d import Axes3D
    
    for i, channel_idx in enumerate(channels_to_plot[:2]):  # Limit to 2 channels for 3D
        channel_data = sample[channel_idx]
        
        # Downsample if too large
        if channel_data.shape[0] > 50 or channel_data.shape[1] > 50:
            step_h = max(1, channel_data.shape[0] // 50)
            step_w = max(1, channel_data.shape[1] // 50)
            channel_data = channel_data[::step_h, ::step_w]
        
        # Handle NaN/Inf
        finite_mask = np.isfinite(channel_data)
        if finite_mask.any():
            vis_data = channel_data.copy()
            vis_data[~finite_mask] = 0
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create meshgrid
            h, w = channel_data.shape
            H, W = np.meshgrid(np.arange(w), np.arange(h))
            
            # Plot surface
            surf = ax.plot_surface(H, W, vis_data, cmap='viridis', alpha=0.8)
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
            ax.set_zlabel('Signal Intensity')
            ax.set_title(f'Channel {channel_idx} 3D Surface (Sample {sample_idx})')
            
            plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            if show_plots:
                plt.show()
            plt.savefig(os.path.join(output_dir, f'sample_{sample_idx}_channel_{channel_idx}_3d.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize signal intensity from N,C,H,W tensor')
    parser.add_argument('--x_path', default='/home/smatsubara/documents/airlift/data/sandbox/results/x_train_real.npy',
                       help='Path to input data file')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Index of sample to visualize (0-based)')
    parser.add_argument('--channel_idx', type=int, default=None,
                       help='Specific channel to visualize (None for all)')
    parser.add_argument('--output_dir', default='signal_plots',
                       help='Output directory for plots')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save plots to files')
    
    args = parser.parse_args()
    
    visualize_signal_sample(
        x_path=args.x_path,
        sample_idx=args.sample_idx,
        channel_idx=args.channel_idx,
        output_dir=args.output_dir,
        save_plots=not args.no_save,
        show_plots=args.show
    )


if __name__ == "__main__":
    main()









