#!/usr/bin/env python3
"""
Plot all channels from a single sample of the dataset.
Saves the plot to /mnt/matsubara/pictures
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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


def plot_signal_sample(x_path, output_dir, sample_idx=0, channel_idx=0, x_key=None):
    """
    Plot a single channel from a single sample.
    
    Args:
        x_path: Path to X dataset file
        output_dir: Directory to save plot
        sample_idx: Sample index (default: 0)
        channel_idx: Channel index (default: 0)
        x_key: Key for X data if .npz file
    """
    print(f"[INFO] Loading dataset from: {x_path}")
    
    # Load data
    try:
        x = _load_np_any(x_path, x_key)
        print(f"[OK] Loaded X dataset: shape={x.shape}, dtype={x.dtype}")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return False
    
    # Check dimensions
    if x.ndim == 2:
        # (N, L) -> extract 1D signal
        if sample_idx >= x.shape[0]:
            print(f"[ERROR] Sample index {sample_idx} out of range (max: {x.shape[0]-1})")
            return False
        signal = x[sample_idx, :]
        signal_name = f"Sample {sample_idx}, 1D signal"
        xlabel = "Time/Position"
        
    elif x.ndim == 3:
        # (N, C, L) -> extract 1D signal from specific channel
        if sample_idx >= x.shape[0]:
            print(f"[ERROR] Sample index {sample_idx} out of range (max: {x.shape[0]-1})")
            return False
        if channel_idx >= x.shape[1]:
            print(f"[ERROR] Channel index {channel_idx} out of range (max: {x.shape[1]-1})")
            return False
        signal = x[sample_idx, channel_idx, :]
        signal_name = f"Sample {sample_idx}, Channel {channel_idx}"
        xlabel = "Time/Position"
        
    elif x.ndim == 4:
        # (N, C, H, W) or (N, H, W, C)
        if sample_idx >= x.shape[0]:
            print(f"[ERROR] Sample index {sample_idx} out of range (max: {x.shape[0]-1})")
            return False
        
        # Determine format
        if x.shape[1] in [1, 3, 4]:
            # (N, C, H, W) format
            if channel_idx >= x.shape[1]:
                print(f"[ERROR] Channel index {channel_idx} out of range (max: {x.shape[1]-1})")
                return False
            
            # Extract middle row from the channel
            h_mid = x.shape[2] // 2
            signal = x[sample_idx, channel_idx, h_mid, :]
            signal_name = f"Sample {sample_idx}, Channel {channel_idx}, Row {h_mid}"
            xlabel = f"Width (W={x.shape[3]})"
            
        else:
            # (N, H, W, C) format - need to transpose
            if channel_idx >= x.shape[3]:
                print(f"[ERROR] Channel index {channel_idx} out of range (max: {x.shape[3]-1})")
                return False
            
            # Extract middle row from the channel
            h_mid = x.shape[1] // 2
            signal = x[sample_idx, h_mid, :, channel_idx]
            signal_name = f"Sample {sample_idx}, Channel {channel_idx}, Row {h_mid}"
            xlabel = f"Width (W={x.shape[2]})"
    else:
        print(f"[ERROR] Unsupported dimensionality: {x.ndim}D")
        return False
    
    print(f"[INFO] Extracted signal: {signal_name}")
    print(f"[INFO] Signal shape: {signal.shape}")
    print(f"[INFO] Signal statistics:")
    print(f"  Min: {signal.min():.6f}")
    print(f"  Max: {signal.max():.6f}")
    print(f"  Mean: {signal.mean():.6f}")
    print(f"  Std: {signal.std():.6f}")
    print(f"  NaN count: {np.isnan(signal).sum()}")
    print(f"  Inf count: {np.isinf(signal).sum()}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Full signal
    axes[0].plot(signal, linewidth=0.5, alpha=0.7)
    axes[0].set_title(f'{signal_name} - Full Signal', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(xlabel, fontsize=10)
    axes[0].set_ylabel('Value', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Add statistics text
    stats_text = f"Min: {signal.min():.2f}, Max: {signal.max():.2f}, Mean: {signal.mean():.2f}, Std: {signal.std():.2f}"
    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=9)
    
    # Plot 2: Zoomed in (first 1000 points or 10% of signal, whichever is smaller)
    zoom_length = min(1000, len(signal) // 10)
    if zoom_length > 0:
        axes[1].plot(signal[:zoom_length], linewidth=1.0, alpha=0.8, color='green')
        axes[1].set_title(f'{signal_name} - Zoomed (First {zoom_length} points)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel(xlabel, fontsize=10)
        axes[1].set_ylabel('Value', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    else:
        axes[1].text(0.5, 0.5, 'Signal too short for zoom', 
                    transform=axes[1].transAxes, ha='center', va='center')
        axes[1].set_title(f'{signal_name} - Zoomed', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_filename = f"signal_sample{sample_idx}_channel{channel_idx}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Plot saved to: {output_path}")
    
    plt.close()
    
    # Also create a histogram to understand value distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Remove NaN and Inf for histogram
    valid_signal = signal[np.isfinite(signal)]
    if len(valid_signal) > 0:
        ax.hist(valid_signal, bins=100, alpha=0.7, edgecolor='black')
        ax.set_title(f'{signal_name} - Value Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add statistics
        stats_text = f"Mean: {valid_signal.mean():.2f}, Std: {valid_signal.std():.2f}\n"
        stats_text += f"Min: {valid_signal.min():.2f}, Max: {valid_signal.max():.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9)
    
    plt.tight_layout()
    
    # Save histogram
    hist_filename = f"signal_sample{sample_idx}_channel{channel_idx}_histogram.png"
    hist_path = os.path.join(output_dir, hist_filename)
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Histogram saved to: {hist_path}")
    
    plt.close()
    
    return True


def plot_all_channels(x_path, output_dir, sample_idx=0, x_key=None):
    """
    Plot all channels from a single sample and create comparison plots.
    
    Args:
        x_path: Path to X dataset file
        output_dir: Directory to save plot
        sample_idx: Sample index (default: 0)
        x_key: Key for X data if .npz file
    """
    print(f"[INFO] Loading dataset from: {x_path}")
    
    # Load data
    try:
        x = _load_np_any(x_path, x_key)
        print(f"[OK] Loaded X dataset: shape={x.shape}, dtype={x.dtype}")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return False
    
    # Determine number of channels
    if x.ndim == 2:
        # (N, L) -> single channel
        n_channels = 1
    elif x.ndim == 3:
        # (N, C, L)
        n_channels = x.shape[1]
    elif x.ndim == 4:
        # (N, C, H, W) or (N, H, W, C)
        if x.shape[1] in [1, 3, 4]:
            n_channels = x.shape[1]
            is_nchw = True
        else:
            n_channels = x.shape[3]
            is_nchw = False
    else:
        print(f"[ERROR] Unsupported dimensionality: {x.ndim}D")
        return False
    
    if sample_idx >= x.shape[0]:
        print(f"[ERROR] Sample index {sample_idx} out of range (max: {x.shape[0]-1})")
        return False
    
    print(f"[INFO] Number of channels: {n_channels}")
    
    # Extract signals for all channels
    signals = []
    signal_names = []
    
    for ch_idx in range(n_channels):
        if x.ndim == 2:
            signal = x[sample_idx, :]
            signal_name = f"Channel {ch_idx}"
            xlabel = "Time/Position"
            
        elif x.ndim == 3:
            signal = x[sample_idx, ch_idx, :]
            signal_name = f"Channel {ch_idx}"
            xlabel = "Time/Position"
            
        elif x.ndim == 4:
            if is_nchw:
                # (N, C, H, W) format
                h_mid = x.shape[2] // 2
                signal = x[sample_idx, ch_idx, h_mid, :]
                signal_name = f"Channel {ch_idx}, Row {h_mid}"
                xlabel = f"Width (W={x.shape[3]})"
            else:
                # (N, H, W, C) format
                h_mid = x.shape[1] // 2
                signal = x[sample_idx, h_mid, :, ch_idx]
                signal_name = f"Channel {ch_idx}, Row {h_mid}"
                xlabel = f"Width (W={x.shape[2]})"
        
        signals.append(signal)
        signal_names.append(signal_name)
        
        # Print statistics for each channel
        print(f"\n[INFO] {signal_name}:")
        print(f"  Shape: {signal.shape}")
        print(f"  Min: {signal.min():.6f}, Max: {signal.max():.6f}")
        print(f"  Mean: {signal.mean():.6f}, Std: {signal.std():.6f}")
        print(f"  NaN: {np.isnan(signal).sum()}, Inf: {np.isinf(signal).sum()}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot individual channels
    for ch_idx, (signal, signal_name) in enumerate(zip(signals, signal_names)):
        print(f"\n[INFO] Plotting {signal_name}...")
        plot_signal_sample(
            x_path=None,  # Not used when signal is provided
            output_dir=output_dir,
            sample_idx=sample_idx,
            channel_idx=ch_idx,
            x_key=None,
            signal=signal,
            signal_name=signal_name,
            xlabel=xlabel
        )
    
    # Create comparison plot with all channels
    print(f"\n[INFO] Creating comparison plot with all channels...")
    
    # Determine subplot layout
    if n_channels <= 4:
        n_rows = 2
        n_cols = 2
    else:
        n_rows = (n_channels + 3) // 4
        n_cols = 4
    
    # Full signal comparison
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_channels == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for ch_idx, (signal, signal_name) in enumerate(zip(signals, signal_names)):
        ax = axes[ch_idx]
        color = colors[ch_idx % len(colors)]
        ax.plot(signal, linewidth=0.5, alpha=0.7, color=color, label=signal_name)
        ax.set_title(f'{signal_name}', fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Add statistics
        stats_text = f"Min: {signal.min():.2f}, Max: {signal.max():.2f}\n"
        stats_text += f"Mean: {signal.mean():.2f}, Std: {signal.std():.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_channels, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Sample {sample_idx} - All Channels Comparison (Full Signal)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save comparison plot
    comparison_filename = f"signal_sample{sample_idx}_all_channels_comparison.png"
    comparison_path = os.path.join(output_dir, comparison_filename)
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Comparison plot saved to: {comparison_path}")
    plt.close()
    
    # Zoomed comparison (first portion)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_channels == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    zoom_length = min(1000, len(signals[0]) // 10)
    
    for ch_idx, (signal, signal_name) in enumerate(zip(signals, signal_names)):
        ax = axes[ch_idx]
        color = colors[ch_idx % len(colors)]
        if zoom_length > 0:
            ax.plot(signal[:zoom_length], linewidth=1.0, alpha=0.8, color=color, label=signal_name)
            ax.set_title(f'{signal_name} - Zoomed', fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Signal too short', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{signal_name} - Zoomed', fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Hide unused subplots
    for idx in range(n_channels, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Sample {sample_idx} - All Channels Comparison (Zoomed: First {zoom_length} points)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save zoomed comparison plot
    zoomed_filename = f"signal_sample{sample_idx}_all_channels_zoomed.png"
    zoomed_path = os.path.join(output_dir, zoomed_filename)
    plt.savefig(zoomed_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Zoomed comparison plot saved to: {zoomed_path}")
    plt.close()
    
    # Overlay plot (all channels on same axes)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Full signal overlay
    for ch_idx, (signal, signal_name) in enumerate(zip(signals, signal_names)):
        color = colors[ch_idx % len(colors)]
        ax1.plot(signal, linewidth=0.5, alpha=0.6, color=color, label=signal_name)
    ax1.set_title(f'Sample {sample_idx} - All Channels Overlay (Full Signal)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel(xlabel, fontsize=10)
    ax1.set_ylabel('Value', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.legend(loc='best', fontsize=8)
    
    # Zoomed overlay
    if zoom_length > 0:
        for ch_idx, (signal, signal_name) in enumerate(zip(signals, signal_names)):
            color = colors[ch_idx % len(colors)]
            ax2.plot(signal[:zoom_length], linewidth=1.0, alpha=0.7, color=color, label=signal_name)
    ax2.set_title(f'Sample {sample_idx} - All Channels Overlay (Zoomed: First {zoom_length} points)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel(xlabel, fontsize=10)
    ax2.set_ylabel('Value', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    
    # Save overlay plot
    overlay_filename = f"signal_sample{sample_idx}_all_channels_overlay.png"
    overlay_path = os.path.join(output_dir, overlay_filename)
    plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Overlay plot saved to: {overlay_path}")
    plt.close()
    
    return True


def plot_signal_sample(x_path=None, output_dir=None, sample_idx=0, channel_idx=0, x_key=None, 
                       signal=None, signal_name=None, xlabel=None):
    """
    Plot a single channel from a single sample.
    Can be called with pre-extracted signal or will load from file.
    
    Args:
        x_path: Path to X dataset file (if signal not provided)
        output_dir: Directory to save plot
        sample_idx: Sample index (default: 0)
        channel_idx: Channel index (default: 0)
        x_key: Key for X data if .npz file
        signal: Pre-extracted signal array (optional)
        signal_name: Name for the signal (optional)
        xlabel: X-axis label (optional)
    """
    # If signal is provided, use it directly
    if signal is not None and signal_name is not None and xlabel is not None:
        # Use provided signal
        pass
    else:
        # Load from file
        if x_path is None or output_dir is None:
            return False
        
        print(f"[INFO] Loading dataset from: {x_path}")
        
        # Load data
        try:
            x = _load_np_any(x_path, x_key)
            print(f"[OK] Loaded X dataset: shape={x.shape}, dtype={x.dtype}")
        except Exception as e:
            print(f"[ERROR] Failed to load dataset: {e}")
            return False
        
        # Check dimensions
        if x.ndim == 2:
            # (N, L) -> extract 1D signal
            if sample_idx >= x.shape[0]:
                print(f"[ERROR] Sample index {sample_idx} out of range (max: {x.shape[0]-1})")
                return False
            signal = x[sample_idx, :]
            signal_name = f"Sample {sample_idx}, 1D signal"
            xlabel = "Time/Position"
            
        elif x.ndim == 3:
            # (N, C, L) -> extract 1D signal from specific channel
            if sample_idx >= x.shape[0]:
                print(f"[ERROR] Sample index {sample_idx} out of range (max: {x.shape[0]-1})")
                return False
            if channel_idx >= x.shape[1]:
                print(f"[ERROR] Channel index {channel_idx} out of range (max: {x.shape[1]-1})")
                return False
            signal = x[sample_idx, channel_idx, :]
            signal_name = f"Sample {sample_idx}, Channel {channel_idx}"
            xlabel = "Time/Position"
            
        elif x.ndim == 4:
            # (N, C, H, W) or (N, H, W, C)
            if sample_idx >= x.shape[0]:
                print(f"[ERROR] Sample index {sample_idx} out of range (max: {x.shape[0]-1})")
                return False
            
            # Determine format
            if x.shape[1] in [1, 3, 4]:
                # (N, C, H, W) format
                if channel_idx >= x.shape[1]:
                    print(f"[ERROR] Channel index {channel_idx} out of range (max: {x.shape[1]-1})")
                    return False
                
                # Extract middle row from the channel
                h_mid = x.shape[2] // 2
                signal = x[sample_idx, channel_idx, h_mid, :]
                signal_name = f"Sample {sample_idx}, Channel {channel_idx}, Row {h_mid}"
                xlabel = f"Width (W={x.shape[3]})"
                
            else:
                # (N, H, W, C) format - need to transpose
                if channel_idx >= x.shape[3]:
                    print(f"[ERROR] Channel index {channel_idx} out of range (max: {x.shape[3]-1})")
                    return False
                
                # Extract middle row from the channel
                h_mid = x.shape[1] // 2
                signal = x[sample_idx, h_mid, :, channel_idx]
                signal_name = f"Sample {sample_idx}, Channel {channel_idx}, Row {h_mid}"
                xlabel = f"Width (W={x.shape[2]})"
        else:
            print(f"[ERROR] Unsupported dimensionality: {x.ndim}D")
            return False
    
    print(f"[INFO] Extracted signal: {signal_name}")
    print(f"[INFO] Signal shape: {signal.shape}")
    print(f"[INFO] Signal statistics:")
    print(f"  Min: {signal.min():.6f}")
    print(f"  Max: {signal.max():.6f}")
    print(f"  Mean: {signal.mean():.6f}")
    print(f"  Std: {signal.std():.6f}")
    print(f"  NaN count: {np.isnan(signal).sum()}")
    print(f"  Inf count: {np.isinf(signal).sum()}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Full signal
    axes[0].plot(signal, linewidth=0.5, alpha=0.7)
    axes[0].set_title(f'{signal_name} - Full Signal', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(xlabel, fontsize=10)
    axes[0].set_ylabel('Value', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Add statistics text
    stats_text = f"Min: {signal.min():.2f}, Max: {signal.max():.2f}, Mean: {signal.mean():.2f}, Std: {signal.std():.2f}"
    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=9)
    
    # Plot 2: Zoomed in (first 1000 points or 10% of signal, whichever is smaller)
    zoom_length = min(1000, len(signal) // 10)
    if zoom_length > 0:
        axes[1].plot(signal[:zoom_length], linewidth=1.0, alpha=0.8, color='green')
        axes[1].set_title(f'{signal_name} - Zoomed (First {zoom_length} points)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel(xlabel, fontsize=10)
        axes[1].set_ylabel('Value', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    else:
        axes[1].text(0.5, 0.5, 'Signal too short for zoom', 
                    transform=axes[1].transAxes, ha='center', va='center')
        axes[1].set_title(f'{signal_name} - Zoomed', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_filename = f"signal_sample{sample_idx}_channel{channel_idx}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Plot saved to: {output_path}")
    
    plt.close()
    
    # Also create a histogram to understand value distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Remove NaN and Inf for histogram
    valid_signal = signal[np.isfinite(signal)]
    if len(valid_signal) > 0:
        ax.hist(valid_signal, bins=100, alpha=0.7, edgecolor='black')
        ax.set_title(f'{signal_name} - Value Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add statistics
        stats_text = f"Mean: {valid_signal.mean():.2f}, Std: {valid_signal.std():.2f}\n"
        stats_text += f"Min: {valid_signal.min():.2f}, Max: {valid_signal.max():.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9)
    
    plt.tight_layout()
    
    # Save histogram
    hist_filename = f"signal_sample{sample_idx}_channel{channel_idx}_histogram.png"
    hist_path = os.path.join(output_dir, hist_filename)
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Histogram saved to: {hist_path}")
    
    plt.close()
    
    return True


def main():
    """Main function."""
    # Load config
    config_path = "config/config_real_updated.yaml"
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)
    
    cfg = OmegaConf.load(config_path)
    
    # Get dataset path
    x_path = cfg.dataset.x_train
    x_key = cfg.dataset.get('x_key', None)
    
    # Output directory
    output_dir = "/mnt/matsubara/pictures"
    
    print(f"[INFO] X dataset path: {x_path}")
    print(f"[INFO] Output directory: {output_dir}")
    
    # Plot all channels for first sample
    success = plot_all_channels(
        x_path=x_path,
        output_dir=output_dir,
        sample_idx=0,
        x_key=x_key
    )
    
    if success:
        print(f"\n[SUCCESS] Plotting completed successfully!")
        sys.exit(0)
    else:
        print(f"\n[FAILURE] Plotting failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

