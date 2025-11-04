import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def inspect_dataset(x_path: str, t_path: str, x_key: str = None, t_key: str = None, 
                   sample_limit: int = 10, save_plots: bool = True, output_dir: str = "debug_plots"):
    """
    Comprehensive dataset inspection for troubleshooting NaN issues.
    
    Args:
        x_path: Path to input data file (.npy or .npz)
        t_path: Path to target data file (.npy or .npz)
        x_key: Key for x data if .npz (None for auto-detect)
        t_key: Key for t data if .npz (None for auto-detect)
        sample_limit: Number of samples to inspect in detail
        save_plots: Whether to save diagnostic plots
        output_dir: Directory to save plots
    """
    print("=" * 60)
    print("DATASET INSPECTION REPORT")
    print("=" * 60)
    
    # Create output directory
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"\n1. LOADING DATA")
    print(f"   X file: {x_path}")
    print(f"   T file: {t_path}")
    
    x = load_data_robust(x_path, x_key)
    t = load_data_robust(t_path, t_key)
    
    print(f"   X shape: {x.shape}, dtype: {x.dtype}")
    print(f"   T shape: {t.shape}, dtype: {t.dtype}")
    
    # Basic statistics
    print(f"\n2. BASIC STATISTICS")
    print_x_stats(x, "X")
    print_x_stats(t, "T")
    
    # Check for problematic values
    print(f"\n3. PROBLEMATIC VALUES CHECK")
    check_problematic_values(x, "X")
    check_problematic_values(t, "T")
    
    # Data range analysis
    print(f"\n4. DATA RANGE ANALYSIS")
    analyze_data_ranges(x, "X")
    analyze_data_ranges(t, "T")
    
    # Sample inspection
    print(f"\n5. SAMPLE INSPECTION (first {sample_limit} samples)")
    inspect_samples(x, t, sample_limit)
    
    # Visualization
    if save_plots:
        print(f"\n6. GENERATING DIAGNOSTIC PLOTS")
        create_diagnostic_plots(x, t, sample_limit, output_dir)
    
    # Model input compatibility check
    print(f"\n7. MODEL INPUT COMPATIBILITY")
    check_model_compatibility(x, t)
    
    print(f"\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


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


def print_x_stats(data, name):
    """Print comprehensive statistics for data array."""
    print(f"   {name} Statistics:")
    print(f"     Min: {np.min(data):.6f}")
    print(f"     Max: {np.max(data):.6f}")
    print(f"     Mean: {np.mean(data):.6f}")
    print(f"     Std: {np.std(data):.6f}")
    print(f"     Median: {np.median(data):.6f}")
    
    if data.ndim > 1:
        print(f"     Shape: {data.shape}")
        print(f"     Memory usage: {data.nbytes / 1024**2:.2f} MB")


def check_problematic_values(data, name):
    """Check for NaN, Inf, and other problematic values."""
    print(f"   {name} Problematic Values:")
    
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    neg_inf_count = np.isneginf(data).sum()
    pos_inf_count = np.isposinf(data).sum()
    
    print(f"     NaN count: {nan_count}")
    print(f"     Inf count: {inf_count}")
    print(f"     -Inf count: {neg_inf_count}")
    print(f"     +Inf count: {pos_inf_count}")
    
    if nan_count > 0:
        print(f"     ⚠️  WARNING: {nan_count} NaN values detected!")
    if inf_count > 0:
        print(f"     ⚠️  WARNING: {inf_count} Inf values detected!")
    
    # Check for extremely large values
    large_values = np.abs(data) > 1e10
    if large_values.any():
        print(f"     ⚠️  WARNING: {large_values.sum()} values > 1e10 detected!")
    
    # Check for extremely small values
    small_values = (np.abs(data) < 1e-10) & (data != 0)
    if small_values.any():
        print(f"     ℹ️  INFO: {small_values.sum()} very small values (< 1e-10) detected")


def analyze_data_ranges(data, name):
    """Analyze data ranges and potential issues."""
    print(f"   {name} Range Analysis:")
    
    if data.ndim > 1:
        # Per-dimension analysis
        for i in range(min(3, data.ndim)):  # Check first 3 dimensions
            if data.ndim == 2:
                dim_data = data[:, i] if i < data.shape[1] else data[:, 0]
                dim_name = f"dim {i}"
            elif data.ndim == 3:
                dim_data = data[:, i, :].flatten() if i < data.shape[1] else data[:, 0, :].flatten()
                dim_name = f"channel {i}"
            elif data.ndim == 4:
                dim_data = data[:, i, :, :].flatten() if i < data.shape[1] else data[:, 0, :, :].flatten()
                dim_name = f"channel {i}"
            else:
                continue
                
            print(f"     {dim_name}: min={np.min(dim_data):.6f}, max={np.max(dim_data):.6f}, mean={np.mean(dim_data):.6f}")
    
    # Check for constant values
    if data.size > 0:
        unique_vals = len(np.unique(data))
        print(f"     Unique values: {unique_vals}")
        if unique_vals < 10:
            print(f"     Values: {np.unique(data)}")


def inspect_samples(x, t, sample_limit):
    """Inspect individual samples in detail."""
    n_samples = min(sample_limit, x.shape[0])
    
    for i in range(n_samples):
        print(f"   Sample {i}:")
        
        # X sample
        x_sample = x[i]
        print(f"     X shape: {x_sample.shape}")
        print(f"     X range: [{np.min(x_sample):.6f}, {np.max(x_sample):.6f}]")
        print(f"     X has NaN: {np.isnan(x_sample).any()}")
        print(f"     X has Inf: {np.isinf(x_sample).any()}")
        
        # T sample
        t_sample = t[i]
        print(f"     T shape: {t_sample.shape}")
        print(f"     T values: {t_sample}")
        print(f"     T has NaN: {np.isnan(t_sample).any()}")
        print(f"     T has Inf: {np.isinf(t_sample).any()}")
        
        # Check for extreme values
        if np.any(np.abs(x_sample) > 1e6):
            print(f"     ⚠️  X has extreme values!")
        if np.any(np.abs(t_sample) > 1e6):
            print(f"     ⚠️  T has extreme values!")


def create_diagnostic_plots(x, t, sample_limit, output_dir):
    """Create diagnostic plots for data visualization."""
    n_samples = min(sample_limit, x.shape[0])
    
    # Plot 1: Data distribution
    plt.figure(figsize=(15, 10))
    
    # X distribution
    plt.subplot(2, 3, 1)
    x_flat = x.flatten()
    # Filter out NaN and Inf values for plotting
    x_finite = x_flat[np.isfinite(x_flat)]
    if len(x_finite) > 0:
        plt.hist(x_finite, bins=50, alpha=0.7, density=True)
        plt.title(f'X Data Distribution (finite values only)\n{len(x_finite)}/{len(x_flat)} finite')
    else:
        plt.text(0.5, 0.5, 'No finite values found!', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('X Data Distribution - NO FINITE VALUES')
    plt.xlabel('Value')
    plt.ylabel('Density')
    
    # T distribution
    plt.subplot(2, 3, 2)
    t_flat = t.flatten()
    plt.hist(t_flat, bins=50, alpha=0.7, density=True)
    plt.title('T Data Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    
    # Sample X values over time/space
    plt.subplot(2, 3, 3)
    for i in range(min(3, n_samples)):
        if x.ndim == 2:
            x_sample = x[i]
        elif x.ndim == 3:
            x_sample = x[i, 0]
        elif x.ndim == 4:
            x_sample = x[i, 0, :, 0]
        else:
            continue
            
        # Filter finite values for plotting
        finite_mask = np.isfinite(x_sample)
        if np.any(finite_mask):
            finite_indices = np.where(finite_mask)[0]
            finite_values = x_sample[finite_mask]
            # Plot only a subset to avoid overcrowding
            step = max(1, len(finite_indices) // 1000)
            plt.plot(finite_indices[::step], finite_values[::step], alpha=0.7, label=f'Sample {i}')
    plt.title('Sample X Values (finite only)')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    
    # T values
    plt.subplot(2, 3, 4)
    plt.plot(t[:n_samples])
    plt.title('T Values (first samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('T Value')
    
    # X vs T scatter (if 1D targets)
    plt.subplot(2, 3, 5)
    if t.ndim == 1:
        # Calculate mean X values, handling NaN
        mean_x_values = []
        for i in range(n_samples):
            x_sample = x[i]
            finite_x = x_sample[np.isfinite(x_sample)]
            if len(finite_x) > 0:
                mean_x_values.append(np.mean(finite_x))
            else:
                mean_x_values.append(np.nan)
        
        # Filter out NaN means for plotting
        valid_mask = np.isfinite(mean_x_values)
        if np.any(valid_mask):
            plt.scatter(np.array(t[:n_samples])[valid_mask], np.array(mean_x_values)[valid_mask])
            plt.xlabel('T Value')
            plt.ylabel('Mean X Value (finite)')
            plt.title('T vs Mean X (finite values)')
        else:
            plt.text(0.5, 0.5, 'No valid X means found!', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('T vs Mean X - NO VALID DATA')
    else:
        plt.plot(t[:n_samples])
        plt.title('T Values (multi-target)')
        plt.xlabel('Sample Index')
        plt.ylabel('T Value')
    
    # Data quality heatmap
    plt.subplot(2, 3, 6)
    if x.ndim >= 2:
        # Show first sample as heatmap
        if x.ndim == 2:
            im_data = x[0].reshape(1, -1)
        elif x.ndim == 3:
            im_data = x[0]
        elif x.ndim == 4:
            im_data = x[0, 0]  # First channel
        
        # Handle NaN/Inf in heatmap
        finite_mask = np.isfinite(im_data)
        if np.any(finite_mask):
            # Replace NaN/Inf with 0 for visualization
            im_data_clean = im_data.copy()
            im_data_clean[~finite_mask] = 0
            plt.imshow(im_data_clean, aspect='auto', cmap='viridis')
            plt.title(f'First Sample (Heatmap)\n{finite_mask.sum()}/{im_data.size} finite')
        else:
            plt.text(0.5, 0.5, 'No finite values\nfor heatmap!', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('First Sample - NO FINITE VALUES')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_diagnostic.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Diagnostic plots saved to {output_dir}/data_diagnostic.png")


def check_model_compatibility(x, t):
    """Check if data is compatible with model requirements."""
    print(f"   Model Compatibility Check:")
    
    # Check X dimensions
    if x.ndim == 2:
        print(f"     X: 2D array -> will be reshaped to (N, 1, L)")
    elif x.ndim == 3:
        print(f"     X: 3D array -> Conv1D model")
    elif x.ndim == 4:
        print(f"     X: 4D array -> Conv2D model")
    else:
        print(f"     ⚠️  WARNING: Unsupported X dimensions: {x.ndim}")
    
    # Check T dimensions
    if t.ndim == 1:
        print(f"     T: 1D array -> single target")
    elif t.ndim == 2:
        print(f"     T: 2D array -> multi-target ({t.shape[1]} targets)")
    else:
        print(f"     ⚠️  WARNING: Unsupported T dimensions: {t.ndim}")
    
    # Check for potential issues
    if np.isnan(x).any():
        print(f"     ❌ CRITICAL: X contains NaN values!")
    if np.isnan(t).any():
        print(f"     ❌ CRITICAL: T contains NaN values!")
    if np.isinf(x).any():
        print(f"     ❌ CRITICAL: X contains Inf values!")
    if np.isinf(t).any():
        print(f"     ❌ CRITICAL: T contains Inf values!")
    
    # Check data ranges
    if np.max(np.abs(x)) > 1e6:
        print(f"     ⚠️  WARNING: X has very large values (max: {np.max(np.abs(x)):.2e})")
    if np.max(np.abs(t)) > 1e6:
        print(f"     ⚠️  WARNING: T has very large values (max: {np.max(np.abs(t)):.2e})")
    
    # Check for constant data
    if len(np.unique(x)) < 10:
        print(f"     ⚠️  WARNING: X has very few unique values ({len(np.unique(x))})")
    if len(np.unique(t)) < 10:
        print(f"     ⚠️  WARNING: T has very few unique values ({len(np.unique(t))})")


if __name__ == "__main__":
    # Example usage
    x_path = "/home/smatsubara/documents/airlift/data/sandbox/results/x_train_real.npy"
    t_path = "/home/smatsubara/documents/airlift/data/sandbox/results/t_train_real.npy"
    
    inspect_dataset(x_path, t_path, sample_limit=5, save_plots=True)
