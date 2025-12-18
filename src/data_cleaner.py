import numpy as np
import os
from typing import Tuple, Optional


def clean_dataset(x_path: str, t_path: str, 
                 x_key: str = None, t_key: str = None,
                 output_dir: str = "cleaned_data",
                 strategy: str = "nan_to_zero",
                 clip_extreme: bool = True,
                 extreme_threshold: float = 1e6) -> Tuple[str, str]:
    """
    Clean dataset by handling NaN, Inf, and extreme values.
    
    Args:
        x_path: Path to input data file
        t_path: Path to target data file  
        x_key: Key for x data if .npz
        t_key: Key for t data if .npz
        output_dir: Directory to save cleaned data
        strategy: Strategy for handling NaN/Inf ("nan_to_zero", "nan_to_mean", "drop_samples")
        clip_extreme: Whether to clip extreme values
        extreme_threshold: Threshold for extreme values
        
    Returns:
        Tuple of (cleaned_x_path, cleaned_t_path)
    """
    print("🧹 Dataset Cleaning Tool")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original data
    print("Loading original data...")
    x = load_data_robust(x_path, x_key)
    t = load_data_robust(t_path, t_key)
    
    print(f"Original X shape: {x.shape}, dtype: {x.dtype}")
    print(f"Original T shape: {t.shape}, dtype: {t.dtype}")
    
    # Analyze problematic values
    print("\nAnalyzing problematic values...")
    x_nan_count = np.isnan(x).sum()
    x_inf_count = np.isinf(x).sum()
    t_nan_count = np.isnan(t).sum()
    t_inf_count = np.isinf(t).sum()
    
    print(f"X: {x_nan_count} NaN, {x_inf_count} Inf")
    print(f"T: {t_nan_count} NaN, {t_inf_count} Inf")
    
    # Clean X data
    print(f"\nCleaning X data using strategy: {strategy}")
    x_cleaned = clean_array(x, strategy, clip_extreme, extreme_threshold)
    
    # Clean T data (usually T is already clean, but let's be safe)
    print("Cleaning T data...")
    t_cleaned = clean_array(t, strategy, clip_extreme, extreme_threshold)
    
    # Verify cleaning
    print("\nVerifying cleaned data...")
    x_nan_after = np.isnan(x_cleaned).sum()
    x_inf_after = np.isinf(x_cleaned).sum()
    t_nan_after = np.isnan(t_cleaned).sum()
    t_inf_after = np.isinf(t_cleaned).sum()
    
    print(f"X after cleaning: {x_nan_after} NaN, {x_inf_after} Inf")
    print(f"T after cleaning: {t_nan_after} NaN, {t_inf_after} Inf")
    
    # Save cleaned data
    print(f"\nSaving cleaned data to {output_dir}/")
    x_cleaned_path = os.path.join(output_dir, "x_train_real_cleaned.npy")
    t_cleaned_path = os.path.join(output_dir, "t_train_real_cleaned.npy")
    
    np.save(x_cleaned_path, x_cleaned)
    np.save(t_cleaned_path, t_cleaned)
    
    print(f"✅ Cleaned X saved to: {x_cleaned_path}")
    print(f"✅ Cleaned T saved to: {t_cleaned_path}")
    
    # Print statistics
    print(f"\nCleaned data statistics:")
    print(f"X: min={np.min(x_cleaned):.6f}, max={np.max(x_cleaned):.6f}, mean={np.mean(x_cleaned):.6f}")
    print(f"T: min={np.min(t_cleaned):.6f}, max={np.max(t_cleaned):.6f}, mean={np.mean(t_cleaned):.6f}")
    
    return x_cleaned_path, t_cleaned_path


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


def clean_array(data: np.ndarray, strategy: str, clip_extreme: bool, extreme_threshold: float) -> np.ndarray:
    """Clean a single array by handling NaN, Inf, and extreme values."""
    data_cleaned = data.copy()
    
    # Handle extreme values first
    if clip_extreme:
        extreme_mask = np.abs(data_cleaned) > extreme_threshold
        if extreme_mask.any():
            print(f"   Clipping {extreme_mask.sum()} extreme values (>{extreme_threshold})")
            data_cleaned = np.clip(data_cleaned, -extreme_threshold, extreme_threshold)
    
    # Handle NaN and Inf values
    if strategy == "nan_to_zero":
        nan_inf_mask = ~np.isfinite(data_cleaned)
        if nan_inf_mask.any():
            print(f"   Replacing {nan_inf_mask.sum()} NaN/Inf values with 0")
            data_cleaned[nan_inf_mask] = 0.0
                
    elif strategy == "nan_to_mean":
        # Calculate mean of finite values
        finite_mask = np.isfinite(data_cleaned)
        if finite_mask.any():
            finite_mean = np.mean(data_cleaned[finite_mask])
            nan_inf_mask = ~finite_mask
            if nan_inf_mask.any():
                print(f"   Replacing {nan_inf_mask.sum()} NaN/Inf values with mean ({finite_mean:.6f})")
                data_cleaned[nan_inf_mask] = finite_mean
        else:
            print("   Warning: No finite values found, using 0")
            data_cleaned[~np.isfinite(data_cleaned)] = 0.0
            
    elif strategy == "drop_samples":
        # For multi-dimensional data, drop samples with any NaN/Inf
        if data_cleaned.ndim > 1:
            # Check each sample (first dimension)
            valid_samples = []
            for i in range(data_cleaned.shape[0]):
                sample = data_cleaned[i]
                if np.isfinite(sample).all():
                    valid_samples.append(i)
            
            if len(valid_samples) < data_cleaned.shape[0]:
                print(f"   Dropping {data_cleaned.shape[0] - len(valid_samples)} samples with NaN/Inf")
                data_cleaned = data_cleaned[valid_samples]
        else:
            # For 1D data, just replace NaN/Inf
            nan_inf_mask = ~np.isfinite(data_cleaned)
            if nan_inf_mask.any():
                print(f"   Replacing {nan_inf_mask.sum()} NaN/Inf values with 0")
                data_cleaned[nan_inf_mask] = 0.0
    
    return data_cleaned


def create_cleaning_report(original_x_path: str, original_t_path: str,
                          cleaned_x_path: str, cleaned_t_path: str,
                          output_dir: str = "cleaned_data"):
    """Create a detailed report comparing original and cleaned data."""
    print("\n📊 Creating cleaning report...")
    
    # Load both datasets
    x_orig = np.load(original_x_path)
    t_orig = np.load(original_t_path)
    x_clean = np.load(cleaned_x_path)
    t_clean = np.load(cleaned_t_path)
    
    report_path = os.path.join(output_dir, "cleaning_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("Dataset Cleaning Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("ORIGINAL DATA:\n")
        f.write(f"X shape: {x_orig.shape}, dtype: {x_orig.dtype}\n")
        f.write(f"T shape: {t_orig.shape}, dtype: {t_orig.dtype}\n")
        f.write(f"X NaN count: {np.isnan(x_orig).sum()}\n")
        f.write(f"X Inf count: {np.isinf(x_orig).sum()}\n")
        f.write(f"T NaN count: {np.isnan(t_orig).sum()}\n")
        f.write(f"T Inf count: {np.isinf(t_orig).sum()}\n\n")
        
        f.write("CLEANED DATA:\n")
        f.write(f"X shape: {x_clean.shape}, dtype: {x_clean.dtype}\n")
        f.write(f"T shape: {t_clean.shape}, dtype: {t_clean.dtype}\n")
        f.write(f"X NaN count: {np.isnan(x_clean).sum()}\n")
        f.write(f"X Inf count: {np.isinf(x_clean).sum()}\n")
        f.write(f"T NaN count: {np.isnan(t_clean).sum()}\n")
        f.write(f"T Inf count: {np.isinf(t_clean).sum()}\n\n")
        
        f.write("STATISTICS COMPARISON:\n")
        f.write("X Original - Min: {:.6f}, Max: {:.6f}, Mean: {:.6f}, Std: {:.6f}\n".format(
            np.nanmin(x_orig), np.nanmax(x_orig), np.nanmean(x_orig), np.nanstd(x_orig)))
        f.write("X Cleaned  - Min: {:.6f}, Max: {:.6f}, Mean: {:.6f}, Std: {:.6f}\n".format(
            np.min(x_clean), np.max(x_clean), np.mean(x_clean), np.std(x_clean)))
        f.write("T Original - Min: {:.6f}, Max: {:.6f}, Mean: {:.6f}, Std: {:.6f}\n".format(
            np.min(t_orig), np.max(t_orig), np.mean(t_orig), np.std(t_orig)))
        f.write("T Cleaned  - Min: {:.6f}, Max: {:.6f}, Mean: {:.6f}, Std: {:.6f}\n".format(
            np.min(t_clean), np.max(t_clean), np.mean(t_clean), np.std(t_clean)))
    
    print(f"✅ Cleaning report saved to: {report_path}")


if __name__ == "__main__":
    # Example usage
    x_path = "/home/smatsubara/documents/airlift/data/sandbox/results/x_train_real.npy"
    t_path = "/home/smatsubara/documents/airlift/data/sandbox/results/t_train_real.npy"
    
    # Clean the dataset
    x_cleaned_path, t_cleaned_path = clean_dataset(
        x_path=x_path,
        t_path=t_path,
        strategy="nan_to_zero",  # Replace NaN/Inf with 0
        clip_extreme=True,       # Clip extreme values
        extreme_threshold=1e6    # Threshold for extreme values
    )
    
    # Create cleaning report
    create_cleaning_report(x_path, t_path, x_cleaned_path, t_cleaned_path)
    
    print("\n🎯 Next Steps:")
    print("1. Use the cleaned data files for training")
    print("2. Update your config files to point to cleaned data")
    print("3. Test training with the cleaned dataset")







