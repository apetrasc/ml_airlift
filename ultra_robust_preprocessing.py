#!/usr/bin/env python3
"""
Ultra-Robust Data Preprocessing
Handles extreme data quality issues including infinity values.
"""

import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ultra_robust_preprocessing(x_data, t_data):
    """Ultra-robust data preprocessing."""
    
    logger.info(f"Original data shapes: x={x_data.shape}, t={t_data.shape}")
    
    # 1. Handle infinity and extreme values
    logger.info("Handling infinity and extreme values...")
    
    # Replace infinity with NaN
    x_data = np.where(np.isfinite(x_data), x_data, np.nan)
    t_data = np.where(np.isfinite(t_data), t_data, np.nan)
    
    # Replace extremely large values
    x_data = np.where(np.abs(x_data) < 1e10, x_data, np.nan)
    t_data = np.where(np.abs(t_data) < 1e10, t_data, np.nan)
    
    # 2. Handle NaN values with simple replacement
    logger.info("Handling NaN values...")
    
    # For x_data, replace NaN with 0
    x_clean = np.nan_to_num(x_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # For t_data, replace NaN with median
    t_median = np.nanmedian(t_data)
    t_clean = np.nan_to_num(t_data, nan=t_median, posinf=t_median, neginf=t_median)
    
    # 3. Remove samples that are all zeros (likely corrupted)
    logger.info("Removing corrupted samples...")
    
    # Check for samples that are mostly zeros
    x_flat = x_clean.reshape(x_clean.shape[0], -1)
    zero_ratio = np.sum(x_flat == 0, axis=1) / x_flat.shape[1]
    
    # Keep samples with less than 90% zeros
    valid_samples = zero_ratio < 0.9
    x_clean = x_clean[valid_samples]
    t_clean = t_clean[valid_samples]
    
    logger.info(f"Kept {len(x_clean)} samples after removing corrupted ones")
    
    # 4. Simple normalization per sample
    logger.info("Normalizing data...")
    x_normalized = np.zeros_like(x_clean)
    
    for i in range(len(x_clean)):
        sample = x_clean[i]
        for c in range(sample.shape[0]):
            channel = sample[c]
            if np.std(channel) > 1e-8:  # Avoid division by very small numbers
                x_normalized[i, c] = (channel - np.mean(channel)) / np.std(channel)
            else:
                x_normalized[i, c] = channel
    
    # Normalize targets
    if np.std(t_clean) > 1e-8:
        t_normalized = (t_clean - np.mean(t_clean)) / np.std(t_clean)
    else:
        t_normalized = t_clean
    
    # 5. Add small regularization noise
    logger.info("Adding regularization noise...")
    noise_scale = 0.01
    x_normalized += np.random.normal(0, noise_scale, x_normalized.shape)
    t_normalized += np.random.normal(0, noise_scale * 0.1, t_normalized.shape)
    
    logger.info("Ultra-robust preprocessing completed")
    return x_normalized, t_normalized

def create_stable_split(x_data, t_data, test_ratio=0.2, random_state=42):
    """Create stable train-test split."""
    
    np.random.seed(random_state)
    n_samples = len(x_data)
    indices = np.random.permutation(n_samples)
    
    test_size = int(n_samples * test_ratio)
    train_indices = indices[:-test_size]
    test_indices = indices[-test_size:]
    
    x_train = x_data[train_indices]
    t_train = t_data[train_indices]
    x_test = x_data[test_indices]
    t_test = t_data[test_indices]
    
    logger.info(f"Stable split: train={len(x_train)}, test={len(x_test)}")
    
    return x_train, t_train, x_test, t_test

def main():
    """Main preprocessing function."""
    logger.info("Starting ultra-robust data preprocessing...")
    
    # Load original data
    x_path = "/home/smatsubara/documents/airlift/data/sandbox/results/x_train_real.npy"
    t_path = "/home/smatsubara/documents/airlift/data/sandbox/results/t_train_real.npy"
    
    try:
        x_data = np.load(x_path)
        t_data = np.load(t_path)
        logger.info(f"Loaded data: x={x_data.shape}, t={t_data.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Ultra-robust preprocessing
    x_clean, t_clean = ultra_robust_preprocessing(x_data, t_data)
    
    # Create stable split
    x_train, t_train, x_test, t_test = create_stable_split(x_clean, t_clean)
    
    # Save processed data
    output_dir = "/home/smatsubara/documents/airlift/data/stable"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training data
    np.save(os.path.join(output_dir, "x_train_stable.npy"), x_train)
    np.save(os.path.join(output_dir, "t_train_stable.npy"), t_train)
    
    # Save test data
    np.save(os.path.join(output_dir, "x_test_stable.npy"), x_test)
    np.save(os.path.join(output_dir, "t_test_stable.npy"), t_test)
    
    logger.info("Ultra-robust preprocessing completed!")
    logger.info(f"Processed data saved to: {output_dir}")
    
    # Print statistics
    logger.info("Data statistics:")
    logger.info(f"X_train: mean={np.mean(x_train):.4f}, std={np.std(x_train):.4f}")
    logger.info(f"T_train: mean={np.mean(t_train):.4f}, std={np.std(t_train):.4f}")
    logger.info(f"X_test: mean={np.mean(x_test):.4f}, std={np.std(x_test):.4f}")
    logger.info(f"T_test: mean={np.mean(t_test):.4f}, std={np.std(t_test):.4f}")
    
    # Check for any remaining issues
    logger.info("Final data quality check:")
    logger.info(f"X_train NaN count: {np.isnan(x_train).sum()}")
    logger.info(f"T_train NaN count: {np.isnan(t_train).sum()}")
    logger.info(f"X_train Inf count: {np.isinf(x_train).sum()}")
    logger.info(f"T_train Inf count: {np.isinf(t_train).sum()}")

if __name__ == "__main__":
    main()
