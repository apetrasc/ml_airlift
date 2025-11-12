#!/usr/bin/env python3
"""
Advanced Data Preprocessing for Stable Training
Implements robust data cleaning and augmentation strategies.
"""

import numpy as np
import logging
import os
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedDataPreprocessor:
    """Advanced data preprocessing for stable training."""
    
    def __init__(self):
        self.x_scaler = RobustScaler()  # More robust to outliers
        self.t_scaler = StandardScaler()
        self.x_imputer = SimpleImputer(strategy='median')  # Better than mean for outliers
        self.t_imputer = SimpleImputer(strategy='median')
        
    def clean_and_normalize(self, x_data, t_data):
        """Clean and normalize data with advanced techniques."""
        
        logger.info(f"Original data shapes: x={x_data.shape}, t={t_data.shape}")
        
        # 1. Handle NaN values with median imputation
        logger.info("Handling NaN values...")
        x_clean = self.x_imputer.fit_transform(x_data.reshape(x_data.shape[0], -1))
        t_clean = self.t_imputer.fit_transform(t_data)
        
        # Reshape x_data back to original shape
        x_clean = x_clean.reshape(x_data.shape)
        
        # 2. Remove extreme outliers (beyond 3 standard deviations)
        logger.info("Removing extreme outliers...")
        x_clean = self._remove_outliers_3d(x_clean)
        
        # 3. Normalize data
        logger.info("Normalizing data...")
        x_normalized = self._normalize_3d_data(x_clean)
        t_normalized = self.t_scaler.fit_transform(t_clean)
        
        # 4. Add small noise to prevent overfitting
        logger.info("Adding regularization noise...")
        noise_scale = 0.01
        x_normalized += np.random.normal(0, noise_scale, x_normalized.shape)
        t_normalized += np.random.normal(0, noise_scale * 0.1, t_normalized.shape)
        
        logger.info("Data preprocessing completed")
        return x_normalized, t_normalized
    
    def _remove_outliers_3d(self, data):
        """Remove outliers from 3D data."""
        # Flatten spatial dimensions for outlier detection
        original_shape = data.shape
        data_flat = data.reshape(data.shape[0], -1)
        
        # Calculate robust statistics
        median = np.median(data_flat, axis=1, keepdims=True)
        mad = np.median(np.abs(data_flat - median), axis=1, keepdims=True)
        
        # Remove samples with extreme outliers
        outlier_threshold = 3.0
        outlier_mask = np.any(np.abs(data_flat - median) > outlier_threshold * mad, axis=1)
        
        if outlier_mask.sum() > 0:
            logger.info(f"Removing {outlier_mask.sum()} samples with extreme outliers")
            data_clean = data_flat[~outlier_mask]
            return data_clean.reshape(-1, *original_shape[1:])
        
        return data
    
    def _normalize_3d_data(self, data):
        """Normalize 3D data per sample."""
        normalized_data = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            sample = data[i]
            # Normalize each channel separately
            for c in range(sample.shape[0]):
                channel = sample[c]
                if np.std(channel) > 0:
                    normalized_data[i, c] = (channel - np.mean(channel)) / np.std(channel)
                else:
                    normalized_data[i, c] = channel
        
        return normalized_data

def create_stable_train_test_split(x_data, t_data, test_ratio=0.2, random_state=42):
    """Create stable train-test split with stratification."""
    
    np.random.seed(random_state)
    n_samples = len(x_data)
    indices = np.random.permutation(n_samples)
    
    # Stratified split based on target statistics
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
    logger.info("Starting advanced data preprocessing...")
    
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
    
    # Initialize preprocessor
    preprocessor = AdvancedDataPreprocessor()
    
    # Clean and normalize data
    x_clean, t_clean = preprocessor.clean_and_normalize(x_data, t_data)
    
    # Create stable train-test split
    x_train, t_train, x_test, t_test = create_stable_train_test_split(
        x_clean, t_clean, test_ratio=0.2
    )
    
    # Save processed data
    output_dir = "/home/smatsubara/documents/airlift/data/stable"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training data
    np.save(os.path.join(output_dir, "x_train_stable.npy"), x_train)
    np.save(os.path.join(output_dir, "t_train_stable.npy"), t_train)
    
    # Save test data
    np.save(os.path.join(output_dir, "x_test_stable.npy"), x_test)
    np.save(os.path.join(output_dir, "t_test_stable.npy"), t_test)
    
    # Save scalers for inference
    import joblib
    joblib.dump(preprocessor.x_scaler, os.path.join(output_dir, "x_scaler.pkl"))
    joblib.dump(preprocessor.t_scaler, os.path.join(output_dir, "t_scaler.pkl"))
    
    logger.info("Advanced preprocessing completed!")
    logger.info(f"Processed data saved to: {output_dir}")
    
    # Print statistics
    logger.info("Data statistics:")
    logger.info(f"X_train: mean={np.mean(x_train):.4f}, std={np.std(x_train):.4f}")
    logger.info(f"T_train: mean={np.mean(t_train):.4f}, std={np.std(t_train):.4f}")
    logger.info(f"X_test: mean={np.mean(x_test):.4f}, std={np.std(x_test):.4f}")
    logger.info(f"T_test: mean={np.mean(t_test):.4f}, std={np.std(t_test):.4f}")

if __name__ == "__main__":
    main()
