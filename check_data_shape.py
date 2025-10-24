#!/usr/bin/env python3
"""
Check the actual shape of the data files.
"""

import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_data_shape():
    """Check the actual shape of the data files."""
    
    try:
        # Load data files
        x_train_real = np.load("/home/smatsubara/documents/airlift/data/sandbox/results/x_train_real.npy")
        t_train_real = np.load("/home/smatsubara/documents/airlift/data/sandbox/results/t_train_real.npy")
        
        logger.info(f"X data shape: {x_train_real.shape}")
        logger.info(f"T data shape: {t_train_real.shape}")
        logger.info(f"X data dtype: {x_train_real.dtype}")
        logger.info(f"T data dtype: {t_train_real.dtype}")
        
        # Check if we can access individual samples
        if x_train_real.ndim > 1:
            logger.info(f"First sample shape: {x_train_real[0].shape}")
            logger.info(f"First sample dtype: {x_train_real[0].dtype}")
        
        if t_train_real.ndim > 1:
            logger.info(f"First target shape: {t_train_real[0].shape}")
            logger.info(f"First target dtype: {t_train_real[0].dtype}")
        
        # Check memory usage
        x_size_mb = x_train_real.nbytes / (1024 * 1024)
        t_size_mb = t_train_real.nbytes / (1024 * 1024)
        logger.info(f"X data size: {x_size_mb:.2f} MB")
        logger.info(f"T data size: {t_size_mb:.2f} MB")
        
        return x_train_real.shape, t_train_real.shape
        
    except Exception as e:
        logger.error(f"Error checking data shape: {e}")
        raise

if __name__ == "__main__":
    check_data_shape()
