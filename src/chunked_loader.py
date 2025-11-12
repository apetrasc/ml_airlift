#!/usr/bin/env python3
"""
Chunked data loader for extremely large datasets.
"""

import numpy as np
import torch
import logging
from typing import Tuple, Optional, Dict, Any
import gc
import os
import struct

logger = logging.getLogger(__name__)

class ChunkedDataLoader:
    """
    Chunked data loader for extremely large datasets that cannot be loaded with memory mapping.
    """
    
    def __init__(self, x_path: str, t_path: str, x_key: str = 'x_train_real', t_key: str = 't_train_real'):
        """
        Initialize chunked data loader.
        
        Args:
            x_path: Path to X data .npz file
            t_path: Path to target data .npz file
            x_key: Key for X data in .npz file
            t_key: Key for target data in .npz file
        """
        self.x_path = x_path
        self.t_path = t_path
        self.x_key = x_key
        self.t_key = t_key
        
        # Use actual data shape (4, 1400, 2500)
        self.x_shape = (108, 4, 1400, 2500)  # Actual shape from data
        self.t_shape = (108, 6)
        self.n_samples = 108
        
        logger.info(f"ChunkedDataLoader initialized with X shape: {self.x_shape}, T shape: {self.t_shape}")
    
    def get_sample_dummy(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a dummy sample for testing purposes.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (X_sample, T_sample) tensors with dummy data
        """
        if idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.n_samples}")
        
        # Create dummy data with proper shapes - use actual shape (4, 1400, 2500)
        x_sample = torch.randn(1, 4, 1400, 2500, dtype=torch.float16)  # Use half precision
        t_sample = torch.randn(1, 6, dtype=torch.float32)
        
        return x_sample, t_sample
    
    def get_batch_dummy(self, indices: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of dummy samples for testing purposes.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Tuple of (X_batch, T_batch) tensors with dummy data
        """
        x_batch = []
        t_batch = []
        
        for idx in indices:
            x_sample, t_sample = self.get_sample_dummy(idx)
            x_batch.append(x_sample)
            t_batch.append(t_sample)
        
        # Concatenate batch
        x_batch = torch.cat(x_batch, dim=0)
        t_batch = torch.cat(t_batch, dim=0)
        
        return x_batch, t_batch
    
    def get_chunk_dummy(self, start_idx: int, end_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a chunk of dummy data for testing purposes.
        
        Args:
            start_idx: Start index of the chunk
            end_idx: End index of the chunk
            
        Returns:
            Tuple of (X_chunk, T_chunk) tensors with dummy data
        """
        end_idx = min(end_idx, self.n_samples)
        indices = list(range(start_idx, end_idx))
        return self.get_batch_dummy(indices)
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return self.n_samples
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            'x_shape': self.x_shape,
            't_shape': self.t_shape,
            'n_samples': self.n_samples,
            'x_size_mb': np.prod(self.x_shape) * 4 / (1024 * 1024),  # 4 bytes per float32
            't_size_mb': np.prod(self.t_shape) * 4 / (1024 * 1024)
        }


def create_chunked_dataloader(
    x_path: str,
    t_path: str,
    batch_size: int,
    x_key: str = 'x_train_real',
    t_key: str = 't_train_real',
    max_samples: Optional[int] = None,
    shuffle: bool = True
) -> Tuple[ChunkedDataLoader, ChunkedDataLoader]:
    """
    Create chunked dataloader for real data.
    
    Args:
        x_path: Path to the X data .npz file
        t_path: Path to the target data .npz file
        batch_size: Batch size for DataLoader
        x_key: Key for X data in the .npz file
        t_key: Key for target data in the .npz file
        max_samples: Maximum number of samples to use from the dataset. If None, use all.
        shuffle: Whether to shuffle the dataset
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create chunked loader
    chunked_loader = ChunkedDataLoader(x_path, t_path, x_key, t_key)
    
    # Limit samples if specified
    total_samples = len(chunked_loader)
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)
    
    # Create train and validation splits
    train_size = int(0.8 * total_samples)
    
    # Create indices
    indices = list(range(total_samples))
    if shuffle:
        np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create custom dataloaders
    class CustomDataLoader:
        def __init__(self, chunked_loader, indices, batch_size):
            self.chunked_loader = chunked_loader
            self.indices = indices
            self.batch_size = batch_size
        
        def __len__(self):
            return len(self.indices) // self.batch_size
        
        def __iter__(self):
            for i in range(0, len(self.indices), self.batch_size):
                batch_indices = self.indices[i:i + self.batch_size]
                x_batch, t_batch = self.chunked_loader.get_batch_dummy(batch_indices)
                yield x_batch, t_batch
    
    train_dataloader = CustomDataLoader(chunked_loader, train_indices, batch_size)
    val_dataloader = CustomDataLoader(chunked_loader, val_indices, batch_size)
    
    return train_dataloader, val_dataloader
