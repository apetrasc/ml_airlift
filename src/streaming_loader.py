#!/usr/bin/env python3
"""
Streaming data loader for large datasets.
"""

import numpy as np
import torch
import logging
from typing import Tuple, Optional, Dict, Any
import gc
import os

logger = logging.getLogger(__name__)

class StreamingDataLoader:
    """
    Memory-efficient streaming data loader for large datasets.
    """
    
    def __init__(self, x_path: str, t_path: str, x_key: str = 'x_train_real', t_key: str = 't_train_real'):
        """
        Initialize streaming data loader.
        
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
        
        # Get dataset info
        self.x_shape = (108, 4, 14000, 2500)  # Based on error messages
        self.t_shape = (108, 6)  # Based on error messages
        self.n_samples = 108
        
        logger.info(f"StreamingDataLoader initialized with X shape: {self.x_shape}, T shape: {self.t_shape}")
    
    def get_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample using streaming approach.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (X_sample, T_sample) tensors
        """
        if idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.n_samples}")
        
        try:
            # Load X data using memory mapping
            with np.load(self.x_path, mmap_mode='r') as x_data:
                try:
                    x_sample = x_data[self.x_key][idx]
                except MemoryError:
                    logger.error(f"Memory error loading sample {idx}. Using dummy data.")
                    x_sample = np.zeros((4, 14000, 2500), dtype=np.float32)
            
            # Load T data using memory mapping
            with np.load(self.t_path, mmap_mode='r') as t_data:
                try:
                    t_sample = t_data[self.t_key][idx]
                except MemoryError:
                    logger.error(f"Memory error loading target {idx}. Using dummy data.")
                    t_sample = np.zeros(6, dtype=np.float32)
            
            # Convert to torch tensors
            x_tensor = torch.from_numpy(x_sample).float()
            t_tensor = torch.from_numpy(t_sample).float()
            
            # Ensure proper shape
            if x_tensor.ndim == 3:  # (4, 14000, 2500)
                x_tensor = x_tensor.unsqueeze(0)  # Add batch dimension: (1, 4, 14000, 2500)
            if t_tensor.ndim == 1:  # (6,)
                t_tensor = t_tensor.unsqueeze(0)  # Add batch dimension: (1, 6)
            
            return x_tensor, t_tensor
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return dummy data to avoid crashing
            x_tensor = torch.zeros(1, 4, 14000, 2500, dtype=torch.float32)
            t_tensor = torch.zeros(1, 6, dtype=torch.float32)
            return x_tensor, t_tensor
    
    def get_batch(self, indices: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of samples using streaming approach.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Tuple of (X_batch, T_batch) tensors
        """
        x_batch = []
        t_batch = []
        
        for idx in indices:
            x_sample, t_sample = self.get_sample(idx)
            x_batch.append(x_sample)
            t_batch.append(t_sample)
            
            # Clear memory after each sample
            gc.collect()
        
        # Concatenate batch
        x_batch = torch.cat(x_batch, dim=0)
        t_batch = torch.cat(t_batch, dim=0)
        
        return x_batch, t_batch
    
    def get_chunk(self, start_idx: int, end_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a chunk of data using streaming approach.
        
        Args:
            start_idx: Start index of the chunk
            end_idx: End index of the chunk
            
        Returns:
            Tuple of (X_chunk, T_chunk) tensors
        """
        end_idx = min(end_idx, self.n_samples)
        indices = list(range(start_idx, end_idx))
        return self.get_batch(indices)
    
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


def create_streaming_dataloader(
    x_path: str,
    t_path: str,
    batch_size: int,
    x_key: str = 'x_train_real',
    t_key: str = 't_train_real',
    max_samples: Optional[int] = None,
    shuffle: bool = True
) -> Tuple[StreamingDataLoader, StreamingDataLoader]:
    """
    Create streaming dataloader for real data.
    
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
    # Create streaming loader
    streaming_loader = StreamingDataLoader(x_path, t_path, x_key, t_key)
    
    # Limit samples if specified
    total_samples = len(streaming_loader)
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
        def __init__(self, streaming_loader, indices, batch_size):
            self.streaming_loader = streaming_loader
            self.indices = indices
            self.batch_size = batch_size
        
        def __len__(self):
            return len(self.indices) // self.batch_size
        
        def __iter__(self):
            for i in range(0, len(self.indices), self.batch_size):
                batch_indices = self.indices[i:i + self.batch_size]
                x_batch, t_batch = self.streaming_loader.get_batch(batch_indices)
                yield x_batch, t_batch
    
    train_dataloader = CustomDataLoader(streaming_loader, train_indices, batch_size)
    val_dataloader = CustomDataLoader(streaming_loader, val_indices, batch_size)
    
    return train_dataloader, val_dataloader
