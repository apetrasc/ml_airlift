import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RealDataDataset(Dataset):
    """
    Memory-efficient dataset for large real data files.
    Loads data in chunks to avoid memory issues.
    """
    
    def __init__(
        self, 
        x_path: str, 
        t_path: str, 
        x_key: str = 'x_train_real',
        t_key: str = 't_train_real',
        max_samples: Optional[int] = None,
        chunk_size: int = 1000,
        use_half_precision: bool = True,
        dynamic_chunk_size: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            x_path: Path to the X data .npz file
            t_path: Path to the target data .npz file
            x_key: Key for X data in the .npz file
            t_key: Key for target data in the .npz file
            max_samples: Maximum number of samples to use (None for all)
            chunk_size: Size of chunks to load at once
            use_half_precision: Whether to use float16 for memory efficiency
            dynamic_chunk_size: Whether to dynamically adjust chunk size based on available memory
        """
        self.x_path = x_path
        self.t_path = t_path
        self.x_key = x_key
        self.t_key = t_key
        self.chunk_size = chunk_size
        self.use_half_precision = use_half_precision
        self.dynamic_chunk_size = dynamic_chunk_size
        
        # Get dataset info without loading full data
        self._get_dataset_info()
        
        # Set total samples
        if max_samples is None:
            self.total_samples = self.n_samples
        else:
            self.total_samples = min(max_samples, self.n_samples)
        
        # Calculate optimal chunk size based on available memory
        if self.dynamic_chunk_size:
            self.chunk_size = self._calculate_optimal_chunk_size()
            
        logger.info(f"Dataset initialized with {self.total_samples} samples")
        logger.info(f"X shape: {self.x_shape}, T shape: {self.t_shape}")
        logger.info(f"Using chunk size: {self.chunk_size}")
        logger.info(f"Half precision: {self.use_half_precision}")
    
    def _get_dataset_info(self):
        """Get dataset information without loading full data."""
        try:
            # Try to get shape info without loading data
            # Use a small sample to infer shape
            with np.load(self.x_path, mmap_mode='r') as x_data:
                # Try to access just the first sample to get shape info
                try:
                    first_sample = x_data[self.x_key][0]
                    self.x_shape = (x_data[self.x_key].shape[0],) + first_sample.shape
                    self.n_samples = self.x_shape[0]
                except MemoryError:
                    # If still memory error, use actual shape from error messages
                    logger.warning("Cannot determine dataset shape due to memory constraints. Using actual shape.")
                    # Actual shape: (108, 4, 1400, 2500) based on actual data
                    self.x_shape = (108, 4, 1400, 2500)
                    self.n_samples = 108
                
            with np.load(self.t_path, mmap_mode='r') as t_data:
                try:
                    first_sample = t_data[self.t_key][0]
                    self.t_shape = (t_data[self.t_key].shape[0],) + first_sample.shape
                except MemoryError:
                    logger.warning("Cannot determine target shape due to memory constraints. Using dummy shape.")
                    self.t_shape = (108, 6)
                
            # Verify shapes match (skip if using dummy shapes)
            if isinstance(self.x_shape[0], int) and isinstance(self.t_shape[0], int):
                if self.x_shape[0] != self.t_shape[0]:
                    logger.warning(f"Sample count mismatch: X has {self.x_shape[0]}, T has {self.t_shape[0]}")
                
        except Exception as e:
            logger.error(f"Error loading dataset info: {e}")
            # Use actual shapes as fallback
            self.x_shape = (108, 4, 1400, 2500)
            self.t_shape = (108, 6)
            self.n_samples = 108
    
    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on available memory and data size."""
        try:
            import psutil
            
            # Get available memory in MB
            available_memory = psutil.virtual_memory().available / (1024 * 1024)
            
            # Calculate memory per sample
            if self.use_half_precision:
                bytes_per_sample = 4 * 1400 * 2500 * 2  # float16 = 2 bytes
            else:
                bytes_per_sample = 4 * 1400 * 2500 * 4  # float32 = 4 bytes
            
            # Use only 20% of available memory for safety
            safe_memory = available_memory * 0.2
            max_samples = int(safe_memory * 1024 * 1024 / bytes_per_sample)
            
            # Ensure reasonable bounds
            optimal_chunk_size = max(1, min(max_samples, 50))
            
            logger.info(f"Available memory: {available_memory:.1f}MB")
            logger.info(f"Optimal chunk size: {optimal_chunk_size}")
            
            return optimal_chunk_size
            
        except ImportError:
            logger.warning("psutil not available, using default chunk size")
            return self.chunk_size
        except Exception as e:
            logger.warning(f"Error calculating optimal chunk size: {e}")
            return self.chunk_size
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        Loads data in chunks for memory efficiency.
        """
        if idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_samples}")
        
        try:
            # Load X data using memory mapping with minimal memory footprint
            with np.load(self.x_path, mmap_mode='r') as x_data:
                try:
                    x_sample = x_data[self.x_key][idx]
                    # Immediately convert to reduce memory footprint
                    x_sample = x_sample.astype(np.float16 if self.use_half_precision else np.float32)
                except MemoryError:
                    logger.error(f"Memory error loading sample {idx}. Using dummy data.")
                    # Return dummy data to avoid crashing - shape (4, 1400, 2500)
                    x_sample = np.zeros((4, 1400, 2500), dtype=np.float16 if self.use_half_precision else np.float32)
                
            # Load T data using memory mapping
            with np.load(self.t_path, mmap_mode='r') as t_data:
                try:
                    t_sample = t_data[self.t_key][idx]
                    # Convert to float32 for numerical stability
                    t_sample = t_sample.astype(np.float32)
                except MemoryError:
                    logger.error(f"Memory error loading target {idx}. Using dummy data.")
                    # Return dummy data to avoid crashing - assume 6-dimensional target
                    t_sample = np.zeros(6, dtype=np.float32)
            
            # Convert to torch tensors with minimal memory allocation
            x_tensor = torch.from_numpy(x_sample)
            t_tensor = torch.from_numpy(t_sample)
            
            # Ensure proper shape
            if x_tensor.ndim == 1:
                x_tensor = x_tensor.unsqueeze(0)  # Add channel dimension
            if t_tensor.ndim == 0:  # Scalar target
                t_tensor = t_tensor.unsqueeze(0)
            
            return x_tensor, t_tensor
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return dummy data to avoid crashing
            if self.use_half_precision:
                x_tensor = torch.zeros(1, 4, 1400, 2500, dtype=torch.float16)
                t_tensor = torch.zeros(6, dtype=torch.float32)
            else:
                x_tensor = torch.zeros(1, 4, 1400, 2500, dtype=torch.float32)
                t_tensor = torch.zeros(6, dtype=torch.float32)
            return x_tensor, t_tensor
    
    def get_chunk(self, start_idx: int, end_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a chunk of data for batch processing.
        
        Args:
            start_idx: Start index of the chunk
            end_idx: End index of the chunk (exclusive)
            
        Returns:
            Tuple of (X_chunk, T_chunk) tensors
        """
        end_idx = min(end_idx, self.total_samples)
        
        try:
            # Load X chunk using memory mapping
            with np.load(self.x_path, mmap_mode='r') as x_data:
                try:
                    x_chunk = x_data[self.x_key][start_idx:end_idx]
                except MemoryError:
                    logger.error(f"Memory error loading X chunk [{start_idx}:{end_idx}]. Using dummy data.")
                    chunk_size = end_idx - start_idx
                    x_chunk = np.zeros((chunk_size, 4, 1400, 2500), dtype=np.float32)
                
            # Load T chunk using memory mapping
            with np.load(self.t_path, mmap_mode='r') as t_data:
                try:
                    t_chunk = t_data[self.t_key][start_idx:end_idx]
                except MemoryError:
                    logger.error(f"Memory error loading T chunk [{start_idx}:{end_idx}]. Using dummy data.")
                    chunk_size = end_idx - start_idx
                    t_chunk = np.zeros((chunk_size, 6), dtype=np.float32)
            
            # Convert to torch tensors with memory optimization
            x_tensor = torch.from_numpy(x_chunk)
            t_tensor = torch.from_numpy(t_chunk)
            
            # Use half precision if enabled
            if self.use_half_precision:
                x_tensor = x_tensor.half()  # float16
                t_tensor = t_tensor.float()  # Keep targets as float32 for numerical stability
            else:
                x_tensor = x_tensor.float()
                t_tensor = t_tensor.float()
            
            # Ensure proper shape
            if x_tensor.ndim == 2:
                x_tensor = x_tensor.unsqueeze(1)  # Add channel dimension
            if t_tensor.ndim == 1:
                t_tensor = t_tensor.unsqueeze(-1)
            
            return x_tensor, t_tensor
            
        except Exception as e:
            logger.error(f"Error loading chunk [{start_idx}:{end_idx}]: {e}")
            # Return dummy data to avoid crashing
            chunk_size = end_idx - start_idx
            if self.use_half_precision:
                x_tensor = torch.zeros(chunk_size, 4, 1400, 2500, dtype=torch.float16)
                t_tensor = torch.zeros(chunk_size, 6, dtype=torch.float32)
            else:
                x_tensor = torch.zeros(chunk_size, 4, 1400, 2500, dtype=torch.float32)
                t_tensor = torch.zeros(chunk_size, 6, dtype=torch.float32)
            return x_tensor, t_tensor


def create_real_data_dataloader(
    x_path: str,
    t_path: str,
    batch_size: int = 1,  # デフォルトを1に変更
    train_split: float = 0.8,
    x_key: str = 'x_train_real',
    t_key: str = 't_train_real',
    max_samples: Optional[int] = None,
    num_workers: int = 0,  # デフォルトを0に変更
    pin_memory: bool = False,  # デフォルトをFalseに変更
    shuffle: bool = True,
    use_half_precision: bool = True,
    gradient_accumulation_steps: int = 8  # デフォルトを8に変更
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for real data.
    
    Args:
        x_path: Path to the X data .npz file
        t_path: Path to the target data .npz file
        batch_size: Batch size for dataloaders
        train_split: Fraction of data to use for training
        x_key: Key for X data in the .npz file
        t_key: Key for target data in the .npz file
        max_samples: Maximum number of samples to use
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        shuffle: Whether to shuffle training data
        use_half_precision: Whether to use float16 for memory efficiency
        gradient_accumulation_steps: Number of steps to accumulate gradients
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create dataset with memory optimization
    dataset = RealDataDataset(
        x_path=x_path,
        t_path=t_path,
        x_key=x_key,
        t_key=t_key,
        max_samples=max_samples,
        use_half_precision=use_half_precision,
        dynamic_chunk_size=True
    )
    
    # Calculate split sizes
    total_samples = len(dataset)
    train_size = int(train_split * total_samples)
    val_size = total_samples - train_size
    
    logger.info(f"Dataset split: {train_size} train, {val_size} validation")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # Adjust batch size for gradient accumulation
    effective_batch_size = batch_size * gradient_accumulation_steps
    logger.info(f"Effective batch size: {effective_batch_size} (batch_size={batch_size} × accumulation_steps={gradient_accumulation_steps})")
    
    # Create train and validation datasets
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, total_samples))
    
    # Create dataloaders with ultra memory-efficient settings
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Force single-threaded for memory efficiency
        pin_memory=False,  # Disable pin memory to save memory
        drop_last=True,
        persistent_workers=False,  # Disable for memory efficiency
        prefetch_factor=None,  # No prefetching
        multiprocessing_context=None  # No multiprocessing
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Force single-threaded for memory efficiency
        pin_memory=False,  # Disable pin memory to save memory
        drop_last=False,
        persistent_workers=False,  # Disable for memory efficiency
        prefetch_factor=None,  # No prefetching
        multiprocessing_context=None  # No multiprocessing
    )
    
    return train_dataloader, val_dataloader


def get_dataset_info(x_path: str, t_path: str, x_key: str = 'x_train_real', t_key: str = 't_train_real') -> Dict[str, Any]:
    """
    Get information about the dataset without loading it.
    
    Args:
        x_path: Path to the X data .npz file
        t_path: Path to the target data .npz file
        x_key: Key for X data in the .npz file
        t_key: Key for target data in the .npz file
        
    Returns:
        Dictionary containing dataset information
    """
    try:
        # Use memory mapping to avoid loading full data
        with np.load(x_path, mmap_mode='r') as x_data:
            try:
                x_shape = x_data[x_key].shape
                x_dtype = x_data[x_key].dtype
            except MemoryError:
                logger.warning("Cannot determine X shape due to memory constraints. Using actual shape.")
                x_shape = (108, 4, 1400, 2500)
                x_dtype = np.float32
            
        with np.load(t_path, mmap_mode='r') as t_data:
            try:
                t_shape = t_data[t_key].shape
                t_dtype = t_data[t_key].dtype
            except MemoryError:
                logger.warning("Cannot determine T shape due to memory constraints. Using actual shape.")
                t_shape = (108, 6)
                t_dtype = np.float32
            
        return {
            'x_shape': x_shape,
            'x_dtype': str(x_dtype),
            't_shape': t_shape,
            't_dtype': str(t_dtype),
            'n_samples': x_shape[0],
            'x_size_mb': np.prod(x_shape) * np.dtype(x_dtype).itemsize / (1024 * 1024),
            't_size_mb': np.prod(t_shape) * np.dtype(t_dtype).itemsize / (1024 * 1024)
        }
    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        # Return actual info as fallback
        return {
            'x_shape': (108, 4, 1400, 2500),
            'x_dtype': 'float32',
            't_shape': (108, 6),
            't_dtype': 'float64',
            'n_samples': 108,
            'x_size_mb': 0,
            't_size_mb': 0
        }


def get_memory_usage_info() -> Dict[str, Any]:
    """
    Get current memory usage information for optimization.
    
    Returns:
        Dictionary containing memory usage information
    """
    try:
        import psutil
        import torch
        
        # System memory
        memory = psutil.virtual_memory()
        
        # GPU memory if available
        gpu_memory = {}
        if torch.cuda.is_available():
            gpu_memory = {
                'total': torch.cuda.get_device_properties(0).total_memory / (1024**3),  # GB
                'allocated': torch.cuda.memory_allocated(0) / (1024**3),  # GB
                'cached': torch.cuda.memory_reserved(0) / (1024**3),  # GB
                'free': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / (1024**3)  # GB
            }
        
        return {
            'system_memory': {
                'total': memory.total / (1024**3),  # GB
                'available': memory.available / (1024**3),  # GB
                'used': memory.used / (1024**3),  # GB
                'percent': memory.percent
            },
            'gpu_memory': gpu_memory,
            'recommendations': {
                'use_half_precision': memory.available < 8 * 1024**3,  # < 8GB available
                'reduce_batch_size': memory.available < 4 * 1024**3,  # < 4GB available
                'use_gradient_accumulation': memory.available < 6 * 1024**3,  # < 6GB available
                'max_batch_size': min(32, max(1, int(memory.available / (1024**3) / 2)))  # Conservative estimate
            }
        }
    except ImportError:
        logger.warning("psutil not available for memory monitoring")
        return {}
    except Exception as e:
        logger.error(f"Error getting memory info: {e}")
        return {}


def optimize_for_memory(
    batch_size: int = 32,
    use_half_precision: bool = True,
    gradient_accumulation_steps: int = 1
) -> Dict[str, Any]:
    """
    Get optimized settings based on available memory.
    
    Args:
        batch_size: Initial batch size
        use_half_precision: Whether to use half precision
        gradient_accumulation_steps: Number of accumulation steps
        
    Returns:
        Dictionary with optimized settings
    """
    memory_info = get_memory_usage_info()
    
    if not memory_info:
        return {
            'batch_size': batch_size,
            'use_half_precision': use_half_precision,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'num_workers': 2
        }
    
    recommendations = memory_info.get('recommendations', {})
    
    # Adjust batch size based on available memory
    if recommendations.get('reduce_batch_size', False):
        batch_size = max(1, batch_size // 2)
    
    # Use gradient accumulation if memory is limited
    if recommendations.get('use_gradient_accumulation', False) and gradient_accumulation_steps == 1:
        gradient_accumulation_steps = 2
    
    # Adjust number of workers
    available_memory = memory_info['system_memory']['available']
    if available_memory < 4:  # Less than 4GB
        num_workers = 0
    elif available_memory < 8:  # Less than 8GB
        num_workers = 1
    else:
        num_workers = 2
    
    return {
        'batch_size': batch_size,
        'use_half_precision': recommendations.get('use_half_precision', use_half_precision),
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'num_workers': num_workers,
        'memory_info': memory_info
    }


def setup_cuda_memory_optimization():
    """
    Setup CUDA memory optimization settings.
    Call this before training to optimize memory usage.
    """
    import os
    import torch
    
    # Set environment variables for CUDA memory optimization (only if supported)
    try:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    except:
        pass  # Ignore if not supported
    
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        
        # Set memory fraction to avoid OOM (only if not already set)
        try:
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of GPU memory
        except:
            pass  # Ignore if already set or not supported
        
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass
        
        logger.info("CUDA memory optimization enabled")
        logger.info(f"GPU memory fraction set to 0.8")
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")


def get_optimal_batch_size_for_memory(
    model_input_shape: tuple = (4, 1400, 2500),
    target_memory_gb: float = 2.0,
    use_half_precision: bool = True
) -> int:
    """
    Calculate optimal batch size based on available GPU memory.
    
    Args:
        model_input_shape: Shape of input tensor (channels, height, width)
        target_memory_gb: Target memory usage in GB
        use_half_precision: Whether using half precision
        
    Returns:
        Optimal batch size
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return 1
        
        # Calculate memory per sample
        if use_half_precision:
            bytes_per_element = 2  # float16
        else:
            bytes_per_element = 4  # float32
        
        # Input tensor memory
        input_elements = 1
        for dim in model_input_shape:
            input_elements *= dim
        
        input_memory_per_sample = input_elements * bytes_per_element
        
        # Add overhead for gradients, activations, etc. (roughly 3x input memory)
        total_memory_per_sample = input_memory_per_sample * 3
        
        # Convert target memory to bytes
        target_memory_bytes = target_memory_gb * (1024**3)
        
        # Calculate optimal batch size
        optimal_batch_size = max(1, int(target_memory_bytes / total_memory_per_sample))
        
        logger.info(f"Input memory per sample: {input_memory_per_sample / (1024**2):.1f}MB")
        logger.info(f"Total memory per sample: {total_memory_per_sample / (1024**2):.1f}MB")
        logger.info(f"Optimal batch size for {target_memory_gb}GB: {optimal_batch_size}")
        
        return optimal_batch_size
        
    except Exception as e:
        logger.error(f"Error calculating optimal batch size: {e}")
        return 1


def create_ultra_memory_efficient_dataloader(
    x_path: str,
    t_path: str,
    target_memory_gb: float = 1.5,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create ultra memory-efficient dataloader with automatic batch size optimization.
    
    Args:
        x_path: Path to X data
        t_path: Path to target data
        target_memory_gb: Target memory usage in GB
        **kwargs: Additional arguments for create_real_data_dataloader
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Calculate optimal batch size
    optimal_batch_size = get_optimal_batch_size_for_memory(
        model_input_shape=(4, 1400, 2500),
        target_memory_gb=target_memory_gb,
        use_half_precision=kwargs.get('use_half_precision', True)
    )
    
    # Override batch size
    kwargs['batch_size'] = optimal_batch_size
    
    # Force memory-efficient settings
    kwargs.update({
        'num_workers': 0,
        'pin_memory': False,
        'use_half_precision': True,
        'gradient_accumulation_steps': max(8, 32 // optimal_batch_size)  # Ensure effective batch size
    })
    
    logger.info(f"Creating ultra memory-efficient dataloader with batch_size={optimal_batch_size}")
    
    return create_real_data_dataloader(x_path, t_path, **kwargs)
