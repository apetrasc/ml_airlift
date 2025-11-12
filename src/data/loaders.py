"""
Data loading utilities.
"""

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from typing import Tuple

# Import from existing modules
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

# Import existing data loaders (for backward compatibility)
try:
    from src.data_loader import RealDataDataset
except ImportError:
    RealDataDataset = None

try:
    from src.chunked_loader import ChunkedDataLoader
except ImportError:
    ChunkedDataLoader = None

try:
    from src.streaming_loader import StreamingDataLoader
except ImportError:
    StreamingDataLoader = None


def _load_np_any(path: str, prefer_key: str = None):
    """Load numpy array from .npy or .npz file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found: {path}")
    obj = np.load(path)
    # npz: dict-like, npy: ndarray
    if hasattr(obj, 'keys'):
        if prefer_key and prefer_key in obj:
            return obj[prefer_key]
        elif prefer_key:
            print(f"[INFO] {path} is an npz file. Key '{prefer_key}' not found. Using first key.")
            return obj[list(obj.keys())[0]]
        else:
            return obj
    else:
        if prefer_key:
            print(f"[INFO] {path} is an array (npy). Ignoring key '{prefer_key}'.")
        return obj


def load_npz_pair(x_path: str, t_path: str, x_key: str = "x_train_real", t_key: str = "t_train_real"):
    """Load x and t from npz/npy files robustly and return numpy arrays."""
    x = _load_np_any(x_path, x_key)
    t = _load_np_any(t_path, t_key)
    print(f"Loaded x: shape={getattr(x, 'shape', None)}, dtype={getattr(x, 'dtype', None)}")
    print(f"Loaded t: shape={getattr(t, 'shape', None)}, dtype={getattr(t, 'dtype', None)}")
    return x, t


def to_tensor_dataset(x: np.ndarray, t: np.ndarray, device: str = "cuda:0"):
    """Convert numpy arrays to torch tensors and TensorDataset."""
    if x.ndim == 2:
        # (N, L) -> (N, 1, L)
        x = x[:, None, :]
    elif x.ndim == 3:
        # Either (N, C, L) for 1D or (N, H, W) image-like; we decide later.
        pass
    elif x.ndim == 4:
        # (N, C, H, W) or (N, H, W, C)
        pass
    else:
        raise RuntimeError(f"Unsupported x shape: {x.shape}")

    # Allow multi-target (N, M)
    if t.ndim == 2 and t.shape[1] == 1:
        t = t[:, 0]
    elif t.ndim == 2 and t.shape[1] > 1:
        pass
    elif t.ndim != 1:
        raise RuntimeError(f"Expected t to be 1D or (N,M), got {t.shape}")

    x_t = torch.from_numpy(x).float()
    y_t = torch.from_numpy(t).float()
    print(f"Tensor x: {tuple(x_t.shape)}  Tensor y: {tuple(y_t.shape)}")
    return TensorDataset(x_t, y_t)


def split_dataset(dataset: TensorDataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split dataset into train/validation/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    N = len(dataset)
    
    # Ensure minimum sizes for small datasets
    n_train = max(1, int(N * train_ratio))
    n_val = max(1, int(N * val_ratio)) if N >= 3 else 0
    n_test = N - n_train - n_val
    
    # Adjust if test set would be negative
    if n_test < 0:
        n_test = 0
        n_val = max(0, N - n_train)
    
    # Ensure we don't exceed total size
    if n_train + n_val + n_test > N:
        n_test = N - n_train - n_val
    
    print(f"[INFO] Split sizes: train={n_train}, val={n_val}, test={n_test}")
    
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=g)

