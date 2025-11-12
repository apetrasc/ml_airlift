"""
Data loading and preprocessing modules.
"""

from .loaders import (
    load_npz_pair,
    to_tensor_dataset,
    split_dataset
)

# Optional imports for advanced data loaders (from existing src modules)
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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

# Preprocessing and validation (will be implemented)
try:
    from .preprocessing import create_dropped_dataset, preprocess_data
except ImportError:
    create_dropped_dataset = None
    preprocess_data = None

try:
    from .validation import validate_dataset_pair, validate_array
except ImportError:
    validate_dataset_pair = None
    validate_array = None

__all__ = [
    'load_npz_pair',
    'to_tensor_dataset',
    'split_dataset',
]

# Add optional exports if available
if RealDataDataset is not None:
    __all__.append('RealDataDataset')
if ChunkedDataLoader is not None:
    __all__.append('ChunkedDataLoader')
if StreamingDataLoader is not None:
    __all__.append('StreamingDataLoader')
if create_dropped_dataset is not None:
    __all__.extend(['create_dropped_dataset', 'preprocess_data'])
if validate_dataset_pair is not None:
    __all__.extend(['validate_dataset_pair', 'validate_array'])

