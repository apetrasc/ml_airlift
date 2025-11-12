"""
Utility functions.
"""

from .device import get_valid_device
from .memory import clear_gpu_memory, log_gpu_memory_usage

try:
    from .config import load_config, print_config_summary
except ImportError:
    load_config = None
    print_config_summary = None

__all__ = [
    'get_valid_device',
    'clear_gpu_memory',
    'log_gpu_memory_usage',
]

if load_config is not None:
    __all__.extend(['load_config', 'print_config_summary'])

