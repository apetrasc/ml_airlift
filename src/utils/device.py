"""
Device management utilities.
"""

import torch


def get_valid_device(device_str: str) -> torch.device:
    """Get valid torch device from string, with fallback to CPU if CUDA unavailable."""
    if device_str.startswith('cuda'):
        if torch.cuda.is_available():
            return torch.device(device_str)
        else:
            print(f"[WARN] CUDA requested but not available. Falling back to CPU.")
            return torch.device('cpu')
    else:
        return torch.device(device_str)

