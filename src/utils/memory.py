"""
Memory management utilities.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def clear_gpu_memory():
    """Clear GPU memory and reset CUDA state."""
    if torch.cuda.is_available():
        print("Clearing GPU memory...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Reset CUDA state
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        
        print(f"GPU memory cleared. Available devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = props.total_memory / 1024**3
            print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
    else:
        print("CUDA not available")


def log_gpu_memory_usage(stage: str = ""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / 1024**3
            logger.info(f"GPU {i} {stage}: {allocated:.2f}GB/{total:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        logger.info(f"GPU not available {stage}")

