#!/usr/bin/env python3
"""Clear GPU memory and reset CUDA state."""

import torch

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

