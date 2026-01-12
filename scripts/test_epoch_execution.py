#!/usr/bin/env python3
"""
Test script to check if one epoch can be executed successfully.
This checks CPU and GPU memory usage before and after one epoch training.
If the first epoch can be executed, subsequent epochs should be fine too.
"""

import os
import sys
import time

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[WARN] psutil not available. CPU memory monitoring will be limited.")

# Set PyTorch CUDA memory allocator configuration
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import load_npz_pair, to_tensor_dataset, split_dataset
from src.training.trainer import train_one_epoch, evaluate, create_model


def _get_actual_gpu_memory_usage():
    """Get actual GPU memory usage from nvidia-smi or nvidia-ml-py."""
    if not torch.cuda.is_available():
        return None
    
    try:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pid = os.getpid()
            
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for proc in procs:
                if proc.pid == pid:
                    return {
                        'used_mb': proc.usedGpuMemory / 1024**2,
                        'method': 'pynvml'
                    }
            
            procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
            for proc in procs:
                if proc.pid == pid:
                    return {
                        'used_mb': proc.usedGpuMemory / 1024**2,
                        'method': 'pynvml'
                    }
            
            pynvml.nvmlShutdown()
        except ImportError:
            pass
        
        import subprocess
        pid = os.getpid()
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2 and parts[0].strip() == str(pid):
                        return {
                            'used_mb': float(parts[1].strip()),
                            'method': 'nvidia-smi'
                        }
    except Exception:
        pass
    
    return None


def _get_cpu_memory_info():
    """Get CPU memory usage information."""
    if not PSUTIL_AVAILABLE:
        return None
    
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        system_mem = psutil.virtual_memory()
        
        return {
            'process_rss_gb': mem_info.rss / 1024**3,
            'process_vms_gb': mem_info.vms / 1024**3,
            'system_total_gb': system_mem.total / 1024**3,
            'system_available_gb': system_mem.available / 1024**3,
            'system_used_gb': system_mem.used / 1024**3,
            'system_percent': system_mem.percent,
            'process_percent': (mem_info.rss / system_mem.total) * 100
        }
    except Exception as e:
        print(f"[WARN] Failed to get CPU memory info: {e}")
        return None


def _print_memory_status(stage: str):
    """Print CPU and GPU memory status."""
    print(f"\n{'='*60}")
    print(f"Memory Status: {stage}")
    print(f"{'='*60}")
    
    # CPU memory
    cpu_mem = _get_cpu_memory_info()
    if cpu_mem:
        print(f"[CPU Memory]")
        print(f"  Process RSS (physical): {cpu_mem['process_rss_gb']:.2f} GB ({cpu_mem['process_percent']:.1f}% of system)")
        print(f"  Process VMS (virtual):  {cpu_mem['process_vms_gb']:.2f} GB")
        print(f"  System total:           {cpu_mem['system_total_gb']:.2f} GB")
        print(f"  System available:       {cpu_mem['system_available_gb']:.2f} GB")
        print(f"  System used:            {cpu_mem['system_used_gb']:.2f} GB ({cpu_mem['system_percent']:.1f}%)")
    else:
        print(f"[CPU Memory] Information not available (psutil not installed)")
    
    # GPU memory
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        mem_free = mem_total - mem_reserved
        
        actual_mem = _get_actual_gpu_memory_usage()
        
        print(f"\n[GPU Memory - PyTorch]")
        print(f"  Total:     {mem_total:.2f} GB")
        print(f"  Allocated: {mem_allocated:.2f} GB")
        print(f"  Reserved:  {mem_reserved:.2f} GB")
        print(f"  Free:      {mem_free:.2f} GB")
        
        if actual_mem:
            actual_gb = actual_mem['used_mb'] / 1024
            print(f"\n[GPU Memory - Actual (nvidia-smi)]")
            print(f"  Process usage: {actual_gb:.2f} GB ({actual_mem['used_mb']:.0f} MB)")
            print(f"  Method: {actual_mem['method']}")
    else:
        print(f"\n[GPU Memory] GPU not available")
    
    print(f"{'='*60}\n")


def main():
    """Main function to test epoch execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test if one epoch can be executed successfully')
    parser.add_argument('--config', type=str, default='config/config_real.yaml',
                        help='Path to config file')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size from config')
    parser.add_argument('--downsample-factor', type=int, default=None,
                        help='Override downsample factor from config')
    args = parser.parse_args()
    
    # Load config
    cfg = OmegaConf.load(args.config)
    
    # Override config if specified
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.downsample_factor:
        cfg.dataset.downsample_factor = args.downsample_factor
    
    print(f"[INFO] Testing epoch execution with:")
    print(f"  Config: {args.config}")
    print(f"  Batch size: {cfg.training.batch_size}")
    print(f"  Downsample factor: {cfg.dataset.downsample_factor}")
    print()
    
    # Clear GPU memory
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
    
    # Print initial memory status
    _print_memory_status("Initial state (before data loading)")
    
    try:
        # 1. Load data
        print("[STEP] Loading dataset files...")
        x, t = load_npz_pair(
            cfg.dataset.x_train,
            cfg.dataset.t_train,
            cfg.dataset.x_key,
            cfg.dataset.t_key
        )
        print(f"[OK] Loaded. x.shape={x.shape}, t.shape={t.shape}")
        
        # Exclude Channel 1 and Channel 3 (keep only channels 0, 2)
        if x.ndim == 4 and x.shape[1] == 4:
            print(f"[INFO] Excluding Channel 1 and Channel 3 (keeping channels 0, 2)")
            x = x[:, [0, 2], :, :]
            print(f"[OK] After excluding: x.shape={x.shape}")
            cfg.model.in_channels = 2
        elif x.ndim == 3 and x.shape[1] == 4:
            print(f"[INFO] Excluding Channel 1 and Channel 3 (keeping channels 0, 2)")
            x = x[:, [0, 2], :]
            print(f"[OK] After excluding: x.shape={x.shape}")
            cfg.model.in_channels = 2
        
        # Limit samples if specified
        if cfg.dataset.limit_samples > 0:
            n = min(cfg.dataset.limit_samples, x.shape[0])
            x = x[:n]
            t = t[:n]
            print(f"[INFO] Limited to first {n} samples")
        
        # Optional downsampling
        if x.ndim == 4 and cfg.dataset.downsample_factor > 1:
            h0 = x.shape[2]
            x = x[:, :, ::cfg.dataset.downsample_factor, :]
            print(f"[INFO] Downsampled H: {h0} -> {x.shape[2]} (factor={cfg.dataset.downsample_factor})")
        
        # Create dataset on CPU
        print("[STEP] Creating dataset tensors on CPU...")
        dataset = to_tensor_dataset(x, t, "cpu")
        
        dataset_size_mb = sum(
            tensor.element_size() * tensor.nelement() / 1024**2
            for tensor in dataset.tensors
        )
        print(f"[INFO] Dataset size: {dataset_size_mb:.2f} MB ({dataset_size_mb/1024:.2f} GB)")
        
        # Print memory status after dataset creation
        _print_memory_status("After dataset creation (on CPU)")
        
        # Delete original numpy arrays
        del x, t
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Split dataset
        print("[STEP] Splitting dataset...")
        train_set, val_set, test_set = split_dataset(
            dataset,
            cfg.data_split.train_ratio,
            cfg.data_split.val_ratio,
            cfg.data_split.test_ratio,
            cfg.training.seed
        )
        print(f"[OK] Sizes -> train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
        
        # Create dataloaders
        print("[STEP] Creating dataloaders...")
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False
        )
        
        # Print memory status after dataloader creation
        _print_memory_status("After dataloader creation")
        
        # Create model
        print("[STEP] Creating model...")
        x_sample = dataset.tensors[0][:1].clone()
        device = torch.device(cfg.training.device)
        out_dim = dataset.tensors[1].shape[1] if dataset.tensors[1].ndim == 2 else 1
        model = create_model(cfg, x_sample, out_dim, device)
        del x_sample
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Print memory status after model creation
        _print_memory_status("After model creation")
        
        # Create optimizer and loss
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay if 'weight_decay' in cfg.training else 0.0
        )
        
        # Print memory status before training
        _print_memory_status("Before training (ready to train)")
        
        # Test one epoch
        print("[STEP] Testing one epoch execution...")
        t_start = time.time()
        
        try:
            # Train one epoch
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            train_time = time.time() - t_start
            
            # Evaluate validation set
            if len(val_loader.dataset) > 0:
                val_mse, val_mae, _, _ = evaluate(model, val_loader, criterion, device)
            else:
                val_mse, val_mae = 0.0, 0.0
            
            # Print memory status after one epoch
            _print_memory_status("After one epoch training")
            
            # Summary
            print(f"\n{'='*60}")
            print("EPOCH EXECUTION TEST: SUCCESS")
            print(f"{'='*60}")
            print(f"Training completed successfully!")
            print(f"  Training time: {train_time:.2f} seconds")
            print(f"  Train loss: {train_loss:.6f}")
            print(f"  Validation MSE: {val_mse:.6f}")
            print(f"  Validation MAE: {val_mae:.6f}")
            print()
            print("[RESULT] Epoch execution is possible!")
            print("[INFO] If the first epoch can be executed, subsequent epochs should work fine.")
            print(f"{'='*60}\n")
            
            return 0
            
        except RuntimeError as e:
            error_msg = str(e)
            print(f"\n{'='*60}")
            print("EPOCH EXECUTION TEST: FAILED")
            print(f"{'='*60}")
            print(f"[ERROR] Failed to execute one epoch: {error_msg}")
            if 'out of memory' in error_msg.lower():
                print(f"[SOLUTION] GPU out of memory. Try:")
                print(f"  1. Reduce batch size (current: {cfg.training.batch_size})")
                print(f"  2. Increase downsample_factor (current: {cfg.dataset.downsample_factor})")
                print(f"  3. Limit samples (current: {cfg.dataset.limit_samples})")
            print(f"{'='*60}\n")
            
            # Print memory status after error
            _print_memory_status("After error")
            
            return 1
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("EPOCH EXECUTION TEST: FAILED")
        print(f"{'='*60}")
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())



