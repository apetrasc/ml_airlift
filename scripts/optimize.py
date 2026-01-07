#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for CNN model training.
Based on train_real.py structure with automatic parameter tuning.
"""

import os
import sys
import time
import datetime
import json
import shutil
import numpy as np
import pywt

# Tee class to output to both stdout and file
class Tee:
    """Class to write to both stdout and a file."""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self
    
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        if self.file:
            self.file.close()
        sys.stdout = self.stdout
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Try to import psutil for memory monitoring, fallback if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[WARN] psutil not available. CPU memory monitoring will be limited.")

# Set PyTorch CUDA memory allocator configuration to reduce fragmentation
# This must be set before importing torch
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("[INFO] Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

import scipy.signal as signal
import scipy.ndimage as ndimage

import optuna
from optuna.trial import Trial
from omegaconf import OmegaConf, DictConfig

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load config using OmegaConf to enable attribute access
config = OmegaConf.load('config/config_real.yaml')

# Import functions from new module structure
from src.data.loaders import (
    load_npz_pair,
    to_tensor_dataset,
    split_dataset,
)
from src.training.trainer import (
    train_one_epoch,
    evaluate,
    create_model,
    create_learning_curves,
)
try:
    from src.evaluation.visualizations import create_prediction_plots
except ImportError:
    # Fallback to old location
    from src.evaluate_predictions import create_prediction_plots

# Paths
OPTUNA_DIR = os.path.join(config.output.model_save_dir, 'optuna')
OUTPUTS_ROOT = os.path.join(config.output.model_save_dir, 'optuna', 'runs')
BASE_CONFIG_PATH = 'config/config_real.yaml'

# Epoch range configuration
# Change these values to create a new study for different epoch ranges
# Previous range: 50-200 (step=50)
# Current range: 100-300 (step=100)
EPOCH_MIN = 100  # Minimum epoch value
EPOCH_MAX = 300  # Maximum epoch value
EPOCH_STEP = 100  # Step size for epoch values


def suggest_hyperparameters(trial: Trial, base_cfg: DictConfig) -> DictConfig:
    """
    Suggest hyperparameters using Optuna and update config.
    
    Args:
        trial: Optuna trial object
        base_cfg: Base configuration from YAML
    
    Returns:
        Updated configuration with suggested hyperparameters
    """
    # Create a copy of the config
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    
    # Model hyperparameters
    cfg.model.hidden = trial.suggest_int('model.hidden', 32, 128, step=16)
    cfg.model.dropout_rate = trial.suggest_float('model.dropout_rate', 0.0, 0.5, step=0.05)
    cfg.model.use_residual = trial.suggest_categorical('model.use_residual', [True, False])
    
    # Training hyperparameters
    cfg.training.learning_rate = trial.suggest_float(
        'training.learning_rate', 1e-5, 1e-2, log=True
    )
    # Reduce batch size options to avoid OOM with large images
    cfg.training.batch_size = trial.suggest_categorical('training.batch_size', [2, 4, 8])
    
    # Add weight_decay if not exists
    if 'weight_decay' not in cfg.training:
        cfg.training.weight_decay = trial.suggest_float(
            'training.weight_decay', 1e-6, 1e-3, log=True
        )
    else:
        cfg.training.weight_decay = trial.suggest_float(
            'training.weight_decay', 1e-6, 1e-3, log=True
        )
    
    # Data hyperparameters
    # cfg.dataset.downsample_factor = trial.suggest_int('dataset.downsample_factor', 1, 4)
    cfg.dataset.downsample_factor = trial.suggest_int('dataset.downsample_factor', 2, 4)
    
    # Limit epochs for faster optimization (can be adjusted)
    # IMPORTANT: These values must match EPOCH_MIN, EPOCH_MAX, EPOCH_STEP defined at module level
    # to ensure the study name correctly reflects the epoch range being used
    global EPOCH_MIN, EPOCH_MAX, EPOCH_STEP
    cfg.training.epochs = trial.suggest_int(
        'training.epochs', 
        EPOCH_MIN, 
        EPOCH_MAX, 
        step=EPOCH_STEP
    )

    # Preprocess selection
    # Scaling method
    cfg.preprocess.scaling = trial.suggest_categorical(
        'preprocess.scaling',
        ['zscore','max']
    )
    # Noise reduction
    cfg.preprocess.noise_reduction = trial.suggest_categorical(
        'preprocess.noise_reduction',
        ['bandpass','wavelet','raw']
    )
    # Harmonic decomposition
    cfg.preprocess.harmonic_decomposition = trial.suggest_categorical(
        'preprocess.harmonic_decomposition',
        [True, False]
    )
    # Envelope
    # cfg.preprocess.envelope = trial.suggest_categorical(
    #     'preprocess.envelope',
    #     [True, False]
    # )
    cfg.preprocess.envelope = True

    if cfg.preprocess.envelope:
        cfg.preprocess.resize_ratio = trial.suggest_float(
            'preprocess.resize_ratio',
            0.05,0.55,step=0.05
        )
    
    return cfg


class GradCAM:
    def __init__(self,model,target_layer):
        self.model=model
        self.target_layer=target_layer
        self.gradient = None
        self.activations=None
        self.hook_handles=[]
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(output):
            self.activations=output.detach()
        def backward_hook(grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def __call__(self, input_tensor, batch=False):
        """
        Compute Grad-CAM map(s) for the provided input tensor.

        Args:
            input_tensor (torch.Tensor): Tensor of shape [B, C, L] or [1, C, L].
            batch (bool): If True, compute Grad-CAM for each sample independently
                          and return a NumPy array of shape [B, L].
                          If False, return a NumPy array of shape [L].
        """
        if batch:
            cam_list = []
            for sample in input_tensor:
                cam = self._compute_single(sample.unsqueeze(0))
                cam_list.append(cam)
            return np.stack(cam_list, axis=0)

        return self._compute_single(input_tensor)

    def _compute_single(self, input_tensor):
        self.model.eval()
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.detach().to(device)
        input_tensor.requires_grad_(True)
        self.model.zero_grad()
        out = self.model(input_tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        target = out.squeeze()
        if target.ndim > 0:
            target = target.sum()
        self.model.zero_grad()
        target.backward(retain_graph=True)
        gradients = self.gradients         # [B, C, L]
        activations = self.activations     # [B, C, L]
        weights = gradients.mean(dim=2, keepdim=True)  # [B, C, 1]
        grad_cam_map = (weights * activations).sum(dim=1, keepdim=True)  # (B,1,L)
        grad_cam_map = torch.relu(grad_cam_map)
        grad_cam_map = torch.nn.functional.interpolate(
            grad_cam_map, size=input_tensor.shape[2], mode='linear', align_corners=False
        )
        grad_cam_map = grad_cam_map.squeeze().cpu().numpy()
        grad_cam_map = (grad_cam_map - grad_cam_map.min()) / (grad_cam_map.max() - grad_cam_map.min() + 1e-8)
        return grad_cam_map

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def create_trial_output_dir(trial_number: int) -> str:
    """
    Create output directory for a trial following train_real.py structure.
    
    Args:
        trial_number: Optuna trial number
    
    Returns:
        Output directory path
    """
    now = datetime.datetime.now()
    run_dir = os.path.join(
        OUTPUTS_ROOT,
        now.strftime('%Y-%m-%d'),
        now.strftime('%H-%M-%S')
    )
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories
    logs_dir = os.path.join(run_dir, "logs")
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    
    return run_dir


def _get_actual_gpu_memory_usage():
    """
    Get actual GPU memory usage from nvidia-smi or nvidia-ml-py.
    This shows the real memory usage that matches nvidia-smi output.
    
    Returns:
        dict with actual GPU memory info in MB, or None if not available
    """
    if not torch.cuda.is_available():
        return None
    
    try:
        # Try to use nvidia-ml-py if available
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get process ID
            import os
            pid = os.getpid()
            
            # Get memory info for this process
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for proc in procs:
                if proc.pid == pid:
                    return {
                        'used_mb': proc.usedGpuMemory / 1024**2,
                        'method': 'pynvml'
                    }
            
            # If process not found in compute processes, try graphics processes
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
        
        # Fallback: try nvidia-smi command
        import subprocess
        import os
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
    except Exception as e:
        # Silently fail and return None
        pass
    
    return None


def _get_cpu_memory_info():
    """
    Get CPU memory usage information.
    
    Returns:
        dict with memory information in GB
    """
    if not PSUTIL_AVAILABLE:
        return None
    
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        system_mem = psutil.virtual_memory()
        
        return {
            'process_rss_gb': mem_info.rss / 1024**3,  # Resident Set Size (actual physical memory)
            'process_vms_gb': mem_info.vms / 1024**3,  # Virtual Memory Size
            'system_total_gb': system_mem.total / 1024**3,
            'system_available_gb': system_mem.available / 1024**3,
            'system_used_gb': system_mem.used / 1024**3,
            'system_percent': system_mem.percent,
            'process_percent': (mem_info.rss / system_mem.total) * 100
        }
    except Exception as e:
        print(f"[WARN] Failed to get CPU memory info: {e}")
        return None


def _print_memory_status(stage: str = ""):
    """
    Print both CPU and GPU memory status.
    
    Args:
        stage: Description of current stage (e.g., "after dataset creation")
    """
    print(f"\n{'='*60}")
    print(f"Memory Status {stage}")
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
    
    # GPU memory
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        mem_free = mem_total - mem_reserved
        
        # Get actual GPU memory usage (matches nvidia-smi)
        actual_mem = _get_actual_gpu_memory_usage()
        
        print(f"\n[GPU Memory - PyTorch]")
        print(f"  Total:     {mem_total:.2f} GB")
        print(f"  Allocated: {mem_allocated:.2f} GB (PyTorch explicit allocation)")
        print(f"  Reserved:  {mem_reserved:.2f} GB (PyTorch memory pool)")
        print(f"  Free:      {mem_free:.2f} GB")
        
        if actual_mem:
            actual_gb = actual_mem['used_mb'] / 1024
            print(f"\n[GPU Memory - Actual (nvidia-smi)]")
            print(f"  Process usage: {actual_gb:.2f} GB ({actual_mem['used_mb']:.0f} MB)")
            print(f"  Method: {actual_mem['method']}")
            if actual_gb > mem_allocated * 1.5:
                extra = actual_gb - mem_allocated
                print(f"  Note: Extra {extra:.2f} GB used by cuDNN, CUDA runtime, and other GPU resources")
    
    print(f"{'='*60}\n")


def _check_dataset_on_cpu(dataset, stage: str = ""):
    """
    Verify that dataset tensors are on CPU memory.
    
    Args:
        dataset: Dataset object to check
        stage: Description of current stage
    
    Returns:
        bool: True if all tensors are on CPU, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Dataset CPU Memory Check {stage}")
    print(f"{'='*60}")
    
    all_on_cpu = True
    total_size_mb = 0
    
    if hasattr(dataset, 'tensors'):
        for i, tensor in enumerate(dataset.tensors):
            device = tensor.device if hasattr(tensor, 'device') else 'unknown'
            size_mb = tensor.element_size() * tensor.nelement() / 1024**2
            total_size_mb += size_mb
            
            if device.type != 'cpu' if hasattr(device, 'type') else device != 'cpu':
                print(f"[ERROR] Tensor {i} is on {device}, expected CPU!")
                all_on_cpu = False
            else:
                print(f"[OK] Tensor {i}: {tuple(tensor.shape)}, {tensor.dtype}, on CPU, size: {size_mb:.2f} MB")
    
    print(f"[INFO] Total dataset size: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
    
    if all_on_cpu:
        print(f"[OK] All dataset tensors are on CPU memory")
    else:
        print(f"[ERROR] Some dataset tensors are not on CPU memory!")
    
    print(f"{'='*60}\n")
    return all_on_cpu


def _check_batch_transfer_to_gpu(dataloader, device, batch_size: int):
    """
    Test if a batch can be transferred to GPU memory.
    
    Args:
        dataloader: DataLoader object
        device: Target device (should be GPU)
        batch_size: Expected batch size
    
    Returns:
        tuple: (success: bool, batch_size_actual: int, error_message: str)
    """
    print(f"\n{'='*60}")
    print(f"GPU Batch Transfer Check")
    print(f"{'='*60}")
    
    if not torch.cuda.is_available() or device.type != 'cuda':
        print(f"[SKIP] GPU not available or device is not CUDA")
        return True, 0, "GPU not available"
    
    try:
        # Get initial GPU memory
        mem_before = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved_before = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"[INFO] GPU memory before batch transfer:")
        print(f"  Allocated: {mem_before:.2f} GB")
        print(f"  Reserved:  {mem_reserved_before:.2f} GB")
        
        # Try to get a batch
        print(f"[INFO] Attempting to load batch from DataLoader...")
        batch_iter = iter(dataloader)
        xb, yb = next(batch_iter)
        
        print(f"[OK] Batch loaded from DataLoader")
        print(f"  X batch shape: {tuple(xb.shape)}, dtype: {xb.dtype}")
        print(f"  Y batch shape: {tuple(yb.shape)}, dtype: {yb.dtype}")
        print(f"  Batch size: {xb.shape[0]}")
        
        # Check if batch is on CPU
        if xb.device.type != 'cpu':
            print(f"[WARN] X batch is already on {xb.device}, expected CPU")
        else:
            print(f"[OK] X batch is on CPU")
        
        # Calculate batch memory requirement
        xb_size_mb = xb.element_size() * xb.nelement() / 1024**2
        yb_size_mb = yb.element_size() * yb.nelement() / 1024**2
        print(f"[INFO] Batch memory requirement:")
        print(f"  X batch: {xb_size_mb:.2f} MB")
        print(f"  Y batch: {yb_size_mb:.2f} MB")
        print(f"  Total: {(xb_size_mb + yb_size_mb):.2f} MB")
        
        # Try to transfer to GPU
        print(f"[INFO] Attempting to transfer batch to GPU ({device})...")
        xb_gpu = xb.to(device, non_blocking=False)
        yb_gpu = yb.to(device, non_blocking=False)
        
        print(f"[OK] Batch transferred to GPU successfully")
        
        # Check GPU memory after transfer
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved_after = torch.cuda.memory_reserved(0) / 1024**3
        mem_increase = mem_after - mem_before
        
        print(f"[INFO] GPU memory after batch transfer:")
        print(f"  Allocated: {mem_after:.2f} GB (increased by {mem_increase:.2f} GB)")
        print(f"  Reserved:  {mem_reserved_after:.2f} GB")
        
        # Store batch size before cleanup
        actual_batch_size = xb.shape[0]
        
        # Clean up test batch
        del xb_gpu, yb_gpu, xb, yb
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        mem_final = torch.cuda.memory_allocated(0) / 1024**3
        print(f"[INFO] GPU memory after cleanup: {mem_final:.2f} GB")
        
        print(f"[OK] Batch transfer test successful!")
        print(f"{'='*60}\n")
        return True, actual_batch_size, ""
        
    except RuntimeError as e:
        error_msg = str(e)
        print(f"[ERROR] Failed to transfer batch to GPU: {error_msg}")
        if 'out of memory' in error_msg.lower():
            print(f"[ERROR] GPU out of memory! Batch size {batch_size} may be too large.")
            print(f"[SOLUTION] Try reducing batch size or using downsample_factor")
        
        # Try to clean up variables that might exist
        try:
            if 'xb_gpu' in locals():
                del xb_gpu
            if 'yb_gpu' in locals():
                del yb_gpu
            if 'xb' in locals():
                del xb
            if 'yb' in locals():
                del yb
        except:
            pass
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print(f"{'='*60}\n")
        return False, 0, error_msg
        
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Unexpected error during batch transfer test: {error_msg}")
        
        # Try to clean up variables that might exist
        try:
            if 'xb_gpu' in locals():
                del xb_gpu
            if 'yb_gpu' in locals():
                del yb_gpu
            if 'xb' in locals():
                del xb
            if 'yb' in locals():
                del yb
        except:
            pass
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print(f"{'='*60}\n")
        return False, 0, error_msg


def _cleanup_trial_memory(model, optimizer, criterion,
                          train_loader, val_loader, test_loader,
                          dataset, train_set, val_set, test_set,
                          x, t, y_pred, y_true):
    """
    Comprehensive cleanup of GPU memory after each trial.
    
    This function explicitly deletes all objects that may hold GPU memory,
    forces garbage collection, and clears CUDA cache.
    """
    import gc
    
    if not torch.cuda.is_available():
        return
    
    # Move model to CPU and delete
    if model is not None:
        try:
            if isinstance(model, nn.DataParallel):
                model = model.module
            # Move model to CPU to free GPU memory
            model = model.cpu()
            del model
        except:
            pass
    
    # Delete optimizer and criterion
    if optimizer is not None:
        try:
            del optimizer
        except:
            pass
    
    if criterion is not None:
        try:
            del criterion
        except:
            pass
    
    # Delete DataLoaders
    for loader in [train_loader, val_loader, test_loader]:
        if loader is not None:
            try:
                del loader
            except:
                pass
    
    # Delete datasets
    for ds in [dataset, train_set, val_set, test_set]:
        if ds is not None:
            try:
                # If it's a TensorDataset, move tensors to CPU before deletion
                if hasattr(ds, 'tensors'):
                    for tensor in ds.tensors:
                        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                            tensor = tensor.cpu()
                del ds
            except:
                pass
    
    # Delete data arrays (move to CPU if on GPU)
    for data in [x, t, y_pred, y_true]:
        if data is not None:
            try:
                if isinstance(data, torch.Tensor):
                    if data.is_cuda:
                        data = data.cpu()
                    del data
                elif isinstance(data, np.ndarray):
                    del data
            except:
                pass
    
    # Force Python garbage collection
    gc.collect()
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Additional cleanup: clear all CUDA cache again
    torch.cuda.empty_cache()


def save_trial_results(trial: Trial, cfg: DictConfig, output_dir: str, 
                       train_losses: list, val_losses: list,
                       best_val_loss: float,
                       test_mse: float, test_mae: float,
                       y_pred: np.ndarray, y_true: np.ndarray):
    """
    Save trial results following train_real.py structure.
    
    Args:
        trial: Optuna trial object
        cfg: Configuration used for this trial
        output_dir: Output directory for this trial
        train_losses: Training loss history
        val_losses: Validation loss history
        best_val_loss: Best validation loss for this trial
        test_mse: Test MSE
        test_mae: Test MAE
        y_pred: Predictions
        y_true: Ground truth
    """
    # Save configuration
    config_path = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(cfg, config_path)
    
    # Save trial information
    # Note: trial.state is not available during objective execution,
    # so we set it to "RUNNING" and it will be updated after completion
    trial_state = "RUNNING"
    if hasattr(trial, 'state') and trial.state is not None:
        trial_state = trial.state.name
    
    trial_info = {
        'trial_number': trial.number,
        'params': trial.params,
        'value': float(best_val_loss),  # Validation loss
        'state': trial_state,
        'user_attrs': trial.user_attrs,
        'datetime': datetime.datetime.now().isoformat()
    }
    trial_info_path = os.path.join(output_dir, 'trial_info.yaml')
    OmegaConf.save(trial_info, trial_info_path)
    
    # Save metrics
    metrics = {
        'validation_loss': float(best_val_loss),
        'test_mse': float(test_mse),
        'test_mae': float(test_mae),
        'train_losses': [float(l) for l in train_losses],
        'val_losses': [float(l) for l in val_losses],
        'final_train_loss': float(train_losses[-1]) if train_losses else None,
        'final_val_loss': float(val_losses[-1]) if val_losses else None,
    }
    
    # Per-target metrics
    if y_pred.shape[1] > 1:
        per_target_metrics = {}
        for i in range(y_pred.shape[1]):
            target_mse = np.mean((y_pred[:, i] - y_true[:, i])**2)
            target_mae = np.mean(np.abs(y_pred[:, i] - y_true[:, i]))
            per_target_metrics[f'target_{i+1}_mse'] = float(target_mse)
            per_target_metrics[f'target_{i+1}_mae'] = float(target_mae)
        metrics['per_target'] = per_target_metrics
    
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions and ground truth
    weights_dir = os.path.join(output_dir, "weights")
    np.save(os.path.join(output_dir, cfg.output.predictions_filename), y_pred)
    np.save(os.path.join(output_dir, cfg.output.ground_truth_filename), y_true)
    
    # Create learning curves
    create_learning_curves(train_losses, val_losses, output_dir)
    
    # Create evaluation plots
    if cfg.evaluation.create_plots and y_pred.shape[1] > 1:
        plots_dir = os.path.join(output_dir, cfg.output.evaluation_plots_dir)
        create_prediction_plots(y_pred, y_true, plots_dir, cfg.evaluation.target_names)
    
    print(f"[OK] Trial {trial.number} results saved to {output_dir}")


def preprocess(x: np.ndarray, cfg: DictConfig):
    """
    preprocess signals for machine learning.

    Flow
    1. Filter raw signal (i.e. bandpass, wavelet)
    2. Take amplitude using hilbert transform.
    3. Filter image (i.e. opencv)
    4. Take log1p

    Args:
        x: ndarray (shape: (number of experiment, transducer channel, number of pulse, data length per pulse))
        cfg: DictConfig (decide how to preprocess)

    Return:
        x: ndarray (same shape as x)
    """
    fs = 52083333.842615336
    x=x.astype('float32')

    # Noise reduction
    if cfg.preprocess.noise_reduction == 'bandpass':
        sos = signal.butter(N=16, Wn=[1e6, 10e6], btype="bandpass",
                            output="sos", fs=fs)
        x = signal.sosfiltfilt(sos=sos, x=x, axis=3)
        del sos 
        print(f"[OK] Bandpass filter done.")
    if cfg.preprocess.noise_reduction == 'wavelet':
        sos = signal.butter(N=16, Wn=1e6, btype='lowpass',
                            output='sos', fs=fs)
        coeff = pywt.wavedec(x, wavelet='db9', mode='per', axis=3)
        sigma = np.std(coeff[-1],axis=3,keepdims=True)
        thres = sigma * np.sqrt(2*np.log(x.shape[3]))
        coeff[1:] = (pywt.threshold(i, value=thres, mode='hard')
                    for i in coeff[1:])
        x = pywt.waverec(coeff, wavelet='db9', mode='per', axis=3)
        del coeff
        del thres
        del sos
        del sigma
        print(f"[OK] Wavelet filter done.")
    
    # Harmonic decomposition
    if cfg.preprocess.harmonic_decomposition:
        sos1 = signal.butter(N=16, Wn=[2e6, 6e6], btype="bandpass",
                            output="sos", fs=fs)
        sos2 = signal.butter(N=16, Wn=[6e6, 10e6], btype="bandpass",
                            output="sos", fs=fs)
        x[:,:x.shape[1],:,:] = signal.sosfiltfilt(
            sos1, x[:,:x.shape[1],:,:], axis=3
        )
        x[:,x.shape[1]:,:,:] = signal.sosfiltfilt(
            sos2, x[:,x.shape[1]:,:,:], axis=3
        )
        del sos1
        del sos2
        print(f"[OK] Harmonic decomposition done.")

    # Scaling
    if cfg.preprocess.scaling == 'zscore':
        mean = np.mean(x,axis=3,keepdims=True)
        std = np.std(x,axis=3,keepdims=True)+1e-8
        x = (x-mean)/std
        del mean
        del std
        print(f"[OK] Scaling done.")
    if cfg.preprocess.scaling == 'max':
        max = np.max(x,axis=3,keepdims=True)
        x = x/max
        del max
        print(f"[OK] Scaling done.")
    
    # Envelope
    if cfg.preprocess.envelope:
        x = np.abs(signal.hilbert(x, axis=3))
        print(f"[OK] Envelope done.")

    return x

def resize(x: np.ndarray, cfg: DictConfig):
    if cfg.preprocess.envelope:
        x = ndimage.zoom(
            x,(1.0,1.0,1.0/float(cfg.dataset.downsample_factor),cfg.preprocess.resize_ratio)
        )
        print(f"[OK] Resize done.")
    return x

def objective(trial: Trial, base_config_path: str) -> float:
    """
    Optuna objective function.
    
    Flow:
    1. Load base configuration
    2. Suggest hyperparameters
    3. Create trial output directory
    4. Load and preprocess data
    5. Create model
    6. Train model (with pruning support)
    7. Evaluate on test set
    8. Save results
    
    Args:
        trial: Optuna trial object
        base_config_path: Path to base configuration file
    
    Returns:
        Validation loss (optimization target)
    """
    # Clear GPU memory before starting trial
    if torch.cuda.is_available():
        # Force garbage collection first
        import gc
        gc.collect()
        # Clear CUDA cache multiple times to ensure cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # Reset memory stats
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
    
    t0 = time.time()
    
    # 1. Load base configuration
    base_cfg = OmegaConf.load(base_config_path)
    
    # 2. Suggest hyperparameters
    cfg = suggest_hyperparameters(trial, base_cfg)
    
    # 3. Create trial output directory
    output_dir = create_trial_output_dir(trial.number)
    cfg.run_dir = output_dir
    cfg.trial_number = trial.number
    
    # Create log file path
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, "training.log")
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number} started")
    print(f"Output directory: {output_dir}")
    print(f"Log file: {log_file_path}")
    print(f"Hyperparameters: {trial.params}")
    print(f"{'='*60}\n")
    
    # Set reproducibility
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Redirect stdout to both console and log file
    with Tee(log_file_path):
        try:
            # 4. Load and preprocess data
            print("[STEP] Loading dataset files...")
            x, t = load_npz_pair(
            cfg.dataset.x_train, 
            cfg.dataset.t_train, 
                cfg.dataset.x_key, 
                cfg.dataset.t_key
            )
            print(f"[OK] Loaded. x.shape={x.shape}, t.shape={t.shape}")
            # raise ValueError("error!")
            # Exclude Channel 1 and Channel 3 (keep only channels 0, 2)
            if x.ndim == 4 and x.shape[1] == 4:
                print(f"[INFO] Excluding Channel 1 and Channel 3 (keeping channels 0, 2)")
                x = x[:, [0, 2], :, :]  # Keep only channels 0, 2
                print(f"[OK] After excluding Channel 1 and 3: x.shape={x.shape}")
                # Update model config to reflect 2 channels
                cfg.model.in_channels = 2
            elif x.ndim == 4 and x.shape[1] == 1:
                print(f"[INFO] Using Channel 0")
                x = x[:, :, :, :]
                print(f"[OK] After excluding Channel 1 and 3: x.shape={x.shape}")
                # Update model config to reflect 2 channels
                cfg.model.in_channels = 2
            elif x.ndim == 3 and x.shape[1] == 4:
                print(f"[INFO] Excluding Channel 1 and Channel 3 (keeping channels 0, 2)")
                x = x[:, [0, 2], :]  # Keep only channels 0, 2
                print(f"[OK] After excluding Channel 1 and 3: x.shape={x.shape}")
                # Update model config to reflect 2 channels
                cfg.model.in_channels = 2
            
            # Limit samples if specified
            if cfg.dataset.limit_samples > 0:
                n = min(cfg.dataset.limit_samples, x.shape[0])
                x = x[:n]
                t = t[:n]
                print(f"[INFO] Limited to first {n} samples")
            
            # # Optional downsampling
            # if x.ndim == 4 and cfg.dataset.downsample_factor > 1:
            #     h0 = x.shape[2]
            #     x = x[:, :, ::cfg.dataset.downsample_factor, :]
            #     print(f"[INFO] Downsampled H: {h0} -> {x.shape[2]} (factor={cfg.dataset.downsample_factor})")

            # Preprocess before tensor transform
            if cfg.preprocess.harmonic_decomposition:
                x = np.repeat(x, 2 ,axis=1)
            for i in range(10):
                extract_idx = np.arange((i*x.shape[0])//10, ((i+1)*x.shape[0])//10).astype(int)
                x[extract_idx,:,:,:] = preprocess(x[extract_idx,:,:,:], cfg)
            del extract_idx
            print("[STEP] Resizing...")
            x = resize(x, cfg)
            # Create dataset (keep data on CPU, move to GPU only during training)
            print("[STEP] Build dataset tensors...")
            # Note: to_tensor_dataset creates tensors on CPU by default
            # They will be moved to GPU in batches during training
            dataset = to_tensor_dataset(x, t, "cpu")  # Force CPU storage
            
            # Delete original numpy arrays to free memory
            # Tensors have their own memory copy, so we can safely delete numpy arrays
            del x, t
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Split dataset (use fixed seed for consistency)
            print("[STEP] Split dataset...")
            train_set, val_set, test_set = split_dataset(
                dataset,
                cfg.data_split.train_ratio,
                cfg.data_split.val_ratio,
                cfg.data_split.test_ratio,
                cfg.training.seed  # Fixed seed for all trials
            )
            print(f"[OK] Sizes -> train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
            
            # Clear memory before creating dataloaders
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create dataloaders
            print("[STEP] Build dataloaders...")
            # Set num_workers=0 to avoid multiprocessing issues with large images
            # pin_memory=False for large images to reduce memory pressure
            # persistent_workers=False to avoid keeping workers in memory
            train_loader = DataLoader(
                train_set,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=0,  # Disable multiprocessing to avoid deadlocks
                pin_memory=False,  # Disable pin_memory for large images
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
            test_loader = DataLoader(
                test_set,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False
            )
            
            # 5. Create model
            print("[STEP] Build model...")
            # Use a small sample for model creation to save memory
            # Only take the first sample, don't keep reference to full dataset
            x_sample = dataset.tensors[0][:1].clone()  # Take only first sample and clone
            device = torch.device(cfg.training.device)
            out_dim = dataset.tensors[1].shape[1] if dataset.tensors[1].ndim == 2 else 1
            model = create_model(cfg, x_sample, out_dim, device)
            
            # Clear sample tensor from GPU memory
            del x_sample
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Disable DataParallel for large images to avoid deadlock/hanging issues
            # DataParallel can cause deadlocks with very large input images (1400x2500)
            # Use single GPU instead for stability
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if num_gpus > 1:
                print(f"[WARN] Multiple GPUs available ({num_gpus}), but DataParallel disabled for large images")
                print(f"[WARN] Using single GPU: {device} to avoid potential deadlocks")
                print(f"[INFO] If you need multi-GPU, consider using DistributedDataParallel instead")
            else:
                print(f"[INFO] Using single GPU: {device}")
            
            # Disable cudnn benchmark for large images to avoid memory issues
            torch.backends.cudnn.benchmark = False
            
            # Create optimizer and loss
            criterion = nn.MSELoss()
            optimizer = optim.Adam(
                model.parameters(),
                lr=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay
            )
            
            # 6. Pre-training checks
            print("[STEP] Pre-training checks...")
            
            # Check 1: Verify dataset is on CPU
            print("\n[CHECK 1] Verifying dataset is on CPU memory...")
            dataset_on_cpu = _check_dataset_on_cpu(train_set, "train_set")
            if not dataset_on_cpu:
                print("[ERROR] Dataset is not on CPU memory. Cannot proceed with training.")
                raise RuntimeError("Dataset tensors are not on CPU memory")
            
            # Check 2: Test batch transfer to GPU
            print("\n[CHECK 2] Testing batch transfer to GPU...")
            if torch.cuda.is_available():
                batch_transfer_success, actual_batch_size, error_msg = _check_batch_transfer_to_gpu(
                    train_loader, device, cfg.training.batch_size
                )
                if not batch_transfer_success:
                    print(f"[ERROR] Batch transfer test failed: {error_msg}")
                    print(f"[ERROR] Training cannot proceed. Please reduce batch_size or use downsample_factor")
                    raise RuntimeError(f"Batch transfer to GPU failed: {error_msg}")
                print(f"[OK] Pre-training checks passed!")
            else:
                print("[INFO] GPU not available, skipping GPU batch transfer test")
            
            # 7. Training with pruning support
            print("[STEP] Training...")
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            
            for epoch in range(1, cfg.training.epochs + 1):
                t_ep = time.time()
                
                # Train one epoch
                tr = train_one_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                
                # Evaluate validation set
                if len(val_loader.dataset) > 0:
                    val_mse, val_mae, _, _ = evaluate(model, val_loader, criterion, device)
                else:
                    val_mse, val_mae = 0.0, 0.0
                
                train_losses.append(tr)
                val_losses.append(val_mse)
                
                # Update best validation loss
                if val_mse < best_val_loss:
                    best_val_loss = val_mse
                
                # Report to Optuna for pruning
                trial.report(val_mse, step=epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    print(f"[PRUNED] Trial {trial.number} pruned at epoch {epoch}")
                    raise optuna.TrialPruned()
                
                # Print progress (loss changes only)
                if epoch % cfg.logging.print_every_n_epochs == 0:
                    print(f"Epoch {epoch:03d}/{cfg.training.epochs} | "
                          f"train MSE={tr:.6f} | val MSE={val_mse:.6f} | "
                          f"val MAE={val_mae:.6f} | {time.time()-t_ep:.2f}s")
            
            # 7. Evaluate on test set
            print("[STEP] Testing...")
            test_mse, test_mae, y_pred, y_true = evaluate(model, test_loader, criterion, device)
            print(f"Test  | MSE={test_mse:.6f} | MAE={test_mae:.6f}")
            
            # Print per-target results
            if y_pred.shape[1] > 1:
                print(f"[INFO] Per-target results:")
                for i in range(y_pred.shape[1]):
                    target_mse = np.mean((y_pred[:, i] - y_true[:, i])**2)
                    target_mae = np.mean(np.abs(y_pred[:, i] - y_true[:, i]))
                    print(f"  Target {i+1}: MSE={target_mse:.6f}, MAE={target_mae:.6f}")
            
            # 8. Save model
            weights_dir = os.path.join(output_dir, "weights")
            # Save underlying module state_dict if wrapped by DataParallel
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, os.path.join(weights_dir, cfg.output.model_filename))
            
            # 9. Save trial results
            save_trial_results(
                trial, cfg, output_dir,
                train_losses, val_losses,
                best_val_loss,
                test_mse, test_mae,
                y_pred, y_true
            )
            
            # Set user attributes for Optuna
            trial.set_user_attr('test_mse', float(test_mse))
            trial.set_user_attr('test_mae', float(test_mae))
            trial.set_user_attr('best_val_loss', float(best_val_loss))
            trial.set_user_attr('output_dir', output_dir)
            
            elapsed = time.time() - t0
            print(f"\n[OK] Trial {trial.number} completed in {elapsed:.2f}s")
            print(f"Validation Loss: {best_val_loss:.6f}")
            print(f"Test MSE: {test_mse:.6f}, Test MAE: {test_mae:.6f}\n")
            
            # Clear GPU memory after trial
            # Note: x and t are already deleted, so pass None
            _cleanup_trial_memory(
                model, optimizer, criterion,
                train_loader, val_loader, test_loader,
                dataset, train_set, val_set, test_set,
                None, None, y_pred, y_true  # x and t already deleted
            )
            
            return best_val_loss
        
        except optuna.TrialPruned:
            # Clean up if pruned
            print(f"[INFO] Cleaning up pruned trial {trial.number}")
            # Try to clean up if variables exist
            try:
                _cleanup_trial_memory(
                    locals().get('model'), locals().get('optimizer'), locals().get('criterion'),
                    locals().get('train_loader'), locals().get('val_loader'), locals().get('test_loader'),
                    locals().get('dataset'), locals().get('train_set'), locals().get('val_set'), locals().get('test_set'),
                    locals().get('x'), locals().get('t'), None, None
                )
            except:
                pass
            raise
        except Exception as e:
            print(f"[ERROR] Trial {trial.number} failed: {e}")
            # Clear GPU memory on error
            try:
                _cleanup_trial_memory(
                    locals().get('model'), locals().get('optimizer'), locals().get('criterion'),
                    locals().get('train_loader'), locals().get('val_loader'), locals().get('test_loader'),
                    locals().get('dataset'), locals().get('train_set'), locals().get('val_set'), locals().get('test_set'),
                    locals().get('x'), locals().get('t'), None, None
                )
            except:
                pass
            raise


def find_trial_output_dir(trial_number: int) -> str:
    """
    Find output directory for a given trial number.
    
    Args:
        trial_number: Optuna trial number
    
    Returns:
        Output directory path or None if not found
    """
    for root, dirs, files in os.walk(OUTPUTS_ROOT):
        if 'trial_info.yaml' in files:
            trial_info_path = os.path.join(root, 'trial_info.yaml')
            try:
                trial_info = OmegaConf.load(trial_info_path)
                if trial_info.get('trial_number') == trial_number:
                    return root
            except Exception:
                continue
    return None


def generate_study_summary(study: optuna.Study, output_dir: str):
    """
    Generate study summary JSON file.
    
    Args:
        study: Optuna study object
        output_dir: Output directory for Optuna files
    """
    summary = {
        'study_name': study.study_name,
        'n_trials': len(study.trials),
        'best_trial': {
            'number': study.best_trial.number,
            'value': study.best_trial.value,
            'params': study.best_trial.params,
            'user_attrs': study.best_trial.user_attrs
        },
        'trials_summary': [
            {
                'number': t.number,
                'value': t.value,
                'state': t.state.name,
                'params': t.params,
                'user_attrs': t.user_attrs
            }
            for t in study.trials
        ]
    }
    
    summary_path = os.path.join(output_dir, 'study_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Also save to outputs_real
    outputs_summary_path = os.path.join(OUTPUTS_ROOT, 'optuna_study_summary.json')
    with open(outputs_summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[OK] Study summary saved to {summary_path}")


def generate_study_name(base_config_path: str, epoch_min: int = None, epoch_max: int = None) -> str:
    """
    Generate study name based on configuration to avoid mixing different experimental settings.
    
    Args:
        base_config_path: Path to base configuration file
        epoch_min: Minimum epoch value (e.g., 100)
        epoch_max: Maximum epoch value (e.g., 300)
    
    Returns:
        Study name string
    """
    cfg = OmegaConf.load(base_config_path)
    
    # Extract key settings that affect results
    in_channels = cfg.model.get('in_channels', 4)
    dataset_path = cfg.dataset.get('x_train', '')
    
    # Create a hash from dataset path to distinguish different datasets
    import hashlib
    dataset_hash = hashlib.md5(dataset_path.encode()).hexdigest()[:8]
    
    # Generate study name based on key settings
    study_name = f'cnn_opt_c{in_channels}_{dataset_hash}'
    
    # Add epoch range to study name if provided
    if epoch_min is not None and epoch_max is not None:
        study_name += f'_ep{epoch_min}-{epoch_max}'
    
    return study_name


def main():
    """
    Main function to run Optuna optimization.
    """
    # Create Optuna directory
    os.makedirs(OPTUNA_DIR, exist_ok=True)
    
    # Create study database
    study_db_path = os.path.join(OPTUNA_DIR, 'study.db')
    storage = optuna.storages.RDBStorage(
        url=f'sqlite:///{study_db_path}',
        engine_kwargs={'pool_size': 20}
    )
    
    # Generate study name based on configuration
    # This ensures different experimental settings use different studies
    # Epoch range is included to distinguish studies with different training durations
    study_name = generate_study_name(BASE_CONFIG_PATH, epoch_min=EPOCH_MIN, epoch_max=EPOCH_MAX)
    print(f"[INFO] Generated study name: {study_name}")
    print(f"[INFO] This ensures different settings (channels, dataset, epoch range, etc.) use separate studies")
    print(f"[INFO] Epoch range: {EPOCH_MIN}-{EPOCH_MAX} (step={EPOCH_STEP})")
    try:
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',  # Minimize validation loss
            storage=storage,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,      # Don't prune first 5 trials
                n_warmup_steps=10,        # Wait 10 epochs before pruning
                interval_steps=1          # Check every epoch
            )
        )
        n_existing = len(study.trials)
        print(f"[INFO] Loaded existing study: {study_name}")
        print(f"[INFO] Found {n_existing} existing trial(s)")
        if n_existing > 0:
            print(f"[WARN] Continuing optimization with existing trials.")
            print(f"[WARN] Make sure the configuration (channels, dataset, etc.) matches previous runs!")
    except Exception as e:
        print(f"[INFO] Creating new study: {study_name}")
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            storage=storage,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
    
    print(f"[INFO] Study database: {study_db_path}")
    print(f"[INFO] Number of existing trials: {len(study.trials)}\n")
    
    # Clear GPU memory before starting optimization
    if torch.cuda.is_available():
        print("[INFO] Clearing GPU memory before optimization...")
        import gc
        
        # Multiple rounds of garbage collection and cache clearing
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        
        # Print initial memory status
        _print_memory_status("before optimization starts")
        
        print("[OK] GPU memory cleared")
    
    # Optimize
    print(f"[INFO] Starting optimization...")
    print(f"[INFO] Base config: {BASE_CONFIG_PATH}\n")
    
    study.optimize(
        lambda trial: objective(trial, BASE_CONFIG_PATH),
        n_trials=20,  # Adjust as needed
        n_jobs=1,      # Set to 1 for GPU usage, increase for parallel CPU trials
        show_progress_bar=True
    )
    
    # Print best trial results
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETED")
    print(f"{'='*60}")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Value (Validation Loss): {study.best_trial.value:.6f}")
    print(f"Best Params:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    print(f"\nTest Metrics:")
    print(f"  Test MSE: {study.best_trial.user_attrs.get('test_mse', 'N/A')}")
    print(f"  Test MAE: {study.best_trial.user_attrs.get('test_mae', 'N/A')}")
    print(f"{'='*60}\n")
    
    # Find and create symlink to best trial
    best_trial_dir = find_trial_output_dir(study.best_trial.number)
    if best_trial_dir:
        best_link = os.path.join(OUTPUTS_ROOT, 'optuna_best')
        if os.path.exists(best_link) or os.path.islink(best_link):
            if os.path.islink(best_link):
                os.remove(best_link)
            else:
                shutil.rmtree(best_link)
        os.symlink(best_trial_dir, best_link)
        print(f"[OK] Best trial directory linked to: {best_link}")
        print(f"      Original: {best_trial_dir}\n")
    
    # Generate summary
    generate_study_summary(study, OPTUNA_DIR)
    
    # Save best trial info
    best_trial_info = {
        'trial_number': study.best_trial.number,
        'validation_loss': study.best_trial.value,
        'params': study.best_trial.params,
        'user_attrs': study.best_trial.user_attrs,
        'output_dir': best_trial_dir if best_trial_dir else None
    }
    best_trial_info_path = os.path.join(OPTUNA_DIR, 'best_trial_info.yaml')
    OmegaConf.save(best_trial_info, best_trial_info_path)
    print(f"[OK] Best trial info saved to {best_trial_info_path}")
    
    print(f"\n[INFO] All results saved to:")
    print(f"  Optuna files: {OPTUNA_DIR}")
    print(f"  Trial outputs: {OUTPUTS_ROOT}")
    print(f"  Best trial link: {os.path.join(OUTPUTS_ROOT, 'optuna_best')}")


if __name__ == "__main__":
    main()

