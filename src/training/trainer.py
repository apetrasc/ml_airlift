"""
Training functions and utilities.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

from src.models.cnn import SimpleCNNReal, SimpleCNNReal2D


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total = 0.0
    use_cuda = device.type == 'cuda'
    scaler = getattr(train_one_epoch, "_scaler", None)
    if scaler is None:
        scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)
        setattr(train_one_epoch, "_scaler", scaler)
    
    # Clear GPU memory before training
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    for i, (xb, yb) in enumerate(dataloader):
        try:
            # Ensure tensors are contiguous and on correct device
            if not xb.is_contiguous():
                xb = xb.contiguous()
            if not yb.is_contiguous():
                yb = yb.contiguous()
            
            xb = xb.to(device, non_blocking=False)  # Use blocking for stability
            yb = yb.to(device, non_blocking=False)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_cuda):
                pred = model(xb)
                # Verify shapes match
                if pred.shape[0] != yb.shape[0]:
                    raise RuntimeError(f"Batch size mismatch: pred={pred.shape[0]}, yb={yb.shape[0]}")
                loss = criterion(pred, yb)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loss_value = loss.item()
            total += loss_value * xb.size(0)
            
            # Clear intermediate tensors and free memory
            del pred, loss
            if use_cuda:
                # Synchronize all GPUs if using DataParallel
                if isinstance(model, nn.DataParallel):
                    # Synchronize all devices used by DataParallel
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.synchronize(i)
                else:
                    torch.cuda.synchronize()  # Single GPU
                
                # Periodically clear cache to prevent fragmentation
                if (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()
        
        except RuntimeError as e:
            if 'CUDA' in str(e) or 'illegal memory access' in str(e).lower():
                print(f"[ERROR] CUDA error at batch {i}: {e}")
                if use_cuda:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                raise
            else:
                raise
    
    # Note: Final step is now printed within the loop if needed, so no need to print here
    # This prevents duplicate printing and ensures messages appear in correct order
    
    return total / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on dataset."""
    model.eval()
    total = 0.0
    preds, targets = [], []
    use_cuda = device.type == 'cuda'
    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)
        with torch.amp.autocast('cuda', enabled=use_cuda):
            pred = model(xb)
            loss = criterion(pred, yb)
        total += loss.item() * xb.size(0)
        preds.append(pred.cpu())
        targets.append(yb.cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    mse = total / len(dataloader.dataset)
    mae = torch.mean(torch.abs(preds - targets)).item()
    return mse, mae, preds.numpy(), targets.numpy()


def create_model(cfg, x_sample: torch.Tensor, out_dim: int, device: torch.device):
    """Create model based on configuration and data shape, with OOM-safe device move."""
    if x_sample.ndim == 3:
        # 1D CNN
        in_channels = x_sample.shape[1]
        length = x_sample.shape[2]
        model = SimpleCNNReal(input_length=length, in_channels=in_channels, out_dim=out_dim)
        print(f"[OK] Using Conv1d model (C={in_channels}, L={length})")
    elif x_sample.ndim == 4:
        # 2D CNN for image data
        valid_channel_counts = {1, 2, 3, 4, cfg.model.get('in_channels', 0)}
        # Remove zeros to avoid false positives when in_channels not set
        valid_channel_counts.discard(0)

        if x_sample.shape[1] not in valid_channel_counts:
            x_nchw = x_sample.permute(0, 3, 1, 2).contiguous()
            print("[INFO] Transposed NHWC -> NCHW")
        else:
            x_nchw = x_sample
        in_channels = x_nchw.shape[1]
        # Get resize parameters from config
        resize_hw = cfg.model.resize_hw
        if resize_hw and resize_hw[0] > 0 and resize_hw[1] > 0:
            resize_hw = tuple(resize_hw)
        else:
            resize_hw = None
        
        # Get model hyperparameters from config
        dropout_rate = cfg.model.get('dropout_rate', 0.2)
        use_residual = cfg.model.get('use_residual', True)
        hidden = cfg.model.get('hidden', 64)
            
        model = SimpleCNNReal2D(
            in_channels=in_channels, 
            out_dim=out_dim, 
            resize_hw=resize_hw,
            dropout_rate=dropout_rate,
            use_residual=use_residual,
            hidden=hidden
        ).to(device)
        print(f"[OK] Using Conv2d model for {in_channels}-channel image data")
        print(f"[OK] Input: (N, {in_channels}, {x_nchw.shape[2]}, {x_nchw.shape[3]}) -> Output: (N, {out_dim})")
        print(f"[OK] Resize: {resize_hw if resize_hw else 'Full resolution'}")
        print(f"[OK] Architecture: {'Residual' if use_residual else 'Simple'}, Hidden: {hidden}, Dropout: {dropout_rate}")
        
    else:
        raise RuntimeError("Unexpected tensor ndim for model selection")

    # Try moving to device; on OOM retry with half precision
    try:
        model = model.to(device)
    except RuntimeError as e:
        if device.type == 'cuda' and 'out of memory' in str(e).lower():
            print("[WARN] OOM while moving model to GPU. Retrying with half precision (fp16)...")
            torch.cuda.empty_cache()
            model = model.half().to(device)
        else:
            raise

    print(f"[OK] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def create_learning_curves(train_losses, val_losses, output_dir):
    """Create learning curve plots."""
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Linear scale plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='red')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(logs_dir, 'learning_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Log scale plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), [np.log(l) for l in train_losses], label='Train Log(Loss)', color='blue')
    plt.plot(range(1, len(val_losses) + 1), [np.log(l) for l in val_losses], label='Validation Log(Loss)', color='red')
    plt.title('Learning Curve (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Log(Loss)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(logs_dir, 'learning_curve_log.png'), dpi=300, bbox_inches='tight')
    plt.close()

