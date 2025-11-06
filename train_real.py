#!/usr/bin/env python3
"""
Train CNN model using Hydra configuration management.
Automatically creates timestamped output directories.
"""

import os
import time
import datetime
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from models.cnn import SimpleCNNReal, SimpleCNNReal2D
from src.evaluate_predictions import create_prediction_plots
import matplotlib.pyplot as plt

import hydra
from omegaconf import OmegaConf


def _load_np_any(path: str, prefer_key: str = None):
    """Load numpy array from .npy or .npz file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found: {path}")
    obj = np.load(path)
    # npz: dict-like, npy: ndarray
    if hasattr(obj, 'keys'):
        keys = list(obj.keys())
        if prefer_key and prefer_key in obj:
            arr = obj[prefer_key]
        else:
            if prefer_key and prefer_key not in obj:
                print(f"[WARN] key '{prefer_key}' not in {path}. Using first key: {keys[0]}")
            arr = obj[keys[0]]
        return arr
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


def train_one_epoch(model, dataloader, criterion, optimizer, device, print_every=50):
    """Train model for one epoch."""
    model.train()
    total = 0.0
    use_cuda = device.type == 'cuda'
    scaler = getattr(train_one_epoch, "_scaler", None)
    if scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
        setattr(train_one_epoch, "_scaler", scaler)
    for i, (xb, yb) in enumerate(dataloader):
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_cuda):
            pred = model(xb)
            loss = criterion(pred, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * xb.size(0)
        if (i + 1) % print_every == 0:
            print(f"  [train] step {i+1}/{len(dataloader)} loss={loss.item():.6f}")
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
        with torch.cuda.amp.autocast(enabled=use_cuda):
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
        if x_sample.shape[1] not in (1, 3, 4):
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
        model = SimpleCNNReal2D(in_channels=in_channels, out_dim=out_dim, resize_hw=resize_hw)
        print(f"[OK] Using Conv2d model for {in_channels}-channel image data")
        print(f"[OK] Input: (N, {in_channels}, {x_nchw.shape[2]}, {x_nchw.shape[3]}) -> Output: (N, {out_dim})")
        print(f"[OK] Resize: {resize_hw if resize_hw else 'Full resolution'}")
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


@hydra.main(config_path="config", config_name="config_real_updated.yaml", version_base=None)
def main(cfg):
    """Main training function with Hydra configuration."""
    
    # Prevent Hydra from creating output directories in sandbox/ml_airlift
    # Get the original working directory (before Hydra changes it)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sandbox_dir = "/home/smatsubara/documents/sandbox/ml_airlift"
    
    # Check and remove Hydra-created output directories in sandbox/ml_airlift
    outputs_dir = os.path.join(sandbox_dir, "outputs")
    if os.path.exists(outputs_dir):
        try:
            shutil.rmtree(outputs_dir)
            print(f"[INFO] Removed Hydra output directory: {outputs_dir}")
        except Exception as e:
            print(f"[WARN] Could not remove Hydra output directory {outputs_dir}: {e}")
    
    # Create time-based output directory under airlift/data/outputs_real/YYYY-MM-DD/HH-MM-SS
    outputs_root = cfg.output.model_save_dir
    now = datetime.datetime.now()
    run_dir = os.path.join(outputs_root, now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))
    base_dir = run_dir
    logs_dir = os.path.join(run_dir, "logs")
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    print(f"[INFO] Run directory: {os.path.abspath(run_dir)}")
    print(f"[INFO] Logs will be saved under: {os.path.abspath(logs_dir)}")
    
    # Print configuration
    print("🔧 Configuration Summary")
    print("=" * 50)
    print(f"Dataset X: {cfg.dataset.x_train}")
    print(f"Dataset T: {cfg.dataset.t_train}")
    print(f"Model: {cfg.model.type}")
    print(f"Epochs: {cfg.training.epochs}")
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Learning rate: {cfg.training.learning_rate}")
    print(f"Device: {cfg.training.device}")
    print("=" * 50)
    
    # Set reproducibility
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    # cuDNN autotune for convs
    torch.backends.cudnn.benchmark = True

    t0 = time.time()
    
    # Load data
    print("[STEP] Loading dataset files...")
    x, t = load_npz_pair(cfg.dataset.x_train, cfg.dataset.t_train, cfg.dataset.x_key, cfg.dataset.t_key)
    print(f"[OK] Loaded. x.shape={x.shape}, t.shape={t.shape} (elapsed {time.time()-t0:.2f}s)")
    
    # Check for NaNs
    if np.isnan(x).any():
        print("[WARNING] NaN values detected in x!")
    else:
        print("[INFO] No NaN values in x.")
    if np.isnan(t).any():
        print("[WARNING] NaN values detected in t!")
    else:
        print("[INFO] No NaN values in t.")
    
    # Print data info for 4D case
    if x.ndim == 4:
        print(f"[INFO] 4D Image Data: N={x.shape[0]}, C={x.shape[1]}, H={x.shape[2]}, W={x.shape[3]}")
        print(f"[INFO] Target shape: {t.shape} (6 targets for multi-output regression)")
        print(f"[INFO] Memory usage: {x.nbytes / 1024**2:.1f} MB")
    
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
    elif x.ndim == 4:
        print(f"[INFO] Using full resolution: H={x.shape[2]}, W={x.shape[3]}")
    
    # Create dataset
    print("[STEP] Build dataset tensors...")
    dataset = to_tensor_dataset(x, t, cfg.training.device)
    print("[OK] Dataset ready.")
    
    # Split dataset
    print("[STEP] Split dataset...")
    train_set, val_set, test_set = split_dataset(
        dataset, 
        cfg.data_split.train_ratio, 
        cfg.data_split.val_ratio, 
        cfg.data_split.test_ratio, 
        cfg.training.seed
    )
    print(f"[OK] Sizes -> train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    
    # Create dataloaders
    print("[STEP] Build dataloaders...")
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.training.workers, 
        pin_memory=cfg.training.pin_memory
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=cfg.training.batch_size, 
        shuffle=False, 
        num_workers=cfg.training.workers, 
        pin_memory=cfg.training.pin_memory
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=cfg.training.batch_size, 
        shuffle=False, 
        num_workers=cfg.training.workers, 
        pin_memory=cfg.training.pin_memory
    )
    print("[OK] Dataloaders ready.")
    
    # Create model
    print("[STEP] Build model...")
    x_sample = dataset.tensors[0]
    device = get_valid_device(cfg.training.device)
    out_dim = dataset.tensors[1].shape[1] if dataset.tensors[1].ndim == 2 else 1
    model = create_model(cfg, x_sample, out_dim, device)
    
    # Create optimizer and loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    # Training
    print("[STEP] Training...")
    print(f"[INFO] Training {x_sample.shape[1]}-channel image data with {out_dim} output targets")
    print(f"[INFO] Batch size: {cfg.training.batch_size}, Epochs: {cfg.training.epochs}")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, cfg.training.epochs + 1):
        t_ep = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, cfg.logging.print_every_n_batches)
        
        # Evaluate validation set if it exists
        if len(val_loader.dataset) > 0:
            val_mse, val_mae, _, _ = evaluate(model, val_loader, criterion, device)
        else:
            val_mse, val_mae = 0.0, 0.0
        
        train_losses.append(tr)
        val_losses.append(val_mse)
        
        if epoch % cfg.logging.print_every_n_epochs == 0:
            if len(val_loader.dataset) > 0:
                print(f"Epoch {epoch:03d} | train MSE={tr:.6f} | val MSE={val_mse:.6f} | val MAE={val_mae:.6f} | {time.time()-t_ep:.2f}s")
            else:
                print(f"Epoch {epoch:03d} | train MSE={tr:.6f} | val MSE=N/A | val MAE=N/A | {time.time()-t_ep:.2f}s")
    
    # Testing
    print("[STEP] Testing...")
    test_mse, test_mae, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    print(f"Test  | MSE={test_mse:.6f} | MAE={test_mae:.6f}")
    
    # Print per-target results for multi-output regression
    if y_pred.shape[1] > 1:
        print(f"[INFO] Per-target results:")
        for i in range(y_pred.shape[1]):
            target_mse = np.mean((y_pred[:, i] - y_true[:, i])**2)
            target_mae = np.mean(np.abs(y_pred[:, i] - y_true[:, i]))
            print(f"  Target {i+1}: MSE={target_mse:.6f}, MAE={target_mae:.6f}")
    
    # Save model and results
    print("[STEP] Saving model and results...")
    torch.save(model.state_dict(), os.path.join(weights_dir, cfg.output.model_filename))
    np.save(os.path.join(base_dir, cfg.output.predictions_filename), y_pred)
    np.save(os.path.join(base_dir, cfg.output.ground_truth_filename), y_true)
    print(f"[OK] Saved to {base_dir}")
    
    # Create learning curves
    print("[STEP] Creating learning curves...")
    create_learning_curves(train_losses, val_losses, base_dir)
    
    # Create evaluation plots if requested
    if cfg.evaluation.create_plots and y_pred.shape[1] > 1:
        print("[STEP] Creating evaluation plots...")
        plots_dir = os.path.join(base_dir, cfg.output.evaluation_plots_dir)
        create_prediction_plots(y_pred, y_true, plots_dir, cfg.evaluation.target_names)
        print(f"[OK] Evaluation plots saved to {plots_dir}")
    
    # Save configuration
    with open(os.path.join(base_dir, "config.yaml"), 'w') as f:
        OmegaConf.save(cfg, f)
    print(f"[OK] Configuration saved to {base_dir}/config.yaml")
    
    print(f"[DONE] Training completed. Total elapsed {time.time()-t0:.2f}s")
    print(f"[DONE] Results saved to: {base_dir}")
    
    # Clean up Hydra-created output directories in sandbox/ml_airlift
    outputs_dir = os.path.join(sandbox_dir, "outputs")
    if os.path.exists(outputs_dir):
        try:
            shutil.rmtree(outputs_dir)
            print(f"[INFO] Cleaned up Hydra output directory: {outputs_dir}")
        except Exception as e:
            print(f"[WARN] Could not remove Hydra output directory {outputs_dir}: {e}")


if __name__ == "__main__":
    main()
