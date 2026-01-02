#!/usr/bin/env python3
"""
Evaluate model and generate Grad-CAM visualizations.
Creates visualizations for gas_velocity, gas_volume_fraction, and liquid_volume_fraction.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.signal import hilbert
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import load_npz_pair, to_tensor_dataset, split_dataset
from src.training.trainer import create_model
from src.models.cnn import SimpleCNNReal2D


class GradCAM2D:
    """Grad-CAM implementation for 2D CNN models."""
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM for 2D CNN.
        
        Args:
            model: PyTorch model
            target_layer: Target layer to compute Grad-CAM (e.g., model.layer3)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))
    
    def __call__(self, input_tensor, output_idx=None):
        """
        Compute Grad-CAM map for the provided input tensor.
        
        Args:
            input_tensor: Input tensor of shape [B, C, H, W]
            output_idx: Index of output dimension to compute Grad-CAM for (None = sum of all outputs)
        
        Returns:
            Grad-CAM map as numpy array of shape [H, W]
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.detach().to(device)
        input_tensor.requires_grad_(True)
        
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if output_idx is None:
            # Sum all outputs for multi-output regression
            target = output.sum()
        else:
            target = output[:, output_idx]
            if target.ndim > 0:
                target = target.sum()
        
        target.backward(retain_graph=True)
        
        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]
        
        # Compute weights: global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Compute Grad-CAM: weighted sum of activations
        grad_cam_map = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        grad_cam_map = torch.relu(grad_cam_map)
        
        # Interpolate to input size if needed
        if grad_cam_map.shape[2:] != input_tensor.shape[2:]:
            grad_cam_map = F.interpolate(
                grad_cam_map, 
                size=input_tensor.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        grad_cam_map = grad_cam_map.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        if grad_cam_map.max() > grad_cam_map.min():
            grad_cam_map = (grad_cam_map - grad_cam_map.min()) / (grad_cam_map.max() - grad_cam_map.min() + 1e-8)
        
        return grad_cam_map
    
    def remove_hooks(self):
        """Remove registered hooks."""
        for handle in self.hook_handles:
            handle.remove()


def tensor_to_image(tensor, channel_idx=0, normalize=False):
    """
    Convert tensor to image for visualization.
    
    Args:
        tensor: Image tensor of shape [C, H, W] or [H, W]
        channel_idx: Channel index to visualize (for multi-channel inputs)
        normalize: If True, normalize to [0, 1]. If False, return raw values.
    
    Returns:
        numpy.ndarray: Image array of shape [H, W]
    """
    if tensor.ndim == 3:
        # Multi-channel: select one channel
        image = tensor[channel_idx].cpu().numpy()
    elif tensor.ndim == 2:
        image = tensor.cpu().numpy()
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    
    # Normalize to [0, 1] for visualization if requested
    if normalize and image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    return image


def visualize_sample_gradcam(input_tensor, gradcam_map, target_name, save_path,
                              sample_idx, target_value=None, pred_value=None):
    """
    Visualize sample image, Grad-CAM heatmap, and overlay in one figure.
    
    Args:
        input_tensor: Input tensor of shape [1, C, H, W]
        gradcam_map: Grad-CAM map of shape [H, W]
        target_name: Name of target variable
        save_path: Path to save visualization
        sample_idx: Sample index for title
        target_value: True target value (optional)
        pred_value: Predicted value (optional)
    """
    # Apply Hilbert transform to input for visualization
    x_numpy = input_tensor[0].cpu().numpy()  # [C, H, W]
    hilbert_images = []
    for c in range(x_numpy.shape[0]):
        hilbert_channel = np.zeros((x_numpy.shape[1], x_numpy.shape[2]))
        for h in range(x_numpy.shape[1]):
            hilbert_signal = hilbert(x_numpy[c, h, :])
            hilbert_channel[h, :] = np.abs(hilbert_signal)
        hilbert_images.append(hilbert_channel)
    
    # Use first channel for sample image visualization
    # Normalize by max value, then apply log1p compression
    sample_image_raw = hilbert_images[0]
    max_val = sample_image_raw.max()
    if max_val > 0:
        sample_image_normalized = sample_image_raw / max_val  # Normalize to [0, 1]
    else:
        sample_image_normalized = sample_image_raw
    sample_image = np.log1p(sample_image_normalized)  # Log compression: log(1 + x)
    
    # Create figure with 3 columns: Sample, Grad-CAM, Overlay
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1], wspace=0.3)
    
    # Build title with target and prediction values
    title_suffix = ""
    if target_value is not None and pred_value is not None:
        error = abs(target_value - pred_value)
        title_suffix = f"\nTarget: {target_value:.4f} | Pred: {pred_value:.4f} | Error: {error:.4f}"
    elif pred_value is not None:
        title_suffix = f"\nPred: {pred_value:.4f}"
    elif target_value is not None:
        title_suffix = f"\nTarget: {target_value:.4f}"
    
    # Column 1: Sample image (Hilbert transform)
    ax_sample = fig.add_subplot(gs[0, 0])
    im_sample = ax_sample.imshow(sample_image, cmap='jet', aspect='auto', vmin=0.01, vmax=0.06)
    ax_sample.set_title(f'Sample {sample_idx} | {target_name}{title_suffix}', fontsize=12)
    ax_sample.axis('off')
    plt.colorbar(im_sample, ax=ax_sample, fraction=0.046, pad=0.04)
    
    # Column 2: Grad-CAM heatmap
    ax_gradcam = fig.add_subplot(gs[0, 1])
    im_gradcam = ax_gradcam.imshow(gradcam_map, cmap='jet', aspect='auto', vmin=0, vmax=1)
    ax_gradcam.set_title(f'Grad-CAM Heatmap\nSample {sample_idx} | {target_name}', fontsize=12)
    ax_gradcam.axis('off')
    plt.colorbar(im_gradcam, ax=ax_gradcam, fraction=0.046, pad=0.04)
    
    # Column 3: Overlay
    ax_overlay = fig.add_subplot(gs[0, 2])
    ax_overlay.imshow(sample_image, cmap='jet', aspect='auto', alpha=0.7)
    im_overlay = ax_overlay.imshow(gradcam_map, cmap='jet', alpha=0.5, aspect='auto', vmin=0, vmax=1)
    ax_overlay.set_title(f'Overlay\nSample {sample_idx} | {target_name}', fontsize=12)
    ax_overlay.axis('off')
    plt.colorbar(im_overlay, ax=ax_overlay, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Main function to evaluate model and generate Grad-CAM visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model and generate Grad-CAM visualizations')
    parser.add_argument('--trial_dir', type=str, 
                        default='/home/smatsubara/documents/airlift/data/outputs_real/optuna/runs/2025-12-21/22-18-50',
                        help='Path to trial directory containing config.yaml and weights')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to visualize (None = all test samples)')
    args = parser.parse_args()
    
    trial_dir = Path(args.trial_dir)
    config_path = trial_dir / "config.yaml"
    model_path = trial_dir / "weights" / "model_simplecnn_real.pth"
    
    # Setup device (CUDA:1)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load configuration
    print(f"[INFO] Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # Load data
    print(f"[INFO] Loading dataset...")
    x, t = load_npz_pair(
        cfg.dataset.x_train,
        cfg.dataset.t_train,
        cfg.dataset.x_key,
        cfg.dataset.t_key
    )
    print(f"[OK] Loaded. x.shape={x.shape}, t.shape={t.shape}")
    
    # Exclude Channel 1 and Channel 3 (keep only channels 0, 2) if needed
    if x.ndim == 4 and x.shape[1] == 4:
        print(f"[INFO] Excluding Channel 1 and Channel 3 (keeping channels 0, 2)")
        x = x[:, [0, 2], :, :]
        print(f"[OK] After excluding: x.shape={x.shape}")
    elif x.ndim == 3 and x.shape[1] == 4:
        print(f"[INFO] Excluding Channel 1 and Channel 3 (keeping channels 0, 2)")
        x = x[:, [0, 2], :]
        print(f"[OK] After excluding: x.shape={x.shape}")
    
    # Optional downsampling
    if x.ndim == 4 and cfg.dataset.downsample_factor > 1:
        h0 = x.shape[2]
        x = x[:, :, ::cfg.dataset.downsample_factor, :]
        print(f"[INFO] Downsampled H: {h0} -> {x.shape[2]} (factor={cfg.dataset.downsample_factor})")
    
    # Create dataset and split
    print(f"[INFO] Creating dataset and splitting...")
    dataset = to_tensor_dataset(x, t, "cpu")
    train_set, val_set, test_set = split_dataset(
        dataset,
        cfg.data_split.train_ratio,
        cfg.data_split.val_ratio,
        cfg.data_split.test_ratio,
        cfg.training.seed
    )
    print(f"[OK] Sizes -> train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    
    # Load model
    print(f"[INFO] Loading model from: {model_path}")
    # Get a sample from test_set (Subset object)
    x_sample, _ = test_set[0]
    x_sample = x_sample.unsqueeze(0)  # Add batch dimension: [C, H, W] -> [1, C, H, W]
    # Get out_dim from dataset
    _, y_sample = dataset[0]
    if isinstance(y_sample, torch.Tensor):
        out_dim = y_sample.shape[0] if y_sample.ndim >= 1 else 1
    else:
        out_dim = 1
    model = create_model(cfg, x_sample, out_dim, device)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[OK] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Select target layer for Grad-CAM
    if cfg.model.use_residual:
        target_layer = model.layer3
        print("[INFO] Using layer3 as target layer for Grad-CAM")
    else:
        target_layer = model.features[-2]
        print("[INFO] Using last conv layer as target layer for Grad-CAM")
    
    # Initialize Grad-CAM
    gradcam = GradCAM2D(model, target_layer)
    
    # Get target names and output indices
    target_names = cfg.evaluation.target_names
    output_indices = {
        'gas_velocity': None,
        'gas_volume_fraction': None,
        'liquid_volume_fraction': None
    }
    
    # Find output indices
    for idx, name in enumerate(target_names):
        if 'Gas Velocity' in name:
            output_indices['gas_velocity'] = idx
        elif 'Gas Volume Fraction' in name:
            output_indices['gas_volume_fraction'] = idx
        elif 'Liquid Volume Fraction' in name:
            output_indices['liquid_volume_fraction'] = idx
    
    print(f"[INFO] Output indices: {output_indices}")
    
    # Create output directories
    base_output_dir = trial_dir / "logs" / "CAMS"
    output_dirs = {
        'gas_velocity': base_output_dir / "gas_velocity",
        'gas_volume_fraction': base_output_dir / "gas_volume_fraction",
        'liquid_volume_fraction': base_output_dir / "liquid_volume_fraction"
    }
    
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Determine number of samples to process
    num_samples = args.num_samples if args.num_samples else len(test_set)
    num_samples = min(num_samples, len(test_set))
    print(f"[INFO] Processing {num_samples} samples from test set")
    
    # Generate visualizations
    print(f"\n[INFO] Generating Grad-CAM visualizations...")
    
    # Create DataLoader for test_set to access samples
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    test_iter = iter(test_loader)
    
    for sample_idx in range(num_samples):
        # Get sample from DataLoader iterator
        input_sample, target_sample = next(test_iter)
        input_sample = input_sample.to(device)  # [1, C, H, W]
        target_sample = target_sample[0]  # Remove batch dimension: [out_dim]
        
        # Get prediction
        with torch.no_grad():
            pred = model(input_sample)
            pred = pred.cpu().numpy()[0]
        
        # Process each target variable
        for target_key, output_idx in output_indices.items():
            if output_idx is None:
                continue
            
            target_name = target_names[output_idx]
            output_dir = output_dirs[target_key]
            
            print(f"  Sample {sample_idx}, {target_name} (output {output_idx})...")
            
            # Get target and prediction values
            target_val = target_sample[output_idx].item() if target_sample is not None else None
            pred_val = pred[output_idx]
            
            # Compute Grad-CAM
            gradcam_map = gradcam(input_sample, output_idx=output_idx)
            
            # Save visualization
            save_path = output_dir / f"sample{sample_idx:04d}_output{output_idx}.png"
            visualize_sample_gradcam(
                input_sample.cpu(),
                gradcam_map,
                target_name,
                save_path,
                sample_idx=sample_idx,
                target_value=target_val,
                pred_value=pred_val
            )
            
            # Print prediction info
            if target_val is not None:
                error = abs(target_val - pred_val)
                print(f"    Target: {target_val:.4f} | Prediction: {pred_val:.4f} | Error: {error:.4f}")
            else:
                print(f"    Prediction: {pred_val:.4f}")
    
    # Cleanup
    gradcam.remove_hooks()
    
    print(f"\n✅ Grad-CAM visualization complete!")
    print(f"Results saved to:")
    for target_key, dir_path in output_dirs.items():
        print(f"  - {target_key}: {dir_path}")


if __name__ == "__main__":
    main()

