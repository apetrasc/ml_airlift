#!/usr/bin/env python3
"""
Grad-CAM visualization script for SimpleCNNReal2D model.
Visualizes Grad-CAM heatmaps for 2D CNN inputs.
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.signal import hilbert

# Add project root to path
sys.path.append('/home/smatsubara/documents/sandbox/ml_airlift')
from models.cnn import SimpleCNNReal2D


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


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config, model_path, device):
    """Load model from configuration and weights."""
    # Create model
    model = SimpleCNNReal2D(
        in_channels=config['model']['in_channels'],
        hidden=config['model']['hidden'],
        out_dim=config['model']['out_dim'],
        resize_hw=config['model']['resize_hw'],
        dropout_rate=config['model']['dropout_rate'],
        use_residual=config['model']['use_residual']
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def load_data(x_path, t_path=None):
    """Load input data and optionally target data."""
    x = np.load(x_path)
    print(f"Loaded data: shape={x.shape}, dtype={x.dtype}")
    
    t = None
    if t_path is not None and os.path.exists(t_path):
        t = np.load(t_path)
        print(f"Loaded targets: shape={t.shape}, dtype={t.dtype}")
    
    return x, t


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


def visualize_gradcam(input_tensor, gradcam_map, output_idx, target_name, save_path, 
                      channel_idx=0, sample_idx=0, target_value=None, pred_value=None,
                      hilbert_tensor=None):
    """
    Visualize Grad-CAM heatmap.
    
    Args:
        input_tensor: Input tensor of shape [1, C, H, W] (for Grad-CAM computation)
        gradcam_map: Grad-CAM map of shape [H, W]
        output_idx: Index of output dimension
        target_name: Name of target variable
        save_path: Path to save visualization
        channel_idx: Channel index to visualize
        sample_idx: Sample index for title
        target_value: True target value (optional)
        pred_value: Predicted value (optional)
        hilbert_tensor: Hilbert transformed tensor of shape [1, C, H, W] for display (optional)
    """
    # Convert original input to image (for Overlay background)
    original_image = tensor_to_image(input_tensor[0], channel_idx=channel_idx)
    
    # Get number of channels
    num_channels = input_tensor.shape[1]
    
    # Prepare Hilbert transformed images for each channel
    hilbert_images = []
    if hilbert_tensor is not None:
        # hilbert_tensor shape: [1, C, H, W] or [1, 1, H, W]
        if hilbert_tensor.shape[1] == num_channels:
            # Each channel has its own Hilbert transform
            for c in range(num_channels):
                hilbert_img = tensor_to_image(hilbert_tensor[0], channel_idx=c, normalize=False)
                hilbert_images.append(hilbert_img)
        else:
            # Single channel Hilbert transform (averaged)
            hilbert_img = tensor_to_image(hilbert_tensor[0], channel_idx=0, normalize=False)
            hilbert_images = [hilbert_img] * num_channels
    else:
        # Fallback: apply Hilbert transform to each channel
        x_numpy = input_tensor[0].cpu().numpy()  # [C, H, W]
        for c in range(num_channels):
            hilbert_channel = np.zeros((x_numpy.shape[1], x_numpy.shape[2]))
            for h in range(x_numpy.shape[1]):
                hilbert_signal = hilbert(x_numpy[c, h, :])
                hilbert_channel[h, :] = np.abs(hilbert_signal)
            hilbert_images.append(hilbert_channel)
    
    # Create visualization with custom layout
    # Left: 2 rows (TDX1, TDX3), Right: 1 row (Grad-CAM, Overlay)
    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1], 
                  hspace=0.3, wspace=0.3)
    
    # Build title with target and prediction values
    title_suffix = ""
    if target_value is not None and pred_value is not None:
        error = abs(target_value - pred_value)
        title_suffix = f"\nTarget: {target_value:.4f} | Pred: {pred_value:.4f} | Error: {error:.4f}"
    elif pred_value is not None:
        title_suffix = f"\nPred: {pred_value:.4f}"
    elif target_value is not None:
        title_suffix = f"\nTarget: {target_value:.4f}"
    
    # Left side: TDX1 and TDX3 (stacked vertically)
    # NOTE: In the dropped dataset, channel 0 corresponds to TDX1 and channel 1 corresponds to TDX3.
    channel_names = ['TDX1', 'TDX3']
    for c in range(min(num_channels, 2)):
        ax = fig.add_subplot(gs[c, 0])
        hilbert_img = hilbert_images[c]
        hilbert_min = hilbert_img.min()
        hilbert_max = hilbert_img.max()
        hilbert_mean = hilbert_img.mean()
        hilbert_std = hilbert_img.std()
        
        # Also check original signal for comparison
        original_channel = tensor_to_image(input_tensor[0], channel_idx=c, normalize=False)
        orig_min = original_channel.min()
        orig_max = original_channel.max()
        
        im = ax.imshow(hilbert_img, cmap='jet', aspect='auto', vmin=hilbert_min, vmax=hilbert_max)
        ax.set_title(f'Sample {sample_idx} | {channel_names[c]} (Hilbert)\n'
                    f'Original: [{orig_min:.4f}, {orig_max:.4f}] | '
                    f'Hilbert: [{hilbert_min:.4f}, {hilbert_max:.4f}]\n'
                    f'Mean: {hilbert_mean:.4f}, Std: {hilbert_std:.4f}', 
                    fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # If only one channel, hide the second subplot
    if num_channels < 2:
        ax = fig.add_subplot(gs[1, 0])
        ax.axis('off')
    
    # Middle: Grad-CAM heatmap (spans both rows)
    ax_gradcam = fig.add_subplot(gs[:, 1])
    im1 = ax_gradcam.imshow(gradcam_map, cmap='jet', aspect='auto', vmin=0, vmax=1)
    ax_gradcam.set_title(f'Sample {sample_idx} | Grad-CAM Heatmap\n{target_name}{title_suffix}', fontsize=12)
    ax_gradcam.axis('off')
    plt.colorbar(im1, ax=ax_gradcam, fraction=0.046, pad=0.04)
    
    # Right: Overlay (spans both rows)
    ax_overlay = fig.add_subplot(gs[:, 2])
    ax_overlay.imshow(original_image, cmap='jet', aspect='auto', vmin=0, vmax=0.5)
    im2 = ax_overlay.imshow(gradcam_map, cmap='jet', alpha=0.5, aspect='auto', vmin=0, vmax=1)
    ax_overlay.set_title(f'Overlay\nSample {sample_idx} | {target_name}{title_suffix}', fontsize=12)
    ax_overlay.axis('off')
    plt.colorbar(im2, ax=ax_overlay, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Main function to run Grad-CAM visualization."""
    # Paths
    base_dir = Path("/home/smatsubara/documents/airlift/data/outputs_real/optuna/runs/optuna_best")
    config_path = base_dir / "config.yaml"
    model_path = base_dir / "weights" / "model_simplecnn_real.pth"
    x_path = "/home/smatsubara/documents/airlift/data/cleaned/x_train_dropped.npy"
    t_path = "/home/smatsubara/documents/airlift/data/cleaned/t_train_real_cleaned.npy"
    base_output_dir = Path("/home/smatsubara/documents/airlift/data/sandbox/visualize/images")
    
    # Create output directories for each output variable
    gas_velocity_dir = base_output_dir / "gas_velocity"
    gas_void_fraction_dir = base_output_dir / "gas_void_fraction"
    gas_velocity_dir.mkdir(parents=True, exist_ok=True)
    gas_void_fraction_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)
    print(f"Model: {config['model']['type']}")
    print(f"Output dimensions: {config['model']['out_dim']}")
    
    # Setup device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(config, model_path, device)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Select target layer for Grad-CAM
    if config['model']['use_residual']:
        # Use layer3 or layer4 for deeper features
        target_layer = model.layer3
        print("Using layer3 as target layer for Grad-CAM")
    else:
        # Use last conv layer in features
        target_layer = model.features[-2]  # Last Conv2d before ReLU
        print("Using last conv layer as target layer for Grad-CAM")
    
    # Initialize Grad-CAM
    gradcam = GradCAM2D(model, target_layer)
    
    # Load data
    print("Loading data...")
    x, t = load_data(x_path, t_path)
    
    # Convert to tensor
    x_tensor = torch.from_numpy(x).float()
    print(f"Original data shape: {x.shape}")
    
    # Data is already in [N, C, H, W] format
    if x_tensor.ndim == 4:
        print(f"Data already in [N, C, H, W] format: {x_tensor.shape}")
        # Verify channel dimension matches
        if x_tensor.shape[1] != config['model']['in_channels']:
            print(f"Warning: Channel dimension mismatch. Expected {config['model']['in_channels']}, got {x_tensor.shape[1]}")
            print(f"Taking first {config['model']['in_channels']} channels...")
            x_tensor = x_tensor[:, :config['model']['in_channels'], :, :]
    elif x_tensor.ndim == 3:
        # Check if it's [N, C, L] (1D) or [N, H, W] (2D)
        if x_tensor.shape[1] == config['model']['in_channels']:
            # It's [N, C, L] - need to reshape to 2D
            # Add spatial dimension: [N, C, L] -> [N, C, 1, L]
            x_tensor = x_tensor.unsqueeze(2)
            print(f"Reshaped from [N, C, L] to [N, C, 1, L]: {x_tensor.shape}")
        else:
            # It's [N, H, W] - add channel dimension: [N, H, W] -> [N, 1, H, W]
            x_tensor = x_tensor.unsqueeze(1)
            print(f"Added channel dimension: {x_tensor.shape}")
    else:
        raise ValueError(f"Unexpected data shape: {x.shape}")
    
    print(f"Final data tensor shape: {x_tensor.shape}")
    
    # Apply Hilbert transform to all samples for visualization
    # Convert to numpy for Hilbert transform
    x_numpy = x_tensor.cpu().numpy()  # [N, C, H, W]
    print("Applying Hilbert transform to all samples...")
    
    # Apply Hilbert transform along the last axis (W) for each sample, channel, and row
    # Keep each channel separate for visualization
    hilbert_tensor = np.zeros((x_numpy.shape[0], x_numpy.shape[1], x_numpy.shape[2], x_numpy.shape[3]))
    for n in range(x_numpy.shape[0]):
        for c in range(x_numpy.shape[1]):
            for h in range(x_numpy.shape[2]):
                # Apply Hilbert transform along W axis
                hilbert_signal = hilbert(x_numpy[n, c, h, :])
                # Take absolute value (envelope)
                hilbert_tensor[n, c, h, :] = np.abs(hilbert_signal)
    
    # Convert back to tensor
    x_hilbert_tensor = torch.from_numpy(hilbert_tensor).float()
    print(f"Hilbert transformed tensor shape: {x_hilbert_tensor.shape}")
    
    # Select samples to visualize - use more samples
    num_samples = min(100, x_tensor.shape[0])  # Visualize up to 100 samples
    sample_indices = np.linspace(0, x_tensor.shape[0] - 1, num_samples, dtype=int)
    
    # Get target names
    target_names = config['evaluation']['target_names']
    
    print(f"\nGenerating Grad-CAM visualizations for {num_samples} samples...")
    
    # Generate visualizations for each sample and output
    for sample_idx in sample_indices:
        input_sample = x_tensor[sample_idx:sample_idx+1].to(device)
        
        # Get prediction
        with torch.no_grad():
            pred = model(input_sample)
            pred = pred.cpu().numpy()[0]
        
        # Compute Grad-CAM for each output dimension (only Gas Velocity and Gas Volume Fraction)
        for output_idx, target_name in enumerate(target_names):
            # Filter: only process Gas Velocity and Gas Volume Fraction
            if target_name not in ["Gas Velocity", "Gas Volume Fraction"]:
                continue
                
            print(f"  Sample {sample_idx}, Output {output_idx} ({target_name})...")
            
            # Get target value if available
            target_val = None
            if t is not None and sample_idx < len(t):
                target_val = t[sample_idx, output_idx]
            
            pred_val = pred[output_idx]
            
            # Compute Grad-CAM
            gradcam_map = gradcam(input_sample, output_idx=output_idx)
            print(f"input_sample.shape: {input_sample.shape}")
            print(f"min: {input_sample.min()}, max: {input_sample.max()}")
            
            # Get Hilbert transformed tensor for this sample
            hilbert_sample = x_hilbert_tensor[sample_idx:sample_idx+1]  # [1, C, H, W]
            
            # Determine output directory based on output_idx
            if output_idx == 1:  # Gas Velocity
                output_subdir = gas_velocity_dir
            elif output_idx == 4:  # Gas Volume Fraction
                output_subdir = gas_void_fraction_dir
            else:
                continue  # Skip other outputs
            
            # Save visualization
            save_path = output_subdir / f"gradcam_sample{sample_idx}_output{output_idx}.png"
            visualize_gradcam(
                input_sample.cpu(),
                gradcam_map,
                output_idx,
                target_name,
                save_path,
                channel_idx=0,  # Visualize first channel
                sample_idx=sample_idx,
                target_value=target_val,
                pred_value=pred_val,
                hilbert_tensor=hilbert_sample
            )
            
            # Also save prediction info
            if target_val is not None:
                error = abs(target_val - pred_val)
                print(f"    Target: {target_val:.4f} | Prediction: {pred_val:.4f} | Error: {error:.4f}")
            else:
                print(f"    Prediction: {pred_val:.4f}")
    
    # Cleanup
    gradcam.remove_hooks()
    
    print(f"\n✅ Grad-CAM visualization complete!")
    print(f"Results saved to:")
    print(f"  - Gas Velocity: {gas_velocity_dir}")
    print(f"  - Gas Void Fraction: {gas_void_fraction_dir}")


if __name__ == "__main__":
    main()

