#!/usr/bin/env python3
"""
Main inference script for ML Airlift project.

This script loads trained models and performs inference on input data.
Supports both 3-phase and 2-phase models.

Usage:
    python main.py --3phase    # Use 3-phase model (models/sota)
    python main.py --2phase     # Use 2-phase model (models/layernorm)
"""

import os
import sys
import argparse
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from pathlib import Path
from scipy.signal import resample_poly

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import for 2-phase (eval_tutorial.py style)
from src import preprocess_and_predict
from models import SimpleCNN

# Import for 3-phase
from src.data.loaders import _load_np_any, to_tensor_dataset
from src.training.trainer import create_model, evaluate
from torch.utils.data import DataLoader
import torch.nn as nn


def compute_phase_fractions_3phase(predictions: np.ndarray):
    """
    Compute phase fractions for 3-phase model.
    
    Model outputs 6 values:
    - Index 0: Solid Velocity
    - Index 1: Gas Velocity
    - Index 2: Liquid Velocity
    - Index 3: Solid Volume Fraction
    - Index 4: Gas Volume Fraction
    - Index 5: Liquid Volume Fraction
    
    For 3-phase:
    - Solid Volume Fraction = 1 - (Gas Volume Fraction + Liquid Volume Fraction)
    - If Solid Volume Fraction <= 0.03, set to 0
    
    Args:
        predictions: Model predictions (N, 6) or (6,)
    
    Returns:
        Phase fractions as [solid, gas, liquid]
    """
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    
    # Extract volume fractions
    gas_vf = predictions[:, 4]  # Gas Volume Fraction
    liquid_vf = predictions[:, 5]  # Liquid Volume Fraction
    
    # Compute solid volume fraction
    solid_vf = 1.0 - (gas_vf + liquid_vf)
    
    # Set to 0 if <= 0.03
    solid_vf = np.where(solid_vf <= 0.03, 0.0, solid_vf)
    
    # Stack as [solid, gas, liquid]
    phase_fractions = np.stack([solid_vf, gas_vf, liquid_vf], axis=1)
    
    return phase_fractions


def compute_phase_fractions_2phase(predictions: np.ndarray):
    """
    Compute phase fractions for 2-phase model.
    
    Model outputs 1 value (single output regression).
    For 2-phase:
    - Solid Volume Fraction = prediction (first dimension)
    - Gas Volume Fraction = 0
    - Liquid Volume Fraction = 1 - Solid Volume Fraction
    
    Args:
        predictions: Model predictions (N,) or scalar
    
    Returns:
        Phase fractions as [solid, gas, liquid]
    """
    if predictions.ndim == 0:
        predictions = np.array([predictions])
    elif predictions.ndim == 1:
        pass
    else:
        predictions = predictions.flatten()
    
    # Extract solid volume fraction (first dimension)
    solid_vf = predictions
    
    # Gas is 0 for 2-phase
    gas_vf = np.zeros_like(solid_vf)
    
    # Liquid is 1 - solid
    liquid_vf = 1.0 - solid_vf
    
    # Stack as [solid, gas, liquid]
    phase_fractions = np.stack([solid_vf, gas_vf, liquid_vf], axis=1)
    
    return phase_fractions


def preprocess_input_to_nchw(x: np.ndarray):
    """
    Preprocess input data to (N, C, H, W) format and select channels 0 and 2.
    
    Args:
        x: Input data in various shapes
    
    Returns:
        Preprocessed array in (N, C=2, H, W) format
    """
    # Convert to (N, C, H, W) format
    if x.ndim == 2:
        # (H, W) -> (1, 1, H, W)
        x = x[np.newaxis, np.newaxis, :, :]
    elif x.ndim == 3:
        # Could be (H, W, C) or (C, H, W) or (N, H, W)
        if x.shape[-1] <= 4:
            # (H, W, C) -> (1, C, H, W)
            x = np.transpose(x[np.newaxis, :, :, :], (0, 3, 1, 2))
        elif x.shape[0] <= 4:
            # (C, H, W) -> (1, C, H, W)
            x = x[np.newaxis, :, :, :]
        else:
            # (N, H, W) -> (N, 1, H, W)
            x = x[:, np.newaxis, :, :]
    elif x.ndim == 4:
        # (N, C, H, W) or (N, H, W, C)
        if x.shape[1] > 4 or (x.shape[1] != 2 and x.shape[-1] <= 4):
            # Might be NHWC, transpose to NCHW
            x = np.transpose(x, (0, 3, 1, 2))
    
    # Exclude Channel 1 and Channel 3 (keep only channels 0, 2)
    if x.ndim == 4 and x.shape[1] == 4:
        x = x[:, [0, 2], :, :]  # Keep only channels 0, 2
    elif x.ndim == 3 and x.shape[0] == 4:
        # (C, H, W) -> select channels
        x = x[[0, 2], :, :]
        x = x[np.newaxis, :, :, :]  # Add N dimension
    
    # Ensure we have N dimension
    if x.ndim == 3:
        x = x[np.newaxis, :, :, :]
    
    # Take only first sample for inference
    if x.shape[0] > 1:
        x = x[0:1, :, :, :]
    
    return x


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --3phase    # Use 3-phase model
  python main.py --2phase    # Use 2-phase model
        """
    )
    parser.add_argument(
        '--3phase',
        dest='three_phase',
        action='store_true',
        help='Use 3-phase model (models/sota)'
    )
    parser.add_argument(
        '--2phase',
        dest='two_phase',
        action='store_true',
        help='Use 2-phase model (models/layernorm)'
    )
    
    args = parser.parse_args()
    
    # Check that exactly one mode is specified
    if args.three_phase == args.two_phase:
        parser.error("Please specify either --3phase or --2phase")
    
    # Load inference config
    config_path = project_root / "config" / "config_inference.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = OmegaConf.load(str(config_path))
    
    print("=" * 60)
    print("ML Airlift Inference")
    print("=" * 60)
    
    if args.two_phase:
        # ========== 2-phase model: same as eval_tutorial.py ==========
        print("[INFO] Using 2-phase model (models/layernorm)")
        
        # Load config.yaml
        config_yaml_path = project_root / "config" / "config.yaml"
        if not os.path.exists(config_yaml_path):
            raise FileNotFoundError(f"config.yaml not found: {config_yaml_path}")
        with open(config_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        device = config["evaluation"]["device"]
        input_length = config['hyperparameters']['input_length']
        
        # Load model
        model_path = project_root / "models" / "layernorm" / "weights" / "model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = SimpleCNN(input_length).to(device)
        model.load_state_dict(
            torch.load(str(model_path), map_location=device, weights_only=True)
        )
        model.eval()
        print(f"[OK] Model loaded from: {model_path}")
        
        # Run inference
        input_file = cfg.input_file
        print(f"\n[STEP] Running inference...")
        mean, var = preprocess_and_predict(input_file, model, device=device)
        
        # Convert to numpy for phase fraction calculation
        predictions = np.array([mean.item()])
        
        # Compute phase fractions
        phase_fractions = compute_phase_fractions_2phase(predictions)
        
    else:  # args.three_phase
        # ========== 3-phase model: inference on single sample ==========
        print("[INFO] Using 3-phase model (models/sota)")
        
        # Load model config
        model_config_path = project_root / "models" / "sota" / "config.yaml"
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Model config not found: {model_config_path}")
        model_cfg = OmegaConf.load(str(model_config_path))
        
        device = torch.device(cfg.device)
        print(f"[INFO] Device: {device}")
        
        # Load input data
        input_file = cfg.input_file
        input_key = cfg.get('input_key', 'processed_data')
        
        print(f"\n[STEP] Loading input data...")
        x = _load_np_any(input_file, input_key)
        print(f"[OK] Loaded. x.shape={x.shape}")
        
        # Apply resample_poly downsampling (same as dataset creation in main.ipynb)
        # This converts H dimension from 14000 to 1400 using polynomial interpolation
        target_length = 1400
        original_length = x.shape[0] if x.ndim >= 1 else x.shape[-3] if x.ndim >= 3 else None
        
        if original_length is not None and original_length != target_length:
            print(f"[STEP] Applying resample_poly downsampling...")
            print(f"[INFO] Original H dimension: {original_length}, Target: {target_length}")
            
            # Determine axis for downsampling based on data shape
            if x.ndim == 2:
                # (H, W) -> downsample H dimension (axis=0)
                x = resample_poly(x, target_length, original_length, axis=0)
            elif x.ndim == 3:
                # Could be (H, W, C) or (C, H, W) or (N, H, W)
                if x.shape[-1] <= 4:
                    # (H, W, C) -> downsample H dimension (axis=0)
                    x = resample_poly(x, target_length, original_length, axis=0)
                elif x.shape[0] <= 4:
                    # (C, H, W) -> downsample H dimension (axis=1)
                    x = resample_poly(x, target_length, original_length, axis=1)
                else:
                    # (N, H, W) -> downsample H dimension (axis=1)
                    x = resample_poly(x, target_length, original_length, axis=1)
            elif x.ndim == 4:
                # (N, C, H, W) or (N, H, W, C)
                if x.shape[1] > 4 or (x.shape[1] != 2 and x.shape[-1] <= 4):
                    # Might be (N, H, W, C), downsample H dimension (axis=1)
                    x = resample_poly(x, target_length, original_length, axis=1)
                else:
                    # (N, C, H, W), downsample H dimension (axis=2)
                    x = resample_poly(x, target_length, original_length, axis=2)
            
            print(f"[OK] After resample_poly: x.shape={x.shape}")
        else:
            print(f"[INFO] No resample_poly needed (H dimension already {original_length})")
        
        # Preprocess to (N, C=2, H, W) format
        x = preprocess_input_to_nchw(x)
        print(f"[OK] Preprocessed. x.shape={x.shape}")
        
        # Apply downsampling if configured (same as training)
        downsample_factor = model_cfg.dataset.get('downsample_factor', 1)
        if downsample_factor > 1 and x.ndim == 4:
            h0 = x.shape[2]
            x = x[:, :, ::downsample_factor, :]
            print(f"[INFO] Downsampled H: {h0} -> {x.shape[2]} (factor={downsample_factor})")
        
        # Convert to tensor and create dataset
        x_tensor = torch.from_numpy(x).float()
        
        # Create dummy targets for evaluation function (not used for inference)
        n_samples = x_tensor.shape[0]
        out_dim = 6
        t_dummy = torch.zeros((n_samples, out_dim))
        
        # Create dataset and dataloader
        dataset = to_tensor_dataset(x_tensor.numpy(), t_dummy.numpy(), "cpu")
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # Load model
        print("[STEP] Loading model...")
        model_weights_path = project_root / "models" / "sota" / "weights" / "model_simplecnn_real.pth"
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Model weights not found: {model_weights_path}")
        
        # Create model (use downsampled x_sample)
        x_sample = x_tensor[:1].clone()
        model = create_model(model_cfg, x_sample, out_dim, device)
        
        # Load weights
        model.load_state_dict(torch.load(str(model_weights_path), map_location=device, weights_only=True))
        model.to(device)
        print(f"[OK] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Run inference using evaluate function
        print("[STEP] Running inference...")
        criterion = nn.MSELoss()
        _, _, predictions_np, _ = evaluate(model, dataloader, criterion, device)
        
        print(f"[OK] Inference complete. Predictions: {predictions_np}")
        
        # Compute phase fractions
        phase_fractions = compute_phase_fractions_3phase(predictions_np)
    
    # Output results
    print("\n" + "=" * 60)
    print("Results: Phase Fractions [Solid, Gas, Liquid]")
    print("=" * 60)
    
    if phase_fractions.shape[0] == 1:
        result = phase_fractions[0]
        print(f"Phase fractions: {result}")
        print(f"  Solid:  {result[0]:.4f}")
        print(f"  Gas:    {result[1]:.4f}")
        print(f"  Liquid: {result[2]:.4f}")
    else:
        for i, pf in enumerate(phase_fractions):
            print(f"Sample {i}: {pf}")
            print(f"  Solid:  {pf[0]:.4f}, Gas: {pf[1]:.4f}, Liquid: {pf[2]:.4f}")
    
    print("\n" + "=" * 60)
    print("[DONE] Inference completed successfully")
    print("=" * 60)
    
    return phase_fractions


if __name__ == "__main__":
    main()
