#!/usr/bin/env python3
"""
Test script to verify that cleaned data works with the training pipeline.
"""

import sys
import os
sys.path.append('/home/smatsubara/documents/sandbox/ml_airlift')

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.cnn import SimpleCNNReal2D


def test_cleaned_data():
    """Test the cleaned data with a simple forward pass."""
    print("🧪 Testing Cleaned Data")
    print("=" * 40)
    
    # Load cleaned data
    x_path = "/home/smatsubara/documents/sandbox/ml_airlift/cleaned_data/x_train_real_cleaned.npy"
    t_path = "/home/smatsubara/documents/sandbox/ml_airlift/cleaned_data/t_train_real_cleaned.npy"
    
    if not os.path.exists(x_path) or not os.path.exists(t_path):
        print("❌ Cleaned data files not found. Run data_cleaner.py first.")
        return False
    
    print("Loading cleaned data...")
    x = np.load(x_path)
    t = np.load(t_path)
    
    print(f"X shape: {x.shape}, dtype: {x.dtype}")
    print(f"T shape: {t.shape}, dtype: {t.dtype}")
    
    # Check for problematic values
    x_nan = np.isnan(x).sum()
    x_inf = np.isinf(x).sum()
    t_nan = np.isnan(t).sum()
    t_inf = np.isinf(t).sum()
    
    print(f"X: {x_nan} NaN, {x_inf} Inf")
    print(f"T: {t_nan} NaN, {t_inf} Inf")
    
    if x_nan > 0 or x_inf > 0 or t_nan > 0 or t_inf > 0:
        print("❌ Cleaned data still has problematic values!")
        return False
    
    print("✅ Data is clean!")
    
    # Test with PyTorch tensors
    print("\nTesting PyTorch conversion...")
    try:
        x_tensor = torch.from_numpy(x).float()
        t_tensor = torch.from_numpy(t).float()
        print(f"✅ PyTorch conversion successful")
        print(f"X tensor shape: {x_tensor.shape}, dtype: {x_tensor.dtype}")
        print(f"T tensor shape: {t_tensor.shape}, dtype: {t_tensor.dtype}")
    except Exception as e:
        print(f"❌ PyTorch conversion failed: {e}")
        return False
    
    # Test model creation and forward pass
    print("\nTesting model forward pass...")
    try:
        # Use only first few samples to save memory
        x_sample = x_tensor[:2]  # First 2 samples
        t_sample = t_tensor[:2]
        
        # Convert to NCHW format for Conv2D
        if x_sample.ndim == 4:  # (N, C, H, W)
            x_nchw = x_sample
        else:
            print("❌ Unexpected tensor dimensions")
            return False
        
        # Create model
        in_channels = x_nchw.shape[1]
        out_dim = t_sample.shape[1] if t_sample.ndim == 2 else 1
        
        print(f"Creating model: in_channels={in_channels}, out_dim={out_dim}")
        model = SimpleCNNReal2D(in_channels=in_channels, out_dim=out_dim, resize_hw=(256, 256))
        
        # Test forward pass
        print("Running forward pass...")
        with torch.no_grad():
            output = model(x_nchw)
        
        print(f"✅ Forward pass successful!")
        print(f"Input shape: {x_nchw.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")
        
        # Check for NaN in output
        if torch.isnan(output).any():
            print("❌ Model output contains NaN!")
            return False
        
        print("✅ Model output is clean!")
        
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        return False
    
    # Test loss calculation
    print("\nTesting loss calculation...")
    try:
        criterion = torch.nn.MSELoss()
        loss = criterion(output, t_sample)
        print(f"✅ Loss calculation successful: {loss.item():.6f}")
        
        if torch.isnan(loss):
            print("❌ Loss is NaN!")
            return False
        
        print("✅ Loss is finite!")
        
    except Exception as e:
        print(f"❌ Loss calculation failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Cleaned data is ready for training.")
    return True


def test_training_loop():
    """Test a minimal training loop with cleaned data."""
    print("\n🚀 Testing Minimal Training Loop")
    print("=" * 40)
    
    # Load cleaned data
    x_path = "/home/smatsubara/documents/sandbox/ml_airlift/cleaned_data/x_train_real_cleaned.npy"
    t_path = "/home/smatsubara/documents/sandbox/ml_airlift/cleaned_data/t_train_real_cleaned.npy"
    
    x = np.load(x_path)
    t = np.load(t_path)
    
    # Use only first 4 samples for testing
    x = x[:4]
    t = t[:4]
    
    print(f"Using {x.shape[0]} samples for testing")
    
    # Convert to tensors
    x_tensor = torch.from_numpy(x).float()
    t_tensor = torch.from_numpy(t).float()
    
    # Create dataset and dataloader
    dataset = TensorDataset(x_tensor, t_tensor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Create model
    in_channels = x_tensor.shape[1]
    out_dim = t_tensor.shape[1]
    model = SimpleCNNReal2D(in_channels=in_channels, out_dim=out_dim, resize_hw=(256, 256))
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    print("Running 3 training steps...")
    
    for epoch in range(3):
        model.train()
        total_loss = 0
        
        for batch_idx, (batch_x, batch_t) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch_x)
            loss = criterion(output, batch_t)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"❌ NaN loss at epoch {epoch}, batch {batch_idx}")
                return False
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")
    
    print("✅ Training loop test successful!")
    return True


if __name__ == "__main__":
    print("🧪 Cleaned Data Test Suite")
    print("=" * 50)
    
    # Test 1: Basic data validation
    if not test_cleaned_data():
        print("❌ Basic data test failed!")
        exit(1)
    
    # Test 2: Training loop
    if not test_training_loop():
        print("❌ Training loop test failed!")
        exit(1)
    
    print("\n🎉 All tests passed! You can now use the cleaned data for training.")
    print("\nTo use cleaned data in training:")
    print("1. Update config files to point to cleaned data")
    print("2. Or modify train_real.py to use cleaned data paths")
    print("3. Run training with: python train_real.py --epochs 5 --batch 4")




