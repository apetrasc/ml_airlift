#!/usr/bin/env python3
"""
Ultra-lightweight PFNet training script for GPU.
Minimal memory usage to avoid segmentation faults.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
import os
from models.cnn import PFNet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pfnet_lightweight.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UltraLightDataLoader:
    """Ultra-lightweight data loader with minimal memory usage."""
    
    def __init__(self, x_path, t_path, batch_size=2, max_samples=None):
        self.x_path = x_path
        self.t_path = t_path
        self.batch_size = batch_size
        self.max_samples = max_samples or 20
        
        # Load only metadata first
        self.x_data = np.load(x_path, mmap_mode='r')  # Memory mapping
        self.t_data = np.load(t_path, mmap_mode='r')
        
        self.max_samples = min(self.max_samples, len(self.x_data))
        
        logger.info(f"UltraLightDataLoader: {self.max_samples} samples, batch_size={batch_size}")
    
    def __len__(self):
        return (self.max_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        for i in range(0, self.max_samples, self.batch_size):
            end_i = min(i + self.batch_size, self.max_samples)
            
            # Load only the current batch
            x_batch = self.x_data[i:end_i].astype(np.float32)
            t_batch = self.t_data[i:end_i].astype(np.float32)
            
            # Convert to tensor
            x_tensor = torch.tensor(x_batch, dtype=torch.float32)
            t_tensor = torch.tensor(t_batch, dtype=torch.float32)
            
            yield x_tensor, t_tensor
            
            # Clear memory
            del x_batch, t_batch, x_tensor, t_tensor

def setup_minimal_cuda():
    """Setup minimal CUDA configuration."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set memory fraction to avoid OOM
        try:
            torch.cuda.set_per_process_memory_fraction(0.5)
        except:
            pass
        logger.info("Minimal CUDA setup completed")

def train_epoch_minimal(model, train_loader, optimizer, criterion, device):
    """Minimal training epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move to GPU
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Clear cache every batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log progress
        if batch_idx % 2 == 0:
            logger.info(f"Batch {batch_idx}, Loss: {loss.item():.6f}")
    
    return total_loss / num_batches

def validate_epoch_minimal(model, val_loader, criterion, device):
    """Minimal validation epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            outputs = model(data)
            loss = criterion(outputs, target)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return total_loss / num_batches

def main():
    """Main training function."""
    logger.info("Starting ultra-lightweight PFNet training...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Setup minimal CUDA
    setup_minimal_cuda()
    
    # Create ultra-lightweight data loaders
    train_loader = UltraLightDataLoader(
        x_path="/home/smatsubara/documents/airlift/data/cleaned/x_train_split.npy",
        t_path="/home/smatsubara/documents/airlift/data/cleaned/t_train_split.npy",
        batch_size=2,  # Very small batch size
        max_samples=32  # Reduced samples
    )
    
    val_loader = UltraLightDataLoader(
        x_path="/home/smatsubara/documents/airlift/data/cleaned/x_test_split.npy",
        t_path="/home/smatsubara/documents/airlift/data/cleaned/t_test_split.npy",
        batch_size=2,
        max_samples=10  # Very small validation set
    )
    
    logger.info(f"Created loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create smaller PFNet model
    model = PFNet(input_channels=4, output_dim=6, hidden_dim=64)  # Reduced hidden_dim
    model = model.to(device)
    
    logger.info(f"PFNet model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    num_epochs = 50  # Reduced epochs
    train_losses = []
    val_losses = []
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training
        train_loss = train_epoch_minimal(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validation
        val_loss = validate_epoch_minimal(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        logger.info(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.1f}s")
        logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"/home/smatsubara/documents/airlift/data/weights/pfnet_lightweight_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Clear cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save final model
    final_model_path = "/home/smatsubara/documents/airlift/data/weights/pfnet_lightweight_final.pth"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
    
    # Log final results
    total_time = time.time() - start_time
    logger.info("Training completed successfully!")
    logger.info(f"Total training time: {total_time:.1f} seconds")
    logger.info(f"Final train loss: {train_losses[-1]:.6f}")
    logger.info(f"Final val loss: {val_losses[-1]:.6f}")
    
    # Save results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'total_time': total_time,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }
    
    import json
    with open('pfnet_lightweight_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved to pfnet_lightweight_results.json")

if __name__ == "__main__":
    main()
