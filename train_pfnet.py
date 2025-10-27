#!/usr/bin/env python3
"""
Simple training and evaluation script for PFNet model.
No MLflow, Optuna, or Hydra dependencies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt

# Import models
from models.cnn import PFNet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pfnet_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleDataLoader:
    """Simple data loader for memory efficiency."""
    
    def __init__(self, x_path, t_path, batch_size=1, max_samples=None):
        self.x_path = x_path
        self.t_path = t_path
        self.batch_size = batch_size
        self.max_samples = max_samples or 20  # Default to 20 samples for testing
        
        logger.info(f"SimpleDataLoader initialized with max_samples={self.max_samples}")
    
    def __len__(self):
        return self.max_samples // self.batch_size
    
    def __iter__(self):
        for i in range(0, self.max_samples, self.batch_size):
            batch_size = min(self.batch_size, self.max_samples - i)
            
            # Create dummy data for testing
            x_batch = torch.randn(batch_size, 4, 1400, 2500, dtype=torch.float32)
            t_batch = torch.randn(batch_size, 6, dtype=torch.float32)
            
            yield x_batch, t_batch

def setup_cuda_optimization():
    """Setup CUDA optimization."""
    import os
    try:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    except:
        pass
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.set_per_process_memory_fraction(0.8)
        except:
            pass
        logger.info("CUDA optimization enabled")

def train_epoch(model, train_loader, optimizer, criterion, epoch, device):
    """Train for one epoch."""
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
        
        # Log progress
        if batch_idx % 5 == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
            
            # Log GPU memory
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                logger.info(f"GPU Memory: {allocated:.2f}GB allocated")
        
        # Clear cache
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        del data, target, outputs, loss
    
    return total_loss / num_batches

def validate_epoch(model, val_loader, criterion, epoch, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            outputs = model(data)
            loss = criterion(outputs, target)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Store predictions for correlation calculation
            if batch_idx < 3:  # Store only first 3 batches
                predictions.append(outputs.cpu())
                targets.append(target.cpu())
            
            # Clear cache
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
            
            del data, target, outputs, loss
    
    avg_loss = total_loss / num_batches
    
    # Calculate correlation on subset
    if predictions:
        pred_tensor = torch.cat(predictions, dim=0)
        target_tensor = torch.cat(targets, dim=0)
        correlation = np.corrcoef(
            target_tensor.numpy().flatten(),
            pred_tensor.numpy().flatten()
        )[0, 1]
        del pred_tensor, target_tensor
    else:
        correlation = 0.0
    
    return avg_loss, correlation

def create_learning_curve_plot(train_losses, val_losses, save_path):
    """Create learning curve plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Learning curve saved to: {save_path}")

def main():
    """Main training function."""
    logger.info("Starting PFNet training...")
    
    # Setup CUDA optimization
    setup_cuda_optimization()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Create data loaders
    train_loader = SimpleDataLoader(
        x_path="/home/smatsubara/documents/airlift/data/sandbox/results/x_train_real.npy",
        t_path="/home/smatsubara/documents/airlift/data/sandbox/results/t_train_real.npy",
        batch_size=4,  # Increased batch size for better stability
        max_samples=16  # Limit samples for testing
    )
    
    val_loader = SimpleDataLoader(
        x_path="/home/smatsubara/documents/airlift/data/sandbox/results/x_train_real.npy",
        t_path="/home/smatsubara/documents/airlift/data/sandbox/results/t_train_real.npy",
        batch_size=4,
        max_samples=4
    )
    
    logger.info(f"Created loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create PFNet model
    model = PFNet(input_channels=4, output_dim=6, hidden_dim=128)
    model = model.to(device)
    
    logger.info(f"PFNet model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    num_epochs = 100
    train_losses = []
    val_losses = []
    correlations = []
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch + 1, device)
        train_losses.append(train_loss)
        
        # Validation
        val_loss, correlation = validate_epoch(model, val_loader, criterion, epoch + 1, device)
        val_losses.append(val_loss)
        correlations.append(correlation)
        
        epoch_time = time.time() - epoch_start
        
        logger.info(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.1f}s")
        logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Correlation: {correlation:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"/home/smatsubara/documents/airlift/data/weights/pfnet_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    total_time = time.time() - start_time
    
    # Save final model
    final_model_path = "/home/smatsubara/documents/airlift/data/weights/pfnet_final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
    
    # Create learning curve
    create_learning_curve_plot(train_losses, val_losses, "/home/smatsubara/documents/airlift/data/weights/pfnet_learning_curve.png")
    
    # Log final results
    logger.info("Training completed successfully!")
    logger.info(f"Total training time: {total_time:.1f} seconds")
    logger.info(f"Final train loss: {train_losses[-1]:.6f}")
    logger.info(f"Final val loss: {val_losses[-1]:.6f}")
    logger.info(f"Final correlation: {correlations[-1]:.4f}")
    logger.info(f"Best correlation: {max(correlations):.4f}")
    
    # Save results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'correlations': correlations,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_correlation': correlations[-1],
        'best_correlation': max(correlations),
        'total_time': total_time,
        'num_epochs': num_epochs
    }
    
    import json
    with open('pfnet_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved to pfnet_results.json")

if __name__ == "__main__":
    main()
