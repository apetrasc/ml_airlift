#!/usr/bin/env python3
"""
Stable PFNet Training Script
Implements advanced techniques for stable training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
import os
from models.cnn import PFNet
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pfnet_stable_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StableDataLoader:
    """Stable data loader with memory optimization."""
    
    def __init__(self, x_path, t_path, batch_size=4, max_samples=None, shuffle=True):
        self.x_path = x_path
        self.t_path = t_path
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.shuffle = shuffle
        
        # Load data with memory mapping
        self.x_data = np.load(x_path, mmap_mode='r')
        self.t_data = np.load(t_path, mmap_mode='r')
        
        if max_samples is None:
            self.max_samples = len(self.x_data)
        else:
            self.max_samples = min(max_samples, len(self.x_data))
        
        logger.info(f"StableDataLoader: {self.max_samples} samples, batch_size={batch_size}")
    
    def __len__(self):
        return (self.max_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        indices = np.arange(self.max_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, self.max_samples, self.batch_size):
            end_i = min(i + self.batch_size, self.max_samples)
            batch_indices = indices[i:end_i]
            
            # Load batch
            x_batch = self.x_data[batch_indices].astype(np.float32)
            t_batch = self.t_data[batch_indices].astype(np.float32)
            
            yield torch.tensor(x_batch), torch.tensor(t_batch)

class StablePFNet(nn.Module):
    """Stable PFNet with improved architecture."""
    
    def __init__(self, input_channels=4, output_dim=6, hidden_dim=128):
        super(StablePFNet, self).__init__()
        
        # Feature extraction with residual connections
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Use LayerNorm instead of BatchNorm for stability
        self.ln1 = nn.LayerNorm([32, 1400, 2500])
        self.ln2 = nn.LayerNorm([64, 700, 1250])
        self.ln3 = nn.LayerNorm([128, 350, 625])
        
        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers with residual connections
        self.fc1 = nn.Linear(128, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction with residual connections
        residual1 = x
        x = F.relu(self.ln1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        residual2 = F.avg_pool2d(residual1, 2)
        if residual2.shape[1] != x.shape[1]:
            residual2 = F.pad(residual2, (0, 0, 0, 0, 0, x.shape[1] - residual2.shape[1]))
        x = x + residual2
        
        x = F.relu(self.ln2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.ln3(self.conv3(x)))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def setup_stable_training():
    """Setup stable training environment."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Set memory fraction
        try:
            torch.cuda.set_per_process_memory_fraction(0.7)
        except:
            pass
        
        logger.info("Stable training setup completed")

def train_epoch_stable(model, train_loader, optimizer, criterion, device, epoch):
    """Stable training epoch with gradient clipping."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # Add L2 regularization
        l2_reg = 0.001
        l2_loss = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + l2_reg * l2_loss
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if batch_idx % 5 == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
    
    return total_loss / num_batches

def validate_epoch_stable(model, val_loader, criterion, device):
    """Stable validation epoch."""
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
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return total_loss / num_batches

def main():
    """Main stable training function."""
    logger.info("Starting stable PFNet training...")
    
    # Setup device and training environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    setup_stable_training()
    
    # Create stable data loaders
    train_loader = StableDataLoader(
        x_path="/home/smatsubara/documents/airlift/data/stable/x_train_stable.npy",
        t_path="/home/smatsubara/documents/airlift/data/stable/t_train_stable.npy",
        batch_size=4,
        max_samples=64,
        shuffle=True
    )
    
    val_loader = StableDataLoader(
        x_path="/home/smatsubara/documents/airlift/data/stable/x_test_stable.npy",
        t_path="/home/smatsubara/documents/airlift/data/stable/t_test_stable.npy",
        batch_size=4,
        max_samples=20,
        shuffle=False
    )
    
    logger.info(f"Created loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create stable model
    model = StablePFNet(input_channels=4, output_dim=6, hidden_dim=128)
    model = model.to(device)
    
    logger.info(f"StablePFNet model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    num_epochs = 100
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training
        train_loss = train_epoch_stable(model, train_loader, optimizer, criterion, device, epoch + 1)
        train_losses.append(train_loss)
        
        # Validation
        val_loss = validate_epoch_stable(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        logger.info(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.1f}s")
        logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            best_model_path = "/home/smatsubara/documents/airlift/data/weights/pfnet_stable_best.pth"
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Best model saved: {best_model_path}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = f"/home/smatsubara/documents/airlift/data/weights/pfnet_stable_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save final model
    final_model_path = "/home/smatsubara/documents/airlift/data/weights/pfnet_stable_final.pth"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
    
    # Log final results
    total_time = time.time() - start_time
    logger.info("Stable training completed successfully!")
    logger.info(f"Total training time: {total_time:.1f} seconds")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Final train loss: {train_losses[-1]:.6f}")
    logger.info(f"Final val loss: {val_losses[-1]:.6f}")
    
    # Save results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'total_time': total_time,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }
    
    import json
    with open('pfnet_stable_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved to pfnet_stable_results.json")

if __name__ == "__main__":
    main()
