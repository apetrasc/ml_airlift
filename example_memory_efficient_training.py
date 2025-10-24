#!/usr/bin/env python3
"""
Example of memory-efficient training with the optimized data loader.
This script demonstrates how to use the ultra memory-efficient data loader
to avoid CUDA out of memory errors.
"""

import torch
import torch.nn as nn
from src.data_loader import (
    create_ultra_memory_efficient_dataloader,
    setup_cuda_memory_optimization,
    get_optimal_batch_size_for_memory
)

def main():
    # Setup CUDA memory optimization BEFORE any other operations
    setup_cuda_memory_optimization()
    
    # Define paths to your data
    x_path = "/home/smatsubara/documents/airlift/data/sandbox/results/x_train_real.npy"
    t_path = "/home/smatsubara/documents/airlift/data/sandbox/results/t_train_real.npy"
    
    # Create ultra memory-efficient dataloaders
    # This will automatically calculate optimal batch size
    train_loader, val_loader = create_ultra_memory_efficient_dataloader(
        x_path=x_path,
        t_path=t_path,
        target_memory_gb=1.5,  # Use only 1.5GB of GPU memory
        train_split=0.8,
        shuffle=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Gradient accumulation steps: {train_loader.dataset.dataset.gradient_accumulation_steps}")
    
    # Example model (replace with your actual model)
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, 6)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Initialize model
    model = SimpleModel()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Use half precision if available
    if device.type == 'cuda':
        model = model.half()
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop with gradient accumulation
    model.train()
    accumulation_steps = 8  # Adjust based on your needs
    
    for epoch in range(5):  # Example: 5 epochs
        optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            if device.type == 'cuda':
                data = data.half()
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item() * accumulation_steps:.6f}')
        
        # Clear cache after each epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
