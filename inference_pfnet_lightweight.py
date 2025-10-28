#!/usr/bin/env python3
"""
PFNet Lightweight Inference Script
Uses the trained lightweight PFNet model for inference.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import os
from models.cnn import PFNet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pfnet_lightweight_inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LightweightTestDataLoader:
    """Lightweight test data loader for inference."""
    
    def __init__(self, x_path, t_path, batch_size=2, max_samples=None):
        self.x_path = x_path
        self.t_path = t_path
        self.batch_size = batch_size
        self.max_samples = max_samples
        
        # Load data with memory mapping
        self.x_data = np.load(x_path, mmap_mode='r')
        self.t_data = np.load(t_path, mmap_mode='r')
        
        if max_samples is None:
            self.max_samples = len(self.x_data)
        else:
            self.max_samples = min(max_samples, len(self.x_data))
        
        logger.info(f"LightweightTestDataLoader: {self.max_samples} samples")
        logger.info(f"Data shape: {self.x_data.shape}, Target shape: {self.t_data.shape}")
    
    def __len__(self):
        return (self.max_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        for i in range(0, self.max_samples, self.batch_size):
            end_i = min(i + self.batch_size, self.max_samples)
            
            # Load only current batch
            x_batch = self.x_data[i:end_i].astype(np.float32)
            t_batch = self.t_data[i:end_i].astype(np.float32)
            
            x_tensor = torch.tensor(x_batch, dtype=torch.float32)
            t_tensor = torch.tensor(t_batch, dtype=torch.float32)
            
            yield x_tensor, t_tensor
            
            # Clear memory
            del x_batch, t_batch, x_tensor, t_tensor

def load_lightweight_model(model_path, device):
    """Load trained lightweight PFNet model."""
    # Create model with same architecture as training
    model = PFNet(input_channels=4, output_dim=6, hidden_dim=64)  # Same as training
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info(f"Lightweight model loaded from {model_path}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def perform_inference_lightweight(model, test_loader, device):
    """Perform inference on test data."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(data)
            
            # Store results
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            
            if batch_idx % 2 == 0:
                logger.info(f"Processed batch {batch_idx}/{len(test_loader)}")
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Concatenate all results
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    return predictions, targets

def calculate_metrics_robust(predictions, targets):
    """Calculate evaluation metrics with robust handling."""
    # MSE
    mse = np.mean((predictions - targets) ** 2)
    
    # MAE
    mae = np.mean(np.abs(predictions - targets))
    
    # Correlation coefficient for each output dimension
    correlations = []
    for i in range(predictions.shape[1]):
        pred_i = predictions[:, i]
        target_i = targets[:, i]
        
        # Check for constant values
        if np.std(pred_i) == 0 or np.std(target_i) == 0:
            logger.warning(f"Dimension {i}: constant values detected")
            correlations.append(0.0)
        else:
            corr = np.corrcoef(pred_i, target_i)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            correlations.append(corr)
    
    # Overall correlation (average)
    avg_correlation = np.mean(correlations)
    
    return {
        'mse': mse,
        'mae': mae,
        'correlations': correlations,
        'avg_correlation': avg_correlation
    }

def save_inference_results(predictions, targets, metrics, output_dir):
    """Save inference results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions and targets
    np.save(os.path.join(output_dir, 'lightweight_predictions.npy'), predictions)
    np.save(os.path.join(output_dir, 'lightweight_targets.npy'), targets)
    
    # Save metrics
    metrics_file = os.path.join(output_dir, 'lightweight_inference_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("PFNet Lightweight Inference Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"MSE: {metrics['mse']:.6f}\n")
        f.write(f"MAE: {metrics['mae']:.6f}\n")
        f.write(f"Average Correlation: {metrics['avg_correlation']:.6f}\n")
        f.write("\nPer-dimension correlations:\n")
        for i, corr in enumerate(metrics['correlations']):
            f.write(f"  Dimension {i}: {corr:.6f}\n")
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Metrics saved to {metrics_file}")

def main():
    """Main inference function."""
    logger.info("Starting PFNet lightweight inference...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Model path (use the final lightweight model)
    model_path = "/home/smatsubara/documents/airlift/data/weights/pfnet_lightweight_final.pth"
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    # Load model
    model = load_lightweight_model(model_path, device)
    
    # Create test data loader
    test_loader = LightweightTestDataLoader(
        x_path="/home/smatsubara/documents/airlift/data/cleaned/x_test_split.npy",
        t_path="/home/smatsubara/documents/airlift/data/cleaned/t_test_split.npy",
        batch_size=2,  # Small batch size for memory efficiency
        max_samples=22  # Use all test samples
    )
    
    logger.info(f"Test data: {len(test_loader)} batches")
    
    # Perform inference
    logger.info("Performing inference...")
    predictions, targets = perform_inference_lightweight(model, test_loader, device)
    
    logger.info(f"Inference completed. Predictions shape: {predictions.shape}")
    
    # Calculate metrics
    metrics = calculate_metrics_robust(predictions, targets)
    
    # Print results
    logger.info("=" * 50)
    logger.info("LIGHTWEIGHT INFERENCE RESULTS")
    logger.info("=" * 50)
    logger.info(f"MSE: {metrics['mse']:.6f}")
    logger.info(f"MAE: {metrics['mae']:.6f}")
    logger.info(f"Average Correlation: {metrics['avg_correlation']:.6f}")
    logger.info("Per-dimension correlations:")
    for i, corr in enumerate(metrics['correlations']):
        logger.info(f"  Dimension {i}: {corr:.6f}")
    
    # Save results
    output_dir = "/home/smatsubara/documents/airlift/data/inference_results"
    save_inference_results(predictions, targets, metrics, output_dir)
    
    logger.info("Lightweight inference completed successfully!")

if __name__ == "__main__":
    main()
