#!/usr/bin/env python3
"""
PFNet Inference Script
Loads trained PFNet model and performs inference on test data.
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
        logging.FileHandler('pfnet_inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestDataLoader:
    """Test data loader for inference."""
    
    def __init__(self, x_path, t_path, batch_size=4, start_idx=0, end_idx=None):
        self.x_path = x_path
        self.t_path = t_path
        self.batch_size = batch_size
        self.start_idx = start_idx
        self.end_idx = end_idx
        
        # Load data
        self.x_data = np.load(x_path)
        self.t_data = np.load(t_path)
        
        if end_idx is None:
            end_idx = len(self.x_data)
        
        self.x_data = self.x_data[start_idx:end_idx]
        self.t_data = self.t_data[start_idx:end_idx]
        
        logger.info(f"TestDataLoader initialized with {len(self.x_data)} samples")
        logger.info(f"Data shape: {self.x_data.shape}, Target shape: {self.t_data.shape}")
    
    def __len__(self):
        return (len(self.x_data) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        for i in range(0, len(self.x_data), self.batch_size):
            end_i = min(i + self.batch_size, len(self.x_data))
            
            x_batch = torch.tensor(self.x_data[i:end_i], dtype=torch.float32)
            t_batch = torch.tensor(self.t_data[i:end_i], dtype=torch.float32)
            
            yield x_batch, t_batch

def load_model(model_path, device):
    """Load trained PFNet model."""
    model = PFNet(input_channels=4, output_dim=6, hidden_dim=128)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def perform_inference(model, test_loader, device):
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
            
            if batch_idx % 5 == 0:
                logger.info(f"Processed batch {batch_idx}/{len(test_loader)}")
    
    # Concatenate all results
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    return predictions, targets

def calculate_metrics(predictions, targets):
    """Calculate evaluation metrics."""
    # MSE
    mse = np.mean((predictions - targets) ** 2)
    
    # MAE
    mae = np.mean(np.abs(predictions - targets))
    
    # Correlation coefficient for each output dimension
    correlations = []
    for i in range(predictions.shape[1]):
        corr = np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
        correlations.append(corr)
    
    # Overall correlation (average)
    avg_correlation = np.mean(correlations)
    
    return {
        'mse': mse,
        'mae': mae,
        'correlations': correlations,
        'avg_correlation': avg_correlation
    }

def save_results(predictions, targets, metrics, output_dir):
    """Save inference results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions and targets
    np.save(os.path.join(output_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(output_dir, 'targets.npy'), targets)
    
    # Save metrics
    metrics_file = os.path.join(output_dir, 'inference_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("PFNet Inference Results\n")
        f.write("=" * 30 + "\n")
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
    logger.info("Starting PFNet inference...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Model path (use the final model)
    model_path = "/home/smatsubara/documents/airlift/data/weights/pfnet_final_model.pth"
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    # Load model
    model = load_model(model_path, device)
    
    # Create test data loader (use samples 20-40 as test data for more diversity)
    test_loader = TestDataLoader(
        x_path="/home/smatsubara/documents/airlift/data/sandbox/results/x_train_real.npy",
        t_path="/home/smatsubara/documents/airlift/data/sandbox/results/t_train_real.npy",
        batch_size=4,
        start_idx=20,  # Use samples 20-40 for more diverse test data
        end_idx=40
    )
    
    logger.info(f"Test data: {len(test_loader)} batches")
    
    # Perform inference
    logger.info("Performing inference...")
    predictions, targets = perform_inference(model, test_loader, device)
    
    logger.info(f"Inference completed. Predictions shape: {predictions.shape}")
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, targets)
    
    # Print results
    logger.info("=" * 50)
    logger.info("INFERENCE RESULTS")
    logger.info("=" * 50)
    logger.info(f"MSE: {metrics['mse']:.6f}")
    logger.info(f"MAE: {metrics['mae']:.6f}")
    logger.info(f"Average Correlation: {metrics['avg_correlation']:.6f}")
    logger.info("Per-dimension correlations:")
    for i, corr in enumerate(metrics['correlations']):
        logger.info(f"  Dimension {i}: {corr:.6f}")
    
    # Save results
    output_dir = "/home/smatsubara/documents/airlift/data/inference_results"
    save_results(predictions, targets, metrics, output_dir)
    
    logger.info("Inference completed successfully!")

if __name__ == "__main__":
    main()
