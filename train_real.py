#!/usr/bin/env python3
"""
Training script for real data with MLOps integration.
Uses Optuna for hyperparameter optimization and MLflow for experiment tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import logging
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import json

# Import project modules
from models import SimpleCNN, ResidualCNN, ProposedCNN, BaseCNN
# SimpleViTRegressor is not available due to torchvision import issues
from src import (
    RealDataDataset, 
    create_real_data_dataloader, 
    get_dataset_info,
    OptunaOptimizer, 
    MLflowTracker,
    log_gpu_memory_usage,
    clear_gpu_memory
)
from src.chunked_loader import create_chunked_dataloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealDataTrainer:
    """
    Trainer class for real data with MLOps integration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config['training']['device'])
        self.output_dir = config['hydra']['run']['dir']
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize MLflow tracker (if available)
        if MLflowTracker is not None:
            self.mlflow_tracker = MLflowTracker(
                experiment_name=config['mlflow']['experiment_name'],
                tracking_uri=config['mlflow']['tracking_uri'],
                log_artifacts=config['mlflow']['log_artifacts'],
                log_models=config['mlflow']['log_models']
            )
        else:
            self.mlflow_tracker = None
            logger.warning("MLflowTracker not available - skipping MLflow logging")
        
        # Initialize Optuna optimizer (if available)
        if OptunaOptimizer is not None:
            self.optuna_optimizer = OptunaOptimizer(
                study_name=config['optuna']['study_name'],
                storage=config['optuna']['storage'],
                direction=config['optuna']['direction'],
                n_trials=config['optuna']['n_trials'],
                timeout=config['optuna']['timeout']
            )
        else:
            self.optuna_optimizer = None
            logger.warning("OptunaOptimizer not available - using default hyperparameters")
        
        logger.info(f"Trainer initialized with device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def create_model(self, model_params: Dict[str, Any]) -> nn.Module:
        """
        Create model based on configuration.
        
        Args:
            model_params: Model parameters
            
        Returns:
            PyTorch model
        """
        model_name = self.config['model']['name']
        input_length = self.config['hyperparameters']['input_length']
        
        # Get model architecture parameters
        model_params = self.config['model']['architecture']
        
        if model_name == 'SimpleCNN':
            model = SimpleCNN(input_length, **model_params)
        elif model_name == 'SimpleViTRegressor':
            raise ValueError("SimpleViTRegressor is not available due to torchvision import issues")
        elif model_name == 'ResidualCNN':
            model = ResidualCNN(input_length, **model_params)
        elif model_name == 'ProposedCNN':
            model = ProposedCNN(input_length, **model_params)
        elif model_name == 'BaseCNN':
            model = BaseCNN(input_length, **model_params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Apply custom parameters if provided
        if model_params:
            for key, value in model_params.items():
                if hasattr(model, key):
                    setattr(model, key, value)
        
        # Ensure model outputs 6 dimensions
        if hasattr(model, 'fc'):
            model.fc = torch.nn.Linear(model.fc.in_features, 6)
        elif hasattr(model, 'classifier'):
            model.classifier = torch.nn.Linear(model.classifier.in_features, 6)
        elif hasattr(model, 'head'):
            model.head = torch.nn.Linear(model.head.in_features, 6)
        
        return model
    
    def train_model(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        hyperparams: Dict[str, Any]
    ) -> Tuple[nn.Module, List[float], List[float], Dict[str, Any]]:
        """
        Train a single model.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            hyperparams: Hyperparameters
            
        Returns:
            Tuple of (trained_model, train_losses, val_losses, metrics)
        """
        model = model.to(self.device)
        
        # Use half precision if available
        if self.device.type == 'cuda':
            model = model.half()
        
        # Setup optimizer and loss
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams['learning_rate']
        )
        criterion = nn.MSELoss()
        
        # Training history
        train_losses = []
        val_losses = []
        
        # Training loop with memory-efficient GPU usage
        for epoch in range(hyperparams['num_epochs']):
            # Log memory usage at start of epoch
            if epoch % 10 == 0:
                log_gpu_memory_usage(f"start of epoch {epoch}")
            
            # Training
            model.train()
            train_loss = 0.0
            for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
                # Move batch to GPU
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                # Use half precision if available
                if self.device.type == 'cuda':
                    batch_x = batch_x.half()
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # Add regularization (compute efficiently)
                if hyperparams.get('l1_lambda', 0) > 0 or hyperparams.get('l2_lambda', 0) > 0:
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                    loss = loss + hyperparams.get('l1_lambda', 0) * l1_norm + hyperparams.get('l2_lambda', 0) * l2_norm
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                # Log progress every 10 batches
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.6f}")
                
                # Clear GPU memory after each batch
                del batch_x, batch_y, outputs, loss
                if batch_idx % 50 == 0:  # Clear cache every 50 batches
                    torch.cuda.empty_cache()
            
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            
            # Validation (memory-efficient)
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_idx, (val_x, val_y) in enumerate(val_dataloader):
                    # Move batch to GPU
                    val_x = val_x.to(self.device, non_blocking=True)
                    val_y = val_y.to(self.device, non_blocking=True)
                    
                    # Use half precision if available
                    if self.device.type == 'cuda':
                        val_x = val_x.half()
                    
                    outputs = model(val_x)
                    loss = criterion(outputs, val_y)
                    val_loss += loss.item()
                    
                    # Store only a subset of predictions to save memory
                    if len(val_predictions) < 10:  # Limit to 10 batches
                        val_predictions.append(outputs.cpu().clone())
                        val_targets.append(val_y.cpu().clone())
                    
                    # Clear GPU memory after each batch
                    del val_x, val_y, outputs, loss
                    if batch_idx % 20 == 0:  # Clear cache every 20 batches
                        torch.cuda.empty_cache()
            
            val_loss /= len(val_dataloader)
            val_losses.append(val_loss)
            
            # Calculate correlation on subset to save memory
            if val_predictions:
                val_predictions_tensor = torch.cat(val_predictions, dim=0)
                val_targets_tensor = torch.cat(val_targets, dim=0)
                
                # Calculate correlation
                val_targets_np = val_targets_tensor.numpy().flatten()
                val_predictions_np = val_predictions_tensor.numpy().flatten()
                correlation = np.corrcoef(val_targets_np, val_predictions_np)[0, 1]
                
                # Clear memory
                del val_predictions_tensor, val_targets_tensor
            else:
                correlation = 0.0
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{hyperparams['num_epochs']}, "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Correlation: {correlation:.4f}"
                )
        
        # Calculate final metrics (simplified to save memory)
        final_metrics = {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_correlation': correlation,
            'best_val_loss': min(val_losses),
            'best_correlation': correlation  # Use current correlation instead of complex calculation
        }
        
        return model, train_losses, val_losses, final_metrics
    
    def optimize_hyperparameters(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna (if available).
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            
        Returns:
            Best hyperparameters
        """
        if self.optuna_optimizer is None:
            logger.info("Optuna not available - using default hyperparameters")
            return self.config['hyperparameters']
        
        logger.info("Starting hyperparameter optimization...")
        
        # Run optimization
        study = self.optuna_optimizer.optimize(
            model_factory=self.create_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=self.device,
            input_length=self.config['hyperparameters']['input_length']
        )
        
        # Get best parameters
        best_params = self.optuna_optimizer.get_best_parameters()
        best_value = self.optuna_optimizer.get_best_value()
        
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best validation loss: {best_value}")
        
        # Save optimization results
        optimization_results = {
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': len(study.trials),
            'study_name': self.config['optuna']['study_name']
        }
        
        # Save to file
        optimization_file = os.path.join(self.output_dir, "optimization_results.json")
        with open(optimization_file, 'w') as f:
            json.dump(optimization_results, f, indent=2)
        
        return best_params
    
    def train_final_model(
        self,
        best_params: Dict[str, Any],
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Train final model with best hyperparameters.
        
        Args:
            best_params: Best hyperparameters from optimization
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            
        Returns:
            Tuple of (trained_model, training_metrics)
        """
        logger.info("Training final model with best hyperparameters...")
        
        # Create model
        model = self.create_model(best_params)
        
        # Train model
        trained_model, train_losses, val_losses, metrics = self.train_model(
            model, train_dataloader, val_dataloader, best_params
        )
        
        # Save model
        model_path = os.path.join(self.output_dir, "final_model.pth")
        torch.save(trained_model.state_dict(), model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # Create learning curve plot
        fig = self.mlflow_tracker.create_learning_curve_plot(
            train_losses, val_losses,
            save_path=os.path.join(self.output_dir, "learning_curve.png")
        )
        plt.close(fig)
        
        return trained_model, {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'metrics': metrics
        }
    
    def run_training(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Returns:
            Dictionary containing training results
        """
        logger.info("Starting training pipeline...")
        
        # Skip dataset info to avoid memory issues with large datasets
        # dataset_info = get_dataset_info(
        #     self.config['dataset']['x_train'],
        #     self.config['dataset']['t_train'],
        #     self.config['dataset']['data_keys']['x_key'],
        #     self.config['dataset']['data_keys']['t_key']
        # )
        
        # Create dummy dataset info for logging
        dataset_info = {
            'x_shape': 'Unknown (large dataset)',
            'x_dtype': 'float32',
            't_shape': 'Unknown (large dataset)',
            't_dtype': 'float32',
            'n_samples': 'Unknown (large dataset)',
            'x_size_mb': 'Unknown (large dataset)',
            't_size_mb': 'Unknown (large dataset)'
        }
        
        logger.info(f"Dataset info: {dataset_info}")
        
        # Create dataloaders using ultra memory-efficient loader
        from src.data_loader import create_ultra_memory_efficient_dataloader
        
        train_dataloader, val_dataloader = create_ultra_memory_efficient_dataloader(
            x_path=self.config['dataset']['x_train'],
            t_path=self.config['dataset']['t_train'],
            target_memory_gb=1.5,  # Use only 1.5GB of GPU memory
            x_key=self.config['dataset']['data_keys']['x_key'],
            t_key=self.config['dataset']['data_keys']['t_key'],
            max_samples=self.config['training']['max_samples_per_epoch'],
            shuffle=True
        )
        
        logger.info(f"Created dataloaders: {len(train_dataloader)} train batches, {len(val_dataloader)} val batches")
        
        # Start MLflow run (if available)
        if self.mlflow_tracker is not None:
            self.mlflow_tracker.start_run(
                run_name=f"real_data_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags={
                    "dataset": "real_data",
                    "model": self.config['model']['name'],
                    "optimization": "optuna" if self.optuna_optimizer is not None else "default"
                }
            )
        
        try:
            # Log configuration and dataset info (if MLflow available)
            if self.mlflow_tracker is not None:
                self.mlflow_tracker.log_parameters(self.config['hyperparameters'])
                self.mlflow_tracker.log_parameters(self.config['model'])
                self.mlflow_tracker.log_dataset_info(dataset_info)
            
            # Optimize hyperparameters
            best_params = self.optimize_hyperparameters(train_dataloader, val_dataloader)
            
            # Log optimization results (if MLflow available)
            if self.mlflow_tracker is not None and self.optuna_optimizer is not None:
                optimization_results = {
                    'best_params': best_params,
                    'best_value': self.optuna_optimizer.get_best_value(),
                    'n_trials': self.optuna_optimizer.n_trials
                }
                self.mlflow_tracker.log_optimization_results(optimization_results)
            
            # Train final model
            final_model, training_results = self.train_final_model(
                best_params, train_dataloader, val_dataloader
            )
            
            # Log final metrics (if MLflow available)
            if self.mlflow_tracker is not None:
                self.mlflow_tracker.log_metrics(training_results['metrics'])
                self.mlflow_tracker.log_training_history(
                    training_results['train_losses'],
                    training_results['val_losses']
                )
                
                # Log model
                self.mlflow_tracker.log_model(
                    final_model,
                    model_name="final_model",
                    registered_model_name=f"real_data_{self.config['model']['name']}"
                )
                
                # Log artifacts
                self.mlflow_tracker.log_artifacts(self.output_dir, "training_outputs")
            
            logger.info("Training pipeline completed successfully")
            
            return {
                'model': final_model,
                'best_params': best_params,
                'training_results': training_results,
                'dataset_info': dataset_info,
                'output_dir': self.output_dir
            }
            
        finally:
            # End MLflow run (if available)
            if self.mlflow_tracker is not None:
                self.mlflow_tracker.end_run()


def main():
    """Main function to run training."""
    # Setup CUDA memory optimization FIRST
    from src.data_loader import setup_cuda_memory_optimization
    setup_cuda_memory_optimization()
    
    # Load configuration
    config_path = "/home/smatsubara/documents/sandbox/ml_airlift/config/config_real.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds
    torch.manual_seed(config['hyperparameters']['seed'])
    np.random.seed(config['hyperparameters']['seed'])
    
    # Create trainer
    trainer = RealDataTrainer(config)
    
    # Run training
    results = trainer.run_training()
    
    logger.info(f"Training completed. Results saved to: {results['output_dir']}")
    logger.info(f"Best parameters: {results['best_params']}")
    logger.info(f"Final metrics: {results['training_results']['metrics']}")


if __name__ == "__main__":
    main()
