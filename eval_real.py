#!/usr/bin/env python3
"""
Evaluation script for real data models.
Provides comprehensive evaluation metrics and visualization.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import logging
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import json
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

# Import project modules
from models import SimpleCNN, SimpleViTRegressor, ResidualCNN, ProposedCNN, BaseCNN
from src import (
    RealDataDataset, 
    create_real_data_dataloader, 
    get_dataset_info,
    MLflowTracker
)
from src.chunked_loader import create_chunked_dataloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealDataEvaluator:
    """
    Evaluator class for real data models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config['evaluation']['device'])
        self.output_dir = config['evaluation']['save_csv_path']
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize MLflow tracker
        self.mlflow_tracker = MLflowTracker(
            experiment_name=config['mlflow']['experiment_name'],
            tracking_uri=config['mlflow']['tracking_uri'],
            log_artifacts=config['mlflow']['log_artifacts'],
            log_models=config['mlflow']['log_models']
        )
        
        logger.info(f"Evaluator initialized with device: {self.device}")
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
        
        if model_name == 'SimpleCNN':
            model = SimpleCNN(input_length)
        elif model_name == 'SimpleViTRegressor':
            model = SimpleViTRegressor(input_length)
        elif model_name == 'ResidualCNN':
            model = ResidualCNN(input_length)
        elif model_name == 'ProposedCNN':
            model = ProposedCNN(input_length)
        elif model_name == 'BaseCNN':
            model = BaseCNN(input_length)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Apply custom parameters if provided
        if model_params:
            for key, value in model_params.items():
                if hasattr(model, key):
                    setattr(model, key, value)
        
        return model
    
    def load_model(self, model_path: str, model_params: Dict[str, Any] = None) -> nn.Module:
        """
        Load trained model from file.
        
        Args:
            model_path: Path to the model file
            model_params: Model parameters
            
        Returns:
            Loaded PyTorch model
        """
        model = self.create_model(model_params or {})
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded from: {model_path}")
        return model
    
    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                # Move batch to GPU
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                outputs = model(batch_x)
                
                # Move predictions and targets to CPU before storing
                predictions.append(outputs.cpu().numpy())
                targets.append(batch_y.cpu().numpy())
                
                # Clear GPU memory after each batch
                del batch_x, batch_y, outputs
                torch.cuda.empty_cache()
        
        # Concatenate all predictions and targets
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Flatten arrays for metric calculation
        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()
        
        # Calculate metrics
        mse = mean_squared_error(targets_flat, predictions_flat)
        mae = mean_absolute_error(targets_flat, predictions_flat)
        r2 = r2_score(targets_flat, predictions_flat)
        pearson_corr, pearson_p = pearsonr(targets_flat, predictions_flat)
        spearman_corr, spearman_p = spearmanr(targets_flat, predictions_flat)
        
        # Calculate additional metrics
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((targets_flat - predictions_flat) / (targets_flat + 1e-8))) * 100
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'mape': mape,
            'n_samples': len(targets_flat)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets
        }
    
    def create_evaluation_plots(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        output_dir: str
    ) -> List[str]:
        """
        Create evaluation plots.
        
        Args:
            predictions: Model predictions
            targets: True targets
            output_dir: Directory to save plots
            
        Returns:
            List of saved plot paths
        """
        plot_paths = []
        
        # Flatten arrays
        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()
        
        # 1. Scatter plot: Predictions vs Targets
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(targets_flat, predictions_flat, alpha=0.6, s=20)
        
        # Add diagonal line
        min_val = min(targets_flat.min(), predictions_flat.min())
        max_val = max(targets_flat.max(), predictions_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predictions vs True Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient to plot
        corr = np.corrcoef(targets_flat, predictions_flat)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
                transform=ax.transAxes, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        scatter_path = os.path.join(output_dir, "predictions_vs_targets.png")
        fig.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plot_paths.append(scatter_path)
        plt.close(fig)
        
        # 2. Residual plot
        residuals = targets_flat - predictions_flat
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(predictions_flat, residuals, alpha=0.6, s=20)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals (True - Predicted)')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)
        
        residual_path = os.path.join(output_dir, "residual_plot.png")
        fig.savefig(residual_path, dpi=300, bbox_inches='tight')
        plot_paths.append(residual_path)
        plt.close(fig)
        
        # 3. Histogram of residuals
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Residuals')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        ax.text(0.05, 0.95, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}', 
                transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        histogram_path = os.path.join(output_dir, "residuals_histogram.png")
        fig.savefig(histogram_path, dpi=300, bbox_inches='tight')
        plot_paths.append(histogram_path)
        plt.close(fig)
        
        # 4. Time series plot (if applicable)
        if len(predictions) > 100:
            # Sample for visualization
            n_samples = min(1000, len(predictions))
            indices = np.random.choice(len(predictions), n_samples, replace=False)
            
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(indices, targets_flat[indices], label='True Values', alpha=0.7)
            ax.plot(indices, predictions_flat[indices], label='Predictions', alpha=0.7)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Value')
            ax.set_title('Time Series Comparison (Sampled)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            timeseries_path = os.path.join(output_dir, "timeseries_comparison.png")
            fig.savefig(timeseries_path, dpi=300, bbox_inches='tight')
            plot_paths.append(timeseries_path)
            plt.close(fig)
        
        return plot_paths
    
    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        output_dir: str
    ) -> str:
        """
        Save evaluation results to files.
        
        Args:
            results: Evaluation results
            output_dir: Directory to save results
            
        Returns:
            Path to saved results file
        """
        # Save metrics to JSON
        metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        # Save predictions and targets to CSV
        predictions_df = pd.DataFrame({
            'predictions': results['predictions'].flatten(),
            'targets': results['targets'].flatten(),
            'residuals': results['targets'].flatten() - results['predictions'].flatten()
        })
        
        csv_file = os.path.join(output_dir, "evaluation_results.csv")
        predictions_df.to_csv(csv_file, index=False)
        
        logger.info(f"Evaluation results saved to: {output_dir}")
        
        return metrics_file
    
    def run_evaluation(
        self,
        model_path: str,
        model_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.
        
        Args:
            model_path: Path to the trained model
            model_params: Model parameters
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting evaluation pipeline...")
        
        # Create output directory for this evaluation
        eval_output_dir = os.path.join(
            self.output_dir,
            f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(eval_output_dir, exist_ok=True)
        
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
        
        # Create dataloaders
        train_dataloader, val_dataloader = create_chunked_dataloader(
            x_path=self.config['dataset']['x_train'],
            t_path=self.config['dataset']['t_train'],
            batch_size=self.config['hyperparameters']['batch_size'],
            x_key=self.config['dataset']['data_keys']['x_key'],
            t_key=self.config['dataset']['data_keys']['t_key'],
            max_samples=108,  # Limit samples for evaluation
            shuffle=False
        )
        
        logger.info(f"Created dataloaders: {len(train_dataloader)} train batches, {len(val_dataloader)} val batches")
        
        # Load model
        model = self.load_model(model_path, model_params)
        
        # Start MLflow run
        self.mlflow_tracker.start_run(
            run_name=f"real_data_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={
                "dataset": "real_data",
                "model": self.config['model']['name'],
                "task": "evaluation"
            }
        )
        
        try:
            # Evaluate on training set
            logger.info("Evaluating on training set...")
            train_results = self.evaluate_model(model, train_dataloader)
            
            # Evaluate on validation set
            logger.info("Evaluating on validation set...")
            val_results = self.evaluate_model(model, val_dataloader)
            
            # Create plots for validation set
            logger.info("Creating evaluation plots...")
            plot_paths = self.create_evaluation_plots(
                val_results['predictions'],
                val_results['targets'],
                eval_output_dir
            )
            
            # Save results
            results_file = self.save_evaluation_results(val_results, eval_output_dir)
            
            # Log to MLflow
            self.mlflow_tracker.log_metrics({
                f"train_{key}": value for key, value in train_results['metrics'].items()
            })
            self.mlflow_tracker.log_metrics({
                f"val_{key}": value for key, value in val_results['metrics'].items()
            })
            
            self.mlflow_tracker.log_dataset_info(dataset_info)
            self.mlflow_tracker.log_artifacts(eval_output_dir, "evaluation_outputs")
            
            logger.info("Evaluation pipeline completed successfully")
            
            return {
                'train_results': train_results,
                'val_results': val_results,
                'plot_paths': plot_paths,
                'results_file': results_file,
                'output_dir': eval_output_dir
            }
            
        finally:
            # End MLflow run
            self.mlflow_tracker.end_run()


def main():
    """Main function to run evaluation."""
    # Load configuration
    config_path = "/home/smatsubara/documents/sandbox/ml_airlift/config/config_real.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = RealDataEvaluator(config)
    
    # Run evaluation
    # Note: You need to provide the path to your trained model
    model_path = input("Enter the path to your trained model (.pth file): ")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    results = evaluator.run_evaluation(model_path)
    
    logger.info(f"Evaluation completed. Results saved to: {results['output_dir']}")
    logger.info(f"Validation metrics: {results['val_results']['metrics']}")


if __name__ == "__main__":
    main()
