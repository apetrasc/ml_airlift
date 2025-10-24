import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import numpy as np
import os
import logging
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow integration for experiment tracking and model management.
    """
    
    def __init__(
        self,
        experiment_name: str = "default_experiment",
        tracking_uri: str = "file:./mlruns",
        log_artifacts: bool = True,
        log_models: bool = True
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: URI for MLflow tracking
            log_artifacts: Whether to log artifacts
            log_models: Whether to log models
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.log_artifacts = log_artifacts
        self.log_models = log_models
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new experiment: {experiment_name}")
            else:
                self.experiment_id = self.experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Error setting up experiment: {e}")
            raise
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        self.run_id = None
        self.run_name = None
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Tags to add to the run
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run_name = run_name
        
        # Start run
        mlflow.start_run(run_name=run_name)
        self.run_id = mlflow.active_run().info.run_id
        
        # Add tags
        if tags:
            mlflow.set_tags(tags)
        
        # Add default tags
        mlflow.set_tags({
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Started MLflow run: {run_name} (ID: {self.run_id})")
    
    def end_run(self):
        """End the current MLflow run."""
        if mlflow.active_run():
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.run_name}")
    
    def log_parameters(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        try:
            # Convert parameters to appropriate types
            loggable_params = {}
            for key, value in params.items():
                if isinstance(value, (int, float, str, bool)):
                    loggable_params[key] = value
                else:
                    loggable_params[key] = str(value)
            
            mlflow.log_params(loggable_params)
            logger.info(f"Logged {len(loggable_params)} parameters")
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number for the metrics
        """
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
                else:
                    logger.warning(f"Skipping non-numeric metric: {key} = {value}")
            
            logger.info(f"Logged {len(metrics)} metrics")
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def log_model(
        self,
        model: nn.Module,
        model_name: str = "model",
        signature: Optional[mlflow.models.ModelSignature] = None,
        input_example: Optional[np.ndarray] = None,
        registered_model_name: Optional[str] = None
    ):
        """
        Log PyTorch model to MLflow.
        
        Args:
            model: PyTorch model to log
            model_name: Name for the model
            signature: Model signature
            input_example: Example input for the model
            registered_model_name: Name for model registration
        """
        if not self.log_models:
            logger.info("Model logging disabled")
            return
        
        try:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
            logger.info(f"Logged model: {model_name}")
        except Exception as e:
            logger.error(f"Error logging model: {e}")
    
    def log_artifacts(self, artifact_path: str, artifact_dir: str):
        """
        Log artifacts to MLflow.
        
        Args:
            artifact_path: Path to artifact file or directory
            artifact_dir: Directory name in MLflow artifacts
        """
        if not self.log_artifacts:
            logger.info("Artifact logging disabled")
            return
        
        try:
            mlflow.log_artifacts(artifact_path, artifact_dir)
            logger.info(f"Logged artifacts from {artifact_path} to {artifact_dir}")
        except Exception as e:
            logger.error(f"Error logging artifacts: {e}")
    
    def log_figure(self, figure: plt.Figure, artifact_file: str):
        """
        Log matplotlib figure to MLflow.
        
        Args:
            figure: Matplotlib figure to log
            artifact_file: Name for the artifact file
        """
        if not self.log_artifacts:
            logger.info("Artifact logging disabled")
            return
        
        try:
            mlflow.log_figure(figure, artifact_file)
            logger.info(f"Logged figure: {artifact_file}")
        except Exception as e:
            logger.error(f"Error logging figure: {e}")
    
    def log_training_history(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_metrics: Optional[Dict[str, List[float]]] = None,
        val_metrics: Optional[Dict[str, List[float]]] = None
    ):
        """
        Log training history to MLflow.
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
        """
        try:
            # Log losses
            for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }, step=epoch)
            
            # Log additional metrics
            if train_metrics:
                for metric_name, values in train_metrics.items():
                    for epoch, value in enumerate(values):
                        mlflow.log_metric(f"train_{metric_name}", value, step=epoch)
            
            if val_metrics:
                for metric_name, values in val_metrics.items():
                    for epoch, value in enumerate(values):
                        mlflow.log_metric(f"val_{metric_name}", value, step=epoch)
            
            logger.info(f"Logged training history for {len(train_losses)} epochs")
        except Exception as e:
            logger.error(f"Error logging training history: {e}")
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """
        Log dataset information to MLflow.
        
        Args:
            dataset_info: Dictionary containing dataset information
        """
        try:
            # Log dataset parameters
            mlflow.log_params({
                f"dataset_{key}": value for key, value in dataset_info.items()
                if isinstance(value, (int, float, str, bool))
            })
            
            # Log dataset info as artifact
            if self.log_artifacts:
                dataset_info_file = "dataset_info.json"
                with open(dataset_info_file, 'w') as f:
                    json.dump(dataset_info, f, indent=2)
                mlflow.log_artifact(dataset_info_file)
                os.remove(dataset_info_file)  # Clean up
            
            logger.info("Logged dataset information")
        except Exception as e:
            logger.error(f"Error logging dataset info: {e}")
    
    def log_optimization_results(self, optimization_results: Dict[str, Any]):
        """
        Log optimization results to MLflow.
        
        Args:
            optimization_results: Dictionary containing optimization results
        """
        try:
            # Log best parameters as parameters
            if "best_params" in optimization_results:
                mlflow.log_params({
                    f"best_{key}": value for key, value in optimization_results["best_params"].items()
                })
            
            # Log optimization metrics
            if "best_value" in optimization_results:
                mlflow.log_metric("best_optimization_value", optimization_results["best_value"])
            
            if "n_trials" in optimization_results:
                mlflow.log_metric("n_optimization_trials", optimization_results["n_trials"])
            
            # Log full optimization results as artifact
            if self.log_artifacts:
                optimization_file = "optimization_results.json"
                with open(optimization_file, 'w') as f:
                    json.dump(optimization_results, f, indent=2)
                mlflow.log_artifact(optimization_file)
                os.remove(optimization_file)  # Clean up
            
            logger.info("Logged optimization results")
        except Exception as e:
            logger.error(f"Error logging optimization results: {e}")
    
    def create_learning_curve_plot(
        self,
        train_losses: List[float],
        val_losses: List[float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create and log learning curve plot.
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label='Training Loss', color='blue')
        ax.plot(epochs, val_losses, label='Validation Loss', color='red')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Curve')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Log to MLflow
        self.log_figure(fig, "learning_curve.png")
        
        return fig
    
    def get_run_info(self) -> Dict[str, Any]:
        """
        Get information about the current run.
        
        Returns:
            Dictionary containing run information
        """
        if not mlflow.active_run():
            return {}
        
        run = mlflow.active_run()
        return {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time
        }
