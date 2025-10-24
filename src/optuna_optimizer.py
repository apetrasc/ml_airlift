import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, Any, Tuple, Callable
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimization for machine learning models.
    """
    
    def __init__(
        self,
        study_name: str = "optimization_study",
        storage: str = "sqlite:///optuna_study.db",
        direction: str = "minimize",
        n_trials: int = 50,
        timeout: int = 3600,
        load_if_exists: bool = True
    ):
        """
        Initialize the Optuna optimizer.
        
        Args:
            study_name: Name of the study
            storage: Storage backend for the study
            direction: Optimization direction ('minimize' or 'maximize')
            n_trials: Number of trials to run
            timeout: Timeout in seconds
            load_if_exists: Whether to load existing study
        """
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.load_if_exists = load_if_exists
        
        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=load_if_exists
        )
        
        logger.info(f"Created Optuna study: {study_name}")
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        hyperparams = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'num_epochs': trial.suggest_int('num_epochs', 50, 200),
            'l1_lambda': trial.suggest_float('l1_lambda', 1e-8, 1e-5, log=True),
            'l2_lambda': trial.suggest_float('l2_lambda', 1e-8, 1e-5, log=True),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
            'hidden_channels_1': trial.suggest_categorical('hidden_channels_1', [32, 64, 128]),
            'hidden_channels_2': trial.suggest_categorical('hidden_channels_2', [64, 128, 256]),
            'hidden_channels_3': trial.suggest_categorical('hidden_channels_3', [128, 256, 512]),
        }
        
        return hyperparams
    
    def objective_function(
        self,
        trial: optuna.Trial,
        model_factory: Callable,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: str,
        input_length: int,
        model_params: Dict[str, Any] = None
    ) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            model_factory: Function to create model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            device: Device to use for training
            input_length: Input sequence length
            model_params: Additional model parameters
            
        Returns:
            Validation loss (to be minimized)
        """
        try:
            # Suggest hyperparameters
            hyperparams = self.suggest_hyperparameters(trial)
            
            # Create model with suggested parameters
            if model_params is None:
                model_params = {}
            
            # Update model parameters with hyperparameters
            model_params.update({
                'hidden_channels': [
                    hyperparams['hidden_channels_1'],
                    hyperparams['hidden_channels_2'],
                    hyperparams['hidden_channels_3']
                ],
                'dropout_rate': hyperparams['dropout_rate']
            })
            
            # Create model using the factory function
            model = model_factory(model_params)
            model = model.to(device)
            
            # Setup optimizer and loss
            optimizer = optim.Adam(
                model.parameters(),
                lr=hyperparams['learning_rate']
            )
            criterion = nn.MSELoss()
            
            # Training loop with memory-efficient GPU usage
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(hyperparams['num_epochs']):
                # Training
                model.train()
                train_loss = 0.0
                for batch_x, batch_y in train_dataloader:
                    # Move batch to GPU
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    # Add regularization
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                    loss = loss + hyperparams['l1_lambda'] * l1_norm + hyperparams['l2_lambda'] * l2_norm
                    
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    
                    # Clear GPU memory after each batch
                    del batch_x, batch_y, outputs, loss
                    torch.cuda.empty_cache()
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_x, val_y in val_dataloader:
                        # Move batch to GPU
                        val_x = val_x.to(device, non_blocking=True)
                        val_y = val_y.to(device, non_blocking=True)
                        outputs = model(val_x)
                        loss = criterion(outputs, val_y)
                        val_loss += loss.item()
                        
                        # Clear GPU memory after each batch
                        del val_x, val_y, outputs, loss
                        torch.cuda.empty_cache()
                
                val_loss /= len(val_dataloader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    break
                
                # Report intermediate result
                trial.report(val_loss, epoch)
                
                # Handle pruning
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            return best_val_loss
            
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            raise
    
    def optimize(
        self,
        model_factory: Callable,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: str,
        input_length: int,
        model_params: Dict[str, Any] = None
    ) -> optuna.Study:
        """
        Run hyperparameter optimization.
        
        Args:
            model_factory: Function to create model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            device: Device to use for training
            input_length: Input sequence length
            model_params: Additional model parameters
            
        Returns:
            Completed Optuna study
        """
        logger.info(f"Starting optimization with {self.n_trials} trials")
        
        # Create objective function with fixed parameters
        objective = lambda trial: self.objective_function(
            trial=trial,
            model_factory=model_factory,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            input_length=input_length,
            model_params=model_params
        )
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        logger.info(f"Optimization completed. Best value: {self.study.best_value}")
        logger.info(f"Best parameters: {self.study.best_params}")
        
        return self.study
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """
        Get the best parameters from the study.
        
        Returns:
            Dictionary of best parameters
        """
        return self.study.best_params
    
    def get_best_value(self) -> float:
        """
        Get the best value from the study.
        
        Returns:
            Best objective value
        """
        return self.study.best_value
    
    def save_study_results(self, output_dir: str) -> str:
        """
        Save study results to files.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Path to saved results file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best parameters
        best_params_file = os.path.join(output_dir, "best_parameters.json")
        with open(best_params_file, 'w') as f:
            json.dump(self.study.best_params, f, indent=2)
        
        # Save study statistics
        stats = {
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'n_trials': len(self.study.trials),
            'study_name': self.study_name,
            'timestamp': datetime.now().isoformat()
        }
        
        stats_file = os.path.join(output_dir, "optimization_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Study results saved to {output_dir}")
        
        return stats_file
    
    def get_trial_results(self) -> list:
        """
        Get results from all trials.
        
        Returns:
            List of trial results
        """
        results = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                results.append({
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                })
        
        return results
