#!/usr/bin/env python3
"""
Configuration utilities for loading and managing YAML config files.
"""

import yaml
import os
from typing import Dict, Any, Optional
import argparse


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge command line arguments with configuration file.
    Command line arguments take precedence over config file values.
    
    Args:
        config: Configuration dictionary from YAML file
        args: Command line arguments
        
    Returns:
        Merged configuration dictionary
    """
    # Convert args to dictionary, excluding None values
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    
    # Merge configurations (args override config)
    merged_config = config.copy()
    
    # Map command line arguments to config structure
    arg_mapping = {
        'x': 'dataset.x_train',
        't': 'dataset.t_train',
        'x_key': 'dataset.x_key',
        't_key': 'dataset.t_key',
        'device': 'training.device',
        'epochs': 'training.epochs',
        'batch': 'training.batch_size',
        'limit': 'dataset.limit_samples',
        'workers': 'training.workers',
        'resize_h': 'model.resize_hw.0',
        'resize_w': 'model.resize_hw.1',
        'ds_factor': 'dataset.downsample_factor',
        'output_dir': 'output.model_save_dir',
        'verbose': 'logging.verbose'
    }
    
    for arg_key, config_path in arg_mapping.items():
        if arg_key in args_dict:
            # Navigate to nested config location
            keys = config_path.split('.')
            current = merged_config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = args_dict[arg_key]
    
    return merged_config


def get_nested_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get value from nested dictionary using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to value (e.g., 'training.epochs')
        default: Default value if key not found
        
    Returns:
        Value at key_path or default
    """
    keys = key_path.split('.')
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def print_config_summary(config: Dict[str, Any]):
    """Print a summary of the configuration."""
    print("🔧 Configuration Summary")
    print("=" * 50)
    
    # Dataset info
    x_path = get_nested_value(config, 'dataset.x_train', 'N/A')
    t_path = get_nested_value(config, 'dataset.t_train', 'N/A')
    limit = get_nested_value(config, 'dataset.limit_samples', 0)
    ds_factor = get_nested_value(config, 'dataset.downsample_factor', 1)
    
    print(f"Dataset:")
    print(f"  X: {x_path}")
    print(f"  T: {t_path}")
    print(f"  Limit: {limit if limit > 0 else 'All samples'}")
    print(f"  Downsample: {ds_factor}x")
    
    # Model info
    model_type = get_nested_value(config, 'model.type', 'N/A')
    in_channels = get_nested_value(config, 'model.in_channels', 'N/A')
    out_dim = get_nested_value(config, 'model.out_dim', 'N/A')
    resize_hw = get_nested_value(config, 'model.resize_hw', None)
    
    print(f"Model:")
    print(f"  Type: {model_type}")
    print(f"  Input channels: {in_channels}")
    print(f"  Output dim: {out_dim}")
    print(f"  Resize: {resize_hw if resize_hw else 'Full resolution'}")
    
    # Training info
    device = get_nested_value(config, 'training.device', 'N/A')
    epochs = get_nested_value(config, 'training.epochs', 'N/A')
    batch_size = get_nested_value(config, 'training.batch_size', 'N/A')
    lr = get_nested_value(config, 'training.learning_rate', 'N/A')
    
    print(f"Training:")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    
    # Output info
    output_dir = get_nested_value(config, 'output.model_save_dir', 'N/A')
    print(f"Output: {output_dir}")
    print("=" * 50)


def create_argparser() -> argparse.ArgumentParser:
    """Create argument parser with common options."""
    parser = argparse.ArgumentParser(description='Train CNN model with configuration file')
    
    # Config file
    parser.add_argument('--config', default='config/config_real_updated.yaml',
                       help='Path to configuration file')
    
    # Data options
    parser.add_argument('--x', help='Override X data path')
    parser.add_argument('--t', help='Override T data path')
    parser.add_argument('--x_key', help='Override X data key')
    parser.add_argument('--t_key', help='Override T data key')
    parser.add_argument('--limit', type=int, help='Limit number of samples')
    parser.add_argument('--ds_factor', type=int, help='Downsample factor')
    
    # Model options
    parser.add_argument('--resize_h', type=int, help='Resize height')
    parser.add_argument('--resize_w', type=int, help='Resize width')
    
    # Training options
    parser.add_argument('--device', help='Device to use')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch', type=int, help='Batch size')
    parser.add_argument('--workers', type=int, help='Number of workers')
    
    # Output options
    parser.add_argument('--output_dir', help='Output directory')
    
    # Other options
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser


if __name__ == "__main__":
    # Test configuration loading
    parser = create_argparser()
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Merge with args
    merged_config = merge_config_with_args(config, args)
    
    # Print summary
    print_config_summary(merged_config)



