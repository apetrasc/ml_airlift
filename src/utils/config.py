"""
Configuration utilities.
"""

import yaml
import argparse
from typing import Dict, Any, Optional

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def print_config_summary(config: Dict[str, Any]):
    """Print a summary of the configuration."""
    print("=" * 50)
    print("Configuration Summary")
    print("=" * 50)
    # Add config printing logic here
    print("=" * 50)

