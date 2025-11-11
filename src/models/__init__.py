"""
Model definitions for airlift project.
"""

from .cnn import (
    SimpleCNNReal,
    SimpleCNNReal2D,
    ResidualBlock2D,
    SimpleCNN,
    ResidualCNN,
    BaseCNN,
    ProposedCNN
)

__all__ = [
    'SimpleCNNReal',
    'SimpleCNNReal2D',
    'ResidualBlock2D',
    'SimpleCNN',
    'ResidualCNN',
    'BaseCNN',
    'ProposedCNN',
]

