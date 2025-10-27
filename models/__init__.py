from .cnn import SimpleCNN, ResidualCNN, BaseCNN, ProposedCNN

# Try to import SimpleViTRegressor, but don't fail if it's not available
try:
    from .transformers import SimpleViTRegressor
    __all__ = ['SimpleCNN', 'SimpleViTRegressor', 'ResidualCNN', 'BaseCNN', 'ProposedCNN']
except ImportError:
    __all__ = ['SimpleCNN', 'ResidualCNN', 'BaseCNN', 'ProposedCNN']