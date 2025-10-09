import numpy as np
import matplotlib.pyplot as plt
import os
from src import debug_pipeline
import yaml

# Example usage
config_path = 'config/config.yaml'
file_path = "/home/smatsubara/documents/airlift/data/experiments/processed/solid_liquid/P20241011-1015_processed.npz"
debug_pipeline(config_path, file_path)