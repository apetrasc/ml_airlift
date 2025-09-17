import torch
import numpy as np
import numpy as np
import torch
import yaml
import hydra
import os
import datetime
from omegaconf import OmegaConf, DictConfig
from models import SimpleCNN
from config import config
# Todo: predict based on trained model
config = OmegaConf.load('config/config.yaml')
model = SimpleCNN(input_length=2500)
model.load_state_dict(torch.load(config['evaluation']['weights_path'] + 'model.pth'))
model.eval()
