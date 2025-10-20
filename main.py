import torch
import numpy as np
import numpy as np
import torch
import yaml
import hydra
import os
import datetime
from omegaconf import OmegaConf, DictConfig

@hydra.main(config_path="/home/smatsubara/documents/sandbox/ml_airlift/config", config_name="config.yaml")
def main(cfg: DictConfig):
    print(cfg)

if __name__ == "__main__":
    main()
