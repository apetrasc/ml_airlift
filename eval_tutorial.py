import polars as pl
from src import preprocess_and_predict
from models import SimpleCNN, SimpleViTRegressor, ResidualCNN
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import math
import argparse
# Load configuration from YAML file
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

dataset_dir = '/mnt/sdb/yyamaguchi/psdata2matlab/simulation/dataset'
device = config["training"]["device"]


# Load the trained model
model = SimpleCNN(config['hyperparameters']['input_length']).to(config['evaluation']['device'])
#model = ResidualCNN(config['hyperparameters']['input_length']).to(config['evaluation']['device'])
#model = SimpleViTRegressor(config['hyperparameters']['input_length']).to(config['evaluation']['device'])
# You can use the argparse library to accept a command-line argument for base_dir (datetime).


parser = argparse.ArgumentParser(description="Run evaluation with specified base directory (datetime).")
parser.add_argument('--datetime', type=str, required=True, help='Base directory for evaluation (e.g., /home/smatsubara/documents/airlift/data/outputs/2025-09-07/14-39-46)')
args = parser.parse_args()

base_dir = args.datetime
model_path = os.path.join(base_dir + '/weights/model.pth')
model.load_state_dict(
    torch.load(
        model_path,
        map_location=config['evaluation']['device'],
        weights_only=True,
    )
)
model.eval()

x_train_path = os.path.join(dataset_dir,'x_train.npy')
t_train_path = os.path.join(dataset_dir,'t_train.npy')

x_train = np.load(x_train_path)
t_train = np.load(t_train_path)
eval_result = []
for x_train_signal in x_train:
    model.eval()
    with torch.no_grad():
        x_test_tensor_cnn = torch.from_numpy(x_train_signal).float()
        x_test_tensor_cnn = x_test_tensor_cnn.to(device)
        prediction = model(x_test_tensor_cnn)
        eval_result.append(prediction)
        del prediction
        torch.cuda.empty_cache

plt.fugyre(figsize=(8,8))
plt.rcParams["font.size"] = 18
plt.plot(t_train, eval_result, 'o',color='blue',label='glass beads')
plt.plot([0,1],[0,1],'r--',label='Ideal (y=x)')
plt.xlabel("Ground Truth")
plt.ylabel("Prediction")
plt.xlim(0,0.2)
plt.ylim(0,0.2)
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(base_dir, 'predicted_vs_ground_truth_sim2sim.png'))