import torch
import numpy as np
import numpy as np
import torch
import yaml
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
x_train = np.load(config['dataset']['x_train'])
t_train = np.load(config['dataset']['t_train'])

x_train_tensor = torch.from_numpy(x_train).float()
t_train_tensor = torch.from_numpy(t_train).float()
if t_train_tensor.ndim == 1:
    t_train_tensor = t_train_tensor.unsqueeze(-1)
if x_train_tensor.ndim == 1:
    x_train_tensor = x_train_tensor.unsqueeze(-1)
print(torch.max(x_train_tensor))
print(x_train_tensor.shape)
print(t_train_tensor.shape)
#print(t_train_tensor)
import matplotlib.pyplot as plt

def plot_x_train(x_train,index):
    plt.figure(figsize=(10, 4))
    plt.plot(x_train[index], label="Sample 0")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("Visualization of One x_train Sample")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
for i in range(5):
    plot_x_train(x_train,i)
# Basic GPyTorch regression tutorial with a simple sine function dataset

import math
import torch
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(23)

# English comment: Set device to GPU cuda:1 if available, otherwise CPU
if torch.cuda.device_count() > 1:
    device = torch.device("cuda:1")
    print("Using device: cuda:1")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using device: cuda:0")
else:
    device = torch.device("cpu")
    print("Using device: cpu")

# Convert numpy arrays to torch tensors and move to device
# GPyTorch expects float32 tensors
t_train_tensor = torch.from_numpy(t_train).float().to(device)
x_train_tensor = torch.from_numpy(x_train).float().to(device)
print(t_train_tensor.shape)
print(x_train_tensor.shape)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Check if x_train_tensor is 2D (samples, features) or needs reshaping for CNN
# For 1D CNN, input should be (batch, channels, length)
# If x_train_tensor is (N, L), reshape to (N, 1, L)
if x_train_tensor.ndim == 2:
    x_train_tensor_cnn = x_train_tensor.unsqueeze(1)  # (N, 1, L)
else:
    # If already 3D, use as is
    x_train_tensor_cnn = x_train_tensor

# If t_train_tensor is 2D, flatten to 1D for regression
if t_train_tensor.ndim > 1 and t_train_tensor.shape[1] == 1:
    t_train_tensor_cnn = t_train_tensor.squeeze(1)
else:
    t_train_tensor_cnn = t_train_tensor
print(x_train_tensor_cnn.shape)
print(t_train_tensor_cnn.shape)
# Create TensorDataset and DataLoader
from torch.utils.data import random_split

dataset = TensorDataset(x_train_tensor_cnn, t_train_tensor_cnn)
total_size = len(dataset)
train_size = int(0.75 * total_size)
val_size = total_size - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
batch_size=8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# Get input length from data
input_length = x_train_tensor_cnn.shape[2]
print(input_length)