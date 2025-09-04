import torch
import numpy as np
import yaml
from models import SimpleCNN
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
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



# Set random seed for reproducibility
torch.manual_seed(23)
device1 = torch.device("cuda:1")

# Convert numpy arrays to torch tensors and move to device
# GPyTorch expects float32 tensors
t_train_tensor = torch.from_numpy(t_train).float().to(device1)
x_train_tensor = torch.from_numpy(x_train).float().to(device1)
print(t_train_tensor.shape)
print(x_train_tensor.shape)


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
batch_size=config['hyperparameters']['batch_size']
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# Get input length from data
#input_length = config['hyperparameters']['input_length']

size=config['hyperparameters']['input_length']
# Instantiate model, loss, and Aoptimizer, and move model to device
model = SimpleCNN(size).to(device1)



# English comment: Define a custom loss function as the sum of (target - prediction) / target
def relative_sum_loss(pred, target):
    # English comment: Take the logarithm of the loss to avoid instability due to very small values
    epsilon = 1e-7  # Avoid division by zero and log(0)
    # 英語でコメント: Calculate the mean of the relative error instead of the sum
    loss = torch.mean(torch.abs(target - pred) / (target + epsilon))
    #loss = torch.log(torch.abs(loss) + epsilon)
    return loss

#criterion = relative_sum_loss
criterion = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters(), lr=config['hyperparameters']['learning_rate'])


loss_history = []

num_epochs = config['hyperparameters']['num_epochs']
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_dataloader:
        batch_x = batch_x.to(device1)
        batch_y = batch_y.to(device1)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)
    epoch_loss = running_loss / len(train_dataloader.dataset)
    loss_history.append(epoch_loss)
    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")



save_weights_path = config['dataset']['weights_path']
torch.save(model.state_dict(), os.path.join(save_weights_path, 'model.pth'))


# Plot and save the learning curve (logarithmic loss)
save_log_path = config['dataset']['log_path']
plt.figure()
plt.plot(range(1, num_epochs + 1), [np.log(l) for l in loss_history])
plt.title('Learning Curve (Log Loss)')
plt.xlabel('Epoch')
plt.ylabel('Log(Loss)')
plt.grid(True)
plt.savefig(os.path.join(save_log_path, 'learning_curve_log.png'))
plt.close()

plt.figure()
plt.plot(range(1, num_epochs + 1), loss_history)
plt.title('Learning Curve (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(os.path.join(save_log_path, 'learning_curve.png'))
plt.close()

import os
from models import SimpleCNN
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
save_path="/home/smatsubara/documents/sandbox/psdata2matlab/tmp"
model.load_state_dict(torch.load(os.path.join(save_path, 'model.pth')))
model = model.to('cuda:0')
model.eval()
val_predictions = []
val_targets = []
with torch.no_grad():
    for val_x, val_y in val_dataloader:
        # English comment: Send both val_x and val_y to cuda:0 for inference
        val_x = val_x.to('cuda:0')
        val_y = val_y.to('cuda:0')
        outputs = model(val_x)
        val_predictions.append(outputs.cpu())
        val_targets.append(val_y.cpu())
val_predictions = torch.cat(val_predictions, dim=0)
val_targets = torch.cat(val_targets, dim=0)
print("Validation predictions shape:", val_predictions.shape)
print("Validation targets shape:", val_targets.shape)
print(val_predictions)
# English comment: Plot the validation predictions and targets for comparison
import matplotlib.pyplot as plt

# English comment: Calculate the correlation coefficient between predicted and actual values
import numpy as np

# Convert tensors to numpy arrays
val_targets_np = val_targets.numpy().flatten()
val_predictions_np = val_predictions.numpy().flatten()

# English comment: Compute the Pearson correlation coefficient
correlation_matrix = np.corrcoef(val_targets_np, val_predictions_np)
correlation_coefficient = correlation_matrix[0, 1]
print(f"Pearson correlation coefficient between predictions and actual values: {correlation_coefficient:.4f}")

# English comment: Plot predicted values (y-axis) against actual values (x-axis) in the xy-plane, with both axes limited to the range [0, 1]
plt.figure(figsize=(8, 8))
plt.scatter(val_targets_np, val_predictions_np, alpha=0.6, marker='o', label='Predictions')
plt.plot([0, 1], [0, 1], 'r--', label='Ideal (y=x)')
plt.title('Predicted vs Actual Values (Validation Set)')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.xlim(0, 0.1)
plt.ylim(0, 0.1)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_path, 'val_pred_vs_actual_scatter.png'))
plt.show()
# English comment: Visualize the kernels (filters) of the first convolutional layer and compute Grad-CAM for a sample input

# --- Visualize kernels of the first convolutional layer (Conv1d version) ---
# English comment: Get the first Conv1d layer (assuming model has attribute 'conv1' or similar)
first_conv1d = None
for m in model.modules():
    if isinstance(m, torch.nn.Conv1d):
        first_conv1d = m
        break

if first_conv1d is not None:
    kernels = first_conv1d.weight.data.cpu().numpy()  # shape: (out_channels, in_channels, kernel_size)
    num_kernels = kernels.shape[0]
    num_channels = kernels.shape[1]
    kernel_size = kernels.shape[2]
    plt.figure(figsize=(num_kernels * 2, 2 * num_channels))
    for i in range(num_kernels):
        for j in range(num_channels):
            plt.subplot(num_kernels, num_channels, i * num_channels + j + 1)
            # English comment: Show the 1D kernel as a line plot
            plt.plot(kernels[i, j, :])
            plt.title(f'Kernel {i}, Channel {j}')
            plt.axis('tight')
            plt.grid(True)
    plt.suptitle('First Conv1d Layer Kernels')
    plt.tight_layout()
    plt.show()
else:
    print("No Conv1d layer found in the model.")

# --- Grad-CAM visualization for Conv1d ---
# English comment: Define a simple Grad-CAM function for the last Conv1d layer
class GradCAM1d:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            target = output.squeeze()
        else:
            target = output[:, class_idx]
        if target.ndim > 0:
            target = target.sum()
        target.backward(retain_graph=True)
        gradients = self.gradients  # shape: (batch, channels, length)
        activations = self.activations  # shape: (batch, channels, length)
        weights = gradients.mean(dim=2, keepdim=True)  # shape: (batch, channels, 1)
        grad_cam_map = (weights * activations).sum(dim=1, keepdim=True)  # shape: (batch, 1, length)
        grad_cam_map = torch.relu(grad_cam_map)
        grad_cam_map = torch.nn.functional.interpolate(grad_cam_map, size=input_tensor.shape[2], mode='linear', align_corners=False)
        grad_cam_map = grad_cam_map.squeeze().cpu().numpy()
        grad_cam_map = (grad_cam_map - grad_cam_map.min()) / (grad_cam_map.max() - grad_cam_map.min() + 1e-8)
        return grad_cam_map

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

target_layer_1d = None
for m in reversed(list(model.modules())):
    if isinstance(m, torch.nn.Conv1d):
        target_layer_1d = m
        break

if target_layer_1d is not None:
    grad_cam_1d = GradCAM1d(model, target_layer_1d)
    # val_x shape: (batch, channels, length)
    sample_input = val_x[0].unsqueeze(0).to('cuda:0')
    grad_cam_map = grad_cam_1d(sample_input)
    grad_cam_1d.remove_hooks()
    # English comment: Plot the Grad-CAM heatmap (1D)
    plt.figure(figsize=(10, 4))
    plt.plot(grad_cam_map)
    plt.title('Grad-CAM Map for Sample Input (Conv1d)')
    plt.xlabel('Time Axis')
    plt.ylabel('Grad-CAM Intensity')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No Conv1d layer found for Grad-CAM.")


