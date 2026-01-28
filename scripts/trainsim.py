import torch
import numpy as np
import yaml
from models import SimpleCNN, SimpleViTRegressor, ResidualCNN, ProposedCNN, BaseCNN
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader,random_split
import matplotlib.pyplot as plt
import os
from src import npz2png
from torch.nn import init

import hydra
import datetime
from omegaconf import OmegaConf

@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg):

    # Get hydra run directory as base path for reading relative inputs
    base_dir = os.getcwd()
    # Create time-based output directory under /mnt/matsubara/outputs/YYYY-MM-DD/HH-MM-SS
    outputs_root = "/mnt/matsubara/outputs"
    now = datetime.datetime.now()
    date_dir = now.strftime('%Y-%m-%d')
    run_dir = os.path.join(outputs_root, date_dir, now.strftime('%H-%M-%S'))
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Run directory: {os.path.abspath(run_dir)}")
    print(f"Logs will be saved under: {os.path.abspath(logs_dir)}")
    # Load config (already loaded as cfg)
    x_train = np.load(os.path.relpath(cfg.dataset.x_train, base_dir))
    t_train = np.load(os.path.relpath(cfg.dataset.t_train, base_dir))

    x_train_tensor = torch.from_numpy(x_train).float()
    t_train_tensor = torch.from_numpy(t_train).float()

    # Helper function to validate and get device
    def get_valid_device(device_str):
        """Validate device string and return valid device, fallback to cuda:0 or CPU if invalid."""
        try:
            device = torch.device(device_str)
            if device.type == 'cuda' and not torch.cuda.is_available():
                print(f"CUDA device '{device_str}' unavailable. Falling back to CPU.")
                return torch.device("cpu")
            elif device.type == 'cuda':
                device_idx = int(str(device).split(":")[-1])
                if device_idx >= torch.cuda.device_count():
                    # If GPU is available but invalid device ordinal, use cuda:0 instead of CPU
                    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                        print(f"CUDA device ordinal {device_idx} invalid (only {torch.cuda.device_count()} devices). Falling back to cuda:0.")
                        return torch.device("cuda:0")
                    else:
                        print(f"CUDA device ordinal {device_idx} invalid. No CUDA devices available. Falling back to CPU.")
                        return torch.device("cpu")
            return device
        except Exception as e:
            # If error occurs but CUDA is available, try cuda:0
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                print(f"Error selecting device '{device_str}': {e}. Falling back to cuda:0.")
                return torch.device("cuda:0")
            else:
                print(f"Error selecting device '{device_str}': {e}. Using CPU.")
                return torch.device("cpu")
    
    # Check for valid CUDA device for training, fallback to CPU if invalid
    device1 = get_valid_device(cfg.training.device)
    if t_train_tensor.ndim == 1:
        t_train_tensor = t_train_tensor.unsqueeze(-1)
    if x_train_tensor.ndim == 1:
        x_train_tensor = x_train_tensor.unsqueeze(-1)
    print(torch.max(x_train_tensor))
    print(x_train_tensor.shape)
    print(t_train_tensor.shape)
    plt.figure(figsize=(10, 4))
    plt.ylim(0, 1)
    plt.plot(x_train_tensor[30, :])
    plt.savefig(os.path.join(logs_dir, 'x_train_tensor.png'))
    plt.close()
    

    torch.manual_seed(cfg.hyperparameters.seed)
    # Move tensors to device after initial checks and unsqueezing (and after seeding)
    x_train_tensor = x_train_tensor.to(device1)
    t_train_tensor = t_train_tensor.to(device1)
    print(t_train_tensor.shape)
    print(x_train_tensor.shape)

    if x_train_tensor.ndim == 2:
        x_train_tensor_cnn = x_train_tensor.unsqueeze(1)
    else:
        x_train_tensor_cnn = x_train_tensor

    if t_train_tensor.ndim > 1 and t_train_tensor.shape[1] == 1:
        t_train_tensor_cnn = t_train_tensor.squeeze(1)
    else:
        t_train_tensor_cnn = t_train_tensor
    print(x_train_tensor_cnn.shape)
    print(t_train_tensor_cnn.shape)

    dataset = TensorDataset(x_train_tensor_cnn, t_train_tensor_cnn)
    total_size = len(dataset)
    train_size = int(0.75 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    batch_size = cfg.hyperparameters.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    size = cfg.hyperparameters.input_length
    # Load pretrained weights and fine-tune only the last layer (fc)
    model = SimpleCNN(size).to(device1)
    # Load pretrained weights (make sure the model definition matches)
    pretrained_path = cfg.training.pretrained_path
    try:
        state_dict = torch.load(pretrained_path, map_location=device1, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Warning: Could not load pretrained weights from {pretrained_path}. Exception: {e}")
        # Optionally: raise

    # Freeze all parameters except the last fully connected (fc) layer
    for name, param in model.named_parameters():
        if '.fc' in name or name.startswith('fc'):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Now only model.fc will be optimized during training
    #model = ProposedCNN(size).to(device1)
    #model = BaseCNN(size).to(device1)
    #model = ResidualCNN(size).to(device1)
    #model = SimpleViTRegressor(size).to(device1)
    # init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    # init.xavier_normal_(model.conv2.weight, gain=1.0)
    # init.xavier_normal_(model.fc.weight, gain=1.0)
    #init.kaiming_normal_(model.conv2.weight, mode='fan_out', nonlinearity='relu')
    #init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
    def relative_sum_loss(pred, target):
        epsilon = 1e-7
        loss = torch.mean(torch.abs(target - pred) / (target + epsilon))
        return loss

    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.hyperparameters.learning_rate)

    loss_history = []
    num_epochs = cfg.hyperparameters.num_epochs
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        l1_lambda = cfg.hyperparameters.l1_lambda
        l2_lambda = cfg.hyperparameters.l2_lambda
        for batch_x, batch_y in train_dataloader:
            batch_x = batch_x.to(device1)
            batch_y = batch_y.to(device1)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        epoch_loss = running_loss / len(train_dataloader.dataset)
        train_loss_history.append(epoch_loss)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_dataloader:
                val_x = val_x.to(device1)
                val_y = val_y.to(device1)
                val_outputs = model(val_x)
                val_loss = criterion(val_outputs, val_y)
                val_running_loss += val_loss.item() * val_x.size(0)
        val_epoch_loss = val_running_loss / len(val_dataloader.dataset)
        val_loss_history.append(val_epoch_loss)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

    # Create weights directory inside the same run directory structure
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    sota_dir=cfg.training.pretrained_path
    #torch.save(model.state_dict(), os.path.join(sota_dir, 'model.pth'))
    torch.save(model.state_dict(), os.path.join(weights_dir, 'model.pth'))
    plt.figure()
    plt.plot(range(1, num_epochs + 1), [np.log(l) for l in train_loss_history], label='Train Log(Loss)')
    plt.plot(range(1, num_epochs + 1), [np.log(l) for l in val_loss_history], label='Validation Log(Loss)')
    plt.title('Learning Curve (Log Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Log(Loss)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(logs_dir, 'learning_curve_log.png'))
    plt.close()

    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_loss_history, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss')
    plt.title('Learning Curve (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(logs_dir, 'learning_curve.png'))
    plt.close()

    #model.load_state_dict(torch.load(os.path.join(sota_dir, 'model.pth')))
    model.load_state_dict(torch.load(os.path.join(weights_dir, 'model.pth'), weights_only=True))
    # Validate evaluation device
    eval_device = get_valid_device(cfg.evaluation.device)
    model = model.to(eval_device)
    model.eval()
    val_predictions = []
    val_targets = []
    with torch.no_grad():
        for val_x, val_y in val_dataloader:
            val_x = val_x.to(eval_device)
            val_y = val_y.to(eval_device)
            outputs = model(val_x)
            val_predictions.append(outputs.cpu())
            val_targets.append(val_y.cpu())
    val_predictions = torch.cat(val_predictions, dim=0)
    val_targets = torch.cat(val_targets, dim=0)
    accuracy = torch.sqrt(torch.mean(torch.abs(val_predictions - val_targets)**2))
    print(f"Root Mean Square Error: {accuracy:.4f}")
    print("Validation predictions shape:", val_predictions.shape)
    print("Validation targets shape:", val_targets.shape)
    print(val_predictions)

    val_targets_np = val_targets.numpy().flatten()
    val_predictions_np = val_predictions.numpy().flatten()

    # Save validation targets and predictions as .npy files
    np.save(os.path.join(logs_dir, 'val_targets.npy'), val_targets_np)
    np.save(os.path.join(logs_dir, 'val_predictions.npy'), val_predictions_np)

    correlation_matrix = np.corrcoef(val_targets_np, val_predictions_np)
    correlation_coefficient = correlation_matrix[0, 1]
    print(f"Pearson correlation coefficient between predictions and actual values: {correlation_coefficient:.4f}")

    plt.figure(figsize=(8, 8))
    plt.scatter(val_targets_np, val_predictions_np, alpha=0.6, marker='o', label='Predictions')
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal (y=x)')
    plt.title('Predicted vs Actual Values (Validation Set)')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.xlim(0, 0.20)
    plt.ylim(0, 0.20)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(logs_dir, 'val_pred_vs_actual_scatter.png'))
    plt.show()

    first_conv1d = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv1d):
            first_conv1d = m
            break

    if first_conv1d is not None:
        kernels = first_conv1d.weight.data.cpu().numpy()
        num_kernels = kernels.shape[0]
        num_channels = kernels.shape[1]
        kernel_size = kernels.shape[2]
        plt.figure(figsize=(num_kernels * 2, 2 * num_channels))
        for i in range(num_kernels):
            for j in range(num_channels):
                plt.subplot(num_kernels, num_channels, i * num_channels + j + 1)
                plt.plot(kernels[i, j, :])
                plt.title(f'Kernel {i}, Channel {j}')
                plt.axis('tight')
                plt.grid(True)
        plt.suptitle('First Conv1d Layer Kernels')
        plt.tight_layout()
        plt.savefig(os.path.join(logs_dir, 'kernels.png'))
        plt.show()
    else:
        print("No Conv1d layer found in the model.")

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
            self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

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
            gradients = self.gradients
            activations = self.activations
            weights = gradients.mean(dim=2, keepdim=True)
            grad_cam_map = (weights * activations).sum(dim=1, keepdim=True)
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
        # Enable gradients for the target conv layer
        if hasattr(target_layer_1d, 'weight') and target_layer_1d.weight is not None:
            target_layer_1d.weight.requires_grad_(True)
        if hasattr(target_layer_1d, 'bias') and target_layer_1d.bias is not None:
            target_layer_1d.bias.requires_grad_(True)
        grad_cam_1d = GradCAM1d(model, target_layer_1d)
        for val_x, _ in val_dataloader:
            sample_input = val_x[0].unsqueeze(0).to(eval_device)
            break
        sample_input.requires_grad_(True)
        grad_cam_map = grad_cam_1d(sample_input)
        grad_cam_1d.remove_hooks()
        plt.figure(figsize=(10, 4))
        plt.plot(grad_cam_map)
        plt.title('Grad-CAM Map for Sample Input (Conv1d)')
        plt.xlabel('Time Axis')
        plt.ylabel('Grad-CAM Intensity')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(logs_dir, 'grad_cam_map.png'))
        plt.show()
    else:
        print("No Conv1d layer found for Grad-CAM.")

if __name__ == "__main__":
    main()