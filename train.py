import torch
import numpy as np
import yaml
from models import SimpleCNN, SimpleViTRegressor
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader,random_split
import matplotlib.pyplot as plt
import os


import hydra
from omegaconf import OmegaConf

@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg):

    # Get hydra run directory as base path
    base_dir = os.getcwd()

    # Load config (already loaded as cfg)
    x_train = np.load(os.path.relpath(cfg.dataset.x_train, base_dir))
    t_train = np.load(os.path.relpath(cfg.dataset.t_train, base_dir))

    x_train_tensor = torch.from_numpy(x_train).float()
    t_train_tensor = torch.from_numpy(t_train).float()
    if t_train_tensor.ndim == 1:
        t_train_tensor = t_train_tensor.unsqueeze(-1)
    if x_train_tensor.ndim == 1:
        x_train_tensor = x_train_tensor.unsqueeze(-1)
    print(torch.max(x_train_tensor))
    print(x_train_tensor.shape)
    print(t_train_tensor.shape)

    def plot_x_train(x_train, index):
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
        plot_x_train(x_train, i)

    torch.manual_seed(cfg.hyperparameters.seed)
    device1 = torch.device(cfg.training.device)

    t_train_tensor = torch.from_numpy(t_train).float().to(device1)
    x_train_tensor = torch.from_numpy(x_train).float().to(device1)
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
    #model = SimpleCNN(size).to(device1)
    model = SimpleViTRegressor(size).to(device1)

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

    weights_dir = os.path.join(base_dir, "weights")
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

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

    model.load_state_dict(torch.load(os.path.join(weights_dir, 'model.pth')))
    model = model.to(cfg.evaluation.device)
    model.eval()
    val_predictions = []
    val_targets = []
    with torch.no_grad():
        for val_x, val_y in val_dataloader:
            val_x = val_x.to(cfg.evaluation.device)
            val_y = val_y.to(cfg.evaluation.device)
            outputs = model(val_x)
            val_predictions.append(outputs.cpu())
            val_targets.append(val_y.cpu())
    val_predictions = torch.cat(val_predictions, dim=0)
    val_targets = torch.cat(val_targets, dim=0)
    print("Validation predictions shape:", val_predictions.shape)
    print("Validation targets shape:", val_targets.shape)
    print(val_predictions)

    val_targets_np = val_targets.numpy().flatten()
    val_predictions_np = val_predictions.numpy().flatten()

    correlation_matrix = np.corrcoef(val_targets_np, val_predictions_np)
    correlation_coefficient = correlation_matrix[0, 1]
    print(f"Pearson correlation coefficient between predictions and actual values: {correlation_coefficient:.4f}")

    plt.figure(figsize=(8, 8))
    plt.scatter(val_targets_np, val_predictions_np, alpha=0.6, marker='o', label='Predictions')
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal (y=x)')
    plt.title('Predicted vs Actual Values (Validation Set)')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.xlim(0, 0.15)
    plt.ylim(0, 0.15)
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
        plt.show()
        plt.savefig(os.path.join(logs_dir, 'kernels.png'))
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
        grad_cam_1d = GradCAM1d(model, target_layer_1d)
        for val_x, _ in val_dataloader:
            sample_input = val_x[0].unsqueeze(0).to(cfg.evaluation.device)
            break
        grad_cam_map = grad_cam_1d(sample_input)
        grad_cam_1d.remove_hooks()
        plt.figure(figsize=(10, 4))
        plt.plot(grad_cam_map)
        plt.title('Grad-CAM Map for Sample Input (Conv1d)')
        plt.xlabel('Time Axis')
        plt.ylabel('Grad-CAM Intensity')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(logs_dir, 'grad_cam_map.png'))
    else:
        print("No Conv1d layer found for Grad-CAM.")

if __name__ == "__main__":
    main()


