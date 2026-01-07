import polars as pl
from src import preprocess_and_predict, preprocess
from models import SimpleCNN, SimpleViTRegressor, ResidualCNN
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import math
import argparse
import torch.nn.functional as F
# Load configuration from YAML file
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

dataset_dir = '/mnt/sdb/yyamaguchi/psdata2matlab/simulation/dataset'
device = config["evaluation"]["device"]
png_save_dir = config['evaluation']['png_save_dir']

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


# x_train_path = os.path.join(dataset_dir,'x_train.npy')
# t_train_path = os.path.join(dataset_dir,'t_train.npy')

# file_path = os.path.join(dataset_dir,'x_train.npy')
# x_train = np.load(x_train_path)
# t_train = np.load(t_train_path)
# print(f'shape of x_train: {x_train.shape}')
# exp_date =""
# sample_index=73
# fs = 50e6
# eval_result = []
# x_test_tensor_cnn = torch.from_numpy(x_train).float()
# x_test_tensor_cnn = x_test_tensor_cnn.unsqueeze(1)


expdata_dir = "/mnt/sdb/yyamaguchi/psdata2matlab/experiments/processed/solid_liquid"
exp_date = "P20241011-1132"
file_path = os.path.join(expdata_dir, exp_date+"_processed.npz")
x_npz = np.load(file_path)
x_test = x_npz["processed_data"][:,:,0]
fs = x_npz["fs"]
sample_index = 1000
# x_train_cnn = preprocess(file_path, device="cuda:0",
#                          fs=fs,x_test=x_test)
x_train_cnn = preprocess(path=file_path,if_log1p=True,if_hilbert=True,
                         x_test=x_test, device="cuda:0",fs=fs,
                         if_drawsignal=True,png_save_dir=png_save_dir,
                         png_name="new",plot_idx=sample_index)
x_train_cnn = x_train_cnn.squeeze(1)
x_train = x_train_cnn.cpu().numpy()



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

    def __call__(self, input_tensor):
        self.model.eval()
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        self.model.zero_grad()
        out = self.model(input_tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        target = out.squeeze()
        if target.ndim > 0:
            target = target.sum()
        self.model.zero_grad()
        target.backward(retain_graph=True)
        gradients = self.gradients         # [B, C, L]
        activations = self.activations     # [B, C, L]
        weights = gradients.mean(dim=2, keepdim=True)  # [B, C, 1]
        grad_cam_map = (weights * activations).sum(dim=1, keepdim=True)  # (B,1,L)
        grad_cam_map = torch.relu(grad_cam_map)
        grad_cam_map = torch.nn.functional.interpolate(
            grad_cam_ma, size=input_tensor.shape[2], mode='linear', align_corners=False
        )
        grad_cam_map = grad_cam_map.squeeze().cpu().numpy()
        grad_cam_map = (grad_cam_map - grad_cam_map.min()) / (grad_cam_map.max() - grad_cam_map.min() + 1e-8)
        return grad_cam_map

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def preprocess_for_gradcam(x, sample_index=50, channel_index=0):
    """
    1サンプル+1chのみ抽出・正規化→[1,1,L] tensorに
    """
    # data = np.load(file_path)
    # if 'arr_0' in data:
    #     x = data['arr_0']p
    # else:
    #     arr_keys = list(data.keys())
    #     if not arr_keys:
    #         raise RuntimeError(f"No arrays found in {file_path}")
    #     x = data[arr_keys[0]]
    # if x.ndim != 3:
    #     raise RuntimeError(f"Expected 3D array, but got shape {x.shape}")
    N, W = x.shape
    if not (0 <= sample_index < N):
        raise RuntimeError(f"Invalid sample_index {sample_index}, N={N}")
    # if not (0 <= channel_index < C):
    #     raise RuntimeError(f"Invalid channel_index {channel_index}, C={C}")
    # signal = x[sample_index, :, channel_index]
    signal = x[sample_index, :]
    # mean = np.mean(signal)
    # std = np.std(signal)
    # std = std if std > 1e-8 else 1e-8
    # signal = (signal - mean) / std
    signal = torch.tensor(signal, dtype=torch.float32)
    signal = signal.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
    print(f'shape of signal {signal.shape}')
    return signal

# ----------- Grad-CAM 実行スクリプト -----------
# モデルのターゲット層を取得 (最後のConv1d)
target_layer = None
if hasattr(model, 'conv3'):
    target_layer = model.conv3
elif hasattr(model, 'layer3'):
    target_layer = model.layer3
else:
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv1d):
            target_layer = m
            break
if target_layer is None:
    raise RuntimeError("Could not find a suitable layer for Grad-CAM.")

gradcam_output_dir = os.path.join(base_dir, "gradcam_outputs")
input_tensor = preprocess_for_gradcam(
    x_train,
    sample_index=sample_index,
    channel_index=0
)

# model.eval()
# with torch.no_grad():
#     x_test_tensor_cnn = input_tensor.to(device)
#     print(f'shape of x_test_tensor_cnn: {x_test_tensor_cnn.shape}')
#     predictions = model(x_test_tensor_cnn)
#     predictions_numpy = predictions.cpu().numpy()
#     mean, var = torch.mean(predictions), torch.var(predictions)
#     print(f"mean: {mean}, var: {var}")
#     # Release memory after computation
#     del predictions
#     torch.cuda.empty_cache()

# print(f'Predicted Fraction: {mean}')

gradcam = GradCAM1d(model, target_layer)
grad_cam_map = gradcam(input_tensor)
gradcam.remove_hooks()

if not os.path.exists(gradcam_output_dir):
    os.makedirs(gradcam_output_dir)
plt.figure(figsize=(10, 4))
plt.rcParams['font.size'] = 16
n_samples = x_train[sample_index,:].shape[0]
t = np.arange(n_samples)
plt.plot(t, x_train[sample_index,:],
         color='blue',label='Processed Signal')
plt.plot(t,grad_cam_map, color='red', label='Saliency-Map')
plt.legend()
plt.fill_between(t, grad_cam_map,
         color='red',alpha=0.1)
plt.title('Saliency-Map for Sample Input (Conv1d)')
plt.xlabel('Time Axis')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(gradcam_output_dir, f'gradcam{exp_date}_{sample_index}.png'))
plt.close()

# model.eval()
# with torch.no_grad():
#     x_test_tensor_cnn = x_test_tensor_cnn.to(device)
#     print(f'shape of x_test_tensor_cnn: {x_test_tensor_cnn.shape}')
#     prediction = model(x_test_tensor_cnn)
#     prediction = prediction.cpu().numpy()
#     eval_result = np.array(prediction)
#     torch.cuda.empty_cache

# plt.figure(figsize=(8,8))
# plt.rcParams["font.size"] = 18
# plt.plot(t_train, eval_result,'o',color='blue',alpha=0.7,label='glass beads')
# plt.plot([-1,1],[-1,1],'r--',label='Ideal (y=x)')
# plt.xlabel("Ground Truth")
# plt.ylabel("Prediction")
# plt.xlim(-0.05,0.2)
# plt.ylim(-0.05,0.2)
# plt.grid(True)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.tight_layout()
# plt.legend()
# plt.savefig(os.path.join(base_dir, 'predicted_vs_ground_truth_sim2sim_new.png'))
