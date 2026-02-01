import os
import sys
from pathlib import Path

# プロジェクトルート（ml_airlift）をパスに追加
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src import preprocess_and_predict, preprocess, npz2png, prepare_cnn_input
from models import SimpleCNN, SimpleViTRegressor, ResidualCNN
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import argparse
from typing import List, Sequence, Tuple, Union
from tqdm import tqdm
# Load configuration from YAML file
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

base_dir = "/home/smatsubara/documents/airlift/data/sandbox/visualize"
file_path = "/home/smatsubara/documents/airlift/data/experiments/processed/solid_liquid/P20241007-1112_processed.npz"
device = config["evaluation"]["device"]

parser = argparse.ArgumentParser(description="Run evaluation with specified base directory (datetime).")
parser.add_argument('--datetime', type=str, required=True, help='Base directory for evaluation (e.g., /home/smatsubara/documents/airlift/data/outputs/2025-09-07/14-39-46)')
args = parser.parse_args()

# Load the trained model
base_dir_model = args.datetime
model_path = os.path.join(base_dir_model + '/weights/model.pth')
model = SimpleCNN(config['hyperparameters']['input_length']).to(config['evaluation']['device'])
#model = ResidualCNN(config['hyperparameters']['input_length']).to(config['evaluation']['device'])
#model = SimpleViTRegressor(config['hyperparameters']['input_length']).to(config['evaluation']['device'])
model.load_state_dict(
    torch.load(
        model_path,
        map_location=config['evaluation']['device'],
        weights_only=True,
    )
)
model.eval()


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

    def __call__(self, input_tensor, batch=False):
        """
        Compute Grad-CAM map(s) for the provided input tensor.

        Args:
            input_tensor (torch.Tensor): Tensor of shape [B, C, L] or [1, C, L].
            batch (bool): If True, compute Grad-CAM for each sample independently
                          and return a NumPy array of shape [B, L].
                          If False, return a NumPy array of shape [L].
        """
        if batch:
            cam_list = []
            for sample in input_tensor:
                cam = self._compute_single(sample.unsqueeze(0))
                cam_list.append(cam)
            return np.stack(cam_list, axis=0)

        return self._compute_single(input_tensor)

    def _compute_single(self, input_tensor):
        self.model.eval()
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.detach().to(device)
        input_tensor.requires_grad_(True)
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
            grad_cam_map, size=input_tensor.shape[2], mode='linear', align_corners=False
        )
        grad_cam_map = grad_cam_map.squeeze().cpu().numpy()
        grad_cam_map = (grad_cam_map - grad_cam_map.min()) / (grad_cam_map.max() - grad_cam_map.min() + 1e-8)
        return grad_cam_map

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def preprocess_for_gradcam(processed_data, sample_index=500, target_length=None):
    """
    前処理後の2次元配列から1サンプル分を抽出し、モデル入力形状[1, 1, L]に整形する。
    target_lengthに合わせてパディングまたはトリミングを行い、元の長さも返却する。
    """
    if isinstance(processed_data, torch.Tensor):
        signal_tensor = processed_data[sample_index]
        signal = signal_tensor.detach().cpu().float()
    else:
        signal = torch.tensor(processed_data[sample_index], dtype=torch.float32)
    
    # Store original length before padding/trimming
    original_length = signal.shape[0]
    
    # Adjust length to match model's expected input length if target_length is specified
    if target_length is not None:
        current_length = signal.shape[0]
        if current_length < target_length:
            # Pad with zeros if shorter
            padding = target_length - current_length
            signal = torch.nn.functional.pad(signal, (0, padding), mode='constant', value=0)
        elif current_length > target_length:
            # Trim if longer
            signal = signal[:target_length]
    
    signal = signal.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
    
    #print(f'shape of signal {signal.shape}')
    return signal, original_length


def preprocess_batch_for_gradcam(
    processed_data: Union[np.ndarray, torch.Tensor, Sequence[np.ndarray]],
    target_length: int = None,
    device: Union[str, torch.device, None] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Vectorised variant of `preprocess_for_gradcam`.

    Args:
        processed_data: Collection of 1-D signals (shape [N, L]) that have already
            passed through the standard preprocessing pipeline.
        target_length: Desired length for the network input. Longer signals are
            trimmed, shorter signals are zero padded. If None, the longest
            sequence length is used.
        device: Target torch device (e.g. "cuda:0" or torch.device("cpu")).

    Returns:
        batch_tensor: Tensor of shape [N, 1, target_length] located on `device`.
        original_lengths: List containing the original length of each sample
            before padding/trimming.
    """
    if isinstance(processed_data, torch.Tensor):
        samples = [
            processed_data[i].detach().cpu().numpy()
            for i in range(processed_data.shape[0])
        ]
    elif isinstance(processed_data, np.ndarray):
        if processed_data.ndim == 1:
            samples = [processed_data]
        elif processed_data.ndim >= 2:
            samples = [processed_data[i] for i in range(processed_data.shape[0])]
        else:
            raise ValueError("processed_data must be a 1-D or 2-D array.")
    elif isinstance(processed_data, Sequence):
        samples = [np.asarray(sample) for sample in processed_data]
    else:
        raise TypeError(
            "processed_data must be a numpy array, torch tensor, or sequence of arrays."
        )

    original_lengths = [int(sample.shape[-1]) for sample in samples]
    if target_length is None:
        target_length = max(original_lengths)

    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    print(f'device: {device}')
    batch_tensor = torch.zeros(
        (len(samples), 1, target_length), dtype=torch.float32, device=device
    )

    for idx, sample in enumerate(samples):
        sample_tensor = torch.as_tensor(sample, dtype=torch.float32, device=device)
        length = sample_tensor.shape[0]
        if length >= target_length:
            batch_tensor[idx, 0, :] = sample_tensor[:target_length]
        else:
            batch_tensor[idx, 0, :length] = sample_tensor

    return batch_tensor, original_lengths


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

# Load data for visualization
x_npz = np.load(file_path)
x_raw = x_npz["processed_data"][:, :, 0]
sample_index = 1000

# Apply preprocessing to match training pipeline (CPU only)
fs = None
if "fs" in x_npz:
    try:
        fs = x_npz["fs"].item() if hasattr(x_npz["fs"], "item") else float(x_npz["fs"])
    except Exception:
        fs = None

x_processed = preprocess(x_raw, fs=fs)

# CNN 入力テンソルへ変換（正規化・log1p・デバイス転送）
x_processed_tensor = prepare_cnn_input(x_processed, device)
x_processed_numpy = x_processed_tensor.squeeze(1).detach().cpu().numpy()

gradcam_output_dir = os.path.join(base_dir, "gradcam_outputs")
input_tensor, original_length = preprocess_for_gradcam(
    x_processed_numpy,
    sample_index=sample_index,
    target_length=config['hyperparameters']['input_length']
)

gradcam = GradCAM1d(model, target_layer)
grad_cam_map = gradcam(input_tensor)
gradcam.remove_hooks()

# Trim grad_cam_map to original length if it was padded
if len(grad_cam_map) > original_length:
    grad_cam_map = grad_cam_map[:original_length]

# すべての行に対して実行してStack
all_grad_cam_maps = []
all_lengths = []
# Speed up using GPU: Batch processing with torch tensors

all_grad_cam_maps = []
all_lengths = []

batch_size = 1024  # Adjust depending on your GPU memory
num_samples = x_processed_numpy.shape[0]
target_len = config['hyperparameters']['input_length']
device_str = config['evaluation']['device']

gradcam.model = gradcam.model.to(device_str)
gradcam.target_layer = gradcam.target_layer.to(device_str)

for start_idx in range(0, num_samples, batch_size):
    end_idx = min(start_idx + batch_size, num_samples)
    batch_array = x_processed_numpy[start_idx:end_idx]

    batch_tensor, orig_lens = preprocess_batch_for_gradcam(
        batch_array,
        target_length=target_len,
        device=device_str,
    )
    print(f'batch_tensor.shape: {batch_tensor.shape}')
    print(f'preprocess ended with device: {device_str}')
    # Run GradCAM on the whole batch (output: [B, L])
    gc_maps_batch = gradcam(batch_tensor, batch=True)

    for i, gc_map in enumerate(
        tqdm(gc_maps_batch, desc=f"Processing GradCAMs {start_idx}-{end_idx-1}")
    ):
        gc_map = np.asarray(gc_map)
        orig_len = orig_lens[i]
        if gc_map.shape[-1] > orig_len:
            gc_map = gc_map[:orig_len]
        all_grad_cam_maps.append(gc_map)
        all_lengths.append(orig_len)

max_length = max(all_lengths)
grad_cam_map_full = np.stack([
    np.pad(
        gc_map,
        (0, max_length - len(gc_map)),
        mode='constant',
    ) if len(gc_map) < max_length else gc_map
    for gc_map in all_grad_cam_maps
])
print(f'grad_cam_map_full.shape: {grad_cam_map_full.shape}')
global_max = grad_cam_map_full.max()
scale_denominator = global_max if global_max > 1e-8 else 1.0
grad_cam_map_full_scaled = grad_cam_map_full / scale_denominator
print(f"x_raw.shape: {x_raw.shape}")
# Align x_raw to match Grad-CAM width and normalise for overlay
if x_raw.shape[1] >= max_length:
    x_raw_aligned = x_raw[:, :max_length]
else:
    pad_width = max_length - x_raw.shape[1]
    x_raw_aligned = np.pad(x_raw, ((0, 0), (0, pad_width)), mode='constant')

x_raw_min = np.min(x_raw_aligned)
x_raw_max = np.max(x_raw_aligned)
if x_raw_max - x_raw_min < 1e-8:
    x_raw_norm = np.zeros_like(x_raw_aligned)
else:
    x_raw_norm = (x_raw_aligned - x_raw_min) / (x_raw_max - x_raw_min)

# Save grad_cam_map_full as an image (each row as one sample)
plt.figure(figsize=(12, 8))
im_full = plt.imshow(
    grad_cam_map_full_scaled,
    aspect='auto',
    interpolation='nearest',
    cmap='viridis',
    vmin=0.0,
    vmax=1.0,
)
clb_full = plt.colorbar(im_full)
clb_full.set_label('Grad-CAM Saliency (scaled)', rotation=270, labelpad=15)
plt.title('Full Grad-CAM map for all samples')
plt.xlabel('Time Axis')
plt.ylabel('Sample Index')
plt.tight_layout()
plt.savefig(os.path.join(gradcam_output_dir, 'gradcam_full_fast2.png'))
plt.close()

# Overlay Grad-CAM heatmap and the normalised raw signal
plt.figure(figsize=(12, 8))
im_overlay = plt.imshow(
    grad_cam_map_full_scaled,
    aspect='auto',
    interpolation='nearest',
    cmap='viridis',
    alpha=0.7,
    vmin=0.0,
    vmax=0.001,
)
plt.imshow(
    x_raw_norm,
    aspect='auto',
    interpolation='nearest',
    cmap='Greys',
    alpha=0.25,
)
clb_overlay = plt.colorbar(im_overlay)
clb_overlay.set_label('Grad-CAM Saliency (scaled)', rotation=270, labelpad=15)
plt.title('Grad-CAM Overlay with Raw Signal')
plt.xlabel('Time Axis')
plt.ylabel('Sample Index')
plt.tight_layout()
plt.savefig(os.path.join(gradcam_output_dir, 'gradcam_signal_overlay.png'))
plt.close()

if not os.path.exists(gradcam_output_dir):
    os.makedirs(gradcam_output_dir)
plt.figure(figsize=(10, 4))
plt.rcParams['font.size'] = 16
processed_signal = x_processed_numpy[sample_index, :]
n_samples = processed_signal.shape[0]
t = np.arange(n_samples)
plt.plot(t, processed_signal,
         color='blue', label='Processed Signal')
plt.plot(t, grad_cam_map, color='red', label='Saliency-Map')
plt.legend()
plt.fill_between(t, grad_cam_map,
         color='red', alpha=0.1)
plt.title('Saliency-Map for Sample Input (Conv1d)')
plt.xlabel('Time Axis')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(gradcam_output_dir, f'gradcam_{sample_index}.png'))
plt.close()

mean, var = preprocess_and_predict(file_path, model, device=config['evaluation']['device'])
npz2png(file_path, gradcam_output_dir, vmin=0, vmax=0.05, max_pulses=30)