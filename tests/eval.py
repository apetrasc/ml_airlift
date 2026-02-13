import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
from src import preprocess_and_predict, preprocess, debug_pipeline, get_valid_data, prepare_cnn_input
from models import SimpleCNN, SimpleViTRegressor, ResidualCNN, BaseCNN, ProposedCNN
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import math
import argparse
from typing import List, Sequence, Tuple, Union
from matplotlib.gridspec import GridSpec

# Parse args first (base_dir needed for paths)
parser = argparse.ArgumentParser(description="Run evaluation with specified base directory (datetime).")
parser.add_argument('--datetime', type=str, required=True, help='Base directory for evaluation (e.g., /home/smatsubara/documents/airlift/data/outputs/2025-09-07/14-39-46)')
parser.add_argument('--skip_gradcam', action='store_true', help='Skip Grad-CAM computation (saves time)')
args = parser.parse_args()

# Load configuration from YAML file
# Note: OmegaConf.resolve() is not used - config has Hydra ${now} which OmegaConf doesn't support.
config_path = 'config/config.yaml'
config_raw = OmegaConf.load(config_path)
config = {
    'base': OmegaConf.to_container(config_raw.get('base', {}), resolve=True),
    'evaluation': OmegaConf.to_container(config_raw['eval']['sim'], resolve=True),
    'hyperparameters': OmegaConf.to_container(config_raw['train']['sim']['hyperparameters'], resolve=True),
}

# =============================================================================
# PATHS (eval.py で用いられるパスをすべてここに集約)
# =============================================================================
base_dir = args.datetime  # 評価対象の run ディレクトリ (--datetime で指定)
base_dir_assets = config['base']['base_dir']
target_variables_path = os.path.join(base_dir_assets, "datasets", "experiments", "target_variables.csv")
processed_dir = os.path.join(base_dir_assets, "datasets", "experiments", "processed", "all")
log_path = os.path.join(base_dir, "logs", "eval_log.txt")
model_path = os.path.join(base_dir, "weights", "model.pth")
save_path = os.path.join(base_dir, "predicted.csv")
gradcam_dir = os.path.join(base_dir, "logs", "gradcam")
path_predicted_vs_ground_truth = os.path.join(base_dir, "predicted_vs_ground_truth.png")
path_predicted_vs_ground_truth_noerrorbars = os.path.join(base_dir, "predicted_vs_ground_truth_noerrorbars.png")
path_predicted_vs_truth_processed = os.path.join(base_dir, "predicted_vs_truth_processed.png")
path_predicted_vs_ground_truth_processed_noerrorbars = os.path.join(base_dir, "predicted_vs_ground_truth_processed_noerrorbars.png")
# =============================================================================

# Read the target variables CSV file (experiment metadata)
target_variables = pl.read_csv(
    target_variables_path,
    encoding="SHIFT_JIS"
)

# Load the trained model
model = SimpleCNN(config['hyperparameters']['input_length']).to(config['evaluation']['device'])

# --- Log all stdout to {log_path} ---
class TeeLogger:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_path, stream):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._stream = stream
        self._file = open(log_path, "w")
    def write(self, msg):
        self._stream.write(msg)
        self._file.write(msg)
        self._file.flush()
    def flush(self):
        self._stream.flush()
        self._file.flush()
    def close(self):
        self._file.close()

_tee = TeeLogger(log_path, sys.stdout)
sys.stdout = _tee
print("[LOG] Logging all output to: %s" % log_path)
# -------------------------------------------------------

if not os.path.exists(model_path):
    print(f"[ERROR] Model file not found: {model_path}")
    sys.exit(1)
try:
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=config['evaluation']['device'],
            weights_only=True,
        )
    )
    model.eval()
    print(f"[OK] Model loaded successfully from {model_path}")
except Exception as e:
    print(f"[ERROR] Failed to load model from {model_path}: {e}")
    sys.exit(1)

# Grad-CAM 用: 推論時シフト（先頭50ゼロ、末尾50削除）
PAD_LEFT_DIM = 50


def apply_inference_shift(tensor, input_length):
    """推論用シフト: 先頭 PAD_LEFT_DIM 個の0を付け、末尾 PAD_LEFT_DIM 個を削除して返す。"""
    pad_left = PAD_LEFT_DIM
    keep_len = input_length - pad_left
    if tensor.dim() == 3:
        zeros_front = torch.zeros(
            tensor.shape[0], tensor.shape[1], pad_left,
            device=tensor.device, dtype=tensor.dtype,
        )
        return torch.cat([zeros_front, tensor[:, :, :keep_len]], dim=2)
    elif tensor.dim() == 2:
        zeros_front = torch.zeros(
            tensor.shape[0], pad_left,
            device=tensor.device, dtype=tensor.dtype,
        )
        return torch.cat([zeros_front, tensor[:, :keep_len]], dim=1)
    return tensor


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
        gradients = self.gradients
        activations = self.activations
        weights = gradients.mean(dim=2, keepdim=True)
        grad_cam_map = (weights * activations).sum(dim=1, keepdim=True)
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


def preprocess_batch_for_gradcam(
    processed_data: Union[np.ndarray, torch.Tensor, Sequence[np.ndarray]],
    target_length: int,
    device: Union[str, torch.device],
) -> Tuple[torch.Tensor, List[int]]:
    """前処理済みデータを target_length に揃え、推論シフトを適用したバッチテンソルを返す。"""
    if isinstance(processed_data, torch.Tensor):
        samples = [processed_data[i].detach().cpu().numpy() for i in range(processed_data.shape[0])]
    elif isinstance(processed_data, np.ndarray):
        samples = [processed_data[i] for i in range(processed_data.shape[0])] if processed_data.ndim >= 2 else [processed_data]
    else:
        samples = [np.asarray(s) for s in processed_data]
    original_lengths = [int(s.shape[-1]) for s in samples]
    if isinstance(device, str):
        device = torch.device(device)
    batch_tensor = torch.zeros((len(samples), 1, target_length), dtype=torch.float32, device=device)
    for idx, sample in enumerate(samples):
        st = torch.as_tensor(sample, dtype=torch.float32, device=device)
        L = st.shape[0]
        if L >= target_length:
            batch_tensor[idx, 0, :] = st[:target_length]
        else:
            batch_tensor[idx, 0, :L] = st
    if batch_tensor.shape[-1] == target_length:
        batch_tensor = apply_inference_shift(batch_tensor, target_length)
    return batch_tensor, original_lengths


def get_gradcam_target_layer(model):
    """Grad-CAM 用のターゲット層（最後の Conv1d）を返す。"""
    if hasattr(model, 'conv3'):
        return model.conv3
    if hasattr(model, 'layer3'):
        return model.layer3
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv1d):
            return m
    raise RuntimeError("Could not find a suitable layer for Grad-CAM.")


def save_gradcam_three_panel(file_path: str, model, target_layer, device: str, config: dict, save_path: str) -> None:
    """
    指定 .npz を読み込み、Grad-CAM を計算し、Original / Saliency / Overlay の3パネル図を保存する。
    """
    data = np.load(file_path)
    x_raw = data["processed_data"][:, :, 0]
    fs = None
    if "fs" in data:
        try:
            fs = data["fs"].item() if hasattr(data["fs"], "item") else float(data["fs"])
        except Exception:
            fs = None
    x_processed = preprocess(x_raw, fs=fs)
    x_processed_tensor = prepare_cnn_input(x_processed, device)
    x_processed_numpy = x_processed_tensor.squeeze(1).detach().cpu().numpy()

    target_len = config["hyperparameters"]["input_length"]
    gradcam = GradCAM1d(model, target_layer)
    gradcam.model = gradcam.model.to(device)
    gradcam.target_layer = gradcam.target_layer.to(device)

    batch_size = 1024
    num_samples = x_processed_numpy.shape[0]
    all_grad_cam_maps = []
    all_lengths = []
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_array = x_processed_numpy[start_idx:end_idx]
        batch_tensor, orig_lens = preprocess_batch_for_gradcam(batch_array, target_len, device)
        gc_maps_batch = gradcam(batch_tensor, batch=True)
        for i, gc_map in enumerate(gc_maps_batch):
            gc_map = np.asarray(gc_map)
            orig_len = orig_lens[i]
            if gc_map.shape[-1] > orig_len:
                gc_map = gc_map[:orig_len]
            all_grad_cam_maps.append(gc_map)
            all_lengths.append(orig_len)
    gradcam.remove_hooks()

    max_length = max(all_lengths)
    grad_cam_map_full = np.stack([
        np.pad(gc_map, (0, max_length - len(gc_map)), mode='constant') if len(gc_map) < max_length else gc_map
        for gc_map in all_grad_cam_maps
    ])
    global_max = grad_cam_map_full.max()
    scale_denom = global_max if global_max > 1e-8 else 1.0
    grad_cam_map_full_scaled = grad_cam_map_full / scale_denom

    pad_left = PAD_LEFT_DIM
    keep_len = target_len - pad_left
    x_processed_shifted = np.zeros((x_processed_numpy.shape[0], target_len), dtype=x_processed_numpy.dtype)
    for i in range(x_processed_numpy.shape[0]):
        row = x_processed_numpy[i]
        row_2500 = row[:target_len] if len(row) >= target_len else np.pad(row, (0, target_len - len(row)), mode='constant', constant_values=0)
        x_processed_shifted[i, pad_left:target_len] = row_2500[:keep_len]
    if max_length <= target_len:
        x_processed_aligned = x_processed_shifted[:, :max_length].copy()
    else:
        x_processed_aligned = np.pad(
            x_processed_shifted, ((0, 0), (0, max_length - target_len)), mode='constant', constant_values=0
        )

    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1], wspace=0.6)
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(x_processed_aligned, aspect='auto', interpolation='nearest', cmap='jet', vmin=0.0, vmax=0.10)
    ax0.set_title('Original', fontsize=20)
    ax0.set_xlabel('Time Axis', fontsize=18)
    ax0.set_ylabel('Sample Index', fontsize=18)
    ax0.tick_params(axis='both', labelsize=16)
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label='Intensity')

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(grad_cam_map_full_scaled, aspect='auto', interpolation='nearest', cmap='jet', vmin=0, vmax=1)
    ax1.set_title('Saliency', fontsize=20)
    ax1.set_xlabel('Time Axis', fontsize=18)
    ax1.set_ylabel('Sample Index', fontsize=18)
    ax1.tick_params(axis='both', labelsize=16)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Saliency')

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(x_processed_aligned, aspect='auto', interpolation='nearest', cmap='jet', alpha=1.0, vmin=0, vmax=0.10)
    im2 = ax2.imshow(grad_cam_map_full_scaled, aspect='auto', interpolation='nearest', cmap='jet', alpha=0.5, vmin=0, vmax=1)
    ax2.set_title('Overlay', fontsize=20)
    ax2.set_xlabel('Time Axis', fontsize=18)
    ax2.set_ylabel('Sample Index', fontsize=18)
    ax2.tick_params(axis='both', labelsize=16)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Saliency')

    plt.tight_layout(pad=1.2)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


target_layer = get_gradcam_target_layer(model)
print(target_variables.head())


# If "IDXX" column exists, drop the "NAME" column (to avoid duplication)
if "IDXX" in target_variables.columns:
    target_variables = target_variables.drop("NAME")

# Create a new "NAME" column based on the first and second columns (date and time)
date_col = target_variables.columns[0]
time_col = target_variables.columns[1]
target_variables = target_variables.with_columns(
    (pl.lit("P") + pl.col(date_col).cast(pl.Utf8) + "-" + pl.col(time_col).cast(pl.Utf8)).alias("NAME")
)

# Prepare the list of columns to display (avoid duplicates)
cols_to_show = [col for col in target_variables.columns[2:] if col != "NAME"] + ["NAME"]

# Add a new column "FullPath" that contains the full path to the corresponding processed .npz file
target_variables = target_variables.with_columns(
    (pl.lit(processed_dir + "/") + pl.col("NAME") + pl.lit("_processed.npz")).alias("FullPath")
)
# target_variables = target_variables.with_columns(
#     (pl.lit(processed_dir_stone + "/") + pl.col("NAME") + pl.lit("_processed.npz")).alias("FullPathStone")
# )

# For each row, load the processed .npz file, run inference, and add mean and variance as new columns
mean_list = []
var_list = []
mean_list_stone = []
var_list_stone = []
for row in target_variables.iter_rows(named=True):
    file_path = row["FullPath"]
    #file_path_stone = row["FullPathStone"]
    # Debug: Check if the file path is exactly as expected
    
    
    if os.path.exists(file_path):
        if file_path == os.path.join(processed_dir, "P20241011-1015_processed.npz"):
            debug_pipeline(base_dir, config, file_path)
        if file_path == os.path.join(processed_dir, "P20240726-1600_processed.npz"):
            debug_pipeline(base_dir, config, file_path)
        try:
            mean, var = preprocess_and_predict(file_path, model, device=config['evaluation']['device'])
            mean_list.append(mean)
            var_list.append(var)
            # mean_list_stone.append(mean_stone)
            # var_list_stone.append(var_stone)
        except Exception as e:
            # Even if the file exists, an error occurred during preprocess_and_predict.
            # The most common reasons are:
            # - The file format is not as expected (e.g., missing keys, wrong structure)
            # - The file is corrupted or empty
            # - preprocess_and_predict expects certain data inside the .npz file, but it is not present
            # - The model or preprocessing code is not compatible with the data
            # - There is a bug in preprocess_and_predict or the model
            # To debug, print the exception:
            print("ERROR: Exception occurred while processing:", file_path)
            print("Exception message:", str(e))
            mean_list.append(None)
            var_list.append(None)
            # mean_list_stone.append(None)
            # var_list_stone.append(None)
    else:
        mean_list.append(None)
        var_list.append(None)
        # mean_list_stone.append(None)
        # var_list_stone.append(None)

# Add the mean and variance as new columns (float or None only)
target_variables = target_variables.with_columns([
    pl.Series("mean", mean_list, dtype=pl.Float64),
    pl.Series("var", var_list, dtype=pl.Float64),
    # pl.Series("mean_stone", mean_list_stone, dtype=pl.Float64),
    # pl.Series("var_stone", var_list_stone, dtype=pl.Float64)
])

# Adjust the column order so that "mean" and "var" come right after "FullPath"
#cols_to_show = [col for col in target_variables.columns[2:] if col not in ["NAME", "FullPath", "mean", "var", "mean_stone", "var_stone"]] + ["NAME", "FullPath", "mean", "var", "mean_stone", "var_stone"]
cols_to_show = [col for col in target_variables.columns[2:] if col not in ["NAME", "FullPath", "mean", "var"]] + ["NAME", "FullPath", "mean", "var"]

# Save the results to a CSV file
target_variables.select(cols_to_show).write_csv(save_path)
print(f"Prediction results saved to {save_path}.")

# 各入力に対して Original / Saliency / Overlay の3パネル図を logs/gradcam に保存
if args.skip_gradcam:
    print("[INFO] Skipping Grad-CAM computation (--skip_gradcam flag set)")
else:
    os.makedirs(gradcam_dir, exist_ok=True)
    device_str = config['evaluation']['device']
    for row in target_variables.iter_rows(named=True):
        file_path = row["FullPath"]
        if not os.path.exists(file_path):
            continue
        basename = os.path.basename(file_path)
        out_name = basename.replace(".npz", "_gradcam.png")
        out_path = os.path.join(gradcam_dir, out_name)
        try:
            save_gradcam_three_panel(file_path, model, target_layer, device_str, config, out_path)
            print(f"Grad-CAM 3-panel saved: {out_path}")
        except Exception as e:
            print(f"ERROR: Grad-CAM failed for {file_path}: {e}")

# Read the CSV file with UTF-8 (with BOM) encoding to prevent character corruption
target_variables = pl.read_csv(
    save_path,
    # encoding="SHIFT_JIS"
)
print(target_variables.head())


glass_diameter_col = target_variables["ガラス球直径"]

is_str = np.array([1 if isinstance(v, str) and not v.replace('.', '', 1).isdigit() else 0 for v in glass_diameter_col.to_numpy()])
print(is_str.shape)
x = target_variables["固相体積率"].to_numpy()
# mean に null が含まれる場合（推論失敗）は np.nan として扱い、get_valid_data で除外
y = target_variables["mean"].cast(pl.Float64, strict=False).to_numpy()
# y_stone = target_variables["mean_stone"].to_numpy()
y_stone = y[is_str == 1]


# var に null が含まれる場合（ファイル未存在・推論失敗）は 0 として扱う
# CSV 読み込みで "null" 文字列になる場合もあるため cast で数値化
var_col = target_variables["var"].cast(pl.Float64, strict=False).fill_null(0.0)
var_arr = var_col.to_numpy()
var_arr = np.where(np.isnan(var_arr), 0.0, np.maximum(var_arr, 0.0))
yerr = np.sqrt(var_arr)
yerr_stone = yerr[is_str == 1]


x_valid, y_valid, yerr_valid = get_valid_data(x, y, yerr)
# x_valid_stone, y_valid_stone, yerr_valid_stone = get_valid_data(x, y_stone, yerr_stone)

print(y_valid)

if len(x_valid) == 0:
    print("[WARN] No valid prediction data (all files missing or inference failed). Skipping evaluation plots and metrics.")
else:
    def calibration(x, y, yerr):
        """
        Calibrate the predicted values using the ground truth values.
        """
        bias = np.min(y) * np.ones(len(y))
        c=(1/3*math.pi+math.sqrt(3)/2)/math.pi
        y_processed = y - bias
        return y_processed

    y_valid_calibrated = calibration(x_valid, y_valid, yerr_valid)
    # y_valid_stone_calibrated = calibration(x_valid_stone, y_valid_stone, yerr_valid_stone)

    if len(x_valid) > 1:
        corr_coef = np.corrcoef(x_valid, y_valid)[0, 1]
        print(f"Correlation coefficient between x and y: {corr_coef:.4f}")
    else:
        print("Not enough valid data to calculate correlation coefficient.")

    x_valid_stone, y_valid_stone_plot, yerr_valid_stone_plot = get_valid_data(x[is_str == 1], y_stone, yerr_stone)

    def calculate_rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    def calculate_mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    def RelativeError(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred) / (y_true + 1e-7))

    rmse = calculate_rmse(x_valid, y_valid)
    mae = calculate_mae(x_valid, y_valid)
    relative_error = RelativeError(x_valid, y_valid)
    relative_error_stone = RelativeError(x_valid_stone, y_valid_stone_plot)
    print(f"RMSE between ground truth and prediction: {rmse:.6f}")
    print(f"MAE between ground truth and prediction: {mae:.6f}")
    print(f"Relative Error between ground truth and prediction: {relative_error:.6f}")
    print(f"Relative Error between ground truth and prediction (stone): {relative_error_stone:.6f}")
    plt.figure(figsize=(8, 8))
    plt.errorbar(
        x_valid, y_valid, yerr=yerr_valid, fmt='o', color='blue', alpha=0.7, ecolor='red', capsize=3, label='All'
    )
    x_valid_stone, y_valid_stone_plot, yerr_valid_stone_plot = get_valid_data(x[is_str == 1], y_stone, yerr_stone)
    plt.errorbar(
        x_valid_stone, y_valid_stone_plot, yerr=yerr_valid_stone_plot, fmt='o', color='orange', alpha=0.7, ecolor='green', capsize=3, label='Stone'
    )
    plt.legend(fontsize=24,loc='upper left')
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal (y=x)')
    plt.xlabel("Ground Truth(Tube Closing)", fontsize=24)
    plt.ylabel("Prediction(machine learning)", fontsize=24)
    plt.xlim(0, 0.2)
    plt.ylim(0, 0.2)
    plt.title("Prediction vs. Ground Truth", fontsize=20)
    plt.tick_params(axis='x', which='major', labelsize=22,length=15, direction='in', top=False, right=False)
    plt.tick_params(axis='y', which='major', labelsize=22,length=15, direction='in', top=False, right=False)
    # Set y-axis ticks to 0.05 intervals
    plt.yticks(np.arange(0, 0.21, 0.05))
    #plt.tick_params(axis='both', which='minor', labelsize=22,length=10, direction='in', top=True, right=True)
    plt.minorticks_on()
    # Add grid lines for both major and minor ticks
    plt.grid(True, which='major', alpha=0.7, linewidth=1.0)
    plt.grid(True, which='minor', alpha=0.3, linewidth=0.5)
    # 余白を自動調整してラベルのはみ出しを防ぐ
    plt.tight_layout()
    plt.savefig(path_predicted_vs_ground_truth, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 8))
    #plt.errorbar(x_valid, y_valid, yerr=yerr_valid, fmt='o', color='blue', alpha=0.7, ecolor='red', capsize=3, label='All')
    plt.plot(x_valid, y_valid, 'o', color='blue', alpha=0.7, label='glass ball')
    #plt.errorbar(x_valid_stone, y_valid_stone_plot, yerr=yerr_valid_stone_plot, fmt='o', color='orange', alpha=0.7, ecolor='green', capsize=3, label='Stone')
    plt.plot(x_valid_stone, y_valid_stone_plot, 'o', color='orange', alpha=0.7, label='Stone')
    plt.legend(fontsize=24,loc='upper left')
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal (y=x)')
    plt.xlabel("Ground Truth(Tube Closing)", fontsize=24)
    plt.ylabel("Prediction(machine learning)", fontsize=24)
    plt.xlim(0, 0.2)
    plt.ylim(0, 0.2)
    plt.title("Prediction vs. Ground Truth", fontsize=20)
    plt.tick_params(axis='x', which='major', labelsize=22, length=15, direction='in', top=False, right=False)
    plt.tick_params(axis='y', which='major', labelsize=22, length=15, direction='in', top=False, right=False)
    # Set y-axis ticks to 0.05 intervals
    plt.yticks(np.arange(0, 0.21, 0.05))
    #plt.tick_params(axis='both', which='minor', labelsize=18, length=8, direction='in', top=True, right=True)
    plt.minorticks_on()
    # Add grid lines for both major and minor ticks
    plt.grid(True, which='major', alpha=0.7, linewidth=1.0)
    plt.grid(True, which='minor', alpha=0.3, linewidth=0.5)
    # 余白を自動調整してラベルのはみ出しを防ぐ
    plt.tight_layout()
    plt.savefig(path_predicted_vs_ground_truth_noerrorbars, bbox_inches='tight')

    # Optionally, display the results
    # print(target_variables.select(cols_to_show))
    plt.figure(figsize=(8, 8))
    # y_valid_calibrated（全体）とy_valid_stone_calibrated（石あり）を色分けして表示
    plt.errorbar(x_valid, y_valid_calibrated, yerr=yerr_valid, fmt='o', color='blue', alpha=0.7, ecolor='red', capsize=3, label='glass ball (calibrated)')
    # y_valid_stone_calibrated用のx, y, yerrを抽出（is_str==1のインデックスを利用）
    x_valid_stone, y_valid_stone_plot, yerr_valid_stone_plot = get_valid_data(x[is_str == 1], y_stone, yerr_stone)
    y_valid_stone_calibrated = calibration(x_valid_stone, y_valid_stone_plot, yerr_valid_stone_plot)
    plt.errorbar(x_valid_stone, y_valid_stone_calibrated, yerr=yerr_valid_stone_plot, fmt='o', color='orange', alpha=0.7, ecolor='green', capsize=3, label='Stone (calibrated)')
    plt.legend(fontsize=24,loc='upper left')
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal (y=x)')
    plt.xlabel("Ground Truth(Tube Closing)", fontsize=24)
    plt.ylabel("Prediction(machine learning)", fontsize=24)
    plt.xlim(-0, 0.2)
    plt.ylim(-0, 0.2)
    plt.title("Prediction vs. Truth (Processed)", fontsize=20)
    plt.tick_params(axis='x', which='major', labelsize=22,length=15, direction='in', top=False, right=False)
    plt.tick_params(axis='y', which='major', labelsize=22,length=15, direction='in', top=False, right=False)
    # Set y-axis ticks to 0.05 intervals
    plt.yticks(np.arange(0, 0.21, 0.05))
    #plt.tick_params(axis='both', which='minor', labelsize=22,length=10, direction='in', top=True, right=True)
    plt.minorticks_on()
    # Add grid lines for both major and minor ticks
    plt.grid(True, which='major', alpha=0.7, linewidth=1.0)
    plt.grid(True, which='minor', alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(path_predicted_vs_truth_processed, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(x_valid, y_valid_calibrated, 'o', color='blue', alpha=0.7, label='glass ball (calibrated)')
    plt.plot(x_valid_stone, y_valid_stone_calibrated, 'o', color='orange', alpha=0.7, label='Stone (calibrated)')
    plt.legend(fontsize=24,loc='upper left')
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal (y=x)')
    plt.xlabel("Ground Truth(Tube Closing)", fontsize=24)
    plt.ylabel("Prediction(machine learning)", fontsize=24)
    plt.xlim(0, 0.2)
    plt.ylim(0, 0.2)
    plt.title("Prediction vs. Ground Truth (Processed)", fontsize=20)
    plt.tick_params(axis='x', which='major', labelsize=22,length=15, direction='in', top=False, right=False)
    plt.tick_params(axis='y', which='major', labelsize=22,length=15, direction='in', top=False, right=False)
    # Set y-axis ticks to 0.05 intervals
    plt.yticks(np.arange(0, 0.21, 0.05))
    #plt.tick_params(axis='both', which='minor', labelsize=22,length=10, direction='in', top=True, right=True)
    plt.minorticks_on()
    # Add grid lines for both major and minor ticks
    plt.grid(True, which='major', alpha=0.7, linewidth=1.0)
    plt.grid(True, which='minor', alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(path_predicted_vs_ground_truth_processed_noerrorbars, bbox_inches='tight')
    plt.close()
    rmse_calibrated = calculate_rmse(x_valid, y_valid_calibrated)
    mae_calibrated = calculate_mae(x_valid, y_valid_calibrated)
    relative_error_calibrated = RelativeError(x_valid, y_valid_calibrated)
    relative_error_stone_calibrated = RelativeError(x_valid_stone, y_valid_stone_calibrated)
    print(f"RMSE between ground truth and prediction (calibrated): {rmse_calibrated:.6f}")
    print(f"MAE between ground truth and prediction (calibrated): {mae_calibrated:.6f}")
    print(f"Relative Error between ground truth and prediction (calibrated): {relative_error_calibrated:.6f}")
    print(f"Relative Error between ground truth and prediction (calibrated) (stone): {relative_error_stone_calibrated:.6f}")
    print("saved all figures to ", base_dir)

    # --- Range-based evaluation (0.00 - 0.10) from predicted.csv ---
    print("\n" + "=" * 60)
    print("RANGE-BASED EVALUATION: ground truth in [0.00, 0.10]")
    print("=" * 60)
    range_mask = (x_valid >= 0.00) & (x_valid <= 0.10)
    n_range = int(range_mask.sum())
    if n_range > 0:
        x_r, y_r = x_valid[range_mask], y_valid[range_mask]
        rmse_range = calculate_rmse(x_r, y_r)
        mae_range = calculate_mae(x_r, y_r)
        bias_range = float(np.mean(y_r - x_r))
        print(f"  Samples : {n_range}")
        print(f"  RMSE    : {rmse_range:.6f}")
        print(f"  MAE     : {mae_range:.6f}")
        print(f"  Bias    : {bias_range:+.6f}")
        # calibrated
        y_r_cal = y_valid_calibrated[range_mask]
        rmse_range_cal = calculate_rmse(x_r, y_r_cal)
        mae_range_cal = calculate_mae(x_r, y_r_cal)
        bias_range_cal = float(np.mean(y_r_cal - x_r))
        print(f"  RMSE (calibrated) : {rmse_range_cal:.6f}")
        print(f"  MAE  (calibrated) : {mae_range_cal:.6f}")
        print(f"  Bias (calibrated) : {bias_range_cal:+.6f}")
    else:
        print("  No samples in range [0.00, 0.10]")
    print("=" * 60)

# Close the tee logger and restore stdout
print("[LOG] Log saved to: %s" % log_path)
sys.stdout = _tee._stream
_tee.close()

