import polars as pl
from src import preprocess_and_predict, preprocess, debug_pipeline,preprocess_liquidonly
from models import SimpleCNN, SimpleViTRegressor, ResidualCNN, BaseCNN, ProposedCNN
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import math
import argparse
import sys
# Load configuration from YAML file
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

png_save_dir = config['evaluation']['png_save_dir']

# Read the target variables CSV file (experiment metadata)
target_variables = pl.read_csv(
    config['evaluation']['target_variables_path'],
    encoding="SHIFT_JIS"
)

# Load the trained model
#model = SimpleCNN(config['hyperparameters']['input_length']).to(config['evaluation']['device'])
#model = ResidualCNN(config['hyperparameters']['input_length']).to(config['evaluation']['device'])
#model = SimpleViTRegressor(config['hyperparameters']['input_length']).to(config['evaluation']['device'])
model = SimpleCNN(config['hyperparameters']['input_length']).to(config['evaluation']['device'])
#model = ProposedCNN(config['hyperparameters']['input_length']).to(config['evaluation']['device'])
#model = BaseCNN(config['hyperparameters']['input_length']).to(config['evaluation']['device'])
# You can use the argparse library to accept a command-line argument for base_dir (datetime).


parser = argparse.ArgumentParser(description="Run evaluation with specified base directory (datetime).")
parser.add_argument('--datetime', type=str, required=True, help='Base directory for evaluation (e.g., /home/smatsubara/documents/airlift/data/outputs/2025-09-07/14-39-46)')
parser.add_argument('--bandpass', nargs=2, type=float, required=False, help=
                    'Low Freq, High Freq. The unit is [MHz]')
parser.add_argument('--log1p', type=int, required=False, default=0,
                    help='If applying log1p')
parser.add_argument('--rolling', type=int, required=False, default=0,
                    help='If applying maximum rolling windows')
parser.add_argument('--rollingparam', nargs=2, type=int, required=False,
                    help='Window Size, Stride')
parser.add_argument('--hilbert', type=int, required=False,default=1,
                    help='If applying hilbert envelope')
parser.add_argument('--reduce', type=int, required=False, default=0,
                    help='If reducing signals by liquidonly ones')
parser.add_argument('--drawsignal', type=int, required=False, default=0,
                    help='If drawing signals')
parser.add_argument('--pngname', type=str, required=False, default='eval',
                    help='Name of png')
args = parser.parse_args()

if args.bandpass:
    low_freq = args.bandpass[0]*1e6
    high_freq = args.bandpass[1]*1e6
    if low_freq > high_freq:
        sys.exit('The first argument should be smaller than the second one!')
else:
    low_freq=0
    high_freq=1.0e9

if_log1p = args.log1p
if_rolling = args.rolling
if_hilbert = args.hilbert
if_reduce = args.reduce
if_drawsignal = args.drawsignal
png_name = args.pngname
window_size = 0
window_stride = 0

if if_rolling:
    if args.rollingparam:
        window_size = args.rollingparam[0]
        window_stride = args.rollingparam[1]
    else:
        sys.exit('You need rollingparam in the argument.')

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
processed_dir = config['evaluation']['processed_dir']
target_variables = target_variables.with_columns(
    (pl.lit(processed_dir + "/") + pl.col("NAME") + pl.lit("_processed.npz")).alias("FullPath")
)

# Get the mean raw signal preprocessed in the same way as solid-liquid signals
x_liquid_only = None
if if_reduce:
    for row in target_variables.iter_rows(named=True):
        if row["気相体積率"]==0 and row["固相体積率"]==0:
            file_path = row["FullPath"]
            x_liquid_only = preprocess_liquidonly(path=file_path,
                                                filter_freq=[low_freq, high_freq],
                                                rolling_window=if_rolling,
                                                window_size=window_size,
                                                window_stride=window_stride,
                                                if_hilbert=if_hilbert,
                                                if_log1p = if_log1p)
            break
        



# For each row, load the processed .npz file, run inference, and add mean and variance as new columns
mean_list = []
var_list = []

print(f'if_hilbert: {if_hilbert}')

for row in target_variables.iter_rows(named=True):
    file_path = row["FullPath"]
    # Debug: Check if the file path is exactly as expected
    if file_path == "/home/smatsubara/documents/airlift/data/experiments/processed/P20241015-1037_processed.npz":
        print("DEBUG: File path matches exactly:", file_path)
        print("DEBUG: File exists:", os.path.exists(file_path))
    
    
    if os.path.exists(file_path):
        if file_path == "/home/smatsubara/documents/airlift/data/experiments/processed/solid_liquid/P20241011-1015_processed.npz":
            debug_pipeline(base_dir, 'config/config.yaml', file_path)
        if file_path == "/home/smatsubara/documents/airlift/data/experiments/processed/solid_liquid/P20240726-1600_processed.npz":
            debug_pipeline(base_dir, 'config/config.yaml', file_path)
        try:
            mean, var = preprocess_and_predict(path=file_path, model=model,
                                               filter_freq=[low_freq, high_freq],
                                               rolling_window=if_rolling,
                                               window_size=window_size,
                                               window_stride=window_stride,
                                               if_hilbert=if_hilbert,
                                               if_log1p = if_log1p,
                                               if_reduce = if_reduce,
                                               x_liquid_only=x_liquid_only,
                                               if_drawsignal=if_drawsignal,
                                               png_save_dir=png_save_dir,
                                               png_name=png_name)
            mean, var = preprocess_and_predict(file_path, model, device=config['evaluation']['device'])
            mean_list.append(mean)
            var_list.append(var)
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
    else:
        mean_list.append(None)
        var_list.append(None)

# Add the mean and variance as new columns (float or None only)
target_variables = target_variables.with_columns([
    pl.Series("mean", mean_list, dtype=pl.Float64),
    pl.Series("var", var_list, dtype=pl.Float64)
])

# Adjust the column order so that "mean" and "var" come right after "FullPath"
cols_to_show = [col for col in target_variables.columns[2:] if col not in ["NAME", "FullPath", "mean", "var"]] + ["NAME", "FullPath", "mean", "var"]

# Save the results to a CSV file

save_path = os.path.join(base_dir, "predicted.csv")
target_variables.select(cols_to_show).write_csv(save_path)
print(f"Prediction results saved to {save_path}.")


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
y = target_variables["mean"].to_numpy()
# y_stone = target_variables["mean_stone"].to_numpy()
y_stone = y[is_str == 1]


yerr = target_variables["var"].to_numpy() ** 0.5
yerr_stone = yerr[is_str == 1]


x_valid, y_valid, yerr_valid = get_valid_data(x, y, yerr)

print(y_valid)
# print(np.max(y_processed))
# print(np.max(y_valid_stone))
# print(np.max(y_processed_stone))
# print(np.min(y_processed_stone))
# print(np.max(y_processed))
# print(np.min(y_processed))
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
plt.legend(fontsize=18)
plt.plot([0, 1], [0, 1], 'r--', label='Ideal (y=x)')
plt.xlabel("Ground Truth(Tube Closing)", fontsize=18)
plt.ylabel("Prediction(machine learning)", fontsize=18)
plt.xlim(0, 0.2)
plt.ylim(0, 0.2)
plt.title("Prediction vs. Ground Truth", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'predicted_vs_ground_truth.png'))
plt.close()




plt.figure(figsize=(8, 8))
#plt.errorbar(x_valid, y_valid, yerr=yerr_valid, fmt='o', color='blue', alpha=0.7, ecolor='red', capsize=3, label='All')
plt.plot(x_valid, y_valid, 'o', color='blue', alpha=0.7, label='glass ball')
#plt.errorbar(x_valid_stone, y_valid_stone_plot, yerr=yerr_valid_stone_plot, fmt='o', color='orange', alpha=0.7, ecolor='green', capsize=3, label='Stone')
plt.plot(x_valid_stone, y_valid_stone_plot, 'o', color='orange', alpha=0.7, label='Stone')
plt.legend(fontsize=18)
plt.plot([0, 1], [0, 1], 'r--', label='Ideal (y=x)')
plt.xlabel("Ground Truth(Tube Closing)", fontsize=18)
plt.ylabel("Prediction(machine learning)", fontsize=18)
plt.xlim(0, 0.2)
plt.ylim(0, 0.2)
plt.title("Prediction vs. Ground Truth", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'predicted_vs_ground_truth_noerrorbars.png'))

# Optionally, display the results
# print(target_variables.select(cols_to_show))
plt.figure(figsize=(8, 8))
# y_valid_calibrated（全体）とy_valid_stone_calibrated（石あり）を色分けして表示
plt.errorbar(x_valid, y_valid_calibrated, yerr=yerr_valid, fmt='o', color='blue', alpha=0.7, ecolor='red', capsize=3, label='glass ball (calibrated)')
# y_valid_stone_calibrated用のx, y, yerrを抽出（is_str==1のインデックスを利用）
x_valid_stone, y_valid_stone_plot, yerr_valid_stone_plot = get_valid_data(x[is_str == 1], y_stone, yerr_stone)
y_valid_stone_calibrated = calibration(x_valid_stone, y_valid_stone_plot, yerr_valid_stone_plot)
plt.errorbar(x_valid_stone, y_valid_stone_calibrated, yerr=yerr_valid_stone_plot, fmt='o', color='orange', alpha=0.7, ecolor='green', capsize=3, label='Stone (calibrated)')
plt.legend()
plt.plot([0, 1], [0, 1], 'r--', label='Ideal (y=x)')
plt.xlabel("Ground Truth(Tube Closing)", fontsize=18)
plt.ylabel("Prediction(machine learning)", fontsize=18)
plt.xlim(-0, 0.2)
plt.ylim(-0, 0.2)
plt.title("Prediction vs. Truth (Processed)", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'predicted_vs_truth_processed.png'))
plt.close()
print("saved all figures")