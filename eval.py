import polars as pl
from src import preprocess_and_predict
from models import SimpleCNN, SimpleViTRegressor
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

# Read the target variables CSV file (experiment metadata)
target_variables = pl.read_csv(
    config['evaluation']['target_variables_path'],
    encoding="SHIFT_JIS"
)

# Load the trained model
model = SimpleCNN(config['hyperparameters']['input_length']).to(config['evaluation']['device'])
#model = SimpleViTRegressor(config['hyperparameters']['input_length']).to(config['evaluation']['device'])
# You can use the argparse library to accept a command-line argument for base_dir (datetime).


parser = argparse.ArgumentParser(description="Run evaluation with specified base directory (datetime).")
parser.add_argument('--datetime', type=str, required=True, help='Base directory for evaluation (e.g., /home/smatsubara/documents/airlift/data/outputs/2025-09-07/14-39-46)')
args = parser.parse_args()

base_dir = args.datetime
model_path = os.path.join(base_dir + '/weights/model.pth')
model.load_state_dict(torch.load(model_path))
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
#processed_dir_stone = config['evaluation']['processed_dir_stone']
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
    if file_path == "/home/smatsubara/documents/airlift/data/experiments/processed/P20241015-1037_processed.npz":
        print("DEBUG: File path matches exactly:", file_path)
        print("DEBUG: File exists:", os.path.exists(file_path))
    if os.path.exists(file_path):
        try:
            mean, var = preprocess_and_predict(file_path, model)
            #mean_stone, var_stone = preprocess_and_predict(file_path_stone, model)
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

#  Calculate the correlation coefficient between x and y
# Remove any NaN values before calculation
# x, y, yerr からNaNを除外した有効なデータのみを抽出
def get_valid_data(x, y, yerr):
    """
    Remove NaN values from x, y, and yerr, and return only valid data.
    """
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(yerr)
    x_valid = x[mask]
    y_valid = y[mask]
    yerr_valid = yerr[mask]
    return x_valid, y_valid, yerr_valid

x_valid, y_valid, yerr_valid = get_valid_data(x, y, yerr)
# x_valid_stone, y_valid_stone, yerr_valid_stone = get_valid_data(x, y_stone, yerr_stone)

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
    y_processed = 2*(y - bias)
    return y_processed

y_valid_calibrated = calibration(x_valid, y_valid, yerr_valid)
# y_valid_stone_calibrated = calibration(x_valid_stone, y_valid_stone, yerr_valid_stone)

if len(x_valid) > 1:
    corr_coef = np.corrcoef(x_valid, y_valid_calibrated)[0, 1]
    print(f"Correlation coefficient between x and y: {corr_coef:.4f}")
else:
    print("Not enough valid data to calculate correlation coefficient.")

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
plt.title("Predicted vs. Ground Truth", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(base_dir, 'predicted_vs_ground_truth.png'))


x_valid_stone, y_valid_stone_plot, yerr_valid_stone_plot = get_valid_data(x[is_str == 1], y_stone, yerr_stone)

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
plt.title("Predicted vs. Ground Truth", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()
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
plt.title("Predicted vs. Truth (Processed)", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(base_dir, 'predicted_vs_truth_processed.png'))



