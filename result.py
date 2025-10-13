import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from src import get_valid_data
# Load configuration from YAML file
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

csv_path = "/home/smatsubara/documents/sandbox/ml_airlift/models/layernorm/predicted.csv"

target_variables = pl.read_csv(
    csv_path,
    # encoding="SHIFT_JIS"
)
print(target_variables.head())

x = target_variables["固相体積率"].to_numpy()
y = target_variables["mean"].to_numpy()
yerr = np.array([
    float(v) ** 0.5 if (v is not None and (isinstance(v, (float, int)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit()))) else np.nan
    for v in target_variables["var"].to_numpy()
])

glass_diameter_col = target_variables["ガラス球直径"]
x = target_variables["固相体積率"].to_numpy()
y = target_variables["mean"].to_numpy()
is_str = np.array([1 if isinstance(v, str) and not v.replace('.', '', 1).isdigit() else 0 for v in glass_diameter_col.to_numpy()])
y_stone = y[is_str == 1]
yerr_stone = yerr[is_str == 1]
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
def RelativeError(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / (y_true + 1e-7))
    
x_valid, y_valid, yerr_valid = get_valid_data(x, y, yerr)
x_valid_stone, y_valid_stone, yerr_valid_stone = get_valid_data(x[is_str == 1], y_stone, yerr_stone)    
print(x_valid.shape)
rmse = calculate_rmse(x_valid, y_valid)
mae = calculate_mae(x_valid, y_valid)
relative_error = RelativeError(x_valid, y_valid)
relative_error_stone = RelativeError(x_valid_stone, y_valid_stone)
print(f"RMSE between ground truth and prediction: {rmse:.6f}")
print(f"MAE between ground truth and prediction: {mae:.6f}")
print(f"Relative Error between ground truth and prediction: {relative_error:.6f}")
print(f"Relative Error between ground truth and prediction (stone): {relative_error_stone:.6f}")