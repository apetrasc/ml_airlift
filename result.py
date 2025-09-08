import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

# Load configuration from YAML file
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

save_results_path = config['evaluation']['results_path']

target_variables = pl.read_csv(
    config['evaluation']['save_path'],
    # encoding="SHIFT_JIS"
)
print(target_variables.head())

x = target_variables["固相体積率"].to_numpy()
y = target_variables["mean"].to_numpy()
yerr = target_variables["var"].to_numpy() ** 0.5    


# Read the CSV file with UTF-8 (with BOM) encoding to prevent character corruption
target_variables = pl.read_csv(
    config['evaluation']['save_path'],
    # encoding="SHIFT_JIS"
)
#print(target_variables.head())

# Plot mean (y-axis) vs. solid phase volume fraction (x-axis)
x = target_variables["固相体積率"].to_numpy()
y = target_variables["mean"].to_numpy()
if np.isnan(y).any():
    print("nan")
    
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    yerr = yerr[mask]
print(f"min: {np.min(y)}")
bias = np.min(y)* np.ones(len(y))
y = y - bias
print(y)
print(y.shape,x.shape, yerr.shape)

#  Calculate the correlation coefficient between x and y
# Remove any NaN values before calculation
mask = ~np.isnan(x) & ~np.isnan(y)
x_valid = x[mask]
y_valid = y[mask]

if len(x_valid) > 1:
    corr_coef = np.corrcoef(x_valid, y_valid)[0, 1]
    print(f"Correlation coefficient between x and y: {corr_coef:.4f}")
else:
    print("Not enough valid data to calculate correlation coefficient.")
save_results_path = config['evaluation']['results_path']
plt.figure(figsize=(8, 8))
plt.errorbar(x, y, yerr=yerr, fmt='o', color='blue', alpha=0.7, ecolor='red', capsize=3)
plt.plot([0, 1], [0, 1], 'r--', label='Ideal (y=x)')
plt.xlabel("Ground Truth")
plt.ylabel("Predicted")
plt.xlim(-0, 0.2)
plt.ylim(-0, 0.2)
plt.title("Predicted vs. Truth (Processed)")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(save_results_path, 'predicted_vs_truth_processed.png'))