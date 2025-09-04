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
print(target_variables.head())

# Plot mean (y-axis) vs. solid phase volume fraction (x-axis)
x = target_variables["固相体積率"].to_numpy()
y = target_variables["mean"].to_numpy()
# bias = 0.2 * np.ones(len(y))
# y = y - bias
print(y)
yerr = target_variables["var"].to_numpy() ** 0.5

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
plt.figure(figsize=(8, 6))
plt.errorbar(x, y, yerr=yerr, fmt='o', color='blue', alpha=0.7, ecolor='red', capsize=3)
plt.plot([0, 1], [0, 1], 'r--', label='Ideal (y=x)')
plt.xlabel("Solid Phase Volume Fraction")
plt.ylabel("Mean")
plt.xlim(-0.2, 0.2)
plt.ylim(-0.2, 0.2)
plt.title("Mean vs. Solid Phase Volume Fraction")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(save_results_path, 'mean_vs_solid_phase_volume_fraction.png'))