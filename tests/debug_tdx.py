#!/usr/bin/env python3
"""
Debug script to investigate TDX1 (channel 0) and TDX3 (channel 2 in original, channel 1 after dropping) data.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import hilbert

# Load data
x_path = "/home/smatsubara/documents/airlift/data/cleaned/x_train_dropped.npy"
print("Loading data...")
x = np.load(x_path)
print(f"Data shape: {x.shape}")
print(f"Data dtype: {x.dtype}")

# Select sample 36
sample_idx = 36
x_sample = x[sample_idx]  # [C, H, W]
print(f"\nSample {sample_idx} shape: {x_sample.shape}")
print(f"Sample {sample_idx} dtype: {x_sample.dtype}")

# Check TDX1 (channel 0) and TDX3 (channel 1 in dropped dataset)
tdx1 = x_sample[0]  # [H, W]
tdx2 = x_sample[1]  # [H, W]

print(f"\nTDX1 (channel 0):")
print(f"  Shape: {tdx1.shape}")
print(f"  Min: {tdx1.min():.6f}, Max: {tdx1.max():.6f}")
print(f"  Mean: {tdx1.mean():.6f}, Std: {tdx1.std():.6f}")
print(f"  Non-zero count: {np.count_nonzero(tdx1)} / {tdx1.size}")

print(f"\nTDX3 (channel 1 in dropped dataset):")
print(f"  Shape: {tdx2.shape}")
print(f"  Min: {tdx2.min():.6f}, Max: {tdx2.max():.6f}")
print(f"  Mean: {tdx2.mean():.6f}, Std: {tdx2.std():.6f}")
print(f"  Non-zero count: {np.count_nonzero(tdx2)} / {tdx2.size}")

# Apply Hilbert transform to a single row to see the difference
row_idx = 700  # Middle row
tdx1_row = tdx1[row_idx, :]
tdx2_row = tdx2[row_idx, :]

hilbert_tdx1 = np.abs(hilbert(tdx1_row))
hilbert_tdx2 = np.abs(hilbert(tdx2_row))

print(f"\nRow {row_idx} - Hilbert transform:")
print(f"  TDX1 original: min={tdx1_row.min():.6f}, max={tdx1_row.max():.6f}")
print(f"  TDX1 Hilbert: min={hilbert_tdx1.min():.6f}, max={hilbert_tdx1.max():.6f}")
print(f"  TDX3 original: min={tdx2_row.min():.6f}, max={tdx2_row.max():.6f}")
print(f"  TDX3 Hilbert: min={hilbert_tdx2.min():.6f}, max={hilbert_tdx2.max():.6f}")

# Apply Hilbert transform to all rows for TDX1 and TDX3
print("\nApplying Hilbert transform to all rows...")
hilbert_tdx1_full = np.zeros((tdx1.shape[0], tdx1.shape[1]))
hilbert_tdx2_full = np.zeros((tdx2.shape[0], tdx2.shape[1]))

for h in range(tdx1.shape[0]):
    hilbert_tdx1_full[h, :] = np.abs(hilbert(tdx1[h, :]))
    hilbert_tdx2_full[h, :] = np.abs(hilbert(tdx2[h, :]))

print(f"\nFull Hilbert transform:")
print(f"  TDX1 Hilbert: min={hilbert_tdx1_full.min():.6f}, max={hilbert_tdx1_full.max():.6f}")
print(f"  TDX2 Hilbert: min={hilbert_tdx2_full.min():.6f}, max={hilbert_tdx2_full.max():.6f}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Original TDX1
im1 = axes[0, 0].imshow(tdx1, cmap='jet', aspect='auto')
axes[0, 0].set_title(f'TDX1 Original (Sample {sample_idx})\nMin: {tdx1.min():.4f}, Max: {tdx1.max():.4f}')
axes[0, 0].axis('off')
plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

# Original TDX2
im2 = axes[1, 0].imshow(tdx2, cmap='jet', aspect='auto')
axes[1, 0].set_title(f'TDX2 Original (Sample {sample_idx})\nMin: {tdx2.min():.4f}, Max: {tdx2.max():.4f}')
axes[1, 0].axis('off')
plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

# Hilbert TDX1
im3 = axes[0, 1].imshow(hilbert_tdx1_full, cmap='jet', aspect='auto')
axes[0, 1].set_title(f'TDX1 Hilbert (Sample {sample_idx})\nMin: {hilbert_tdx1_full.min():.4f}, Max: {hilbert_tdx1_full.max():.4f}')
axes[0, 1].axis('off')
plt.colorbar(im3, ax=axes[0, 1], fraction=0.046, pad=0.04)

# Hilbert TDX2
im4 = axes[1, 1].imshow(hilbert_tdx2_full, cmap='jet', aspect='auto')
axes[1, 1].set_title(f'TDX2 Hilbert (Sample {sample_idx})\nMin: {hilbert_tdx2_full.min():.4f}, Max: {hilbert_tdx2_full.max():.4f}')
axes[1, 1].axis('off')
plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

# Single row comparison
axes[0, 2].plot(tdx1_row, 'b-', alpha=0.7, label='TDX1 Original')
axes[0, 2].plot(hilbert_tdx1, 'r-', linewidth=2, label='TDX1 Hilbert')
axes[0, 2].set_title(f'Row {row_idx} - TDX1')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

axes[1, 2].plot(tdx2_row, 'b-', alpha=0.7, label='TDX2 Original')
axes[1, 2].plot(hilbert_tdx2, 'r-', linewidth=2, label='TDX2 Hilbert')
axes[1, 2].set_title(f'Row {row_idx} - TDX2')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
save_path = Path("/home/smatsubara/documents/airlift/data/sandbox/visualize/debug_tdx_sample36.png")
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nVisualization saved to: {save_path}")

