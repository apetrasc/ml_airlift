#!/usr/bin/env python3
"""
Compare x (true) and y (predicted) values from predicted.csv.

Uses only rows where both columns exist (finite numbers).
Evaluates y vs x using Y=X as the ideal prediction line.
Calculates MAE and RMSE directly from the differences between y and x.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def _to_float_or_nan(v: object) -> float:
    if v is None:
        return float("nan")
    s = str(v).strip()
    if s == "" or s.lower() in ("none", "nan"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def read_xy(csv_path: str, x_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray]:
    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig"):
        try:
            xs: list[float] = []
            ys: list[float] = []
            with open(csv_path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError("CSV has no header row.")
                if x_col not in reader.fieldnames:
                    raise KeyError(f"x_col not found: {x_col}. Available: {reader.fieldnames}")
                if y_col not in reader.fieldnames:
                    raise KeyError(f"y_col not found: {y_col}. Available: {reader.fieldnames}")

                for row in reader:
                    x = _to_float_or_nan(row.get(x_col))
                    y = _to_float_or_nan(row.get(y_col))
                    if np.isfinite(x) and np.isfinite(y):
                        xs.append(x)
                        ys.append(y)

            return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to read CSV: {csv_path}") from last_err


def get_english_label(col_name: str) -> str:
    """Convert column name to English label for plots."""
    label_map = {
        '固相体積率': 'Solid Volume Fraction',
        'mean': 'Mean',
        'predicted': 'Predicted',
        'true': 'True',
    }
    # Check if exact match exists
    if col_name in label_map:
        return label_map[col_name]
    # Check if contains any key
    for key, value in label_map.items():
        if key in col_name:
            return value
    # Return as-is if no mapping found
    return col_name


def plot_comparison(x: np.ndarray, y: np.ndarray, r: float, mae: float, rmse: float,
                    x_col: str, y_col: str, output_path: str):
    """Create a scatter plot comparing x (true) vs y (predicted) with Y=X reference line."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get English labels
    x_label = get_english_label(x_col)
    y_label = get_english_label(y_col)
    
    # Determine axis limits (use same range for both axes to show Y=X properly)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    data_min = min(x_min, y_min)
    data_max = max(x_max, y_max)
    # Add some padding
    padding = (data_max - data_min) * 0.05
    axis_min = data_min - padding
    axis_max = data_max + padding
    
    # Set equal aspect ratio and limits for Y=X evaluation
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect('equal', adjustable='box')
    
    # Y=X reference line (ideal prediction line)
    ax.plot([axis_min, axis_max], [axis_min, axis_max], 
            'k--', linewidth=2, alpha=0.7, label='Y = X (ideal)')
    
    # Scatter plot
    ax.scatter(x, y, alpha=0.6, s=30, color='blue', edgecolors='black', linewidth=0.3, label='Data points')
    
    # Labels and title
    ax.set_xlabel(f'{x_label} (True)', fontsize=12)
    ax.set_ylabel(f'{y_label} (Predicted)', fontsize=12)
    ax.set_title(f'{y_label} vs {x_label}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Add metrics text box
    metrics_text = f'Pearson r = {r:.6f}\nMAE = {mae:.6f}\nRMSE = {rmse:.6f}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare predicted vs true values from predicted.csv")
    parser.add_argument(
        "--csv",
        default="/home/smatsubara/documents/airlift/source/ml_airlift/models/pretrained/predicted.csv",
        help="Path to predicted.csv",
    )
    parser.add_argument("--x_col", default="固相体積率", help="X column name (ground truth)")
    parser.add_argument("--y_col", default="mean", help="Y column name (prediction/stat)")
    parser.add_argument("--output", default=None, help="Output path for plot (default: same dir as CSV with .png extension)")
    args = parser.parse_args()

    x, y = read_xy(args.csv, args.x_col, args.y_col)
    if x.size < 2:
        raise SystemExit(f"Not enough valid pairs (N={x.size}).")

    # Calculate metrics directly comparing y (predicted) vs x (true)
    # Pearson r
    r = float(np.corrcoef(x, y)[0, 1]) if x.size >= 2 else float("nan")
    
    # MAE (Mean Absolute Error): mean(|y - x|)
    mae = float(np.mean(np.abs(y - x)))
    
    # RMSE (Root Mean Squared Error): sqrt(mean((y - x)^2))
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))

    print("==================================================")
    print(f"Using valid pairs only: N={x.size}")
    print(f"X (True): {args.x_col}   Y (Predicted): {args.y_col}")
    print("--------------------------------------------------")
    print(f"Evaluation against Y = X (ideal prediction)")
    print("--------------------------------------------------")
    print(f"Pearson r = {r:.6f}")
    print(f"MAE = {mae:.6f}")
    print(f"RMSE = {rmse:.6f}")
    print("==================================================")
    
    # Generate plot
    if args.output is None:
        csv_path = Path(args.csv)
        output_path = csv_path.parent / f"{csv_path.stem}_comparison_plot.png"
    else:
        output_path = Path(args.output)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(x, y, r, mae, rmse, args.x_col, args.y_col, str(output_path))
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


