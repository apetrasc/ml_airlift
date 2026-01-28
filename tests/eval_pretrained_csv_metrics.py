#!/usr/bin/env python3
"""
Compute regression metrics from a CSV file using the same definitions as scripts/optimize.py.

In scripts/optimize.py, R^2 is computed via sklearn.metrics.r2_score(y_true, y_pred),
and MSE/MAE are computed in the usual way. This script reproduces that behavior
for a (true, pred) column pair stored in a CSV.
"""

from __future__ import annotations

import argparse
import csv
from typing import Tuple

import numpy as np

def _read_csv_columns(csv_path: str, true_col: str, pred_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read two numeric columns from a CSV (UTF-8/UTF-8-SIG)."""
    def _to_float_or_nan(v: str) -> float:
        if v is None:
            return float("nan")
        s = str(v).strip()
        if s == "" or s.lower() == "none" or s.lower() == "nan":
            return float("nan")
        try:
            return float(s)
        except ValueError:
            return float("nan")

    # Try common encodings.
    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig"):
        try:
            y_true_list: list[float] = []
            y_pred_list: list[float] = []
            with open(csv_path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError("CSV has no header row.")
                if true_col not in reader.fieldnames:
                    raise KeyError(f"true_col not found: {true_col}. Available: {reader.fieldnames}")
                if pred_col not in reader.fieldnames:
                    raise KeyError(f"pred_col not found: {pred_col}. Available: {reader.fieldnames}")

                for row in reader:
                    y_true_list.append(_to_float_or_nan(row.get(true_col)))
                    y_pred_list.append(_to_float_or_nan(row.get(pred_col)))

            return np.asarray(y_true_list, dtype=np.float64), np.asarray(y_pred_list, dtype=np.float64)
        except Exception as e:  # pragma: no cover
            last_err = e
            continue

    raise RuntimeError(f"Failed to read CSV: {csv_path}") from last_err


def _filter_valid_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Filter NaN/inf values from (y_true, y_pred) pairs."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def _r2_score_sklearn_compatible(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R^2 (coefficient of determination) compatible with sklearn.metrics.r2_score
    for the single-output case.
    """
    y_true, y_pred = _filter_valid_pairs(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")

    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))

    # Match sklearn behavior for constant y_true:
    # - If ss_tot == 0 and predictions are perfect, return 1.0
    # - If ss_tot == 0 and predictions are not perfect, return 0.0
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    return 1.0 - (ss_res / ss_tot)


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _filter_valid_pairs(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_true - y_pred) ** 2))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _filter_valid_pairs(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute R^2/RMSE/MAE/MSE from models/pretrained/predicted.csv",
    )
    parser.add_argument(
        "--csv",
        default="/home/smatsubara/documents/airlift/source/ml_airlift/models/pretrained/predicted.csv",
        help="Path to predicted.csv",
    )
    parser.add_argument(
        "--true_col",
        default="固相体積率",
        help="Column name to use as ground truth (y_true)",
    )
    parser.add_argument(
        "--pred_col",
        default="mean",
        help="Column name to use as prediction (y_pred)",
    )
    args = parser.parse_args()

    y_true, y_pred = _read_csv_columns(args.csv, args.true_col, args.pred_col)
    y_true_v, y_pred_v = _filter_valid_pairs(y_true, y_pred)

    # Prefer sklearn for exact parity with optimize.py, but keep a fallback.
    try:
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        r2 = float(r2_score(y_true_v, y_pred_v))
        mse = float(mean_squared_error(y_true_v, y_pred_v))
        mae = float(mean_absolute_error(y_true_v, y_pred_v))
    except Exception:
        r2 = _r2_score_sklearn_compatible(y_true_v, y_pred_v)
        mse = _mse(y_true_v, y_pred_v)
        mae = _mae(y_true_v, y_pred_v)

    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")

    print("==================================================")
    print("Target\t\tR²\tRMSE\tMAE\tMSE\tN")
    print("--------------------------------------------------")
    print(f"{args.true_col}\t{r2:.4f}\t{rmse:.4f}\t{mae:.4f}\t{mse:.4f}\t{y_true_v.size}")
    print("==================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


