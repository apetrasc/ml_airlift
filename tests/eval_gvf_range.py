#!/usr/bin/env python3
"""
Detailed Gas Volume Fraction (GVF) evaluation with range-based RMSE analysis.

Evaluates GVF prediction performance with:
  - Overall and range-based RMSE/MAE/R2 metrics
  - Focused evaluation on GVF 0.50-1.00 range
  - Measurement uncertainty (+-) evaluation
  - Publication-quality plots with large fonts

Usage:
    python tests/eval_gvf_range.py --run_dir /path/to/run
    python tests/eval_gvf_range.py --run_dir /path/to/run --meas_uncertainty 0.05
    python tests/eval_gvf_range.py --run_dir /path/to/run --gvf_index 4 --ranges 0.0,0.25,0.50,0.75,1.0
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ---------------------------------------------------------------------------
# Global font / style settings  (publication-quality, large fonts)
# ---------------------------------------------------------------------------
FONTSIZE_TITLE = 22
FONTSIZE_LABEL = 20
FONTSIZE_TICK = 18
FONTSIZE_LEGEND = 16
FONTSIZE_ANNOT = 14
FONTSIZE_TABLE = 13

# Module-level target label (set in main based on --svf_subtraction)
TARGET_LABEL = "GVF"

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": FONTSIZE_TITLE,
    "axes.labelsize": FONTSIZE_LABEL,
    "xtick.labelsize": FONTSIZE_TICK,
    "ytick.labelsize": FONTSIZE_TICK,
    "legend.fontsize": FONTSIZE_LEGEND,
    "figure.titlesize": 24,
    "font.family": "serif",
})


# ===================================================================
# Metrics helpers
# ===================================================================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def bias(y_true, y_pred):
    """Mean bias (pred - true). Positive = over-prediction."""
    return np.mean(y_pred - y_true)


def relative_error(y_true, y_pred, eps=1e-7):
    """Relative error, excluding samples where true value is near zero."""
    nonzero_mask = np.abs(y_true) > 0.01
    if nonzero_mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask])
                   / (np.abs(y_true[nonzero_mask]) + eps))


# ===================================================================
# Range-based metric computation
# ===================================================================
def compute_range_metrics(y_true, y_pred, ranges):
    """
    Compute metrics for each [lo, hi) range defined by *ranges*.

    Parameters
    ----------
    y_true, y_pred : 1-D arrays
    ranges : list of float
        Bin edges, e.g. [0.0, 0.25, 0.50, 0.75, 1.0].

    Returns
    -------
    list of dict  – one entry per bin
    """
    results = []
    for lo, hi in zip(ranges[:-1], ranges[1:]):
        # Use <= for the last bin to include the upper boundary
        if hi == ranges[-1]:
            mask = (y_true >= lo) & (y_true <= hi)
        else:
            mask = (y_true >= lo) & (y_true < hi)
        n = int(mask.sum())
        if n == 0:
            results.append(dict(lo=lo, hi=hi, n=0, rmse=np.nan, mae=np.nan,
                                r2=np.nan, bias=np.nan, rel_err=np.nan))
            continue
        yt, yp = y_true[mask], y_pred[mask]
        results.append(dict(
            lo=lo, hi=hi, n=n,
            rmse=rmse(yt, yp),
            mae=mae(yt, yp),
            r2=r2(yt, yp) if n >= 2 else np.nan,
            bias=bias(yt, yp),
            rel_err=relative_error(yt, yp),
        ))
    return results


# ===================================================================
# +/- Measurement uncertainty evaluation
# ===================================================================
def evaluate_within_uncertainty(y_true, y_pred, abs_unc=0.03, rel_unc=None):
    """
    Evaluate how many predictions fall within the measurement uncertainty
    band of the ground truth.

    Parameters
    ----------
    y_true, y_pred : 1-D arrays
    abs_unc : float or None
        Absolute uncertainty band (e.g. 0.03 = +/-0.03).
    rel_unc : float or None
        Relative uncertainty band (e.g. 0.05 = +/-5% of true value).
        When both abs_unc and rel_unc are given, the wider band is used
        for each sample (max of absolute and relative).

    Returns
    -------
    dict with counts, percentage, per-sample uncertainty half-width array
    """
    n = len(y_true)
    # Build per-sample half-width
    hw = np.zeros(n)
    if abs_unc is not None:
        hw = np.maximum(hw, abs_unc)
    if rel_unc is not None:
        hw = np.maximum(hw, rel_unc * np.abs(y_true))
    within = np.abs(y_pred - y_true) <= hw
    return dict(
        n_total=n,
        n_within=int(within.sum()),
        pct_within=float(within.sum()) / n * 100.0 if n > 0 else 0.0,
        half_width=hw,
        within_mask=within,
    )


# ===================================================================
# Plotting – scatter with uncertainty band
# ===================================================================
def plot_scatter_with_uncertainty(y_true, y_pred, unc_hw, unc_mask,
                                  title, out_path, meas_unc_label,
                                  range_mask=None, range_label=None,
                                  plot_range=None, uniform_style=False):
    """
    Scatter plot (pred vs true) with measurement uncertainty band and
    optional range highlighting.

    Parameters
    ----------
    plot_range : tuple (lo, hi) or None
        If given, force both axes to [lo, hi].  Useful for the focus plot.
    uniform_style : bool
        If True, plot all points with the same colour/marker (blue circles).
    """
    fig, ax = plt.subplots(figsize=(9, 8))

    # y=x perfect line
    if plot_range is not None:
        vmin, vmax = plot_range
    else:
        vmin = min(y_true.min(), y_pred.min()) - 0.05
        vmax = max(y_true.max(), y_pred.max()) + 0.05
    xx = np.linspace(vmin, vmax, 200)
    ax.plot(xx, xx, "k--", lw=2, label="$y = x$ (perfect)", zorder=3)

    # Uncertainty band around y=x
    ax.fill_between(xx, xx - np.mean(unc_hw), xx + np.mean(unc_hw),
                    color="gold", alpha=0.25,
                    label=r"$\pm${} band".format(meas_unc_label), zorder=1)

    # Scatter
    if uniform_style:
        ax.scatter(y_true, y_pred, s=90, c="blue",
                   edgecolors="black", linewidths=0.6, alpha=0.6, zorder=4)
    else:
        outside = ~unc_mask
        ax.scatter(y_true[unc_mask], y_pred[unc_mask], s=90, c="royalblue",
                   edgecolors="black", linewidths=0.6, alpha=0.85,
                   label="Within uncertainty", zorder=4)
        if outside.any():
            ax.scatter(y_true[outside], y_pred[outside], s=90, c="tomato",
                       edgecolors="black", linewidths=0.6, alpha=0.85,
                       marker="^", label="Outside uncertainty", zorder=5)

    # Optional range highlight box
    if range_mask is not None and range_label:
        lo = y_true[range_mask].min() if range_mask.any() else 0
        hi = y_true[range_mask].max() if range_mask.any() else 1
        ax.axvspan(lo - 0.01, hi + 0.01, color="green", alpha=0.07,
                   label=range_label)

    ax.set_xlabel("Ground Truth (%s)" % TARGET_LABEL)
    ax.set_ylabel("Predicted (%s)" % TARGET_LABEL)
    ax.set_title(title)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=FONTSIZE_LEGEND)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: %s" % out_path)


# ===================================================================
# Plotting – range-based bar chart
# ===================================================================
def plot_range_bar(range_metrics, out_path, metric_key="rmse", ylabel="RMSE"):
    """Bar chart of a given metric per GVF range."""
    labels, vals, counts = [], [], []
    for rm in range_metrics:
        labels.append("[%.2f, %.2f)" % (rm["lo"], rm["hi"]))
        vals.append(rm[metric_key] if not np.isnan(rm[metric_key]) else 0.0)
        counts.append(rm["n"])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color="steelblue", edgecolor="black", linewidth=0.8)

    # Annotate count on each bar
    for xi, bar, c, v in zip(x, bars, counts, vals):
        txt = "n=%d" % c
        if c == 0:
            txt += "\n(no data)"
        else:
            txt += "\n%.4f" % v
        ax.text(xi, bar.get_height() + max(vals) * 0.02, txt,
                ha="center", va="bottom", fontsize=FONTSIZE_ANNOT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title("%s %s by Range" % (TARGET_LABEL, ylabel))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: %s" % out_path)


# ===================================================================
# Plotting – residual plot coloured by range
# ===================================================================
def plot_residuals_by_range(y_true, y_pred, ranges, out_path):
    """Residual plot coloured by GVF range."""
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.colormaps.get_cmap("tab10").resampled(len(ranges) - 1)
    for idx, (lo, hi) in enumerate(zip(ranges[:-1], ranges[1:])):
        if hi == ranges[-1]:
            mask = (y_true >= lo) & (y_true <= hi)
        else:
            mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() == 0:
            continue
        ax.scatter(y_true[mask], residuals[mask], s=80, c=[cmap(idx)],
                   edgecolors="black", linewidths=0.5, alpha=0.8,
                   label="[%.2f, %.2f)  n=%d" % (lo, hi, mask.sum()))

    ax.axhline(0, color="black", ls="--", lw=1.5)
    ax.set_xlabel("Ground Truth (%s)" % TARGET_LABEL)
    ax.set_ylabel("Residual (Pred $-$ True)")
    ax.set_title("%s Prediction Residuals by Range" % TARGET_LABEL)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTSIZE_LEGEND - 2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: %s" % out_path)


# ===================================================================
# Plotting – combined summary figure  (scatter + bar + table)
# ===================================================================
def plot_combined_summary(y_true, y_pred, range_metrics, unc_result,
                          overall_metrics, focus_metrics, out_path,
                          meas_unc_label):
    """Three-panel summary figure."""
    fig = plt.figure(figsize=(22, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.35)

    # --- Panel 1: Scatter ---
    ax1 = fig.add_subplot(gs[0])
    vmin = min(y_true.min(), y_pred.min()) - 0.05
    vmax = max(y_true.max(), y_pred.max()) + 0.05
    xx = np.linspace(vmin, vmax, 200)
    ax1.plot(xx, xx, "k--", lw=2, zorder=3)
    unc_hw = unc_result["half_width"]
    ax1.fill_between(xx, xx - np.mean(unc_hw), xx + np.mean(unc_hw),
                     color="gold", alpha=0.25, zorder=1)
    mask_w = unc_result["within_mask"]
    ax1.scatter(y_true[mask_w], y_pred[mask_w], s=60, c="royalblue",
                edgecolors="black", linewidths=0.5, alpha=0.85, zorder=4)
    if (~mask_w).any():
        ax1.scatter(y_true[~mask_w], y_pred[~mask_w], s=60, c="tomato",
                    edgecolors="black", linewidths=0.5, marker="^",
                    alpha=0.85, zorder=5)
    ax1.set_xlabel("Ground Truth (GVF)")
    ax1.set_ylabel("Predicted (GVF)")
    ax1.set_title("Prediction vs Ground Truth")
    ax1.set_xlim(vmin, vmax)
    ax1.set_ylim(vmin, vmax)
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, alpha=0.3)
    # Annotation box
    txt = ("Overall\n"
           "  RMSE = %.4f\n"
           "  MAE  = %.4f\n"
           "  R$^2$  = %.4f\n"
           "GVF 0.50-1.00\n"
           "  RMSE = %.4f\n"
           "  MAE  = %.4f\n"
           "  R$^2$  = %.4f"
           % (overall_metrics["rmse"], overall_metrics["mae"],
              overall_metrics["r2"],
              focus_metrics["rmse"], focus_metrics["mae"],
              focus_metrics["r2"]))
    ax1.text(0.02, 0.98, txt, transform=ax1.transAxes,
             fontsize=FONTSIZE_ANNOT, va="top",
             bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    # --- Panel 2: RMSE bar chart ---
    ax2 = fig.add_subplot(gs[1])
    labels, vals, counts = [], [], []
    for rm in range_metrics:
        labels.append("[%.2f,%.2f)" % (rm["lo"], rm["hi"]))
        vals.append(rm["rmse"] if not np.isnan(rm["rmse"]) else 0.0)
        counts.append(rm["n"])
    x = np.arange(len(labels))
    bars = ax2.bar(x, vals, color="steelblue", edgecolor="black", lw=0.8)
    for xi, bar, c, v in zip(x, bars, counts, vals):
        ax2.text(xi, bar.get_height() + max(vals) * 0.02,
                 "n=%d\n%.4f" % (c, v) if c > 0 else "n=0",
                 ha="center", va="bottom", fontsize=FONTSIZE_ANNOT - 1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right")
    ax2.set_ylabel("RMSE")
    ax2.set_title("RMSE by GVF Range")
    ax2.grid(axis="y", alpha=0.3)

    # --- Panel 3: Metrics table ---
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    # Build table data
    col_labels = ["Range", "n", "RMSE", "MAE", "Bias", "R\u00b2"]
    cell_text = []
    for rm in range_metrics:
        row = [
            "[%.2f, %.2f)" % (rm["lo"], rm["hi"]),
            "%d" % rm["n"],
            "%.4f" % rm["rmse"] if rm["n"] > 0 else "-",
            "%.4f" % rm["mae"] if rm["n"] > 0 else "-",
            "%+.4f" % rm["bias"] if rm["n"] > 0 else "-",
            "%.4f" % rm["r2"] if (rm["n"] >= 2 and not np.isnan(rm["r2"])) else "-",
        ]
        cell_text.append(row)
    # Append overall and focus rows
    cell_text.append([
        "Overall", "%d" % len(y_true),
        "%.4f" % overall_metrics["rmse"],
        "%.4f" % overall_metrics["mae"],
        "%+.4f" % overall_metrics["bias"],
        "%.4f" % overall_metrics["r2"],
    ])
    cell_text.append([
        "[0.50, 1.00]", "%d" % focus_metrics["n"],
        "%.4f" % focus_metrics["rmse"],
        "%.4f" % focus_metrics["mae"],
        "%+.4f" % focus_metrics["bias"],
        "%.4f" % focus_metrics["r2"] if focus_metrics["n"] >= 2 else "-",
    ])
    # Uncertainty row
    cell_text.append([
        "Within %s" % meas_unc_label, "",
        "%d / %d" % (unc_result["n_within"], unc_result["n_total"]),
        "%.1f%%" % unc_result["pct_within"], "", "",
    ])

    table = ax3.table(cellText=cell_text, colLabels=col_labels,
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(FONTSIZE_TABLE)
    table.scale(1.0, 1.6)
    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Highlight focus row (second to last data row)
    focus_row_idx = len(range_metrics) + 1  # +1 for overall
    for j in range(len(col_labels)):
        table[focus_row_idx + 1, j].set_facecolor("#E2EFDA")
    ax3.set_title("Evaluation Metrics", pad=20)

    fig.suptitle("GVF Prediction Evaluation — Detailed Range Analysis",
                 fontsize=24, y=1.02)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: %s" % out_path)


# ===================================================================
# Text report
# ===================================================================
def write_report(y_true, y_pred, range_metrics, unc_result,
                 overall_metrics, focus_metrics, low_metrics,
                 out_path, meas_unc_label):
    """Write a plain-text summary report."""
    lines = []
    lines.append("=" * 72)
    lines.append("GVF (Gas Volume Fraction) — Detailed Range-Based Evaluation")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Samples (total) : %d" % len(y_true))
    lines.append("True  range     : [%.4f, %.4f]" % (y_true.min(), y_true.max()))
    lines.append("Pred  range     : [%.4f, %.4f]" % (y_pred.min(), y_pred.max()))
    lines.append("")

    # Overall metrics
    lines.append("-" * 72)
    lines.append("OVERALL METRICS")
    lines.append("-" * 72)
    for k in ("rmse", "mae", "r2", "bias", "rel_err"):
        lines.append("  %-15s : %.6f" % (k.upper(), overall_metrics[k]))
    lines.append("")

    # Focus range
    lines.append("-" * 72)
    lines.append("FOCUS RANGE: GVF 0.50 - 1.00")
    lines.append("-" * 72)
    lines.append("  Samples         : %d" % focus_metrics["n"])
    for k in ("rmse", "mae", "r2", "bias", "rel_err"):
        v = focus_metrics[k]
        lines.append("  %-15s : %.6f" % (k.upper(), v) if not np.isnan(v) else
                      "  %-15s : N/A" % k.upper())
    lines.append("")

    # Low range
    lines.append("-" * 72)
    lines.append("LOW RANGE: GVF 0.00 - 0.20")
    lines.append("-" * 72)
    lines.append("  Samples         : %d" % low_metrics["n"])
    for k in ("rmse", "mae", "r2", "bias", "rel_err"):
        v = low_metrics[k]
        lines.append("  %-15s : %.6f" % (k.upper(), v) if not np.isnan(v) else
                      "  %-15s : N/A" % k.upper())
    lines.append("")

    # Per-range table
    lines.append("-" * 72)
    lines.append("PER-RANGE METRICS")
    lines.append("-" * 72)
    header = "%-16s %5s %10s %10s %10s %10s %10s" % (
        "Range", "n", "RMSE", "MAE", "Bias", "R2", "RelErr")
    lines.append(header)
    lines.append("-" * len(header))
    for rm in range_metrics:
        if rm["n"] == 0:
            lines.append("%-16s %5d %10s %10s %10s %10s %10s" % (
                "[%.2f,%.2f)" % (rm["lo"], rm["hi"]), 0,
                "-", "-", "-", "-", "-"))
        else:
            lines.append("%-16s %5d %10.4f %10.4f %+10.4f %10.4f %10.4f" % (
                "[%.2f,%.2f)" % (rm["lo"], rm["hi"]), rm["n"],
                rm["rmse"], rm["mae"], rm["bias"],
                rm["r2"] if not np.isnan(rm["r2"]) else 0.0,
                rm["rel_err"]))
    lines.append("")

    # Uncertainty evaluation
    lines.append("-" * 72)
    lines.append("+/- MEASUREMENT UNCERTAINTY EVALUATION  (band: %s)" % meas_unc_label)
    lines.append("-" * 72)
    lines.append("  Total samples   : %d" % unc_result["n_total"])
    lines.append("  Within band     : %d" % unc_result["n_within"])
    lines.append("  Percentage      : %.1f%%" % unc_result["pct_within"])
    lines.append("")

    # Per-range uncertainty
    lines.append("  Per-range within-uncertainty breakdown:")
    lines.append("  %-16s %5s %8s %8s" % ("Range", "n", "Within", "Pct"))
    lines.append("  " + "-" * 42)
    for rm in range_metrics:
        lo, hi = rm["lo"], rm["hi"]
        if hi == 1.0:
            mask_r = (y_true >= lo) & (y_true <= hi)
        else:
            mask_r = (y_true >= lo) & (y_true < hi)
        if mask_r.sum() == 0:
            lines.append("  %-16s %5d %8s %8s" % (
                "[%.2f,%.2f)" % (lo, hi), 0, "-", "-"))
            continue
        within_r = unc_result["within_mask"][mask_r].sum()
        pct_r = within_r / mask_r.sum() * 100
        lines.append("  %-16s %5d %8d %7.1f%%" % (
            "[%.2f,%.2f)" % (lo, hi), mask_r.sum(), within_r, pct_r))
    lines.append("")
    lines.append("=" * 72)

    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    print("  Saved: %s" % out_path)
    print()
    print(text)


# ===================================================================
# main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Detailed GVF range-based evaluation with +/- uncertainty analysis")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Run directory with y_pred.npy, y_true.npy, config.yaml")
    parser.add_argument("--gvf_index", type=int, default=4,
                        help="Column index for GVF in the prediction array (0-indexed, default=4)")
    parser.add_argument("--ranges", type=str, default="0.0,0.10,0.25,0.50,0.75,1.0",
                        help="Comma-separated bin edges for range evaluation")
    parser.add_argument("--meas_uncertainty", type=float, default=0.03,
                        help="Absolute measurement uncertainty band (+/- value, default=0.03)")
    parser.add_argument("--rel_uncertainty", type=float, default=None,
                        help="Relative measurement uncertainty (e.g. 0.05 for +/-5%%)")
    parser.add_argument("--output_subdir", type=str, default="evaluation_gvf_range",
                        help="Sub-directory name for outputs (created inside run_dir)")
    parser.add_argument("--svf_subtraction", action="store_true",
                        help="Evaluate SVF as 1-GVF-LVF (subtraction from gas & liquid) "
                             "instead of the default GVF evaluation. Also compares with "
                             "the direct SVF model output (idx 3).")
    parser.add_argument("--svf_threshold", type=float, default=None,
                        help="If set, clamp subtracted SVF predictions <= threshold to 0 "
                             "(mirrors main.py logic, e.g. 0.03)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / args.output_subdir
    os.makedirs(out_dir, exist_ok=True)

    # Parse range edges
    ranges = [float(x) for x in args.ranges.split(",")]
    assert len(ranges) >= 2, "Need at least 2 range edges"

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    y_pred_all = np.load(str(run_dir / "y_pred.npy")).astype(np.float64)
    y_true_all = np.load(str(run_dir / "y_true.npy")).astype(np.float64)
    print("Loaded y_pred %s  y_true %s" % (y_pred_all.shape, y_true_all.shape))

    # ------------------------------------------------------------------
    # Select target: GVF (default) or SVF via subtraction
    # ------------------------------------------------------------------
    global TARGET_LABEL
    if args.svf_subtraction:
        # SVF = 1 - GVF(idx4) - LVF(idx5)
        y_true = 1.0 - y_true_all[:, 4] - y_true_all[:, 5]
        y_pred = 1.0 - y_pred_all[:, 4] - y_pred_all[:, 5]
        if args.svf_threshold is not None:
            y_pred = np.where(y_pred <= args.svf_threshold, 0.0, y_pred)
            print("SVF threshold applied: pred <= %.3f set to 0" % args.svf_threshold)
        target_label = "SVF (1-GVF-LVF)"
        TARGET_LABEL = "SVF"
        # Also load direct SVF for comparison
        svf_direct_true = y_true_all[:, 3]
        svf_direct_pred = y_pred_all[:, 3]
        print("Mode: SVF subtraction (1 - GVF - LVF)")
        print("SVF subtracted:  true [%.4f, %.4f]  pred [%.4f, %.4f]  n=%d"
              % (y_true.min(), y_true.max(), y_pred.min(), y_pred.max(),
                 len(y_true)))
        print("SVF direct(idx3):true [%.4f, %.4f]  pred [%.4f, %.4f]"
              % (svf_direct_true.min(), svf_direct_true.max(),
                 svf_direct_pred.min(), svf_direct_pred.max()))
        print("  Direct   RMSE=%.4f  MAE=%.4f  Bias=%+.4f"
              % (rmse(svf_direct_true, svf_direct_pred),
                 mae(svf_direct_true, svf_direct_pred),
                 bias(svf_direct_true, svf_direct_pred)))
        print("  Subtract RMSE=%.4f  MAE=%.4f  Bias=%+.4f"
              % (rmse(y_true, y_pred), mae(y_true, y_pred),
                 bias(y_true, y_pred)))
        if args.output_subdir == "evaluation_gvf_range":
            out_dir = run_dir / "evaluation_svf_subtraction"
            os.makedirs(out_dir, exist_ok=True)
    else:
        gvf_idx = args.gvf_index
        y_true = y_true_all[:, gvf_idx]
        y_pred = y_pred_all[:, gvf_idx]
        target_label = "GVF"
        svf_direct_true = None
        svf_direct_pred = None
        print("GVF (target %d):  true [%.4f, %.4f]  pred [%.4f, %.4f]  n=%d"
              % (gvf_idx, y_true.min(), y_true.max(), y_pred.min(), y_pred.max(),
                 len(y_true)))
    print()

    # ------------------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------------------
    overall = dict(
        rmse=rmse(y_true, y_pred), mae=mae(y_true, y_pred),
        r2=r2(y_true, y_pred), bias=bias(y_true, y_pred),
        rel_err=relative_error(y_true, y_pred), n=len(y_true))

    # ------------------------------------------------------------------
    # Focus range: 0.50 - 1.00
    # ------------------------------------------------------------------
    focus_mask = (y_true >= 0.50) & (y_true <= 1.00)
    yt_f, yp_f = y_true[focus_mask], y_pred[focus_mask]
    nf = int(focus_mask.sum())
    focus = dict(
        rmse=rmse(yt_f, yp_f), mae=mae(yt_f, yp_f),
        r2=r2(yt_f, yp_f) if nf >= 2 else np.nan,
        bias=bias(yt_f, yp_f),
        rel_err=relative_error(yt_f, yp_f), n=nf)

    # ------------------------------------------------------------------
    # Low range: 0.00 - 0.20
    # ------------------------------------------------------------------
    low_mask = (y_true >= 0.00) & (y_true <= 0.20)
    yt_lo, yp_lo = y_true[low_mask], y_pred[low_mask]
    nl = int(low_mask.sum())
    low = dict(
        rmse=rmse(yt_lo, yp_lo), mae=mae(yt_lo, yp_lo),
        r2=r2(yt_lo, yp_lo) if nl >= 2 else np.nan,
        bias=bias(yt_lo, yp_lo),
        rel_err=relative_error(yt_lo, yp_lo), n=nl)

    # ------------------------------------------------------------------
    # Range-based metrics
    # ------------------------------------------------------------------
    range_metrics = compute_range_metrics(y_true, y_pred, ranges)

    # ------------------------------------------------------------------
    # +/- Measurement uncertainty evaluation
    # ------------------------------------------------------------------
    abs_unc = args.meas_uncertainty
    rel_unc = args.rel_uncertainty
    if abs_unc is not None and rel_unc is not None:
        meas_label = "%.3f / %.0f%%" % (abs_unc, rel_unc * 100)
    elif abs_unc is not None:
        meas_label = "%.3f (abs)" % abs_unc
    else:
        meas_label = "%.0f%% (rel)" % (rel_unc * 100)

    unc_result = evaluate_within_uncertainty(y_true, y_pred,
                                             abs_unc=abs_unc,
                                             rel_unc=rel_unc)

    # Also compute uncertainty for the focus range and low range
    unc_focus = evaluate_within_uncertainty(yt_f, yp_f,
                                            abs_unc=abs_unc,
                                            rel_unc=rel_unc)
    unc_low = evaluate_within_uncertainty(yt_lo, yp_lo,
                                          abs_unc=abs_unc,
                                          rel_unc=rel_unc)

    # ------------------------------------------------------------------
    # Print summary to console
    # ------------------------------------------------------------------
    print("=" * 60)
    print("OVERALL  RMSE=%.4f  MAE=%.4f  R2=%.4f  Bias=%+.4f"
          % (overall["rmse"], overall["mae"], overall["r2"], overall["bias"]))
    print("FOCUS [0.50,1.00]  n=%d  RMSE=%.4f  MAE=%.4f  R2=%.4f  Bias=%+.4f"
          % (nf, focus["rmse"], focus["mae"], focus["r2"], focus["bias"]))
    print("LOW   [0.00,0.20]  n=%d  RMSE=%.4f  MAE=%.4f  R2=%.4f  Bias=%+.4f"
          % (nl, low["rmse"], low["mae"], low["r2"], low["bias"]))
    print("+/- Uncertainty (%s): %d/%d within band (%.1f%%)"
          % (meas_label, unc_result["n_within"], unc_result["n_total"],
             unc_result["pct_within"]))
    print("+/- Uncertainty (focus): %d/%d within band (%.1f%%)"
          % (unc_focus["n_within"], unc_focus["n_total"],
             unc_focus["pct_within"]))
    print("+/- Uncertainty (low):   %d/%d within band (%.1f%%)"
          % (unc_low["n_within"], unc_low["n_total"],
             unc_low["pct_within"]))
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    print("Generating plots...")

    # 1) Scatter with uncertainty – all data
    plot_scatter_with_uncertainty(
        y_true, y_pred,
        unc_result["half_width"], unc_result["within_mask"],
        title="%s Prediction vs Ground Truth (All Data)" % target_label,
        out_path=str(out_dir / "gvf_scatter_all.png"),
        meas_unc_label=meas_label,
        range_mask=focus_mask,
        range_label="Focus: [0.50, 1.00]")

    # 2) Scatter – focus range only (axis range 0.40-1.00 for context)
    plot_scatter_with_uncertainty(
        yt_f, yp_f,
        unc_focus["half_width"], unc_focus["within_mask"],
        title="%s Prediction vs Ground Truth (%s 0.50-1.00)" % (target_label, target_label),
        out_path=str(out_dir / "gvf_scatter_focus.png"),
        meas_unc_label=meas_label,
        plot_range=(0.40, 1.00),
        uniform_style=True)

    # 3) Scatter – low range only (axis range -0.15 to 0.25 for context)
    plot_scatter_with_uncertainty(
        yt_lo, yp_lo,
        unc_low["half_width"], unc_low["within_mask"],
        title="%s Prediction vs Ground Truth (%s 0.00-0.20)" % (target_label, target_label),
        out_path=str(out_dir / "gvf_scatter_low.png"),
        meas_unc_label=meas_label,
        plot_range=(-0.15, 0.25),
        uniform_style=True)

    # 4) RMSE bar chart by range
    plot_range_bar(range_metrics, str(out_dir / "gvf_rmse_by_range.png"),
                   metric_key="rmse", ylabel="RMSE")

    # 4) MAE bar chart by range
    plot_range_bar(range_metrics, str(out_dir / "gvf_mae_by_range.png"),
                   metric_key="mae", ylabel="MAE")

    # 5) Residuals coloured by range
    plot_residuals_by_range(y_true, y_pred, ranges,
                            str(out_dir / "gvf_residuals_by_range.png"))

    # 6) Combined summary figure
    plot_combined_summary(y_true, y_pred, range_metrics, unc_result,
                          overall, focus, str(out_dir / "gvf_summary.png"),
                          meas_label)

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------
    print("\nWriting text report...")
    write_report(y_true, y_pred, range_metrics, unc_result,
                 overall, focus, low,
                 str(out_dir / "gvf_evaluation_report.txt"),
                 meas_label)

    # ------------------------------------------------------------------
    # JSON report (machine-readable)
    # ------------------------------------------------------------------
    json_data = {
        "overall": {k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in overall.items()},
        "focus_0.50_1.00": {k: float(v) if isinstance(v, (float, np.floating)) else v
                            for k, v in focus.items()},
        "low_0.00_0.20": {k: float(v) if isinstance(v, (float, np.floating)) else v
                           for k, v in low.items()},
        "per_range": [{k: (float(v) if isinstance(v, (float, np.floating)) else v)
                       for k, v in rm.items()} for rm in range_metrics],
        "uncertainty": {
            "label": meas_label,
            "abs_unc": abs_unc,
            "rel_unc": rel_unc,
            "n_total": unc_result["n_total"],
            "n_within": unc_result["n_within"],
            "pct_within": unc_result["pct_within"],
        },
    }
    json_path = str(out_dir / "gvf_evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print("  Saved: %s" % json_path)

    print("\nDone. All outputs in: %s" % out_dir)


if __name__ == "__main__":
    main()
