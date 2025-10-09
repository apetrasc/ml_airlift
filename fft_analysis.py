import polars as pl
from src import preprocess_and_predict,preprocess_liquidonly
from models import SimpleCNN
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
model = SimpleCNN(config['hyperparameters']['input_length']).to(config['evaluation']['device'])
# You can use the argparse library to accept a command-line argument for base_dir (datetime).


parser = argparse.ArgumentParser(description="Run evaluation with specified base directory (datetime).")
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
    if os.path.exists(file_path) and row["気相体積率"]==0:
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
