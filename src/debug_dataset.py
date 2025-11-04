#!/usr/bin/env python3
"""
Debug script to investigate NaN issues in the dataset.
Run this to understand what's causing the training problems.
"""

import sys
import os
sys.path.append('/home/smatsubara/documents/sandbox/ml_airlift')

from src.data_inspector import inspect_dataset

def main():
    print("🔍 Dataset Debugging Tool")
    print("=" * 50)
    
    # Paths to your data files
    x_path = "/home/smatsubara/documents/airlift/data/sandbox/results/x_train_real.npy"
    t_path = "/home/smatsubara/documents/airlift/data/sandbox/results/t_train_real.npy"
    
    # Check if files exist
    if not os.path.exists(x_path):
        print(f"❌ X file not found: {x_path}")
        return
    if not os.path.exists(t_path):
        print(f"❌ T file not found: {t_path}")
        return
    
    print(f"✅ Files found, starting inspection...")
    
    # Run comprehensive inspection
    inspect_dataset(
        x_path=x_path,
        t_path=t_path,
        sample_limit=10,  # Inspect first 10 samples in detail
        save_plots=True,
        output_dir="debug_plots"
    )
    
    print("\n🎯 Next Steps:")
    print("1. Check the generated plots in debug_plots/")
    print("2. Look for NaN/Inf values in the report above")
    print("3. Check data ranges and distributions")
    print("4. Based on findings, we'll fix the data preprocessing")

if __name__ == "__main__":
    main()
