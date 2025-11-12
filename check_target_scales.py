#!/usr/bin/env python3
"""Check target variable scales and demonstrate the scaling problem."""

import numpy as np
from omegaconf import OmegaConf
from train_real import load_npz_pair

def main():
    cfg = OmegaConf.load('config/config_real_updated.yaml')
    _, t = load_npz_pair(cfg.dataset.t_train, cfg.dataset.t_train, cfg.dataset.x_key, cfg.dataset.t_key)
    
    print("=" * 70)
    print("ターゲット変数の統計情報")
    print("=" * 70)
    target_names = [
        "Solid Velocity (t1)",
        "Gas Velocity (t2)", 
        "Liquid Velocity (t3)",
        "Solid Volume Fraction (t4)",
        "Gas Volume Fraction (t5)",
        "Liquid Volume Fraction (t6)"
    ]
    
    print(f"\nデータ形状: {t.shape}")
    print(f"サンプル数: {t.shape[0]}\n")
    
    scales = []
    for i in range(min(6, t.shape[1])):
        t_i = t[:, i]
        t_range = t_i.max() - t_i.min()
        scales.append(t_range)
        
        print(f"{target_names[i]} (Target {i}):")
        print(f"  Min: {t_i.min():.6f}")
        print(f"  Max: {t_i.max():.6f}")
        print(f"  Mean: {t_i.mean():.6f}")
        print(f"  Std: {t_i.std():.6f}")
        print(f"  Range: {t_range:.6f}")
        
        # 範囲の10%誤差の場合のMSE
        range_10pct_error = t_range * 0.1
        mse_example = range_10pct_error ** 2
        print(f"  範囲10%誤差時のMSE: {mse_example:.6f}")
        print()
    
    print("=" * 70)
    print("問題の説明: スケールの違いによるMSE損失の偏り")
    print("=" * 70)
    
    # スケール比を計算
    scale_ratio = max(scales[:3]) / max(scales[3:]) if max(scales[3:]) > 0 else 1
    print(f"\n最大スケール比 (t1-t3 vs t4-t6): {scale_ratio:.2f}倍")
    
    print("\n【問題点】")
    print("同じ相対誤差（例: 範囲の10%）でも、スケールが大きいターゲットのMSEが")
    print("圧倒的に大きくなるため、学習がスケールの大きいターゲットに集中してしまう。")
    print("\n【例】")
    print("  t1 (範囲100): 10%誤差 → MSE = 100.0")
    print("  t4 (範囲1.0): 10%誤差 → MSE = 0.01")
    print("  → t1のMSEが10000倍大きい！")
    print("\n【解決策】")
    print("1. ターゲット変数の正規化（StandardScaler/MinMaxScaler）")
    print("2. スケールに応じた重み付け（範囲の逆数など）")
    print("3. 相対誤差ベースの損失関数（MAE/範囲）")

if __name__ == "__main__":
    main()

