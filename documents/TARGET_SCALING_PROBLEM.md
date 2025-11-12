# ターゲット変数のスケール違いによる学習の問題

## 問題の概要

複数ターゲット回帰において、ターゲット変数のスケールが異なる場合、MSE損失で学習するとスケールの大きいターゲットの誤差が支配的になり、スケールの小さいターゲットの学習がうまくいかない問題が発生します。

## 具体例

### ターゲット変数のスケール

- **t1, t2, t3（Velocity系）**: 0～100の範囲
- **t4, t5, t6（Volume Fraction系）**: 0～1の範囲

### 問題の数値例

同じ相対誤差（範囲の10%）でも、MSE損失が大きく異なります：

```python
# ターゲット1（範囲100）の場合
t1_range = 100
t1_error = 10  # 範囲の10%誤差
t1_mse = 10 ** 2 = 100.0

# ターゲット4（範囲1.0）の場合
t4_range = 1.0
t4_error = 0.1  # 範囲の10%誤差
t4_mse = 0.1 ** 2 = 0.01

# スケール比
scale_ratio = t1_mse / t4_mse = 100.0 / 0.01 = 10,000倍
```

**結果**: ターゲット1のMSEがターゲット4の10,000倍大きくなるため、学習がターゲット1に集中し、ターゲット4の予測が改善されない。

## 現在のコードの問題点

### 現在のWeightedMSELoss

```python
class WeightedMSELoss(nn.Module):
    def forward(self, pred, target):
        mse_per_target = (pred - target) ** 2  # (batch_size, n_targets)
        # 重みを掛けているが、スケールの違いは考慮されていない
        if self.weights is not None:
            mse_per_target = mse_per_target * self.weights.unsqueeze(0)
        return mse_per_target.mean()
```

**問題**: スケールの違いを考慮せずにMSEを計算しているため、スケールの大きいターゲットの誤差が支配的になる。

## 解決策

### 解決策1: ターゲット変数の正規化（推奨）

学習時は正規化したターゲットで学習し、予測時は逆変換する。

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import torch

# 学習データでスケーラーを学習
scaler = StandardScaler()
t_train_scaled = scaler.fit_transform(t_train)

# 学習時は正規化したターゲットを使用
# モデルの出力も正規化されたスケール
pred_scaled = model(x_train)
loss = nn.MSELoss()(pred_scaled, t_train_scaled)

# 予測時は逆変換
pred_original = scaler.inverse_transform(pred_scaled.detach().cpu().numpy())
```

**メリット**:
- すべてのターゲットが同じスケール（平均0、標準偏差1）になる
- MSE損失が公平に機能する
- 学習が安定する

**デメリット**:
- スケーラーを保存・読み込みする必要がある
- 予測時に逆変換が必要

### 解決策2: スケールに応じた重み付け

各ターゲットの範囲に応じて重みを自動調整する。

```python
class ScaleAwareWeightedMSELoss(nn.Module):
    """スケールを考慮した重み付けMSE損失"""
    def __init__(self, target_ranges=None, base_weights=None):
        """
        Args:
            target_ranges: 各ターゲットの範囲 (max - min) のリスト
            base_weights: ベースとなる重み（オプション）
        """
        super().__init__()
        if target_ranges is not None:
            # 範囲の逆数を重みとして使用（範囲が小さいほど重みが大きい）
            # 正規化: すべての重みの合計がn_targetsになるように
            inverse_ranges = 1.0 / np.array(target_ranges)
            self.weights = torch.tensor(inverse_ranges / inverse_ranges.mean())
        else:
            self.weights = None
        
        if base_weights is not None:
            # ベース重みとスケール重みを組み合わせ
            self.weights = self.weights * torch.tensor(base_weights)
    
    def forward(self, pred, target):
        mse_per_target = (pred - target) ** 2  # (batch_size, n_targets)
        
        if self.weights is not None:
            weights = self.weights.to(pred.device)
            mse_per_target = mse_per_target * weights.unsqueeze(0)
        
        return mse_per_target.mean()
```

**使用例**:
```python
# ターゲットの範囲を計算（学習データから）
target_ranges = [
    t_train[:, 0].max() - t_train[:, 0].min(),  # t1の範囲
    t_train[:, 1].max() - t_train[:, 1].min(),  # t2の範囲
    # ... など
]

criterion = ScaleAwareWeightedMSELoss(target_ranges=target_ranges)
```

### 解決策3: 相対誤差ベースの損失関数

各ターゲットの範囲で正規化した相対誤差を使用する。

```python
class RelativeMSELoss(nn.Module):
    """相対誤差ベースのMSE損失"""
    def __init__(self, target_ranges):
        """
        Args:
            target_ranges: 各ターゲットの範囲 (max - min) のリスト
        """
        super().__init__()
        self.target_ranges = torch.tensor(target_ranges)
    
    def forward(self, pred, target):
        # 範囲で正規化
        ranges = self.target_ranges.to(pred.device).unsqueeze(0)
        pred_normalized = pred / ranges
        target_normalized = target / ranges
        
        # 正規化されたMSE
        mse_per_target = (pred_normalized - target_normalized) ** 2
        return mse_per_target.mean()
```

### 解決策4: ターゲットごとの正規化MSE損失

各ターゲットのMSEを個別に正規化してから平均を取る。

```python
class NormalizedPerTargetMSELoss(nn.Modoss):
    """ターゲットごとに正規化したMSE損失"""
    def __init__(self, target_stds=None):
        """
        Args:
            target_stds: 各ターゲットの標準偏差のリスト
        """
        super().__init__()
        if target_stds is not None:
            self.target_stds = torch.tensor(target_stds)
        else:
            self.target_stds = None
    
    def forward(self, pred, target):
        mse_per_target = (pred - target) ** 2  # (batch_size, n_targets)
        
        if self.target_stds is not None:
            # 各ターゲットの標準偏差で正規化
            stds = self.target_stds.to(pred.device).unsqueeze(0)
            mse_per_target = mse_per_target / (stds ** 2)
        
        # ターゲットごとの平均MSEを計算
        mse_per_target_mean = mse_per_target.mean(dim=0)  # (n_targets,)
        
        # すべてのターゲットの平均
        return mse_per_target_mean.mean()
```

## 推奨される実装

### 1. ターゲット変数の正規化（最推奨）

学習データでStandardScalerを学習し、予測時に逆変換する。

```python
# 学習時
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
t_train_scaled = scaler.fit_transform(t_train)
t_val_scaled = scaler.transform(t_val)
t_test_scaled = scaler.transform(t_test)

# モデルは正規化されたターゲットを予測
pred_scaled = model(x)
loss = nn.MSELoss()(pred_scaled, t_scaled)

# 評価時は逆変換
pred_original = scaler.inverse_transform(pred_scaled.detach().cpu().numpy())
```

### 2. スケーラーを保存・読み込み

```python
import joblib

# 学習後にスケーラーを保存
joblib.dump(scaler, 'target_scaler.joblib')

# 予測時にスケーラーを読み込み
scaler = joblib.load('target_scaler.joblib')
pred_original = scaler.inverse_transform(pred_scaled)
```

## 実装の優先順位

1. **最優先**: ターゲット変数の正規化（StandardScaler）
2. **高優先度**: スケールに応じた重み付け（正規化が難しい場合）
3. **中優先度**: 相対誤差ベースの損失関数
4. **低優先度**: ターゲットごとの正規化MSE損失

## まとめ

ターゲット変数のスケールが異なる場合、MSE損失ではスケールの大きいターゲットの誤差が支配的になり、スケールの小さいターゲットの学習がうまくいきません。

**最も効果的な解決策は、ターゲット変数を正規化することです。** これにより、すべてのターゲットが同じスケールになり、MSE損失が公平に機能するようになります。

