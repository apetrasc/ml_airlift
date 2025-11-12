# ターゲット変数の正規化実装

## 問題の背景

ターゲット変数のスケールが異なる場合（例: t1-t3が0-100、t4-t6が0-1）、MSE損失ではスケールの大きいターゲットの誤差が支配的になり、スケールの小さいターゲットの学習がうまくいきません。

### 実際のデータでのスケール差

```
Target 0 (Solid Velocity):     範囲 2.84   → 10%誤差時のMSE: 0.081
Target 1 (Gas Velocity):        範囲 65.88  → 10%誤差時のMSE: 43.396
Target 2 (Liquid Velocity):     範囲 3.60   → 10%誤差時のMSE: 0.130
Target 3 (Solid Volume Fraction): 範囲 0.22  → 10%誤差時のMSE: 0.0005
Target 4 (Gas Volume Fraction):   範囲 0.92  → 10%誤差時のMSE: 0.008
Target 5 (Liquid Volume Fraction):範囲 0.92  → 10%誤差時のMSE: 0.009
```

**最大スケール比**: 約71倍（Gas Velocity vs Solid Volume Fraction）
**MSEの違い**: 同じ相対誤差でも、Gas VelocityのMSEがSolid Volume Fractionの約85,000倍大きい

## 実装内容

### 1. ターゲット変数の正規化

`optuna_tutorial.py`の`objective`関数内で、以下の処理を実装：

1. **データの分割**: 固定シードでデータを学習・検証・テストセットに分割
2. **スケーラーの学習**: 学習データのみで`StandardScaler`を学習（データリークを防止）
3. **正規化の適用**: 学習・検証・テストデータにスケーラーを適用
4. **データセットの作成**: 正規化されたターゲットで`TensorDataset`を作成
5. **スケーラーの保存**: 各トライアルの出力ディレクトリに`target_scaler.joblib`として保存

### 2. 学習時の処理

- **モデルの出力**: 正規化されたスケールで予測
- **損失関数**: 正規化されたスケールでMSE損失を計算
- **検証損失**: 正規化されたスケールで計算（Optunaの最適化目標として使用）

### 3. 評価時の処理

- **予測値の逆変換**: テストセットの予測値を元のスケールに逆変換
- **評価指標**: 元のスケールでR²、MSE、MAE、RMSEを計算
- **評価プロット**: 元のスケールでプロットを生成

## コードの流れ

```python
# 1. データの読み込み
x, t = load_npz_pair(...)

# 2. データの分割（固定シード）
indices = np.arange(n_samples)
rng = np.random.RandomState(seed)
rng.shuffle(indices)
train_idx, val_idx, test_idx = split_indices(...)

# 3. スケーラーの学習（学習データのみ）
target_scaler = StandardScaler()
t_train_scaled = target_scaler.fit_transform(t[train_idx])

# 4. 正規化の適用
t_val_scaled = target_scaler.transform(t[val_idx])
t_test_scaled = target_scaler.transform(t[test_idx])

# 5. データセットの作成
t_normalized = concatenate([t_train_scaled, t_val_scaled, t_test_scaled])
dataset = to_tensor_dataset(x, t_normalized, device)

# 6. 学習（正規化されたスケールで）
pred_scaled = model(x)
loss = MSE(pred_scaled, t_scaled)

# 7. 評価（元のスケールに逆変換）
pred_original = target_scaler.inverse_transform(pred_scaled)
r2 = r2_score(t_original, pred_original)
```

## 重要なポイント

### データリークの防止

- スケーラーは**学習データのみ**で学習（`fit_transform`）
- 検証・テストデータには**学習済みスケーラーを適用**（`transform`のみ）
- これにより、検証・テストデータの情報が学習に漏れない

### 検証損失の解釈

- 検証損失は**正規化されたスケール**で計算される
- Optunaの最適化目標として使用されるため、比較可能
- 実際の性能評価（R²、MSE、MAE）は**元のスケール**で計算

### スケーラーの保存

- 各トライアルの出力ディレクトリに`target_scaler.joblib`として保存
- 予測時や推論時にスケーラーを読み込んで使用可能

## Optunaでの最適化

### ハイパーパラメータ

- `training.normalize_targets`: `True` or `False`
  - `True`: ターゲット変数を正規化
  - `False`: 正規化しない（元のスケールで学習）

### 最適化の戦略

1. **正規化あり + 重み付け損失なし**: すべてのターゲットが同じスケールになるため、均等に学習される
2. **正規化あり + 重み付け損失あり**: 正規化後でも、性能の悪いターゲットに高い重みを付けることが可能
3. **正規化なし + 重み付け損失あり**: スケールの違いを重みで補償（ただし、完全には補償できない可能性がある）

## 期待される効果

### 正規化ありの場合

- すべてのターゲットが同じスケール（平均0、標準偏差1）になる
- MSE損失が公平に機能する
- スケールの小さいターゲット（t4-t6）の学習が改善される可能性が高い
- 学習が安定する

### 正規化なしの場合

- スケールの大きいターゲット（t1-t3）の誤差が支配的
- スケールの小さいターゲット（t4-t6）の学習が困難
- 重み付け損失で補償する必要があるが、完全には補償できない可能性がある

## 使用方法

### 設定ファイルでの指定

```yaml
training:
  normalize_targets: true  # または false
```

### Optunaでの自動選択

Optunaが`normalize_targets`を`True`/`False`から自動的に選択します。

## まとめ

ターゲット変数のスケールが異なる場合、正規化することで学習の公平性と安定性が向上します。特に、スケールの小さいターゲット（Volume Fraction系）の予測性能が大幅に改善される可能性が高いです。

