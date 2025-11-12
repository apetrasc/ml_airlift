# 再現性ガイド：optuna_bestのハイパーパラメータを使用した訓練の再現

## 概要

`optuna_best`ディレクトリに保存されているハイパーパラメータを使用して訓練を再実行する場合、**ほぼ同じ結果**が得られるはずです。ただし、完全に同じ結果を得るためには、いくつかの重要な条件を満たす必要があります。

## 保存されている情報

`optuna_best`ディレクトリには以下の情報が保存されています：

1. **`config.yaml`**: すべてのハイパーパラメータとシード値
   - 学習率、バッチサイズ、エポック数
   - モデルのハイパーパラメータ（hidden, dropout_rate, use_residualなど）
   - データ分割の比率
   - **シード値（`training.seed: 42`）**

2. **`trial_info.yaml`**: トライアル情報
   - トライアル番号
   - パラメータ
   - バリデーション損失
   - 各ターゲットのR²値

3. **`target_scaler.joblib`**: ターゲット正規化スケーラー（使用されている場合）
   - 学習データで学習されたスケーラー
   - 予測時に使用するための逆変換パラメータ

4. **`weights/`**: 学習済みモデルの重み
   - ベストモデルの重み
   - 最終エポックの重み

## 再現性を確保するための条件

### 1. シードの設定

`scripts/optimize.py`の`objective`関数では、以下のシード設定が行われています：

```python
# Set reproducibility
torch.manual_seed(cfg.training.seed)
np.random.seed(cfg.training.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

**重要**: 同じシード値（`config.yaml`の`training.seed`）を使用する必要があります。

### 2. データ分割

データ分割は`split_dataset`関数で行われ、`torch.Generator().manual_seed(seed)`を使用しています：

```python
g = torch.Generator().manual_seed(seed)
return random_split(dataset, [n_train, n_val, n_test], generator=g)
```

**重要**: 同じシード値を使用すれば、**同じデータ分割**が得られます。

### 3. DataLoaderのシャッフル

学習用のDataLoaderは`shuffle=True`ですが、シードが設定されていれば、同じシードで同じデータ順序になります。

**注意**: DataLoaderに`generator`を明示的に渡すことで、より確実に再現性を確保できます：

```python
generator = torch.Generator()
generator.manual_seed(cfg.training.seed)
train_loader = DataLoader(
    train_set,
    batch_size=cfg.training.batch_size,
    shuffle=True,
    generator=generator,  # 明示的にgeneratorを渡す
    num_workers=0,
    pin_memory=False
)
```

### 4. モデルの初期化

モデルの重み初期化は、PyTorchのデフォルトの初期化方法に依存します。シードが設定されていれば、同じ初期値になるはずです。

**注意**: モデルの初期化も明示的にシードを設定することで、より確実に再現性を確保できます。

### 5. CUDAの決定論性

`torch.backends.cudnn.deterministic = True`が設定されていますが、CUDAの操作によっては完全に決定論的にならない場合があります。

**注意**: 完全な決定論性を確保するには、以下の環境変数を設定する必要がある場合があります：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
```

### 6. ターゲット正規化

`target_scaler.joblib`が保存されている場合、同じスケーラーを使用する必要があります。

**重要**: スケーラーを読み込んで使用する必要があります：

```python
import joblib
target_scaler = joblib.load('optuna_best/target_scaler.joblib')
```

## 再現性が保証されない場合

以下の場合、完全に同じ結果が得られない可能性があります：

1. **CUDAのバージョンが異なる**: CUDAのバージョンによって、浮動小数点演算の結果が微妙に異なる場合があります。

2. **PyTorchのバージョンが異なる**: PyTorchのバージョンによって、初期化方法や演算の結果が異なる場合があります。

3. **ハードウェアが異なる**: GPUが異なる場合、浮動小数点演算の結果が微妙に異なる場合があります。

4. **マルチスレッド/マルチプロセス**: `num_workers > 0`の場合、マルチプロセスによる並列処理がランダム性を導入する可能性があります。

5. **DataLoaderのシャッフル**: DataLoaderのシャッフルが完全に決定論的でない場合があります。

## 再現性を高めるための推奨事項

### 1. シードの明示的な設定

訓練スクリプトの最初に、すべてのシードを明示的に設定します：

```python
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)  # config.yamlのtraining.seedと同じ値
```

### 2. DataLoaderにgeneratorを渡す

DataLoaderに`generator`を明示的に渡します：

```python
generator = torch.Generator()
generator.manual_seed(cfg.training.seed)
train_loader = DataLoader(
    train_set,
    batch_size=cfg.training.batch_size,
    shuffle=True,
    generator=generator,
    num_workers=0,
    pin_memory=False
)
```

### 3. 環境変数の設定

完全な決定論性を確保するために、環境変数を設定します：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
```

### 4. 同じ環境を使用する

可能であれば、同じCUDAバージョン、PyTorchバージョン、ハードウェアを使用します。

## 実際の再現性テスト

`optuna_best`のハイパーパラメータを使用して訓練を再実行する場合、以下の手順を推奨します：

1. **`config.yaml`を読み込む**: `optuna_best/config.yaml`を読み込んで使用します。

2. **シードを設定する**: `config.yaml`の`training.seed`を使用してシードを設定します。

3. **スケーラーを読み込む**: `target_scaler.joblib`が存在する場合、読み込んで使用します。

4. **同じデータを使用する**: 同じデータセットパスを使用します。

5. **同じ前処理を適用する**: チャンネル除外、ダウンサンプリングなど、同じ前処理を適用します。

6. **結果を比較する**: 訓練後の評価指標（R²、MSE、MAEなど）を比較します。

## 結論

**`optuna_best`のハイパーパラメータを使用して訓練を再実行すれば、ほぼ同じ結果が得られるはずです。**

ただし、完全に同じ結果を得るためには：

1. **同じシード値を使用する**（`config.yaml`の`training.seed`）
2. **同じデータセットを使用する**
3. **同じ前処理を適用する**
4. **同じ環境（CUDA、PyTorch、ハードウェア）を使用する**
5. **DataLoaderにgeneratorを明示的に渡す**
6. **環境変数を設定する**（完全な決定論性が必要な場合）

これらの条件を満たせば、**評価指標（R²、MSE、MAEなど）はほぼ同じ値になるはずです**。ただし、浮動小数点演算の微妙な違いにより、完全に同じ値にならない場合もあります。

## 参考

- PyTorchの再現性: https://pytorch.org/docs/stable/notes/randomness.html
- Optunaのベストトライアル: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.best_trial

