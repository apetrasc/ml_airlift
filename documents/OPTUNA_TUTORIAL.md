# Optuna Tutorial: ハイパーパラメータ最適化ガイド

このドキュメントでは、`optuna_tutorial.py`を使用したハイパーパラメータ最適化の実行方法と結果の確認方法を説明します。

## 目次

1. [前提条件](#前提条件)
2. [実行方法](#実行方法)
3. [結果の確認方法](#結果の確認方法)
4. [ディレクトリ構造](#ディレクトリ構造)
5. [よくある質問](#よくある質問)

---

## 前提条件

### 必要なパッケージ

```bash
pip install optuna
pip install optuna[visualization]  # 可視化機能（オプション）
```

### 必要なファイル

- `train_real.py`: 学習関数が定義されている
- `config/config_real_updated.yaml`: ベース設定ファイル
- データファイル:
  - `/home/smatsubara/documents/airlift/data/experiments/cleaned_data/x_train_real_cleaned.npy`
  - `/home/smatsubara/documents/airlift/data/experiments/cleaned_data/t_train_real_cleaned.npy`

---

## 実行方法

### 1. 基本的な実行

```bash
cd /home/smatsubara/documents/sandbox/ml_airlift
python optuna_tutorial.py
```

### 2. 実行中の確認

実行中は以下のような出力が表示されます：

```
[INFO] Study database: /home/smatsubara/documents/airlift/data/sandbox/optuna/study.db
[INFO] Number of existing trials: 0

[INFO] Starting optimization...
[INFO] Base config: /home/smatsubara/documents/sandbox/ml_airlift/config/config_real_updated.yaml

============================================================
Trial 0 started
Output directory: /home/smatsubara/documents/airlift/data/outputs_real/2025-11-04/15-30-00
Hyperparameters: {'model.hidden': 64, 'model.dropout_rate': 0.2, ...}
============================================================

[STEP] Loading dataset files...
[OK] Loaded. x.shape=(108, 4, 1400, 2500), t.shape=(108, 6)
[STEP] Build dataset tensors...
...
```

### 3. 実行パラメータの調整

`optuna_tutorial.py`内の`main()`関数で調整可能：

```python
# 最適化トライアル数
study.optimize(..., n_trials=20)  # デフォルト: 20

# 並列実行数（GPU使用時は1を推奨）
study.optimize(..., n_jobs=1)  # デフォルト: 1
```

### 4. 中断と再開

OptunaはSQLiteデータベースに結果を保存するため、実行を中断しても再開できます：

```bash
# 中断（Ctrl+C）
# 再実行（前回の結果から続行）
python optuna_tutorial.py
```

---

## 結果の確認方法

### 1. ディレクトリ構造の確認

#### A. Optuna関連ファイル

```bash
# Optunaディレクトリの確認
ls -lh /home/smatsubara/documents/airlift/data/sandbox/optuna/
```

**ファイル一覧:**
- `study.db`: Optuna SQLiteデータベース（全トライアル情報）
- `study_summary.json`: 全トライアルのサマリー
- `best_trial_info.yaml`: ベストトライアルの詳細情報

#### B. 各トライアルの結果

```bash
# 出力ディレクトリの確認
ls -lh /home/smatsubara/documents/airlift/data/outputs_real/
```

**構造:**
```
outputs_real/
├── 2025-11-04/
│   ├── 15-30-00/          # Trial 0
│   │   ├── config.yaml
│   │   ├── trial_info.yaml
│   │   ├── metrics.json
│   │   ├── weights/
│   │   │   └── model_simplecnn_real.pth
│   │   ├── logs/
│   │   │   ├── learning_curve.png
│   │   │   └── learning_curve_log.png
│   │   └── evaluation_plots/
│   │       └── ...
│   ├── 15-35-12/          # Trial 1
│   │   └── ...
│   └── ...
└── optuna_best -> 2025-11-04/15-35-12/  # ベストトライアルへのシンボリックリンク
```

### 2. ベストトライアルの確認

#### A. シンボリックリンクを使用

```bash
# ベストトライアルのディレクトリに移動
cd /home/smatsubara/documents/airlift/data/outputs_real/optuna_best

# 設定ファイルの確認
cat config.yaml

# メトリクスの確認
cat metrics.json | python -m json.tool
```

#### B. ベストトライアル情報ファイル

```bash
# ベストトライアル情報の確認
cat /home/smatsubara/documents/airlift/data/sandbox/optuna/best_trial_info.yaml
```

### 3. Studyサマリーの確認

#### A. JSONファイルを直接確認

```bash
# Studyサマリーの確認
cat /home/smatsubara/documents/airlift/data/sandbox/optuna/study_summary.json | python -m json.tool
```

#### B. Pythonスクリプトで確認

```python
import optuna
import json

# Studyを読み込み
storage = optuna.storages.RDBStorage(
    url='sqlite:////home/smatsubara/documents/airlift/data/sandbox/optuna/study.db'
)
study = optuna.load_study(
    study_name='cnn_hyperparameter_optimization',
    storage=storage
)

# ベストトライアル情報
best_trial = study.best_trial
print(f"Best Trial: #{best_trial.number}")
print(f"Best Validation Loss: {best_trial.value:.6f}")
print(f"Test MSE: {best_trial.user_attrs.get('test_mse', 'N/A')}")
print(f"Test MAE: {best_trial.user_attrs.get('test_mae', 'N/A')}")
print(f"\nBest Parameters:")
for key, value in best_trial.params.items():
    print(f"  {key}: {value}")
```

### 4. 全トライアルの比較

#### A. メトリクス比較

```python
import json
import pandas as pd

# Studyサマリーを読み込み
with open('/home/smatsubara/documents/airlift/data/sandbox/optuna/study_summary.json', 'r') as f:
    summary = json.load(f)

# DataFrameに変換
trials_data = []
for trial in summary['trials_summary']:
    row = {
        'trial_number': trial['number'],
        'validation_loss': trial['value'],
        'test_mse': trial['user_attrs'].get('test_mse', None),
        'test_mae': trial['user_attrs'].get('test_mae', None),
        **trial['params']
    }
    trials_data.append(row)

df = pd.DataFrame(trials_data)
print(df.sort_values('validation_loss').head(10))  # 上位10トライアル
```

### 5. 可視化（オプション）

#### A. Optuna Dashboardの使用

```bash
# Optuna Dashboardを起動
optuna-dashboard sqlite:////home/smatsubara/documents/airlift/data/sandbox/optuna/study.db

# ブラウザで http://localhost:8080 にアクセス
```

#### B. Pythonで可視化

```python
import optuna
import optuna.visualization as vis

# Studyを読み込み
storage = optuna.storages.RDBStorage(
    url='sqlite:////home/smatsubara/documents/airlift/data/sandbox/optuna/study.db'
)
study = optuna.load_study(
    study_name='cnn_hyperparameter_optimization',
    storage=storage
)

# 最適化履歴
fig = vis.plot_optimization_history(study)
fig.show()

# パラメータ重要度
fig = vis.plot_param_importances(study)
fig.show()

# パラレル座標
fig = vis.plot_parallel_coordinate(study)
fig.show()
```

### 6. 特定のトライアルの詳細確認

```python
import os
from omegaconf import OmegaConf
import json

# トライアル番号を指定
trial_number = 5

# トライアルディレクトリを検索
outputs_root = "/home/smatsubara/documents/airlift/data/outputs_real"
for root, dirs, files in os.walk(outputs_root):
    if 'trial_info.yaml' in files:
        trial_info_path = os.path.join(root, 'trial_info.yaml')
        trial_info = OmegaConf.load(trial_info_path)
        if trial_info.get('trial_number') == trial_number:
            trial_dir = root
            break

# 設定ファイルの確認
config = OmegaConf.load(os.path.join(trial_dir, 'config.yaml'))
print("Configuration:")
print(OmegaConf.to_yaml(config))

# メトリクスの確認
with open(os.path.join(trial_dir, 'metrics.json'), 'r') as f:
    metrics = json.load(f)
print("\nMetrics:")
print(json.dumps(metrics, indent=2))

# 学習曲線の確認
print(f"\nLearning curve: {os.path.join(trial_dir, 'logs/learning_curve.png')}")
```

---

## ディレクトリ構造

### 完全なディレクトリ構造

```
/home/smatsubara/documents/airlift/data/
├── sandbox/
│   └── optuna/                          # Optuna関連ファイル
│       ├── study.db                     # SQLiteデータベース
│       ├── study_summary.json           # 全トライアルサマリー
│       └── best_trial_info.yaml         # ベストトライアル情報
│
└── outputs_real/                         # 各トライアルの実行結果
    ├── 2025-11-04/
    │   ├── 15-30-00/                    # Trial 0
    │   │   ├── config.yaml              # このトライアルの設定
    │   │   ├── trial_info.yaml           # トライアル情報
    │   │   ├── metrics.json             # メトリクス
    │   │   ├── y_pred.npy               # 予測結果
    │   │   ├── y_true.npy               # 正解データ
    │   │   ├── weights/
    │   │   │   └── model_simplecnn_real.pth
    │   │   ├── logs/
    │   │   │   ├── learning_curve.png
    │   │   │   └── learning_curve_log.png
    │   │   └── evaluation_plots/
    │   │       ├── overview_prediction_plots.png
    │   │       ├── target_01_prediction_plot.png
    │   │       └── ...
    │   ├── 15-35-12/                    # Trial 1
    │   │   └── ...
    │   └── ...
    ├── optuna_best -> 2025-11-04/15-35-12/  # ベストトライアルへのリンク
    └── optuna_study_summary.json        # Studyサマリー（コピー）
```

---

## よくある質問

### Q1: 実行中に中断した場合、再開できますか？

**A:** はい。OptunaはSQLiteデータベースに結果を保存するため、同じStudy名で再実行すると続行されます。

### Q2: ベストトライアルのモデルをどう使いますか？

**A:** ベストトライアルのモデルは以下のパスに保存されています：

```python
import torch
from models.cnn import SimpleCNNReal2D

# ベストトライアルの設定とモデルを読み込み
config_path = "/home/smatsubara/documents/airlift/data/outputs_real/optuna_best/config.yaml"
cfg = OmegaConf.load(config_path)

# モデルを作成
model = SimpleCNNReal2D(
    in_channels=cfg.model.in_channels,
    hidden=cfg.model.hidden,
    out_dim=cfg.model.out_dim,
    dropout_rate=cfg.model.dropout_rate,
    use_residual=cfg.model.use_residual
)

# 重みを読み込み
model_path = "/home/smatsubara/documents/airlift/data/outputs_real/optuna_best/weights/model_simplecnn_real.pth"
model.load_state_dict(torch.load(model_path))
```

### Q3: 特定のハイパーパラメータの範囲を変更したい

**A:** `optuna_tutorial.py`の`suggest_hyperparameters()`関数を編集：

```python
def suggest_hyperparameters(trial: Trial, base_cfg: DictConfig) -> DictConfig:
    # 例: hiddenチャネルの範囲を変更
    cfg.model.hidden = trial.suggest_int('model.hidden', 64, 256, step=32)
    # 例: 学習率の範囲を変更
    cfg.training.learning_rate = trial.suggest_float(
        'training.learning_rate', 1e-4, 1e-1, log=True
    )
    # ...
```

### Q4: プルーニングを無効化したい

**A:** `main()`関数でプルーニングをNoneに設定：

```python
study = optuna.create_study(
    study_name=study_name,
    direction='minimize',
    storage=storage,
    pruner=None  # プルーニングを無効化
)
```

### Q5: トライアル数を増やしたい

**A:** `main()`関数の`study.optimize()`を編集：

```python
study.optimize(
    lambda trial: objective(trial, BASE_CONFIG_PATH),
    n_trials=100,  # トライアル数を増やす
    n_jobs=1,
    show_progress_bar=True
)
```

---

## まとめ

1. **実行**: `python optuna_tutorial.py`で最適化を開始
2. **結果確認**: 
   - ベストトライアル: `outputs_real/optuna_best/`
   - 全トライアル情報: `sandbox/optuna/study_summary.json`
3. **詳細分析**: Optuna DashboardやPythonスクリプトで可視化・分析

最適化が完了したら、ベストトライアルの設定とモデルを使用して、本番データで推論を行ってください。

