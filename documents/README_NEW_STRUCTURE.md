# ML Airlift Project - New Repository Structure

## ディレクトリ構造

```
ml_airlift/
├── scripts/              # エントリーポイント（実行可能スクリプト）
│   ├── train.py         # 統一された訓練スクリプト
│   ├── evaluate.py      # 評価スクリプト（今後作成）
│   └── optimize.py      # Optuna最適化スクリプト
│
├── src/                  # コアライブラリ
│   ├── models/          # モデル定義
│   ├── data/            # データ処理
│   ├── training/        # 訓練ロジック
│   ├── evaluation/      # 評価機能
│   ├── optimization/    # ハイパーパラメータ最適化
│   ├── utils/           # 汎用ユーティリティ
│   └── visualization/   # 可視化ツール
│
├── config/              # 設定ファイル
│   ├── train_real.yaml
│   ├── train_simulation.yaml
│   └── config_real_updated.yaml
│
├── tools/               # 開発・デバッグツール
│   ├── clear_gpu_memory.py
│   ├── inspect_model.py
│   └── plot_signal_sample.py
│
└── notebooks/           # Jupyterノートブック（オプション）
```

## 使用方法

### 訓練の実行

```bash
# 実データで訓練
python scripts/train.py --config-name=train_real

# シミュレーションデータで訓練
python scripts/train.py --config-name=train_simulation
```

### ハイパーパラメータ最適化

```bash
python scripts/optimize.py
```

### 開発ツール

```bash
# GPUメモリのクリア
python tools/clear_gpu_memory.py

# モデルの検査
python tools/inspect_model.py
```

## 移行ガイド

既存のスクリプトから新しい構造への移行：

- `train_real.py` → `scripts/train.py` (--config-name=train_real)
- `train.py` → `scripts/train.py` (--config-name=train_simulation)
- `optuna_tutorial.py` → `scripts/optimize.py`
- `validate_dataset.py` → `src/data/validation.py`
- `create_dropped_dataset.py` → `src/data/preprocessing.py`

## 注意事項

- 既存のファイルは後方互換性のため残されています
- 新しいコードは新しい構造を使用してください
- インポートパスが変更されているため、既存のコードを更新する必要があります

