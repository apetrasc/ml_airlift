# レポジトリ構造

```
ml_airlift/
├── scripts/              # エントリーポイント（実行可能スクリプト）
│   ├── __init__.py
│   ├── train.py         # 統一された訓練スクリプト
│   └── optimize.py      # Optuna最適化スクリプト
│
├── src/                  # コアライブラリ
│   ├── data/            # データ処理
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   ├── preprocessing.py
│   │   └── validation.py
│   ├── models/          # モデル定義
│   │   ├── __init__.py
│   │   ├── cnn.py
│   │   └── transformers.py
│   ├── training/        # 訓練ロジック
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── evaluation/      # 評価機能
│   │   ├── __init__.py
│   │   └── visualizations.py
│   ├── utils/           # ユーティリティ
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── device.py
│   │   └── memory.py
│   └── [その他のファイル] # 整理が必要
│
├── config/              # 設定ファイル
│   ├── config_real_updated.yaml
│   ├── config_real.yaml
│   └── config.yaml
│
├── tools/               # 開発ツール
│   ├── check_optuna_results.py
│   ├── clear_gpu_memory.py
│   ├── inspect_model.py
│   ├── inspect_model_sim.py
│   └── plot_signal_sample.py
│
├── models/              # モデルファイルと実験結果
│   ├── cnn.py
│   ├── transformers.py
│   ├── layernorm/       # 必須：実験結果（ログ、重み、予測結果）
│   └── pretrained/      # 必須：実験結果（ログ、重み、予測結果）
│
├── documents/           # ドキュメント
│   └── OPTUNA_TUTORIAL.md
│
├── notebooks/           # Jupyterノートブック（空）
├── tests/               # テスト（空）
│
└── [ルートディレクトリの古いファイル] # 整理が必要
```

## 新しい構造 vs 古い構造

### 新しい構造（推奨）
- `scripts/train.py` - 統一された訓練スクリプト
- `scripts/optimize.py` - Optuna最適化
- `src/` - モジュール化されたコード
- `tools/` - 開発ツール

### 古い構造（後方互換性のため残存）
- ルートディレクトリの古いスクリプト
- `models/` ディレクトリの古いファイル

