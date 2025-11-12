# 不要なファイル一覧

## ルートディレクトリの重複ファイル

以下のファイルは新しい構造に移動済みまたは統合済みのため、削除可能です：

### 1. 新しい構造に移動済み（削除推奨）

- `optuna_tutorial.py` → `scripts/optimize.py` に統合済み
- `train_real.py` → `scripts/train.py` に統合済み
- `train.py` → `scripts/train.py` に統合済み
- `clear_gpu_memory.py` → `tools/clear_gpu_memory.py` に移動済み
- `check_optuna_results.py` → `tools/check_optuna_results.py` に移動済み
- `plot_signal_sample.py` → `tools/plot_signal_sample.py` に移動済み

### 2. 新しい構造に統合予定（削除推奨）

- `create_dropped_dataset.py` → `src/data/preprocessing.py` に統合予定
- `validate_dataset.py` → `src/data/validation.py` に統合予定

### 3. 使用されていない可能性（確認が必要）

- `eval.py` - 評価スクリプト（新しい構造に統合されていない）
- `eval_real.py` - 評価スクリプト（新しい構造に統合されていない）
- `evalsim.py` - 評価スクリプト（新しい構造に統合されていない）
- `eval_tutorial.py` - 評価スクリプト（新しい構造に統合されていない）
- `main.py` - メインスクリプト（使用されていない可能性）
- `result.py` - 結果処理スクリプト（使用されていない可能性）

### 4. 一時ファイル（削除推奨）

- `y_pred.npy` - 一時的な予測結果ファイル
- `y_true.npy` - 一時的な真値ファイル

## models/ ディレクトリのファイル

### 5. 必須ファイル（保持）

- `models/layernorm/` - **必須**（実験結果、ログ、重み、予測結果）
- `models/pretrained/` - **必須**（実験結果、ログ、重み、予測結果）

### 6. 重複ファイル（確認が必要）

- `models/cnn.py` - `src/models/cnn.py` と重複の可能性
- `models/transformers.py` - `src/models/transformers.py` と重複の可能性
- `models/__init__.py` - `src/models/__init__.py` と重複の可能性

## src/ ディレクトリの整理が必要なファイル

### 7. ルートレベルのファイル（適切な場所に移動推奨）

- `src/chunked_loader.py` → `src/data/` に移動推奨
- `src/data_loader.py` → `src/data/` に移動推奨
- `src/streaming_loader.py` → `src/data/` に移動推奨
- `src/evaluate_predictions.py` → `src/evaluation/` に移動推奨
- `src/visualize_signal.py` → `src/visualization/` に移動推奨
- `src/optuna_optimizer.py` → `src/optimization/` に移動推奨
- `src/memory_utils.py` → `src/utils/memory.py` と統合推奨
- `src/utils.py` → `src/utils/` に統合推奨

### 8. デバッグ・テストファイル（tests/ に移動推奨）

- `src/debug.py`
- `src/debug_dataset.py`
- `src/test_cleaned_data.py`

### 9. その他のユーティリティ（整理推奨）

- `src/config_utils.py` → `src/utils/config.py` と統合推奨
- `src/data_cleaner.py` → `src/data/preprocessing.py` と統合推奨
- `src/data_inspector.py` → `tools/` に移動推奨
- `src/compare_data.py` → `tools/` に移動推奨
- `src/summary_report.py` → `src/evaluation/` に移動推奨
- `src/image_cnn_models.py` → `src/models/` に移動推奨
- `src/mlflow_tracker.py` → `src/utils/` または `src/training/` に移動推奨

## 削除推奨の優先順位

### 高優先度（即座に削除可能）
1. `y_pred.npy`, `y_true.npy` - 一時ファイル
2. `optuna_tutorial.py` - 完全に統合済み
3. `clear_gpu_memory.py` (ルート) - `tools/` に移動済み
4. `check_optuna_results.py` (ルート) - `tools/` に移動済み
5. `plot_signal_sample.py` (ルート) - `tools/` に移動済み

### 中優先度（確認後削除）
6. `train_real.py`, `train.py` - 新しい `scripts/train.py` を使用
7. `create_dropped_dataset.py` - 新しい構造に統合予定
8. `validate_dataset.py` - 新しい構造に統合予定
9. `eval.py`, `eval_real.py`, `evalsim.py`, `eval_tutorial.py` - 使用状況を確認

### 低優先度（確認後削除）
10. `main.py`, `result.py` - 使用状況を確認

