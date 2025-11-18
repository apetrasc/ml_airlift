# チャンネル数変更の履歴

## 変更日時

### 最終変更（Channel 1と3を除外）
- **日付**: 2025年11月7日
- **時刻**: 18時57分頃
- **変更ファイル**:
  - `create_dropped_dataset.py`: Channel 1と3を除外してデータセットを作成
  - `train_real.py`: Channel 1と3を除外する処理を追加
  - `optuna_tutorial.py`: Channel 1と3を除外する処理を追加
  - `validate_dataset.py`: Channel 1と3を除外したデータセットの検証機能を追加

### 最初の変更（Channel 3のみ除外）
- **日付**: 2025年11月7日
- **時刻**: 17時33分頃
- **変更ファイル**:
  - `validate_dataset.py`: Channel 3の検証機能を追加

## 変更内容

### 変更前
- 入力チャンネル数: 4チャンネル（Channel 0, 1, 2, 3）

### 変更後
- 入力チャンネル数: 2チャンネル（Channel 0, 2）
- 除外されたチャンネル: Channel 1, Channel 3
- 理由: Channel 1と3に極端な値（-1000000, 1000000）が含まれていたため

## 学習開始日時

- **日付**: 2025年11月7日
- **時刻**: 19時42分頃
- **Study名**: `cnn_hyperparameter_optimization_20251107_194253`
- **最初のトライアル**: 19:42:54に開始

## 関連ファイル

- `optuna_tutorial.py`: 2025-11-08 16:41:59 (最新の変更)
- `train_real.py`: 2025-11-07 18:57:26
- `create_dropped_dataset.py`: 2025-11-07 18:57:26
- `validate_dataset.py`: 2025-11-07 17:33:20

