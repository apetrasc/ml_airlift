# 実行時メッセージの分類ガイドライン

## 概要

実行時のメッセージは、**1回だけ表示するもの**と**実行中ずっと表示しておくもの**に分類されます。
この分類は、ユーザーが実行状況を適切に把握できるようにするためのものです。

## 分類基準

### 1. **1回だけ表示すべきもの（One-time Messages）**

**特徴:**
- 実行開始時や各ステップ完了時に1回だけ出力される
- 時間が経っても変わらない固定情報
- 初期化や設定に関する情報
- 最終結果や完了通知

**例:**
- 環境設定（環境変数、パッケージインポート）
- 設定情報（スタディ名、エポック範囲、ベース設定ファイル）
- データロード完了（形状、サイズ情報）
- モデル作成完了
- メモリチェック結果（初期状態）
- 最終結果（ベストトライアル、最適化完了）
- エラー・警告（発生時のみ）

**現在のコード例:**
```python
# 初期設定（1回のみ）
print("[INFO] Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
print(f"[INFO] Generated study name: {study_name}")
print(f"[INFO] Starting optimization...")

# データロード完了（1回のみ）
print(f"[OK] Loaded. x.shape={x.shape}, t.shape={t.shape}")

# 最終結果（1回のみ）
print("OPTIMIZATION COMPLETED")
print(f"Best Trial: #{study.best_trial.number}")
```

---

### 2. **実行中ずっと表示しておくべきもの（Progressive/Real-time Messages）**

**特徴:**
- 時間の経過と共に変化する情報
- 進捗状況や状態を追跡するために必要な情報
- ループ内で繰り返し出力される
- 実行中に値を確認したい情報

**例:**
- エポックごとの進捗（train MSE, val MSE, val MAE）
- GPUメモリ使用状況（各エポック後）
- トライアル開始情報（複数トライアル実行時）
- 進行中のステップ情報（[STEP]メッセージ、ただし各トライアルごとに1回）
- リアルタイムの状態変化

**現在のコード例:**
```python
# エポックごとの進捗（実行中ずっと表示）
for epoch in range(1, cfg.training.epochs + 1):
    # ... トレーニング ...
    if epoch % cfg.logging.print_every_n_epochs == 0:
        print(f"\nEpoch {epoch:03d}/{cfg.training.epochs} | "
              f"train MSE={tr:.6f} | val MSE={val_mse:.6f} | "
              f"val MAE={val_mae:.6f} | {time.time()-t_ep:.2f}s")
        
        # GPUメモリ使用状況（実行中ずっと表示）
        print(f"  GPU memory: PyTorch={mem_allocated:.0f} MB allocated, "
              f"{mem_reserved:.0f} MB reserved | "
              f"Actual={actual_mem['used_mb']:.0f} MB")

# トライアル開始情報（複数トライアル実行時、各トライアルごとに表示）
for trial in study.optimize(...):
    print(f"\n{'='*60}")
    print(f"Trial {trial.number} started")
    print(f"Hyperparameters: {trial.params}")
```

---

## 分類の判断基準

### 判断フローチャート

```
メッセージを出力する場所は？
│
├─ [ループの外] → 1回だけ表示
│   └─ 初期化、設定、最終結果など
│
└─ [ループの中] → 実行中ずっと表示（ただし、以下を考慮）
    │
    ├─ 各イテレーションで値が変わる？
    │   ├─ Yes → 実行中ずっと表示
    │   │   └─ エポックごとの進捗、メモリ使用状況など
    │   │
    │   └─ No → 1回だけ表示（ループ開始時や条件付き）
    │       └─ 初期設定、エラー発生時のみなど
    │
    └─ ユーザーが実行中に確認したい情報か？
        ├─ Yes → 実行中ずっと表示
        │   └─ 進捗、メモリ使用状況など
        │
        └─ No → 1回だけ表示
            └─ 設定情報、完了通知など
```

---

## 実装時のベストプラクティス

### 1. **メッセージプレフィックスの使用**

```python
# 1回だけ表示すべきもの
print("[INFO] ...")      # 情報（設定、初期化など）
print("[OK] ...")        # 完了通知
print("[ERROR] ...")     # エラー（発生時のみ）
print("[WARN] ...")      # 警告（発生時のみ）

# 実行中ずっと表示すべきもの
print("[STEP] ...")      # 進行中のステップ（各トライアルごとに1回）
# エポックごとの進捗はプレフィックスなしで直接出力
print(f"Epoch {epoch:03d} | ...")  # 進捗情報
```

### 2. **条件付き出力**

```python
# エラーや警告は発生時のみ表示（1回だけ表示の原則）
if error_condition:
    print("[ERROR] ...")

# エポックごとの進捗は条件付きで定期的に表示（実行中ずっと表示の原則）
if epoch % print_every_n_epochs == 0:
    print(f"Epoch {epoch:03d} | ...")
```

### 3. **メモリ使用状況の表示**

```python
# 初期状態（1回だけ表示）
_print_memory_status("before optimization starts")

# 実行中の状態（実行中ずっと表示）
for epoch in range(1, epochs + 1):
    # ... トレーニング ...
    if epoch % print_every_n_epochs == 0:
        print(f"  GPU memory: ...")  # 各エポック後
```

---

## 現在のコードにおける分類例

### ✅ 正しい分類例

#### 1回だけ表示
```python
# optimize.py の main() 関数内
print(f"[INFO] Generated study name: {study_name}")           # 初期設定
print(f"[INFO] Starting optimization...")                     # 開始通知
print("[OK] Memory check completed")                          # 完了通知
print("OPTIMIZATION COMPLETED")                               # 最終結果
print(f"Best Trial: #{study.best_trial.number}")              # 最終結果

# objective() 関数内（各トライアルごとに1回）
print(f"[STEP] Loading dataset files...")                     # ステップ開始
print(f"[OK] Loaded. x.shape={x.shape}")                      # 完了通知
print(f"[STEP] Build model...")                               # ステップ開始
print(f"[OK] Trial {trial.number} results saved")             # 完了通知
```

#### 実行中ずっと表示
```python
# objective() 関数内のトレーニングループ
for epoch in range(1, cfg.training.epochs + 1):
    # ... トレーニング ...
    if epoch % cfg.logging.print_every_n_epochs == 0:
        print(f"\nEpoch {epoch:03d}/{cfg.training.epochs} | "  # 進捗（値が変わる）
              f"train MSE={tr:.6f} | val MSE={val_mse:.6f}")
        print(f"  GPU memory: ...")                            # メモリ状況（値が変わる）
```

### ⚠️ 注意が必要なケース

#### [STEP] メッセージについて
- **トライアルごとに1回**：各トライアルの実行中に1回だけ表示される
- **ループ内にあるが、値が変わらない**：各トライアルの初期化時に1回だけ表示
- **結論**：トライアルごとの1回だけ表示として扱う

```python
# 各トライアルごとに1回だけ表示（トライアルループの中にあるが、値は変わらない）
print(f"[STEP] Loading dataset files...")  # トライアルごとに1回
print(f"[STEP] Build model...")            # トライアルごとに1回
```

---

## 実際のコードにおける分類表

### `optimize.py` のメッセージ分類

| ファイル位置 | メッセージ | 分類 | 理由 |
|------------|----------|------|------|
| `main()` 関数 | `[INFO] Generated study name: ...` | **1回だけ表示** | 初期設定、固定情報 |
| `main()` 関数 | `[INFO] Starting optimization...` | **1回だけ表示** | 開始通知、固定情報 |
| `main()` 関数 | `[OK] Memory check completed` | **1回だけ表示** | 完了通知、固定情報 |
| `main()` 関数 | `OPTIMIZATION COMPLETED` | **1回だけ表示** | 最終結果、固定情報 |
| `main()` 関数 | `Best Trial: #...` | **1回だけ表示** | 最終結果、固定情報 |
| `objective()` 関数 | `[STEP] Loading dataset files...` | **1回だけ表示**<br>（トライアルごと） | 各トライアルの初期化、値は変わらない |
| `objective()` 関数 | `[OK] Loaded. x.shape=...` | **1回だけ表示**<br>（トライアルごと） | 完了通知、固定情報 |
| `objective()` 関数 | `[STEP] Build model...` | **1回だけ表示**<br>（トライアルごと） | 各トライアルの初期化、値は変わらない |
| `objective()` 関数 | `Trial {trial.number} started` | **実行中ずっと表示**<br>（トライアルごと） | 複数トライアル実行時、各トライアルで値が変わる |
| `objective()` 関数 | `Epoch {epoch:03d}/... | train MSE=...` | **実行中ずっと表示** | エポックごとに値が変わる、進捗状況 |
| `objective()` 関数 | `GPU memory: ...` | **実行中ずっと表示** | エポックごとに値が変わる、状態追跡 |
| `objective()` 関数 | `[ERROR] ...` | **1回だけ表示**<br>（発生時のみ） | エラー発生時のみ表示 |
| `objective()` 関数 | `[WARN] ...` | **1回だけ表示**<br>（発生時のみ） | 警告発生時のみ表示 |

### `trainer.py` のメッセージ分類

| 関数 | メッセージ | 分類 | 理由 |
|------|----------|------|------|
| `train_one_epoch()` | （現在はメッセージなし） | - | ステップごとの表示を削除済み |
| `evaluate()` | （メッセージなし） | - | 評価結果は呼び出し元で表示 |

---

## まとめ

| 分類 | 特徴 | 例 |
|------|------|-----|
| **1回だけ表示** | 固定情報、初期化、完了通知 | 設定情報、データロード完了、最終結果 |
| **実行中ずっと表示** | 時間経過で変化する情報、進捗状況 | エポックごとの進捗、メモリ使用状況 |

### 判断のポイント

1. **値が変わるか？** → 変わるなら「実行中ずっと表示」
2. **ループの外か？** → 外なら「1回だけ表示」
3. **ユーザーが実行中に確認したいか？** → はいなら「実行中ずっと表示」
4. **完了通知か？** → はいなら「1回だけ表示」

この分類に従うことで、ユーザーは実行状況を適切に把握でき、ログも読みやすくなります。



