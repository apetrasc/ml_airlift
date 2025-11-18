# optuna_tutorial.py 処理内容の解説

## 概要

このスクリプトは、Optunaを使用してCNNモデルのハイパーパラメータを自動的に最適化するためのプログラムです。複数のトライアル（試行）を実行し、最適なハイパーパラメータの組み合わせを探索します。

---

## 全体の処理フロー

```
1. 設定ファイルの読み込み
2. Study（最適化実験）の作成/読み込み
3. 複数のトライアルを実行
   └─ 各トライアル:
      a. ハイパーパラメータの提案
      b. データの読み込み・前処理
      c. モデルの作成
      d. 学習（Early Pruning対応）
      e. 評価
      f. 結果の保存
      g. GPUメモリのクリーンアップ
4. 最適なトライアルの特定
5. 結果のまとめと保存
```

---

## 主要な関数とクラス

### 1. `WeightedMSELoss` クラス (47-78行目)

**役割**: ターゲット変数ごとに異なる重みを付けたMSE損失関数

**機能**:
- 6つのターゲット変数（Solid Velocity, Gas Velocity, Liquid Velocity, Solid Volume Fraction, Gas Volume Fraction, Liquid Volume Fraction）に対して、それぞれ異なる重みを設定可能
- 性能の悪いターゲット（Solid Velocity, Solid Volume Fraction）に高い重みを付けることで、学習を改善

**使用例**:
```python
weights = [5.0, 1.0, 2.0, 5.0, 1.0, 1.0]  # ターゲット0, 3に高い重み
criterion = WeightedMSELoss(weights=weights)
```

---

### 2. `suggest_hyperparameters` 関数 (81-146行目)

**役割**: Optunaがハイパーパラメータを提案し、設定ファイルを更新

**最適化するハイパーパラメータ**:

| パラメータ | 範囲 | 説明 |
|-----------|------|------|
| `model.hidden` | 32-128 (16刻み) | モデルの隠れ層のチャンネル数 |
| `model.dropout_rate` | 0.0-0.5 (0.05刻み) | ドロップアウト率（過学習防止） |
| `model.use_residual` | True/False | 残差接続を使用するか |
| `training.learning_rate` | 1e-5 ～ 1e-2 (対数スケール) | 学習率 |
| `training.batch_size` | [2, 4, 8] | バッチサイズ |
| `training.weight_decay` | 1e-6 ～ 1e-3 (対数スケール) | 重み減衰（正則化） |
| `dataset.downsample_factor` | 1-4 | データのダウンサンプリング係数 |
| `training.epochs` | 50-200 (50刻み) | エポック数 |
| `training.use_weighted_loss` | True/False | 重み付け損失を使用するか |
| `training.weight_target_*` | 0.1-10.0 | 各ターゲットの損失重み |

**処理の流れ**:
1. ベース設定ファイルをコピー
2. Optunaの`trial.suggest_*`メソッドで各パラメータを提案
3. 設定ファイルを更新して返す

---

### 3. `create_trial_output_dir` 関数 (149-173行目)

**役割**: 各トライアルの結果を保存するディレクトリを作成

**作成されるディレクトリ構造**:
```
OUTPUTS_ROOT/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── logs/           # ログファイル
        ├── weights/        # モデルの重み
        ├── config.yaml     # 使用した設定
        ├── trial_info.yaml # トライアル情報
        ├── metrics.json    # 評価指標
        └── evaluation_plots/ # 評価プロット
```

---

### 4. `save_trial_results` 関数 (176-260行目)

**役割**: トライアルの結果をファイルに保存

**保存される内容**:
- **設定ファイル** (`config.yaml`): 使用したハイパーパラメータ
- **トライアル情報** (`trial_info.yaml`): トライアル番号、パラメータ、損失値など
- **評価指標** (`metrics.json`): 
  - 検証損失、テストMSE、テストMAE
  - 学習・検証損失の履歴
  - **ターゲットごとの詳細指標** (MSE, MAE, RMSE, R²)
- **予測結果** (`y_pred.npy`, `y_true.npy`): テストセットの予測値と実測値
- **学習曲線** (`learning_curves.png`): 学習・検証損失の推移
- **評価プロット** (`evaluation_plots/`): 予測値 vs 実測値の散布図

---

### 5. `objective` 関数 (263-571行目) ⭐ **最重要関数**

**役割**: 各トライアルで実行される処理（Optunaの目的関数）

**処理の流れ**:

#### ステップ1: 設定の読み込みとハイパーパラメータの提案 (286-290行目)
```python
base_cfg = OmegaConf.load(base_config_path)  # ベース設定を読み込み
cfg = suggest_hyperparameters(trial, base_cfg)  # ハイパーパラメータを提案
```

#### ステップ2: 出力ディレクトリの作成 (293-295行目)
```python
output_dir = create_trial_output_dir(trial.number)
```

#### ステップ3: 再現性の確保 (304-307行目)
```python
torch.manual_seed(cfg.training.seed)  # 乱数シードを固定
np.random.seed(cfg.training.seed)
```

#### ステップ4: データの読み込みと前処理 (310-345行目)
- データセットファイル（`x_train`, `t_train`）を読み込み
- **Channel 1と3を除外**（Channel 0と2のみを使用）
- サンプル数の制限（オプション）
- ダウンサンプリング（オプション）

#### ステップ5: データセットの分割 (347-360行目)
- 学習セット（70%）
- 検証セット（15%）
- テストセット（15%）

#### ステップ6: データローダーの作成 (363-384行目)
- バッチサイズ、シャッフル、ワーカー数などを設定

#### ステップ7: モデルの作成 (387-391行目)
```python
model = create_model(cfg, x_sample, out_dim, device)
```
- 入力データの形状に基づいてモデルを自動選択（1D CNN or 2D CNN）

#### ステップ8: 複数GPU対応 (394-400行目)
- 複数のGPUが利用可能な場合、DataParallelで並列処理

#### ステップ9: オプティマイザと損失関数の作成 (402-415行目)
```python
# 重み付け損失を使用する場合
if target_weights is not None:
    criterion = WeightedMSELoss(weights=target_weights)
else:
    criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
```

#### ステップ10: 学習ループ (417-457行目)
```python
for epoch in range(1, cfg.training.epochs + 1):
    # 1エポック学習
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    
    # 検証セットで評価
    val_mse, val_mae = evaluate(model, val_loader, criterion, device)
    
    # Optunaに報告（Pruning用）
    trial.report(val_mse, step=epoch)
    
    # 早期終了（Pruning）のチェック
    if trial.should_prune():
        raise optuna.TrialPruned()
```

**Early Pruning（早期終了）**:
- 検証損失が改善しない場合、学習を途中で終了
- 無駄な計算を避けることで、最適化を高速化

#### ステップ11: テストセットで評価 (459-480行目)
```python
test_mse, test_mae, y_pred, y_true = evaluate(model, test_loader, criterion, device)

# ターゲットごとのR²値を計算・表示
for i in range(y_pred.shape[1]):
    target_r2 = r2_score(y_true[:, i], y_pred[:, i])
    print(f"Target {i}: R²={target_r2:.4f}")
```

#### ステップ12: モデルと結果の保存 (482-501行目)
- モデルの重みを保存
- `save_trial_results`で結果を保存
- Optunaのユーザー属性に評価指標を設定

#### ステップ13: GPUメモリのクリーンアップ (520-565行目) ⚠️ **重要**
```python
finally:
    # モデル、オプティマイザ、データローダーなどを削除
    del model, optimizer, train_loader, ...
    
    # GPUキャッシュをクリア
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # ガベージコレクション
    gc.collect()
```

**なぜ重要か**: 各トライアル後にGPUメモリを解放しないと、メモリ不足エラーが発生する

---

### 6. `find_trial_output_dir` 関数 (574-593行目)

**役割**: トライアル番号から、そのトライアルの出力ディレクトリを検索

**処理**:
- `OUTPUTS_ROOT`配下を再帰的に検索
- `trial_info.yaml`ファイルを読み込み、トライアル番号が一致するディレクトリを返す

---

### 7. `generate_study_summary` 関数 (596-634行目)

**役割**: 全てのトライアルの結果をまとめたサマリーファイルを生成

**保存される情報**:
- Study名、トライアル数
- 最良トライアルの情報（番号、損失値、パラメータ）
- 全トライアルのサマリー（番号、損失値、状態、パラメータ）

---

### 8. `main` 関数 (637-782行目) ⭐ **エントリーポイント**

**役割**: プログラムのメイン処理

#### ステップ1: Studyの作成/読み込み (643-691行目)
```python
# Optunaディレクトリを作成
os.makedirs(OPTUNA_DIR, exist_ok=True)

# データベースストレージを作成
storage = optuna.storages.RDBStorage(url=f'sqlite:///{study_db_path}')

# 既存のStudyを読み込む、または新規作成
study = optuna.load_study(study_name=study_name, storage=storage)
# または
study = optuna.create_study(study_name=study_name, storage=storage, ...)
```

**Pruner（早期終了機能）の設定**:
- `MedianPruner`: 検証損失が中央値より悪い場合に早期終了
- `n_startup_trials=5`: 最初の5トライアルは早期終了しない
- `n_warmup_steps=10`: 10エポックまでは早期終了しない

#### ステップ2: 最適化の実行 (700-735行目)
```python
study.optimize(
    lambda trial: objective(trial, BASE_CONFIG_PATH),
    n_trials=20,  # 20回のトライアルを実行
    n_jobs=1,     # 1つのジョブ（GPU使用時は1に設定）
    show_progress_bar=True
)
```

**エラーハンドリング**:
- パラメータ分布の不一致エラーが発生した場合、タイムスタンプ付きの新しいStudyを作成して再試行

#### ステップ3: 最良トライアルの特定と結果の保存 (737-782行目)
- 最良トライアルの情報を表示
- 最良トライアルのディレクトリにシンボリックリンクを作成
- Studyサマリーを生成・保存
- 最良トライアル情報を保存

---

## 重要な概念

### 1. **Optuna Study（研究）**
- 複数のトライアルをまとめて管理するオブジェクト
- 最適化の履歴を保存し、最良のトライアルを追跡

### 2. **Optuna Trial（試行）**
- 1回のハイパーパラメータの組み合わせでの学習・評価
- 各トライアルは独立して実行される

### 3. **Early Pruning（早期終了）**
- 検証損失が改善しない場合、学習を途中で終了
- 無駄な計算を避けることで、最適化を高速化

### 4. **重み付け損失関数**
- 6つのターゲット変数に対して、それぞれ異なる重みを設定
- 性能の悪いターゲットに高い重みを付けることで、学習を改善

### 5. **GPUメモリ管理**
- 各トライアル後にGPUメモリをクリーンアップ
- メモリ不足エラーを防ぐために重要

---

## 実行方法

### 基本的な実行
```bash
python optuna_tutorial.py
```

### 既存のStudyを削除して新規作成
```bash
OPTUNA_DELETE_STUDY=true python optuna_tutorial.py
```

---

## 出力ファイル

### 1. Optunaデータベース
- `OPTUNA_DIR/study.db`: SQLiteデータベース（全トライアルの履歴）

### 2. Studyサマリー
- `OPTUNA_DIR/study_summary.json`: 全トライアルのサマリー
- `OUTPUTS_ROOT/optuna_study_summary.json`: 同じ内容のコピー

### 3. 最良トライアル
- `OPTUNA_DIR/best_trial_info.yaml`: 最良トライアルの情報
- `OUTPUTS_ROOT/optuna_best/`: 最良トライアルのディレクトリへのシンボリックリンク

### 4. 各トライアルの結果
- `OUTPUTS_ROOT/YYYY-MM-DD/HH-MM-SS/`: 各トライアルの出力ディレクトリ
  - `config.yaml`: 使用した設定
  - `trial_info.yaml`: トライアル情報
  - `metrics.json`: 評価指標
  - `weights/model_simplecnn_real.pth`: モデルの重み
  - `y_pred.npy`, `y_true.npy`: 予測結果
  - `learning_curves.png`: 学習曲線
  - `evaluation_plots/`: 評価プロット

---

## トラブルシューティング

### 1. GPUメモリ不足エラー
- **原因**: 各トライアル後にメモリが解放されていない
- **解決策**: `finally`ブロックでメモリをクリーンアップ（既に実装済み）
- **その他**: バッチサイズを小さくする、モデルのサイズを小さくする

### 2. パラメータ分布の不一致エラー
- **原因**: 既存のStudyと新しいコードでパラメータの範囲が異なる
- **解決策**: 自動的に新しいStudyを作成（既に実装済み）
- **その他**: `OPTUNA_DELETE_STUDY=true`で既存のStudyを削除

### 3. 学習が遅い
- **原因**: エポック数が多い、データサイズが大きい
- **解決策**: Early Pruningを有効化（既に実装済み）、ダウンサンプリングを使用

---

## まとめ

このスクリプトは、Optunaを使用してCNNモデルのハイパーパラメータを自動的に最適化する包括的なソリューションです。以下の特徴があります:

1. **自動的なハイパーパラメータ探索**: 複数のパラメータを同時に最適化
2. **Early Pruning**: 無駄な計算を避けて高速化
3. **重み付け損失関数**: 性能の悪いターゲットに重点を置いた学習
4. **詳細な評価指標**: ターゲットごとのR²値などを計算
5. **GPUメモリ管理**: メモリ不足を防ぐためのクリーンアップ
6. **結果の保存**: 全てのトライアルの結果を保存し、後で分析可能

これらの機能により、効率的に最適なハイパーパラメータを見つけることができます。

