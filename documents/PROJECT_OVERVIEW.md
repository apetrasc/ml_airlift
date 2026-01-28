# ML Airlift Project - プロジェクト概要

## プロジェクトの目的

このプロジェクトは、エアリフト（気液固三相流）の数値シミュレーション/実験データから、6つの物理量を回帰予測する機械学習システムです。

### 予測対象（6つの物理量）

1. **Solid Velocity**（固相速度）
2. **Gas Velocity**（ガス速度）
3. **Liquid Velocity**（液相速度）
4. **Solid Volume Fraction**（固相体積率）
5. **Gas Volume Fraction**（ガス体積率）
6. **Liquid Volume Fraction**（液相体積率）

### 入力データ

- 4次元配列 `(N, C, H, W)` の信号データ
  - `N`: サンプル数
  - `C`: チャネル数（元は4チャネル、Channel 1と3を除外して2チャネルにすることも可能）
  - `H`: 高さ方向
  - `W`: 幅方向（元2500点、スライス後2000点）

## ディレクトリ構成

```
ml_airlift/
├── scripts/              # 実行スクリプト
│   ├── optimize.py      # Optunaによるハイパーパラメータ最適化
│   ├── train.py         # 統一訓練スクリプト
│   └── create_nowall_dataset.py  # データセット作成
│
├── src/                  # コアライブラリ
│   ├── models/          # モデル定義
│   │   ├── cnn.py       # CNNモデル（SimpleCNNReal, SimpleCNNReal2D等）
│   │   └── transformers.py  # Transformerモデル
│   ├── data/            # データ処理
│   │   ├── loaders.py   # データローディング
│   │   ├── preprocessing.py  # 前処理
│   │   └── validation.py    # データ検証
│   ├── training/        # 訓練ロジック
│   │   └── trainer.py   # 訓練関数（train_one_epoch, evaluate等）
│   ├── evaluation/      # 評価機能
│   │   └── visualizations.py  # 予測可視化、メトリクス計算
│   └── utils/           # ユーティリティ
│       ├── config.py    # 設定管理
│       ├── device.py    # デバイス管理
│       └── memory.py    # メモリ管理
│
├── models/              # モデル定義と学習済み重み
│   ├── cnn.py          # 各種CNNアーキテクチャ
│   ├── transformers.py # Transformerモデル
│   ├── sota/           # 3相モデル（State-of-the-Art）
│   │   └── weights/    # 学習済み重み
│   └── layernorm/      # 2相モデル
│       └── weights/    # 学習済み重み
│
├── config/              # 設定ファイル
│   ├── config_real.yaml        # 実データ用設定
│   ├── config.yaml             # シミュレーション用設定
│   ├── config_dataset_creation.yaml  # データセット作成用設定
│   └── config_inference.yaml   # 推論用設定
│
├── tools/               # 開発・デバッグツール
│   ├── inspect_model.py        # モデル検査
│   ├── check_optuna_results.py # Optuna結果確認
│   └── plot_signal_sample.py   # 信号可視化
│
├── tests/               # テスト・評価スクリプト
│   ├── evaluate_gradcam.py     # Grad-CAM評価
│   └── compare_datasets.py     # データセット比較
│
├── documents/           # ドキュメント
│   ├── PROJECT_OVERVIEW.md     # このファイル
│   ├── OPTUNA_TUTORIAL.md
│   ├── REPOSITORY_STRUCTURE.md
│   └── MESSAGE_CLASSIFICATION.md
│
└── main.py             # 推論用メインスクリプト
```

## 主要コンポーネント

### 1. モデルアーキテクチャ

#### 1D CNNモデル（信号系列用）
- **SimpleCNNReal**: シンプルな1D CNN
- **ResidualCNN**: 残差ブロック付き1D CNN
- **BaseCNN**, **ProposedCNN**: バリエーション

#### 2D CNNモデル（画像様データ用）
- **SimpleCNNReal2D**: 2D画像用CNN
  - オプションでリサイズ可能
  - 残差ブロック対応
  - ドロップアウト対応

#### Transformerモデル
- **SimpleViTRegressor**: Vision Transformerを1D信号に適用

### 2. データ処理パイプライン

#### データローディング
- `.npy`/`.npz`ファイル対応
- 自動チャネル次元調整
- マルチターゲット対応（6つの物理量）

#### データ前処理
- チャネル除外（Channel 1と3を除外）
- スライシング（`W`次元の500:2500）
- ダウンサンプリング対応

#### データ検証
- NaN/Inf値検出
- 統計情報計算
- 形状・整合性チェック

### 3. 訓練システム

#### 基本訓練
- **train_real.py**: Hydra設定を使用した訓練
- タイムスタンプ付き出力ディレクトリ
- 学習曲線の自動生成

#### ハイパーパラメータ最適化
- **scripts/optimize.py**: Optunaによる自動最適化
- TPE（Tree-structured Parzen Estimator）使用
- Pruning（早期終了）対応
- SQLiteデータベースで結果永続化
- 最良トライアルの自動選択

#### メモリ管理
- GPU/CPUメモリ監視
- バッチ処理時のメモリ最適化
- CUDAキャッシュクリア

### 4. 評価システム

#### 予測評価
- **eval_real.py**: 訓練済みモデルで評価
- 6つの物理量ごとのメトリクス（R², MSE, MAE, RMSE）
- 予測vs実測の散布図生成
- 残差プロット

#### 可視化
- 各ターゲットの個別プロット
- 全ターゲットのオーバービュー
- メトリクスサマリーテーブル

#### Grad-CAM解析
- **tests/evaluate_gradcam.py**: モデルの重要領域可視化
- 各ターゲット（速度・体積率）ごとのGrad-CAM生成
- サンプル画像、Grad-CAM、オーバーレイ画像の統合表示

### 5. 開発ツール

#### データ分析
- **validate_dataset.py**: データセット統計情報
- **tools/plot_signal_sample.py**: 信号サンプルの可視化
- **tests/compare_datasets.py**: データセット間の比較

#### モデル解析
- **tools/inspect_model.py**: モデル重みの可視化
- **tools/check_optuna_results.py**: Optuna結果の確認・エクスポート

## データフロー

```
1. データセット作成
   create_dropped_dataset.py
   → Channel 1,3を除外したデータセット作成
   → x_train_dropped.npy, t_train_dropped.npy

2. データ検証
   validate_dataset.py
   → NaN/Inf値検出、統計情報確認

3. 訓練
   train_real.py または scripts/train.py
   → モデル訓練
   → モデル重み、予測結果、学習曲線を保存

4. ハイパーパラメータ最適化（オプション）
   scripts/optimize.py
   → 複数トライアル実行
   → 最良モデルの自動選択

5. 評価
   eval_real.py
   → テストデータで評価
   → 予測プロット、メトリクス生成

6. 推論
   main.py
   → 学習済みモデルで推論
   → Phase fractionの出力

7. 可視化・解析
   tests/evaluate_gradcam.py
   → Grad-CAM可視化
   → モデルの重要領域解析
```

## 設定管理

### Hydra/OmegaConf使用
- YAMLファイルで設定管理
- 環境ごとに設定ファイルを分離（`config_real.yaml`, `config.yaml`）
- コマンドラインでの上書き可能

### 主要設定項目
- **データパス**: 入力データ、出力ディレクトリ
- **モデル**: タイプ、ハイパーパラメータ
- **訓練**: エポック数、バッチサイズ、学習率、デバイス
- **データ分割**: train/val/test比率

## 特徴的な設計

1. **モジュール化**: `src/`配下に機能を分離
2. **後方互換性**: ルートディレクトリのレガシースクリプトも残存
3. **メモリ効率**: 大規模データ対応の最適化
4. **再現性**: 設定ファイル保存、シード固定
5. **可視化重視**: 評価結果の自動可視化

## 使用方法

### 推論の実行

```bash
# 3相モデルで推論
python main.py --3phase

# 2相モデルで推論
python main.py --2phase
```

推論ファイルのパスは`config/config_inference.yaml`で指定します。

### 訓練の実行

```bash
# 実データで訓練
python train_real.py

# シミュレーションデータで訓練
python train.py
```

### ハイパーパラメータ最適化

```bash
python scripts/optimize.py
```

詳細は `documents/OPTUNA_TUTORIAL.md` を参照してください。

## まとめ

このプロジェクトは、エアリフトの信号データから6つの物理量を予測する回帰タスクの機械学習パイプラインです。CNN/Transformerモデル、Optunaによる最適化、評価・可視化ツールを備え、実験から評価まで一貫して実行できるシステムとなっています。



