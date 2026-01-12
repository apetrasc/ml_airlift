# Machine Learning for airlift project

author : Sadanori Matsubara 
date: 2025/09/04

## インストール

### 1. 仮想環境の作成

**Linuxの場合:**  
ターミナルで以下のコマンドを実行してください：
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision
```

**Windowsの場合:**  
コマンドプロンプトを開いて、以下のコマンドを実行してください：
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision
```

> **注意:**  
> `pip install torch torchvision`コマンドは、標準的な環境（Ubuntu + CUDA 12.8）向けです。  
> 異なるOSやCUDAバージョンを使用している場合は、公式の[PyTorchインストールガイド](https://pytorch.org/get-started/locally/)を参照して、環境に適したバージョンをインストールしてください。

これにより、Python仮想環境がセットアップされ、PyTorchを含むすべての必要な依存関係がインストールされます。

### 2. データセットのダウンロード

データセットをzipファイルとして準備し、設定ファイルまたはコードで指定された適切なディレクトリに展開してください。データセット全体は機密情報のため、必要に応じて連絡してください。

データセットを準備した後、`config/config.yaml`を編集してデータセットの場所を指定してください。Linuxでは直接パスをコピー&ペーストできますが、Windowsではペースト時に""を削除する必要があります。

### 3. 推論の実行

推論には`main.py`を使用することを推奨します（詳細は後述の「推論スクリプト（main.py）」セクションを参照）。

- データセット全体に対して推論を実行するには、`tests/eval.py`を使用してください：
  ```bash
  python tests/eval.py --datetime models/layernorm
  ```
  このスクリプトは、訓練済みモデルを使用してデータセット全体を処理します。

- 単一のデータセットファイルに対して推論を実行するには、`tests/eval_tutorial.py`を使用してください：
  `tests/eval_tutorial.py`の`file_path`変数をデータファイルを指すように編集し、スクリプトを実行してください：
  ```bash
  python tests/eval_tutorial.py
  ```
  
  両方のスクリプトは同じ推論関数を使用します。主な違いは、`tests/eval.py`がバッチで全データを処理するのに対し、`tests/eval_tutorial.py`は単一ファイルのテストやチュートリアル目的で使用されることです。

依存関係やデータセットパスに関する問題が発生した場合は、設定ファイル（例：`config/config.yaml`）を確認し、すべてのパスが正しく設定されていることを確認してください。

## Dataset

### データセットの概要

このプロジェクトでは、数値シミュレーションで作成したデータを訓練データセットとして使用し、実験データをテストデータセットとして使用します。`scripts/train.py`はシミュレーションデータで訓練を行い、`tests/eval.py`は実験データで評価を行います。

訓練と評価では異なるGPUを使用するため、`config/config.yaml`で設定を確認してください。

### データセットの構築(Channel 1とChannel 3を除外)

#### 1. データセットの準備

元のデータセットファイル（`x_train.npy`と`t_train.npy`）を準備します。データセットは機密情報のため、必要に応じて連絡してください。


```bash
# Channel 1とChannel 3を除外したデータセットを作成
python tools/create_dropped_dataset.py
```

このスクリプトは以下の処理を行います：
- 元のデータセット（4チャネル）を読み込み
- Channel 1とChannel 3を除外（Channel 0と2のみを保持）
- 処理済みデータセットを`dropped_data`ディレクトリに保存

**設定ファイル**: `config/config_dataset_creation.yaml`で入力パスと出力パスを指定します。

**出力ファイル**:
- `x_train_dropped.npy`: Channel 1と3を除外した入力データ（2チャネル）
- `t_train_dropped.npy`: ターゲットデータ（変更なし）

#### 2. データセットの検証

データセットを構築した後、以下のスクリプトでデータセットを検証できます：

```bash
# データセットの統計情報と整合性を確認
python tests/validate_dataset.py
```

このスクリプトは以下のチェックを行います：
- NaN値やInf値の検出
- データ形状の確認
- 各チャネルの統計情報（平均、標準偏差、パーセンタイルなど）
- サンプル数の整合性確認
- 極端な値の検出

検証結果を確認し、問題がないことを確認してから訓練を開始してください。

## ハイパーパラメータ最適化（Optuna）

このプロジェクトでは、Optunaを使用したハイパーパラメータ最適化をサポートしています。

### 基本的な使用方法

#### ハイパーパラメータ最適化の実行
```bash
# Optunaを使用したハイパーパラメータ最適化
python scripts/optimize.py

# または、チュートリアル用スクリプト
python optuna_tutorial.py
```

#### 訓練の実行
```bash
# 実データで訓練（Hydra設定を使用）
python scripts/train_real.py

# シミュレーションデータで訓練
python scripts/train.py
```

#### 評価の実行
```bash
# 特定の日時ディレクトリで評価
python eval_real.py --datetime "2025-11-19/20-47-05" --create_plots

# 最良モデル（optuna_best）で評価
python eval_real.py --datetime optuna_best --create_plots
```

### 主な機能
- **Optunaによる自動最適化**: TPE（Tree-structured Parzen Estimator）を使用した効率的なハイパーパラメータ探索
- **Pruning（枝刈り）**: 有望でないトライアルを早期終了して計算リソースを節約
- **結果の永続化**: SQLiteデータベースに結果を保存し、中断後も再開可能
- **最良トライアルの自動選択**: 検証損失が最小のトライアルを自動的に特定

詳細は `documents/OPTUNA_TUTORIAL.md` を参照してください。

## 推論スクリプト（main.py）

### 使用方法

`main.py`は学習済みモデルを使用して推論を行うメインスクリプトです。

```bash
# 3相モデルで推論
python main.py --3phase

# 2相モデルで推論
python main.py --2phase
```

### 設定ファイル

推論に使用する入力ファイルのパスは`config/config_inference.yaml`で指定します：

```yaml
input_file: /path/to/your/processed_data.npz
input_key: processed_data
device: cuda:0
```

### 重要な注意事項

#### 1. データ前処理の2段階ダウンサンプリング

推論時には、データセット作成時と学習時で異なるダウンサンプリング処理が適用されます：

**第1段階（データセット作成時）:**
- `resample_poly`による多項式補間リサンプリング
- H次元（時間軸）を14000 → 1400にダウンサンプリング（1/10）
- データセット作成時に一度だけ実行（`source/psdata2matlab/main.ipynb`で実行）

**第2段階（学習時・推論時）:**
- 単純なスライシング（`::downsample_factor`）
- H次元を1400 → 700にダウンサンプリング（1/2、`downsample_factor=2`の場合）
- 学習時と推論時の両方で実行

**推論時の処理フロー（`main.py --3phase`）:**
```
1. データロード（元データ: 14000サンプル）
   ↓
2. resample_poly (14000 → 1400) ← データセット作成時と同じ処理
   ↓
3. preprocess_input_to_nchw (形状変換 + チャネル0,2選択)
   ↓
4. downsample_factor (1400 → 700) ← 学習時と同じ処理
   ↓
5. evaluate関数で推論
```

**注意:** 
- 推論時には、元の生データ（14000サンプル）に対して`resample_poly`を適用する必要があります
- 既に`resample_poly`で処理済みのデータ（1400サンプル）を使用する場合は、`resample_poly`処理はスキップされます
- `downsample_factor`は学習時に使用された値と同じである必要があります（`models/sota/config.yaml`で確認）

#### 2. チャネル選択

- 3相モデルでは、4チャネルの入力からチャネル0と2のみを選択します
- チャネル1と3は自動的に除外されます

#### 3. モデル設定の確認

推論時には、学習時に使用された設定ファイル（`models/sota/config.yaml`）を参照します：
- `downsample_factor`: 学習時と同じ値を使用
- `model.in_channels`: 2（チャネル0と2のみ）
- `model.out_dim`: 6（3相モデルの場合）

#### 4. 出力形式

3相モデルの場合、出力は以下の形式で表示されます：
```
Phase fractions: [solid, gas, liquid]
  Solid:  0.xxxx
  Gas:    0.xxxx
  Liquid: 0.xxxx
```

固相体積率は以下の計算式で求められます：
- `solid_volume_fraction = 1 - (gas_volume_fraction + liquid_volume_fraction)`
- `solid_volume_fraction <= 0.03`の場合は0として出力されます

## ファイル構成

### ファイル構成

#### メインスクリプト

- `main.py` - 推論用メインスクリプト（推奨）
- `scripts/optimize.py` - Optunaハイパーパラメータ最適化
- `scripts/train_real.py` - 実データでの訓練
- `scripts/train.py` - シミュレーションデータでの訓練

#### テスト・評価スクリプト（`tests/`ディレクトリ）

- `tests/eval.py` - データセット全体の評価
- `tests/eval_tutorial.py` - 単一ファイル評価用
- `tests/evaluate_gradcam.py` - Grad-CAM可視化用（推奨）
- `tests/validate_dataset.py` - データセット検証用
- `tests/debug_tdx.py` - TDXデータのデバッグ用スクリプト
- `tests/compare_datasets.py` - データセット比較用

#### ツールスクリプト（`tools/`ディレクトリ）

- `tools/create_dropped_dataset.py` - データセット作成用（Channel 1と3を除外）
- `tools/visualize_gradcam.py` - Grad-CAM可視化用（`tests/evaluate_gradcam.py`が推奨）
- `tools/evalsim.py` - シミュレーションデータ評価用（使用頻度低）
- `tools/check_optuna_results.py` - Optuna結果確認用
- `tools/clear_gpu_memory.py` - GPUメモリクリア用
- `tools/inspect_model.py` - モデル構造確認用
- `tools/inspect_model_sim.py` - シミュレーションモデル構造確認用
- `tools/plot_signal_sample.py` - 信号サンプルプロット用

#### その他のファイル

- `test.ipynb` - テスト用Jupyterノートブック
- `optuna_tutorial.py` - Optunaチュートリアル用（`scripts/optimize.py`がメイン）

