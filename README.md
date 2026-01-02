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

- データセット全体に対して推論を実行するには、`eval.py`を使用してください：
  ```bash
  python eval.py --datetime models/layernorm
  ```
  このスクリプトは、訓練済みモデルを使用してデータセット全体を処理します。

- 単一のデータセットファイルに対して推論を実行するには、`eval_tutorial.py`を使用してください：
  `eval_tutorial.py`の`file_path`変数をデータファイルを指すように編集し、スクリプトを実行してください：
  ```bash
  python eval_tutorial.py
  ```
  
  両方のスクリプトは同じ推論関数を使用します。主な違いは、`eval.py`がバッチで全データを処理するのに対し、`eval_tutorial.py`は単一ファイルのテストやチュートリアル目的で使用されることです。

依存関係やデータセットパスに関する問題が発生した場合は、設定ファイル（例：`config/config.yaml`）を確認し、すべてのパスが正しく設定されていることを確認してください。

## Dataset

### データセットの概要

このプロジェクトでは、数値シミュレーションで作成したデータを訓練データセットとして使用し、実験データをテストデータセットとして使用します。`train.py`はシミュレーションデータで訓練を行い、`eval.py`は実験データで評価を行います。

訓練と評価では異なるGPUを使用するため、`config/config.yaml`で設定を確認してください。

### データセットの構築(Channel 1とChannel 3を除外)

#### 1. データセットの準備

元のデータセットファイル（`x_train.npy`と`t_train.npy`）を準備します。データセットは機密情報のため、必要に応じて連絡してください。


```bash
# Channel 1とChannel 3を除外したデータセットを作成
python create_dropped_dataset.py
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
python validate_dataset.py
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
python train_real.py

# シミュレーションデータで訓練
python train.py
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

