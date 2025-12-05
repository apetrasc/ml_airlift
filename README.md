# Machine Learning for airlift project

author : Sadanori Matsubara 
date: 2025/09/04

## dataset
 we use numerical simulation for creating training dataset while experiment for test dataset. the `train.py` uses simulatiuon dataset for training and `eval.py` does experiment dataset for evaluating.  
  Please note that while training and evaluating, we use different gpus. such setting have been written in `config/config.yaml`

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

## Installation

1. **Create a virtual environment for inference**  

   **For Linux:**  
   Run the following commands in your terminal:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install torch torchvision
   ```
   **For Windows:**  
   Open Command Prompt and run:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   pip install torch torchvision
   ```
   > **Note:**  
   > The command `pip install torch torchvision` is for the ordinally environment (Ubuntu + CUDA 12.8).  
   > If you are using a different OS or CUDA version, please refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) and install the appropriate version for your environment.
   This will set up a Python virtual environment and install all required dependencies, including PyTorch.

2. **Download the dataset**  
   Prepare the dataset as a zip file and extract it into the appropriate directory as specified in your configuration or code. Whole dataset is confidential so contact me.  
   After you prepared dataset, you should edit `config/config.yaml` and the location of dataset.in Linux, you can directly copy & paste path but in Windows, you must delete "" when pasting.


3. **Run inference**  
   - To perform inference on the entire dataset, use `eval.py`.  
     ```
     python eval.py --datetime models/layernorm
     ```
     This script will process the whole dataset using the trained model.
   - To run inference on a single dataset file, use `eval_tutorial.py`.  
     Edit the `file_path` variable in `eval_tutorial.py` to point to your data file, then execute the script:
     ```
     python eval_tutorial.py
     ```
   Both scripts use the same inference function; the main difference is that `eval.py` processes all data in batch, while `eval_tutorial.py` is intended for testing on a single file or for tutorial purposes.

If you encounter any issues with dependencies or dataset paths, please check your configuration files (e.g., `config/config.yaml`) and ensure all paths are set correctly.

