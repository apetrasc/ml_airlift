# Machine Learning for airlift project

author : Sadanori Matsubara 
date: 2025/09/04

## dataset
 we use numerical simulation for creating training dataset while experiment for test dataset. the `train.py` uses simulatiuon dataset for training and `eval.py` does experiment dataset for evaluating.  
  Please note that while training and evaluating, we use different gpus. such setting have been written in `config/config.yaml`

## MLflow統合による研究効率化

`train_real.py`と`eval_real.py`にMLflowを統合し、実験の自動追跡・管理・比較が可能になりました。

### 基本的な使用方法

#### 学習（MLflow統合）
```bash
# MLflow有効で学習
python train_real.py --use_mlflow --epochs 20 --batch 4 --limit 0

# 実験名とランネームを指定
python train_real.py --use_mlflow --experiment_name "cnn_experiments" --run_name "baseline_model"

# タグを追加
python train_real.py --use_mlflow --tags "model_type=baseline" "data_size=full" "optimizer=adam"
```

#### 評価（MLflow統合）
```bash
# MLflow有効で評価
python eval_real.py --use_mlflow --datetime "2025-10-29/11-39-35" --create_plots

# 最良モデルを自動選択して評価
python eval_real.py --use_mlflow --best_model --training_experiment "cnn_real_data" --create_plots

# 特定のMLflow run IDで評価
python eval_real.py --use_mlflow --mlflow_run_id "abc123def456" --create_plots
```

#### MLflow UIの起動
```bash
# MLflow UIを起動
mlflow ui --backend-store-uri file:/home/smatsubara/documents/airlift/data/outputs_real/mlruns
# ブラウザで http://localhost:5000 にアクセス
```

### 主な機能
- **実験の自動追跡**: ハイパーパラメータ、メトリクス、モデルの自動記録
- **実験結果の比較**: 複数実験を一覧で比較
- **最良モデルの自動選択**: メトリクスに基づく自動選択
- **再現性の確保**: 完全な実験記録で再現可能
- **チーム協力**: 実験結果の共有が容易

詳細は `README_complete_workflow.md` を参照してください。 

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

