# Machine Learning for airlift project

author : Sadanori Matsubara 
date: 2025/09/04

## dataset
 we use numerical simulation for creating training dataset while experiment for test dataset. the `train.py` uses simulatiuon dataset for training and `eval.py` does experiment dataset for evaluating.  
  Please note that while training and evaluating, we use different gpus. such setting have been written in `config/config.yaml` 

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

