# Machine Learning for airlift project

author : Sadanori Matsubara 
date: 2025/09/04

## dataset
 we use numerical simulation for creating training dataset while experiment for test dataset. the `train.py` uses simulatiuon dataset for training and `eval.py` does experiment dataset for evaluating.  
  Please note that while training and evaluating, we use different gpus. such setting have been written in `config/config.yaml` 

## Install 
 1. create virtual environment for inference  
    execute following commands
    ```
     python3 -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     pip3 install torch torchvision
    ```
 2. download dataset  
    prepare zip file 
 3. inference  
    eval.py uses model and new dataset to inference. `eval.py` processes whole dataset and `eval_tutorial.py` does one dataset. They share the same function; only difference is that they are used repeatedly.
    ```
    python eval.py --datetime models/layernorm
    ```
    or, edit `file_path` in `eval_tutorial.py` and execute it.

