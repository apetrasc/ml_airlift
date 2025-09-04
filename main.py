import torch
import numpy as np
import numpy as np
import torch
import yaml
import hydra
import os
import datetime
from omegaconf import OmegaConf

@hydra.main(config_path="/home/smatsubara/documents/sandbox/ml_airlift/config", config_name="config.yaml")
def main(config):
    import pathlib

    # Get current date and time
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")

    base_dir = pathlib.Path("/home/smatsubara/documents/airlift/data/output") / date_str / time_str
    logs_dir = base_dir / "logs"
    weights_dir = base_dir / "weights"

    # Create directories if they do not exist
    logs_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Update config paths for train.py and eval.py outputs
    config['dataset']['log_path'] = str(logs_dir)
    config['dataset']['weights_path'] = str(weights_dir)
    config['evaluation']['results_path'] = str(base_dir)
    config['evaluation']['save_csv_path'] = str(base_dir)
    config['evaluation']['save_path'] = str(base_dir / "predicted.csv")

    print(f"Experiment directories created:\n  logs: {logs_dir}\n  weights: {weights_dir}")
    print(f"Updated config for output paths:\n{config}")
    import shutil
    original_config_path = hydra.utils.to_absolute_path("config/config.yaml")
    copied_config_path = base_dir / "config.yaml"
    shutil.copy2(original_config_path, copied_config_path)
    print(f"Copied config.yaml to {copied_config_path}")

    # Overwrite the project config so train.py reads the updated paths
    overwrite_config_path = "/home/smatsubara/documents/sandbox/ml_airlift/config/config.yaml"
    OmegaConf.save(config, overwrite_config_path)
    print(f"Overwrote config at {overwrite_config_path} with updated paths")

    import subprocess
    train_script_path = hydra.utils.to_absolute_path("train.py")
    project_root = hydra.utils.get_original_cwd()
    # Copy model.py to the experiment directory for reproducibility
    model_py_src = hydra.utils.to_absolute_path("models/cnn.py")
    model_py_dst = base_dir / "cnn.py"
    shutil.copy2(model_py_src, model_py_dst)
    print(f"Copied model to {model_py_dst}")
    cli_overrides = [
        f"dataset.log_path={logs_dir}",
        f"dataset.weights_path={weights_dir}",
        f"evaluation.results_path={base_dir}",
        f"evaluation.save_csv_path={base_dir}",
        f"evaluation.save_path={base_dir / 'predicted.csv'}",
        f"hydra.run.dir={base_dir}"
    ]
    result = subprocess.run(
        ["python", train_script_path, *cli_overrides],
        cwd=project_root
    )
    if result.returncode != 0:
        print("train.py execution failed.")
    else:
        print("train.py executed successfully.")
if __name__ == "__main__":
    main()
