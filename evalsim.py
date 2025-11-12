import os
import sys
import numpy as np
import torch
from models import SimpleCNN

# Ensure project root is on path
sys.path.append('/home/smatsubara/documents/sandbox/ml_airlift')


def evaluate_simulation(
    x_path: str = "/home/smatsubara/documents/airlift/data/simulation/arcaiv/x_train.npy",
    t_path: str = "/home/smatsubara/documents/airlift/data/simulation/arcaiv/t_train.npy",
    weights_path: str = "/home/smatsubara/documents/airlift/data/results/layernorm/weights/model.pth",
    output_dir: str = "/home/smatsubara/documents/airlift/data/results/gradcamsim_outputs",
    device: str = "cuda:0",
    input_length: int = 2500,
):
    """
    Run inference on x_train without any preprocessing, and save predictions.

    - Loads x_train (N, L) and optional t_train (N,)
    - Builds SimpleCNN(input_length)
    - Loads weights and runs model in eval mode
    - Saves predictions to output_dir/predictions_x_train.npy
    - If t_train exists, also saves a CSV with (t_train, prediction)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    x = np.load(x_path)  # expected (N, L)
    t = None
    if os.path.exists(t_path):
        t = np.load(t_path)

    if x.ndim != 2:
        raise RuntimeError(f"Expected x to be 2D (N, L), got {x.shape}")

    # Build model
    model = SimpleCNN(input_length)
    state_dict = torch.load(weights_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        model.load_state_dict(state_dict["state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Prepare tensor: no normalization, no scaling
    x_tensor = torch.from_numpy(x).float()  # (N, L)
    if x_tensor.ndim == 2:
        x_tensor = x_tensor.unsqueeze(1)  # (N, 1, L)
    x_tensor = x_tensor.to(device)

    # 必要なdataloaderを自前で定義し、このスクリプトの目的に合うように推論・保存
    # x_tensor: (N, 1, L), t: (N,) or None

    predictions = []
    with torch.no_grad():
        for i in range(x_tensor.shape[0]):
            input_tensor = x_tensor[i].unsqueeze(0)  # (1, 1, L)
            output = model(input_tensor)
            if isinstance(output, (tuple, list)):
                pred_value = output[0].item()
            else:
                pred_value = output.item()
            predictions.append(pred_value)
    predictions = np.array(predictions)

    print(f"Predictions shape: {predictions.shape}")
    if t is not None:
        print(f"t_train shape: {t.shape}")

    # Save outputs
    pred_path = os.path.join(output_dir, "predictions_x_train.npy")
    np.save(pred_path, predictions)

    # Optionally save paired CSV if t is available
    if t is not None and t.shape[0] == predictions.shape[0]:
        csv_path = os.path.join(output_dir, "predictions_vs_t_train.csv")
        header = "t_train,prediction"
        np.savetxt(
            csv_path,
            np.stack([t.astype(np.float64), predictions.astype(np.float64)], axis=1),
            delimiter=",",
            header=header,
            comments="",
        )
        print(f"Saved predictions and targets CSV to: {csv_path}")

    print(f"Saved predictions to: {pred_path}")
    print(f"Predictions shape: {predictions.shape}")
    if t is not None:
        print(f"t_train shape: {t.shape}")


if __name__ == "__main__":
    evaluate_simulation()


