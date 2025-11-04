import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from models.cnn import SimpleCNNReal, SimpleCNNReal2D
import argparse
from src.evaluate_predictions import create_prediction_plots
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from datetime import datetime


def _load_np_any(path: str, prefer_key: str = None):
    """Load numpy array from .npy or .npz file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found: {path}")
    obj = np.load(path)
    # npz: dict-like, npy: ndarray
    if hasattr(obj, 'keys'):
        keys = list(obj.keys())
        if prefer_key and prefer_key in obj:
            arr = obj[prefer_key]
        else:
            if prefer_key and prefer_key not in obj:
                print(f"[WARN] key '{prefer_key}' not in {path}. Using first key: {keys[0]}")
            arr = obj[keys[0]]
        return arr
    else:
        if prefer_key:
            print(f"[INFO] {path} is an array (npy). Ignoring key '{prefer_key}'.")
        return obj


def load_npz_pair(x_path: str, t_path: str, x_key: str = "x_train_real", t_key: str = "t_train_real"):
    """Load x and t from npz/npy files robustly and return numpy arrays."""
    x = _load_np_any(x_path, x_key)
    t = _load_np_any(t_path, t_key)
    print(f"Loaded x: shape={x.shape}, dtype={x.dtype}")
    print(f"Loaded t: shape={t.shape}, dtype={t.dtype}")
    return x, t


def to_dataset(x: np.ndarray, t: np.ndarray):
    if x.ndim == 2:
        x = x[:, None, :]
    elif x.ndim == 3:
        pass
    elif x.ndim == 4:
        pass
    else:
        raise RuntimeError(f"Unsupported x shape: {x.shape}")
    if t.ndim == 2 and t.shape[1] == 1:
        t = t[:, 0]
    elif t.ndim == 2 and t.shape[1] > 1:
        pass
    elif t.ndim != 1:
        raise RuntimeError(f"Expected t to be 1D or (N,M), got {t.shape}")
    x_t = torch.from_numpy(x).float()
    y_t = torch.from_numpy(t).float()
    return TensorDataset(x_t, y_t)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    preds, targets = [], []
    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        preds.append(pred.cpu())
        targets.append(yb.cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    return mse, mae, preds.numpy(), targets.numpy()


def list_available_datetime_folders(base_dir: str = "/home/smatsubara/documents/airlift/data/outputs_real"):
    """List all available datetime folders."""
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return []
    
    folders = []
    for date_folder in sorted(os.listdir(base_dir)):
        date_path = os.path.join(base_dir, date_folder)
        if not os.path.isdir(date_path) or date_folder.startswith('.'):
            continue
            
        for time_folder in sorted(os.listdir(date_path)):
            time_path = os.path.join(date_path, time_folder)
            if not os.path.isdir(time_path) or time_folder.startswith('.'):
                continue
                
            folder_name = f"{date_folder}/{time_folder}"
            folders.append(folder_name)
    
    return folders


def find_datetime_folder(datetime_str: str, base_dir: str = "/home/smatsubara/documents/airlift/data/outputs_real"):
    """Find the folder matching the datetime string."""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    # Look for folders matching the datetime pattern
    for date_folder in os.listdir(base_dir):
        date_path = os.path.join(base_dir, date_folder)
        if not os.path.isdir(date_path):
            continue
            
        for time_folder in os.listdir(date_path):
            time_path = os.path.join(date_path, time_folder)
            if not os.path.isdir(time_path):
                continue
                
            # Check if this folder matches the datetime string
            folder_name = f"{date_folder}/{time_folder}"
            if datetime_str in folder_name or folder_name == datetime_str:
                return time_path
    
    # If not found, try direct path
    direct_path = os.path.join(base_dir, datetime_str)
    if os.path.exists(direct_path):
        return direct_path
    
    # List available folders for user reference
    available_folders = list_available_datetime_folders(base_dir)
    if available_folders:
        print(f"Available datetime folders:")
        for folder in available_folders:
            print(f"  - {folder}")
    
    raise FileNotFoundError(f"Datetime folder not found: {datetime_str} in {base_dir}")


def load_model_from_datetime_folder(datetime_folder: str, x_sample: torch.Tensor, out_dim: int, device: str):
    """Load model from datetime folder and return the model."""
    # Find model file
    weights_dir = os.path.join(datetime_folder, "weights")
    model_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')] if os.path.exists(weights_dir) else []
    
    if not model_files:
        raise FileNotFoundError(f"No .pth files found in {weights_dir}")
    
    model_path = os.path.join(weights_dir, model_files[0])
    print(f"[INFO] Loading model from: {model_path}")
    
    # Build model matching input shape
    if x_sample.ndim == 3:
        in_channels = x_sample.shape[1]
        length = x_sample.shape[2]
        model = SimpleCNNReal(input_length=length, in_channels=in_channels, out_dim=out_dim)
    elif x_sample.ndim == 4:
        # NHWC -> NCHW if needed
        if x_sample.shape[1] not in (1, 3, 4):
            x_nchw = x_sample.permute(0, 3, 1, 2).contiguous()
            print("[INFO] Transposed NHWC -> NCHW")
        else:
            x_nchw = x_sample
            
        in_channels = x_nchw.shape[1]
        model = SimpleCNNReal2D(in_channels=in_channels, out_dim=out_dim, resize_hw=None)
        print(f"[INFO] Using Conv2d model for {in_channels}-channel image data")
    else:
        raise RuntimeError("Unexpected tensor ndim for model selection")
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    return model


def setup_mlflow_evaluation(experiment_name="cnn_evaluation"):
    """Setup MLflow experiment for evaluation."""
    mlflow.set_tracking_uri("file:/home/smatsubara/documents/airlift/data/outputs_real/mlruns")
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run()


def log_evaluation_metrics(y_pred, y_true, target_names):
    """Log evaluation metrics to MLflow."""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    # Overall metrics
    overall_mse = np.mean((y_pred - y_true) ** 2)
    overall_mae = np.mean(np.abs(y_pred - y_true))
    
    mlflow.log_metrics({
        "eval_mse": overall_mse,
        "eval_mae": overall_mae
    })
    
    # Per-target metrics
    for i, target_name in enumerate(target_names):
        pred_i = y_pred[:, i]
        true_i = y_true[:, i]
        
        r2 = r2_score(true_i, pred_i)
        mse = mean_squared_error(true_i, pred_i)
        mae = mean_absolute_error(true_i, pred_i)
        rmse = np.sqrt(mse)
        
        mlflow.log_metrics({
            f"eval_{target_name}_r2": r2,
            f"eval_{target_name}_mse": mse,
            f"eval_{target_name}_mae": mae,
            f"eval_{target_name}_rmse": rmse
        })


def log_evaluation_artifacts(y_pred, y_true, plots_dir):
    """Log evaluation artifacts to MLflow."""
    # Log prediction results
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        np.save(os.path.join(temp_dir, "eval_y_pred.npy"), y_pred)
        np.save(os.path.join(temp_dir, "eval_y_true.npy"), y_true)
        mlflow.log_artifacts(temp_dir, "evaluation_predictions")
    
    # Log evaluation plots if they exist
    if os.path.exists(plots_dir):
        mlflow.log_artifacts(plots_dir, "evaluation_plots")


def get_best_model_from_experiment(experiment_name, metric="val_loss"):
    """Get the best model from an MLflow experiment."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"]
    )
    
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")
    
    best_run = runs[0]
    return best_run.info.run_id, best_run.data.metrics.get(metric, 0)


def load_model_from_mlflow(run_id, x_sample, out_dim, device):
    """Load model from MLflow run."""
    client = MlflowClient()
    run = client.get_run(run_id)
    
    # Download model
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    
    # Move to device
    model.to(device)
    
    print(f"[MLFLOW] Loaded model from run: {run_id}")
    print(f"[MLFLOW] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on test data")
    parser.add_argument("--x", default="/home/smatsubara/documents/sandbox/ml_airlift/cleaned_data/x_train_real_cleaned.npy",
                        help="Path to input data")
    parser.add_argument("--t", default="/home/smatsubara/documents/sandbox/ml_airlift/cleaned_data/t_train_real_cleaned.npy",
                        help="Path to target data")
    parser.add_argument("--x_key", default="x_train_real", help="Key for x data in .npz file")
    parser.add_argument("--t_key", default="t_train_real", help="Key for t data in .npz file")
    parser.add_argument("--datetime", type=str, help="Datetime folder to load model from (e.g., '2025-10-29/11-39-35')")
    parser.add_argument("--model", default="/home/smatsubara/documents/airlift/data/outputs_real/simple/model_simplecnn_real.pth",
                        help="Path to model file (used if --datetime not specified)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--resize_h", type=int, default=0, help="Height to resize input images to")
    parser.add_argument("--resize_w", type=int, default=0, help="Width to resize input images to")
    parser.add_argument("--ds_factor", type=int, default=1, help="Downsample factor along H for (N,C,H,W) inputs")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples for evaluation (0 for all)")
    parser.add_argument("--create_plots", action="store_true", help="Create evaluation plots")
    
    # MLflow arguments
    parser.add_argument("--use_mlflow", action="store_true", help="Use MLflow for evaluation tracking")
    parser.add_argument("--experiment_name", default="cnn_evaluation", help="MLflow experiment name")
    parser.add_argument("--run_name", default=None, help="MLflow run name")
    parser.add_argument("--tags", nargs="*", help="MLflow tags (format: key=value)")
    parser.add_argument("--best_model", action="store_true", help="Use best model from training experiment")
    parser.add_argument("--training_experiment", default="cnn_real_data", help="Training experiment name for best model")
    parser.add_argument("--mlflow_run_id", type=str, help="Specific MLflow run ID to evaluate")
    
    args = parser.parse_args()

    print("🔍 Model Evaluation Script")
    print("=" * 50)
    print(f"[INFO] MLflow enabled: {args.use_mlflow}")
    
    # Setup MLflow if enabled
    if args.use_mlflow:
        run_context = setup_mlflow_evaluation(args.experiment_name)
        run_name = args.run_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.set_tag("mlflow.runName", run_name)
        
        # Set tags
        if args.tags:
            for tag in args.tags:
                if "=" in tag:
                    key, value = tag.split("=", 1)
                    mlflow.set_tag(key, value)
        
        print(f"[MLFLOW] Experiment: {args.experiment_name}")
        print(f"[MLFLOW] Run: {run_name}")
    
    # Determine output directory
    if args.datetime:
        try:
            datetime_folder = find_datetime_folder(args.datetime)
            print(f"[INFO] Using datetime folder: {datetime_folder}")
            output_dir = os.path.join(datetime_folder, "evaluation_plots")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return
    else:
        output_dir = "/home/smatsubara/documents/airlift/data/outputs_real/simple"
        print(f"[INFO] Using default output directory: {output_dir}")
    
    device = args.device
    print(f"[INFO] Device: {device}")

    # Load data
    print("[STEP] Loading dataset files...")
    x, t = load_npz_pair(args.x, args.t, args.x_key, args.t_key)
    print(f"[OK] Loaded. x.shape={x.shape}, t.shape={t.shape}")
    
    # Limit samples if specified
    if args.limit > 0:
        n = min(args.limit, x.shape[0])
        x = x[:n]
        t = t[:n]
        print(f"[INFO] Limited to first {n} samples")
    
    # Optional downsampling
    if x.ndim == 4 and args.ds_factor > 1:
        h0 = x.shape[2]
        x = x[:, :, ::args.ds_factor, :]
        print(f"[INFO] Downsampled H: {h0} -> {x.shape[2]} (factor={args.ds_factor})")
    elif x.ndim == 4:
        print(f"[INFO] Using full resolution: H={x.shape[2]}, W={x.shape[3]}")
    
    # Create dataset
    print("[STEP] Building dataset...")
    dataset = to_dataset(x, t)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False)
    print(f"[OK] Dataset ready. Samples: {len(dataset)}")

    # Load model
    print("[STEP] Loading model...")
    x_sample = dataset.tensors[0]
    out_dim = dataset.tensors[1].shape[1] if dataset.tensors[1].ndim == 2 else 1
    
    # Determine model source
    if args.use_mlflow and args.best_model:
        # Load best model from MLflow experiment
        try:
            run_id, best_metric = get_best_model_from_experiment(args.training_experiment)
            print(f"[MLFLOW] Using best model from experiment '{args.training_experiment}' (run_id: {run_id}, metric: {best_metric})")
            model = load_model_from_mlflow(run_id, x_sample, out_dim, device)
        except Exception as e:
            print(f"[ERROR] Failed to load best model from MLflow: {e}")
            return
    elif args.use_mlflow and args.mlflow_run_id:
        # Load specific model from MLflow run
        try:
            print(f"[MLFLOW] Loading model from run_id: {args.mlflow_run_id}")
            model = load_model_from_mlflow(args.mlflow_run_id, x_sample, out_dim, device)
        except Exception as e:
            print(f"[ERROR] Failed to load model from MLflow run: {e}")
            return
    elif args.datetime:
        # Load from datetime folder
        model = load_model_from_datetime_folder(datetime_folder, x_sample, out_dim, device)
    else:
        # Load from specified model path
        model_path = args.model
        print(f"[INFO] Loading model from: {model_path}")
        
        if x_sample.ndim == 3:
            in_channels = x_sample.shape[1]
            length = x_sample.shape[2]
            model = SimpleCNNReal(input_length=length, in_channels=in_channels, out_dim=out_dim)
        elif x_sample.ndim == 4:
            if x_sample.shape[1] not in (1, 3, 4):
                x_nchw = x_sample.permute(0, 3, 1, 2).contiguous()
                dataset.tensors = (x_nchw, dataset.tensors[1])
                x_sample = x_nchw
            in_channels = x_sample.shape[1]
            resize_hw = (args.resize_h, args.resize_w) if args.resize_h > 0 and args.resize_w > 0 else None
            model = SimpleCNNReal2D(in_channels=in_channels, out_dim=out_dim, resize_hw=resize_hw)
        else:
            raise RuntimeError("Unexpected tensor ndim for model selection")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    
    print(f"[OK] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Evaluate model
    print("[STEP] Evaluating model...")
    mse, mae, y_pred, y_true = evaluate(model, loader, device)
    print(f"[OK] Evaluation complete!")
    print(f"All-data Eval | MSE={mse:.6f} | MAE={mae:.6f}")
    
    # Print per-target results for multi-output regression
    if y_pred.shape[1] > 1:
        print(f"[INFO] Per-target results:")
        for i in range(y_pred.shape[1]):
            target_mse = np.mean((y_pred[:, i] - y_true[:, i])**2)
            target_mae = np.mean(np.abs(y_pred[:, i] - y_true[:, i]))
            print(f"  Target {i+1}: MSE={target_mse:.6f}, MAE={target_mae:.6f}")

    # Save results
    print("[STEP] Saving results...")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "eval_y_true.npy"), y_true)
    np.save(os.path.join(output_dir, "eval_y_pred.npy"), y_pred)
    print(f"[OK] Results saved to: {output_dir}")

    # Create evaluation plots if requested
    if args.create_plots and y_pred.shape[1] > 1:
        print("[STEP] Creating evaluation plots...")
        target_names = [
            "Solid Velocity", "Gas Velocity", "Liquid Velocity",
            "Solid Volume Fraction", "Gas Volume Fraction", "Liquid Volume Fraction"
        ]
        create_prediction_plots(y_pred, y_true, output_dir, target_names)
        print(f"[OK] Evaluation plots saved to: {output_dir}")
        
        # Log evaluation metrics and artifacts to MLflow
        if args.use_mlflow:
            log_evaluation_metrics(y_pred, y_true, target_names)
            log_evaluation_artifacts(y_pred, y_true, output_dir)
            print(f"[MLFLOW] Evaluation metrics and artifacts logged")
    elif args.use_mlflow and y_pred.shape[1] > 1:
        # Log metrics even without plots
        target_names = [
            "Solid Velocity", "Gas Velocity", "Liquid Velocity",
            "Solid Volume Fraction", "Gas Volume Fraction", "Liquid Volume Fraction"
        ]
        log_evaluation_metrics(y_pred, y_true, target_names)
        print(f"[MLFLOW] Evaluation metrics logged")
    
    print(f"[DONE] Evaluation completed successfully!")
    
    # Close MLflow run if enabled
    if args.use_mlflow:
        run_context.__exit__(None, None, None)
        print(f"[MLFLOW] Evaluation run completed. View results in MLflow UI")


if __name__ == "__main__":
    main()


