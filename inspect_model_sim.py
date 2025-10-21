import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append('/home/smatsubara/documents/sandbox/ml_airlift')

# Import SimpleCNN directly from the file to avoid import issues
import importlib.util
spec = importlib.util.spec_from_file_location("cnn_module", "/home/smatsubara/documents/sandbox/ml_airlift/models/cnn.py")
cnn_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnn_module)
SimpleCNN = cnn_module.SimpleCNN

def create_signal_gradcam_overlay_sim(
    data_path, model_path, model_class, sample_index=0,
    save_path='signal_gradcam_overlay_sim.png', device='cpu', input_length=2500
):
    """
    Create an overlay visualization of simulation signal data and Grad-CAM heatmap.
    
    Args:
        data_path (str): Path to the .npy file containing simulation signal data
        model_path (str): Path to the trained model weights
        model_class (callable): Model construction function
        sample_index (int): Index of the sample to visualize
        save_path (str): Path to save the overlay image
        device (str): Device to use for computation
        input_length (int): Input length for model construction
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import hilbert
    
    # Load the model
    model = model_class(input_length=input_length)
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    
    # Load simulation signal data
    x = np.load(data_path)
    
    if x.ndim != 2:
        raise RuntimeError(f"Expected 2D array for simulation data, but got shape {x.shape}")
    
    N, L = x.shape
    if not (0 <= sample_index < N):
        raise RuntimeError(f"Invalid sample_index {sample_index}, N={N}")
    
    # Extract signal
    signal = x[sample_index, :]
    original_signal = signal.copy()
    
    # Normalize signal (same as in preprocess_for_gradcam)
    mean = np.mean(signal)
    std = np.std(signal)
    std = std if std > 1e-8 else 1e-8
    signal = (signal - mean) / std
    
    # Convert to tensor for model
    signal_tensor = torch.tensor(signal, dtype=torch.float32)
    signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
    
    # Create Grad-CAM
    class GradCAM1d:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None
            self.hook_handles = []
            self._register_hooks()

        def _register_hooks(self):
            def forward_hook(module, input, output):
                self.activations = output.detach()
            def backward_hook(module, grad_in, grad_out):
                self.gradients = grad_out[0].detach()
            self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
            self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

        def __call__(self, input_tensor):
            self.model.eval()
            input_tensor = input_tensor.to(next(self.model.parameters()).device)
            self.model.zero_grad()
            out = self.model(input_tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
            target = out.squeeze()
            if target.ndim > 0:
                target = target.sum()
            self.model.zero_grad()
            target.backward(retain_graph=True)
            gradients = self.gradients         # [B, C, L]
            activations = self.activations     # [B, C, L]
            weights = gradients.mean(dim=2, keepdim=True)  # [B, C, 1]
            grad_cam_map = (weights * activations).sum(dim=1, keepdim=True)  # (B,1,L)
            grad_cam_map = torch.relu(grad_cam_map)
            grad_cam_map = torch.nn.functional.interpolate(
                grad_cam_map, size=input_tensor.shape[2], mode='linear', align_corners=False
            )
            grad_cam_map = grad_cam_map.squeeze().cpu().numpy()
            grad_cam_map = (grad_cam_map - grad_cam_map.min()) / (grad_cam_map.max() - grad_cam_map.min() + 1e-8)
            return grad_cam_map

        def remove_hooks(self):
            for handle in self.hook_handles:
                handle.remove()
    
    # Find target layer (last Conv1d layer)
    target_layer = None
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv1d):
            target_layer = m
            break
    
    if target_layer is None:
        raise RuntimeError("Could not find a suitable layer for Grad-CAM.")
    
    # Generate Grad-CAM
    gradcam = GradCAM1d(model, target_layer)
    grad_cam_map = gradcam(signal_tensor)
    gradcam.remove_hooks()
    
    # Create time axis
    time_axis = np.arange(len(signal))
    
    # Apply Hilbert transform to get envelope (similar to npz2png)
    analytic_signal = np.abs(hilbert(original_signal))
    
    # Create overlay visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Original signal
    ax1.plot(time_axis, original_signal, 'b-', linewidth=1, alpha=0.7, label='Original Signal')
    ax1.plot(time_axis, analytic_signal, 'r-', linewidth=1.5, label='Envelope')
    ax1.set_title(f'Simulation Signal (Sample {sample_index})')
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Grad-CAM heatmap
    ax2.plot(time_axis, grad_cam_map, 'g-', linewidth=2, label='Grad-CAM Intensity')
    ax2.fill_between(time_axis, 0, grad_cam_map, alpha=0.3, color='green')
    ax2.set_title('Grad-CAM Heatmap')
    ax2.set_xlabel('Time Index')
    ax2.set_ylabel('Grad-CAM Intensity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Overlay visualization
    # Create dual y-axis for overlay
    ax3_twin = ax3.twinx()
    
    # Plot signal on left y-axis
    line1 = ax3.plot(time_axis, original_signal, 'b-', linewidth=1, alpha=0.8, label='Original Signal')
    line2 = ax3.plot(time_axis, analytic_signal, 'r-', linewidth=1.5, label='Envelope')
    ax3.set_xlabel('Time Index')
    ax3.set_ylabel('Signal Amplitude', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')
    
    # Plot Grad-CAM on right y-axis
    line3 = ax3_twin.plot(time_axis, grad_cam_map, 'g-', linewidth=2, alpha=0.8, label='Grad-CAM')
    ax3_twin.fill_between(time_axis, 0, grad_cam_map, alpha=0.2, color='green')
    ax3_twin.set_ylabel('Grad-CAM Intensity', color='green')
    ax3_twin.tick_params(axis='y', labelcolor='green')
    ax3_twin.set_ylim(0, 1)
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')
    
    ax3.set_title('Simulation Signal and Grad-CAM Overlay')
    ax3.grid(True, alpha=0.3)
    
    # Highlight high Grad-CAM regions with multiple thresholds
    thresholds = [0.3, 0.5, 0.7]
    colors = ['lightblue', 'orange', 'yellow']
    labels = ['Medium (>0.3)', 'High (>0.5)', 'Very High (>0.7)']
    
    for i, (threshold, color, label) in enumerate(zip(thresholds, colors, labels)):
        mask = grad_cam_map > threshold
        if np.any(mask):
            ax3.axvspan(
                time_axis[mask][0], 
                time_axis[mask][-1], 
                alpha=0.1, color=color, label=f'Grad-CAM {label}'
            )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print detailed analysis with multiple thresholds
    print(f"\nSimulation Signal-GradCAM Overlay Analysis:")
    print(f"Sample: {sample_index}")
    print(f"Signal length: {len(signal)}")
    print(f"Max Grad-CAM intensity: {grad_cam_map.max():.4f}")
    
    for threshold in [0.3, 0.5, 0.7]:
        mask = grad_cam_map > threshold
        if np.any(mask):
            region_start = time_axis[mask][0]
            region_end = time_axis[mask][-1]
            points = np.sum(mask)
            signal_mean = original_signal[mask].mean()
            signal_std = original_signal[mask].std()
            print(f"Grad-CAM > {threshold}: {points} points, region {region_start}-{region_end}, signal: {signal_mean:.4f} ± {signal_std:.4f}")
        else:
            print(f"Grad-CAM > {threshold}: No regions found")
    
    # Find peaks in Grad-CAM
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(grad_cam_map, height=0.3, distance=50)
    if len(peaks) > 0:
        print(f"Grad-CAM peaks found at positions: {peaks}")
        print(f"Peak intensities: {grad_cam_map[peaks]}")
    
    print(f"Overlay image saved to: {save_path}")

def create_all_simulation_gradcams(
    data_path='/home/smatsubara/documents/airlift/data/simulation/dataset/x_train.npy',
    model_path='/home/smatsubara/documents/airlift/data/results/layernorm/weights/model.pth',
    output_dir='/home/smatsubara/documents/airlift/data/results/gradcamsim_outputs',
    device='cpu',
    input_length=2500
):
    """
    Create Grad-CAM visualizations for all simulation samples.
    
    Args:
        data_path (str): Path to simulation data
        model_path (str): Path to model weights
        output_dir (str): Directory to save output images
        device (str): Device to use
        input_length (int): Input length for model
    """
    # Load data to get number of samples
    x = np.load(data_path)
    N = x.shape[0]
    
    print(f"Creating Grad-CAM visualizations for {N} simulation samples...")
    
    # Create all visualizations
    for sample_idx in range(N):
        save_path = os.path.join(output_dir, f'simulation_gradcam_overlay_sample_{sample_idx}.png')
        
        try:
            create_signal_gradcam_overlay_sim(
                data_path=data_path,
                model_path=model_path,
                model_class=lambda input_length=2500: SimpleCNN(input_length=input_length),
                sample_index=sample_idx,
                save_path=save_path,
                device=device,
                input_length=input_length
            )
            print(f"Completed sample {sample_idx + 1}/{N}")
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            continue
    
    print(f"All simulation Grad-CAM visualizations completed and saved to {output_dir}")

def create_prediction_comparison_plot(
    x_data_path='/home/smatsubara/documents/airlift/data/simulation/dataset/x_train.npy',
    t_data_path='/home/smatsubara/documents/airlift/data/simulation/dataset/t_train.npy',
    model_path='/home/smatsubara/documents/airlift/data/results/layernorm/weights/model.pth',
    save_path='/home/smatsubara/documents/airlift/data/results/gradcamsim_outputs/prediction_comparison.png',
    device='cpu',
    input_length=2500
):
    """
    Create a comparison plot between model predictions and actual t_train values.
    
    Args:
        x_data_path (str): Path to simulation input data
        t_data_path (str): Path to simulation target data
        model_path (str): Path to model weights
        save_path (str): Path to save the comparison plot
        device (str): Device to use
        input_length (int): Input length for model
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load the model
    model = SimpleCNN(input_length=input_length)
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    
    # Load data
    x_data = np.load(x_data_path)  # (213, 2500)
    t_data = np.load(t_data_path)  # (213,)
    
    print(f"Loaded data shapes: x_data={x_data.shape}, t_data={t_data.shape}")
    
    # Preprocess data for model input
    predictions = []
    for i in range(len(x_data)):
        signal = x_data[i, :]  # (2500,)
        
        
        # Convert to tensor
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 2500]
        
        # Get prediction
        with torch.no_grad():
            pred = model(signal_tensor)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            pred_value = pred.item()
            predictions.append(pred_value)
    
    predictions = np.array(predictions)
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Scatter plot - Predictions vs Actual
    ax1.scatter(t_data, predictions, alpha=0.7, s=50)
    ax1.plot([t_data.min(), t_data.max()], [t_data.min(), t_data.max()], 'r--', 
             label='Perfect prediction line')
    ax1.set_xlabel('Actual t_train values')
    ax1.set_ylabel('Model predictions')
    ax1.set_title('Predictions vs Actual Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calculate and display correlation
    correlation = np.corrcoef(t_data, predictions)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Residuals plot
    residuals = predictions - t_data
    ax2.scatter(t_data, residuals, alpha=0.7, s=50)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Actual t_train values')
    ax2.set_ylabel('Residuals (Prediction - Actual)')
    ax2.set_title('Residuals Plot')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution comparison
    ax3.hist(t_data, bins=20, alpha=0.7, label='Actual t_train', color='blue')
    ax3.hist(predictions, bins=20, alpha=0.7, label='Predictions', color='red')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Sample index plot
    sample_indices = np.arange(len(t_data))
    ax4.plot(sample_indices, t_data, 'b-o', label='Actual t_train', markersize=3, alpha=0.7)
    ax4.plot(sample_indices, predictions, 'r-s', label='Predictions', markersize=3, alpha=0.7)
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Value')
    ax4.set_title('Values by Sample Index')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\nPrediction Comparison Statistics:")
    print(f"Number of samples: {len(t_data)}")
    print(f"Correlation coefficient: {correlation:.4f}")
    print(f"Mean Absolute Error (MAE): {np.mean(np.abs(residuals)):.6f}")
    print(f"Root Mean Square Error (RMSE): {np.sqrt(np.mean(residuals**2)):.6f}")
    print(f"R² score: {1 - np.sum(residuals**2) / np.sum((t_data - np.mean(t_data))**2):.4f}")
    
    print(f"\nActual t_train statistics:")
    print(f"  Mean: {np.mean(t_data):.6f}, Std: {np.std(t_data):.6f}")
    print(f"  Min: {np.min(t_data):.6f}, Max: {np.max(t_data):.6f}")
    
    print(f"\nPrediction statistics:")
    print(f"  Mean: {np.mean(predictions):.6f}, Std: {np.std(predictions):.6f}")
    print(f"  Min: {np.min(predictions):.6f}, Max: {np.max(predictions):.6f}")
    
    print(f"Comparison plot saved to: {save_path}")

# Execute the analysis for all simulation samples
if __name__ == "__main__":
    create_all_simulation_gradcams()
    create_prediction_comparison_plot()
