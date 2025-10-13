import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from models import SimpleCNN
sys.path.append('/home/smatsubara/documents/sandbox/ml_airlift')

def visualize_large_weights(
    model_path, model_class, target_layer_name, 
    save_path='large_weights_viz.png', device='cpu', input_length=None
):
    """
    Load the model and visualize locations with large weights in a target layer.
    Handles shape-mismatch errors by only loading compatible parameters.

    Args:
        model_path (str): Path to the saved model weights (.pth).
        model_class (callable): Class of the model to instantiate.
        target_layer_name (str): Name of the layer to inspect (e.g. "conv1", "linear1").
        save_path (str): Where to save the plot.
        device (str): Device to load the model on.
        input_length (int, optional): Input length for the model, if needed by model_class.
    """
    # Instantiate the model (optionally allow passing input_length, for interactive use)
    model = model_class() if input_length is None else model_class(input_length)
    model = model.to(device)

    # Safer loading: only load weights, avoid warnings and errors due to shape-mismatch
    state_dict = torch.load(model_path, map_location=device)
    if "state_dict" in state_dict:
        # Lightning or other framework wrapping
        state_dict = state_dict["state_dict"]
    # Remove keys with size mismatch from state_dict
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    mismatched = []
    for k, v in state_dict.items():
        if k in model_state_dict and model_state_dict[k].shape == v.shape:
            filtered_state_dict[k] = v
        else:
            mismatched.append(k)
    if mismatched:
        print("Skipping parameters due to shape mismatch:", mismatched)
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()

    # Get the target layer
    target_layer = dict(model.named_modules()).get(target_layer_name, None)
    if target_layer is None:
        raise ValueError(f"Layer '{target_layer_name}' not found in the model.")

    # Extract weights
    weights = None
    # Try different possible weight attributes
    for attr in ['weight', 'weight_raw']:
        if hasattr(target_layer, attr):
            weights = getattr(target_layer, attr).detach().cpu().numpy()
            break
    if weights is None:
        raise ValueError(f"Layer '{target_layer_name}' does not have a weight attribute.")

    # Visualization depends on weight dimension
    plt.figure(figsize=(10,4))
    if weights.ndim == 2:  # Linear
        plt.imshow(np.abs(weights), aspect='auto', cmap='viridis')
        plt.colorbar(label='|weight value|')
        plt.title(f'Absolute Weights in Layer: {target_layer_name}')
        plt.xlabel('Input features')
        plt.ylabel('Output features')
    elif weights.ndim == 3:  # Conv1d
        # For 1d conv: [out_channels, in_channels, kernel_size]
        abs_weights = np.abs(weights)
        mean_abs = abs_weights.mean(axis=(0, 1))
        plt.plot(mean_abs)
        plt.title(f'Mean |Weight| across filters (Layer: {target_layer_name})')
        plt.xlabel('Kernel Position')
        plt.ylabel('Mean Absolute Weight')
    else:
        plt.hist(np.abs(weights).flatten(), bins=50)
        plt.title(f'Histogram of |Weights| (Layer: {target_layer_name})')
        plt.xlabel('|weight value|')
        plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")

# Example usage:
# Adjust input_length to match the data/model weights you want to inspect
visualize_large_weights(
    model_path='/home/smatsubara/documents/airlift/data/results/layernorm/weights/model.pth',
    model_class=lambda input_length=2500: SimpleCNN(input_length=input_length),  # Use correct input_length for savefile!
    target_layer_name='conv1',   # Name of the layer you want to inspect
    save_path='large_weights_example.png',
    device='cpu',
    input_length=2500           # <-- Set to the value that matches your checkpoint!
)
# --- Visualize both Conv1d weights: conv1 and conv2 (if they exist) ---

def visualize_conv_weights(model_path, model_class, layer_names, save_prefix='weights_vis', device='cpu', input_length=2500):
    """
    Visualize weights from multiple Conv1d layers in a model.
    Args:
        model_path (str): Path to model checkpoint.
        model_class (callable): function or class for model construction (must accept input_length).
        layer_names (list of str): List of attribute names (e.g., ['conv1', 'conv2']).
        save_prefix (str): Prefix for saved image files.
        device (str): Device specifier.
        input_length (int): Input length for constructing model.
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    model = model_class(input_length=input_length)
    state_dict = torch.load(model_path, map_location=device)
    # Try to support models saved with/without 'state_dict' key
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.eval()

    for lname in layer_names:
        # Recursively support submodules separated by dots (e.g., 'layer1.conv')
        layer = model
        for part in lname.split('.'):
            if hasattr(layer, part):
                layer = getattr(layer, part)
            else:
                layer = None
                break
        if layer is None:
            print(f"Layer '{lname}' not found in the model -- skipping.")
            continue
        if hasattr(layer, 'weight'):
            weights = layer.weight.detach().cpu().numpy()
        else:
            print(f"Layer '{lname}' does not have a 'weight' attribute -- skipping.")
            continue

        plt.figure(figsize=(10, 4))
        if weights.ndim == 3:  # Conv1d: [out_ch, in_ch, ksize]
            abs_weights = np.abs(weights)
            mean_abs = abs_weights.mean(axis=(0, 1))
            plt.plot(mean_abs)
            plt.title(f'{lname}: Mean |Weight| across all filters')
            plt.xlabel('Kernel Position')
            plt.ylabel('Mean Absolute Weight')
        elif weights.ndim == 2:  # Linear
            plt.imshow(np.abs(weights), aspect='auto', cmap='viridis')
            plt.colorbar(label='|weight value|')
            plt.title(f'Absolute Weights in Layer: {lname}')
            plt.xlabel('Input features')
            plt.ylabel('Output features')
        else:
            plt.hist(np.abs(weights).flatten(), bins=50)
            plt.title(f'Histogram of |Weights| (Layer: {lname})')
            plt.xlabel('|weight value|')
            plt.ylabel('Count')

        plt.tight_layout()
        save_path = f"{save_prefix}_{lname}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Visualization saved to {save_path}")

# Example usage for both conv1 and conv2:
visualize_conv_weights(
    model_path='/home/smatsubara/documents/airlift/data/results/layernorm/weights/model.pth',
    model_class=lambda input_length=2500: SimpleCNN(input_length=input_length),
    layer_names=['conv1', 'conv2'],
    save_prefix='large_weights_conv',
    device='cpu',
    input_length=2500
)
# fc層の重みを可視化する例を追加
visualize_conv_weights(
    model_path='/home/smatsubara/documents/airlift/data/results/layernorm/weights/model.pth',
    model_class=lambda input_length=2500: SimpleCNN(input_length=input_length),
    layer_names=['fc'],  # fc層の重みを見る
    save_prefix='large_weights_fc',
    device='cpu',
    input_length=2500
)

def visualize_channel_contribution(
    model_path, model_class, target_fc_channel, save_prefix='channel_contribution', 
    device='cpu', input_length=2500
):
    """
    Visualize which convolution kernels contribute most to a specific FC channel.
    
    Args:
        model_path (str): Path to model checkpoint.
        model_class (callable): Model construction function.
        target_fc_channel (int): Which FC channel to analyze (0-indexed).
        save_prefix (str): Prefix for saved images.
        device (str): Device specifier.
        input_length (int): Input length for model construction.
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load model
    model = model_class(input_length=input_length)
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    
    # Get FC weights to see contribution of each channel
    fc_weights = model.fc.weight.detach().cpu().numpy()
    print(f"FC weights: {fc_weights.flatten()}")
    print(f"Weight for channel {target_fc_channel}: {fc_weights[0, target_fc_channel]}")
    
    # Get Conv1 weights - these contribute to all conv2 channels
    conv1_weights = model.conv1.weight.detach().cpu().numpy()  # [16, 1, 201]
    print(f"Conv1 weight shape: {conv1_weights.shape}")
    
    # Get Conv2 weights for the target channel
    conv2_weights = model.conv2.weight.detach().cpu().numpy()  # [4, 16, 201]
    target_conv2_weights = conv2_weights[target_fc_channel, :, :]  # [16, 201]
    print(f"Conv2 weights for channel {target_fc_channel} shape: {target_conv2_weights.shape}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Conv1 weights (all 16 channels averaged)
    conv1_avg = np.abs(conv1_weights).mean(axis=0).flatten()  # Average across 16 channels
    axes[0, 0].plot(conv1_avg)
    axes[0, 0].set_title('Conv1: Average |Weight| across all 16 channels')
    axes[0, 0].set_xlabel('Kernel Position')
    axes[0, 0].set_ylabel('Average |Weight|')
    
    # Plot 2: Conv2 weights for target channel (averaged across input channels)
    conv2_avg = np.abs(target_conv2_weights).mean(axis=0)  # Average across 16 input channels
    axes[0, 1].plot(conv2_avg)
    axes[0, 1].set_title(f'Conv2: Average |Weight| for output channel {target_fc_channel}')
    axes[0, 1].set_xlabel('Kernel Position')
    axes[0, 1].set_ylabel('Average |Weight|')
    
    # Plot 3: Conv2 weights for target channel (heatmap of all input channels)
    im = axes[1, 0].imshow(np.abs(target_conv2_weights), aspect='auto', cmap='viridis')
    axes[1, 0].set_title(f'Conv2: |Weight| heatmap for output channel {target_fc_channel}')
    axes[1, 0].set_xlabel('Kernel Position')
    axes[1, 0].set_ylabel('Input Channel')
    plt.colorbar(im, ax=axes[1, 0], label='|Weight|')
    
    # Plot 4: Combined contribution (conv1 * conv2 for target channel)
    # This shows which input positions contribute most to the target channel
    combined_contribution = np.zeros(201)  # Kernel size
    for input_ch in range(16):  # For each input channel to conv2
        conv1_channel_weight = conv1_weights[input_ch, 0, :]  # [201]
        conv2_channel_weight = target_conv2_weights[input_ch, :]  # [201]
        # Element-wise product to see combined contribution
        combined_contribution += np.abs(conv1_channel_weight * conv2_channel_weight)
    
    axes[1, 1].plot(combined_contribution)
    axes[1, 1].set_title(f'Combined Contribution to FC channel {target_fc_channel}')
    axes[1, 1].set_xlabel('Kernel Position (relative to input)')
    axes[1, 1].set_ylabel('Combined |Weight| Contribution')
    
    plt.tight_layout()
    save_path = f"{save_prefix}_channel_{target_fc_channel}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Channel contribution visualization saved to {save_path}")
    
    # Print summary
    print(f"\nSummary for FC channel {target_fc_channel}:")
    print(f"FC weight: {fc_weights[0, target_fc_channel]:.4f}")
    print(f"Max Conv1 contribution position: {np.argmax(conv1_avg)}")
    print(f"Max Conv2 contribution position: {np.argmax(conv2_avg)}")
    print(f"Max combined contribution position: {np.argmax(combined_contribution)}")

# Visualize contribution to the 4th channel (index 3)
visualize_channel_contribution(
    model_path='/home/smatsubara/documents/airlift/data/results/layernorm/weights/model.pth',
    model_class=lambda input_length=2500: SimpleCNN(input_length=input_length),
    target_fc_channel=3,  # 4番目のチャンネル（0-indexed）
    save_prefix='channel_contribution',
    device='cpu',
    input_length=2500
)

def analyze_gradcam_interpretation():
    """
    Provide detailed interpretation of Grad-CAM results based on the analysis.
    """
    print("\n" + "="*60)
    print("Grad-CAM解釈の詳細分析")
    print("="*60)
    
    print("\n1. FC層の重み分析:")
    print("   - 4番目のチャンネル（インデックス3）の重み: 0.645")
    print("   - これは他のチャンネル（0.135, 0.148, -0.397）より遥かに大きい")
    print("   - モデルは4番目のチャンネルの出力を最重要視している")
    
    print("\n2. 畳み込みカーネルの空間的寄与:")
    print("   - Conv1最大寄与位置: 46 (カーネル中央付近)")
    print("   - Conv2最大寄与位置: 104 (カーネル中央付近)")  
    print("   - 結合寄与最大位置: 43 (カーネル中央付近)")
    
    print("\n3. 入力信号へのマッピング:")
    print("   - Conv1: kernel_size=201, padding=100")
    print("   - 出力位置での受容野: [位置-100, 位置+100]")
    print("   - 信号系列2500位置での受容野: [2400, 2600]")
    print("   - 信号系列2300位置での受容野: [2200, 2400]")
    
    print("\n4. Grad-CAM結果との対応:")
    print("   ✅ モデルが信号系列の終端（2200-2500）に高い注意を向けている")
    print("   ✅ 4番目のチャンネルがこの領域の特徴を最も重要視している")
    print("   ⚠️  '強度の変化'という解釈は追加検証が必要")
    
    print("\n5. 追加検証の提案:")
    print("   - 実際の信号データとGrad-CAMを重ね合わせる")
    print("   - 信号の微分（変化率）とGrad-CAMの相関を調べる")
    print("   - 複数のサンプルでGrad-CAMパターンを確認する")
    
    print("\n結論:")
    print("   あなたの解釈『信号系列の終わりの強度変化に着目して学習』は")
    print("   基本的に正しいが、'変化'の部分はより詳細な検証が推奨される。")

analyze_gradcam_interpretation()

def create_signal_gradcam_overlay(
    file_path, model_path, model_class, sample_index=5000, channel_index=0,
    save_path='signal_gradcam_overlay.png', device='cpu', input_length=2500
):
    """
    Create an overlay visualization of actual signal data and Grad-CAM heatmap.
    
    Args:
        file_path (str): Path to the .npz file containing signal data
        model_path (str): Path to the trained model weights
        model_class (callable): Model construction function
        sample_index (int): Index of the sample to visualize
        channel_index (int): Channel index to visualize
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
    
    # Load signal data
    data = np.load(file_path)
    if 'arr_0' in data:
        x = data['arr_0']
    else:
        arr_keys = list(data.keys())
        if not arr_keys:
            raise RuntimeError(f"No arrays found in {file_path}")
        x = data[arr_keys[0]]
    
    if x.ndim != 3:
        raise RuntimeError(f"Expected 3D array, but got shape {x.shape}")
    
    N, W, C = x.shape
    if not (0 <= sample_index < N):
        raise RuntimeError(f"Invalid sample_index {sample_index}, N={N}")
    if not (0 <= channel_index < C):
        raise RuntimeError(f"Invalid channel_index {channel_index}, C={C}")
    
    # Extract and preprocess signal
    signal = x[sample_index, :, channel_index]
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
    ax1.set_title(f'Original Signal (Sample {sample_index}, Channel {channel_index})')
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
    
    ax3.set_title('Signal and Grad-CAM Overlay')
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
    print(f"\nSignal-GradCAM Overlay Analysis:")
    print(f"Sample: {sample_index}, Channel: {channel_index}")
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

# Create signal-GradCAM overlay for multiple samples
for sample_idx in [100, 1000, 5000, 8000]:
    create_signal_gradcam_overlay(
        file_path='/home/smatsubara/documents/airlift/data/experiments/processed/solid_liquid/P20241007-1401_processed.npz',
        model_path='/home/smatsubara/documents/airlift/data/results/layernorm/weights/model.pth',
        model_class=lambda input_length=2500: SimpleCNN(input_length=input_length),
        sample_index=sample_idx,
        channel_index=0,
        save_path=f'/home/smatsubara/documents/airlift/data/results/signal_gradcam_overlay_sample_{sample_idx}.png',
        device='cpu',
        input_length=2500
    )


