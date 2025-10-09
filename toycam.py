import torch
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
from src import preprocess
from scipy.signal import find_peaks, peak_widths
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
# Import the model definition
from models.cnn import SimpleCNN

base_dir = "/home/smatsubara/documents/airlift/data/results/layernorm"
# Load the trained model
model = SimpleCNN(config['hyperparameters']['input_length']).to(config['evaluation']['device'])
model_path = os.path.join(base_dir + '/weights/model.pth')
model.load_state_dict(torch.load(model_path, map_location=config['evaluation']['device'], weights_only=True))
model.eval()
print("model loaded")
# 1D attribution visualization helper
def visualize_attributions_1d(attributions, original_input):
    import matplotlib.pyplot as plt
    import numpy as np
    attr = attributions.detach().cpu().squeeze().numpy()
    x = original_input.detach().cpu().squeeze().numpy()
    if attr.ndim == 2:
        attr = np.sum(np.abs(attr), axis=0)
    indices = np.arange(x.shape[-1])
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(indices, x, color='black')
    axes[0].set_title('Input signal')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(indices, attr, color='red')
    axes[1].fill_between(indices, 0, attr, color='red', alpha=0.2)
    axes[1].set_title('Attribution (magnitude)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('Index')
    plt.tight_layout()

def visualize_top_attribution_regions(attributions, original_input, top_k=3, min_distance=20, rel_height=0.5, save_path=None):
    """
    Highlight top-k contribution regions on the 1D input using attribution magnitude.
    - attributions: Tensor of shape (1, 1, L) or (1, C, L)
    - original_input: Tensor of same length for plotting the signal
    - top_k: number of regions to highlight
    - min_distance: minimal distance between peaks
    - rel_height: relative height for peak width computation (0..1)
    - save_path: path to save the figure (optional)
    """
    attr = attributions.detach().cpu().squeeze().numpy()
    x = original_input.detach().cpu().squeeze().numpy()
    if attr.ndim == 2:
        # Aggregate channels by absolute sum
        import numpy as np
        attr = np.sum(np.abs(attr), axis=0)
    else:
        attr = np.abs(attr)

    indices = np.arange(x.shape[-1])
    # Peak detection on attribution magnitude
    peaks, _ = find_peaks(attr, distance=min_distance)
    if peaks.size == 0:
        # Fallback: just plot attribution magnitude
        visualize_attributions_1d(attributions, original_input)
        if save_path is not None:
            plt.savefig(save_path)
        return []

    # Sort peaks by height (descending) and keep top_k
    peak_values = attr[peaks]
    order = np.argsort(-peak_values)
    top_peaks = peaks[order][:top_k]

    # Estimate peak widths to get region spans
    widths, width_heights, left_ips, right_ips = peak_widths(attr, top_peaks, rel_height=1.0 - rel_height)
    lefts = left_ips.astype(int)
    rights = right_ips.astype(int)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(indices, x, color='black')
    axes[0].set_title('Input signal with top attribution regions')
    axes[0].grid(True, alpha=0.3)
    for l, r in zip(lefts, rights):
        axes[0].axvspan(l, r, color='orange', alpha=0.25)

    axes[1].plot(indices, attr, color='red')
    axes[1].set_title('Attribution (magnitude) and detected peaks')
    axes[1].grid(True, alpha=0.3)
    axes[1].plot(top_peaks, attr[top_peaks], 'ko')
    for l, r in zip(lefts, rights):
        axes[1].axvspan(l, r, color='orange', alpha=0.25)
    axes[1].set_xlabel('Index')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    # Return regions as list of (left, right)
    return list(zip(lefts.tolist(), rights.tolist()))
# Example usage
if __name__ == "__main__":
    device = torch.device(config['evaluation']['device'])
    length = int(config['hyperparameters']['input_length'])
    file_path = "/home/smatsubara/documents/airlift/data/experiments/processed/solid_liquid/P20241015-1037_processed.npz"

    
    x_raw = np.load(file_path)["processed_data"][:,:,0]
    input_tensor_all = preprocess(x_raw, device)
    # Select a single pulse to avoid huge batch attribution
    pulse_index = config['evaluation'].get('pulse_index', 0)
    pulse_index = int(max(0, min(pulse_index, input_tensor_all.shape[0]-1)))
    input_tensor = input_tensor_all[pulse_index:pulse_index+1]

    input_tensor.requires_grad = True

    # For regression, we can use Integrated Gradients in the same way.
    # The 'target' argument should be set to None for regression tasks.
    ig = IntegratedGradients(model)

    # Compute attributions for regression with smaller memory footprint
    attributions = ig.attribute(
        input_tensor,
        target=None,
        n_steps=32,
        internal_batch_size=8,
        return_convergence_delta=False,
    )

    # Visualize attributions for 1D input
    visualize_attributions_1d(attributions, input_tensor)
    plt.savefig(os.path.join(base_dir, 'toycam.png'))
    # Highlight top-k attribution regions and save
    os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)
    regions = visualize_top_attribution_regions(
        attributions,
        input_tensor,
        top_k=int(config['evaluation'].get('top_k_regions', 3)),
        min_distance=int(config['evaluation'].get('min_region_distance', 20)),
        rel_height=float(config['evaluation'].get('region_rel_height', 0.5)),
        save_path=os.path.join(base_dir, 'logs', 'ig_top_regions.png')
    )
    # Save regions as CSV
    import csv
    with open(os.path.join(base_dir, 'logs', 'ig_top_regions.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['left_index', 'right_index'])
        for l, r in regions:
            writer.writerow([l, r])
