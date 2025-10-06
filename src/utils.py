import numpy as np
import torch
from scipy.signal import hilbert
import matplotlib.pyplot as plt

def hilbert_cuda(img_data_torch, device, if_hilbert = True,
                 low_filter_freq = 0,
                 high_filter_freq = 1.0e9,
                 fs = 52e6):
            """
            Compute the Hilbert envelope of input data using torch (GPU/CPU).

            Parameters
            ----------
            img_data_torch : torch.Tensor
                Input data of shape (n_pulses, n_samples), float32.
            device : torch.device
                Device to perform computation on.

            Returns
            -------
            np.ndarray
                Envelope of the analytic signal, shape (n_pulses, n_samples).
            """
            n_samples = img_data_torch.shape[1]
            # Create the Hilbert transformer in the frequency domain
            h = torch.zeros(n_samples, dtype=torch.complex64, device=device)
            if n_samples % 2 == 0:
                # Even length
                h[0] = 1
                h[1:n_samples//2] = 2
                h[n_samples//2] = 1
                # The rest remain zero
            else:
                # Odd length
                h[0] = 1
                h[1:(n_samples+1)//2] = 2
                # The rest remain zero

            # FFT along the time axis
            Xf = torch.fft.fft(img_data_torch, dim=1)
            low_filter_idx = 0.0
            high_filter_idx = float("Inf")
            low_filter_idx = int(low_filter_freq / fs * n_samples)
            high_filter_idx = int(high_filter_freq /fs * n_samples)
            if high_filter_idx > n_samples-1:
                 high_filter_idx = n_samples-1
            print(f'bandpass: {low_filter_idx}, {high_filter_idx}\n')
            Xf[:,0:low_filter_idx] = 0
            Xf[:,high_filter_idx:n_samples] = 0
            if if_hilbert:
                # Apply the Hilbert transformer
                Xf = Xf * h
                # IFFT to get the analytic signal
                analytic_signal = torch.fft.ifft(Xf, dim=1)
                # Take the amplitude envelope
                img_data_torch_abs = torch.abs(analytic_signal)
                # Move to CPU and convert to numpy array
                return img_data_torch_abs.cpu().numpy()
            else:
                analytic_signal = torch.fft.ifft(Xf, dim=1)
                return analytic_signal.cpu().numpy()

def preprocess_and_predict(path, model, plot_index=80, device='cuda:0',
                           filter_freq=[0, 1.0e9],
                           rolling_window=False, window_size=20,
                           window_stride=10, if_log1p=False, if_hilbert=True):
    """
    Loads data from the given path, applies Hilbert transform and normalization,
    and runs prediction using the provided model.

    Args:
        path (str): Path to the .npz file containing 'processed_data'.
        model (torch.nn.Module): Trained PyTorch model for prediction.
        plot_index (int): Index of the sample to plot.
        device (str): Device to run the model on.

    Returns:
        torch.Tensor: Model predictions.
    """
    import numpy as np
    import torch
    import scipy.signal 
    import hilbert
    import matplotlib.pyplot as plt
    import polars as pl

    # Load and preprocess data
    x_raw = np.load(path)["processed_data"][:,:,0]
    fs = np.load(path)["fs"]
    print(x_raw.shape)
    
    import os
    filename = os.path.basename(path)
    print(f"loading successful and processing {filename}..")
    #npz2png(file_path=path,save_path=output_folder_path,full=False,pulse_index=1)
    #npz2png(file_path=path,save_path=output_folder_path,full=True,pulse_index=2)
    #print(f"max: {np.max(x_raw)}")
    #x_test = np.abs(hilbert(x_raw))
    x_raw_torch = torch.from_numpy(x_raw).float()
    x_raw_torch = x_raw_torch.to(device)
    min_freq = filter_freq[0]
    max_freq = filter_freq[1]

    x_test = hilbert_cuda(x_raw_torch,device, if_hilbert,
                              min_freq, max_freq, fs)
    x_test_tmp = []
    if rolling_window:
        for x_pulse in x_test:
            s =pl.Series(x_pulse)

            rolling_max = s.rolling_max(window_size=window_size)[
                 window_size-1:].gather_every(
                 n=window_stride, offset=0
            )
            x_pulse = rolling_max.to_numpy()
            x_test_tmp.append(x_pulse)
        x_test = np.array(x_test_tmp)
        print('ここまでOK')
    print(f"max: {np.max(x_test[10])}")
    print(f'xtest shap: {x_test.shape}')
    if np.isnan(x_test).any():
        print("nan")
        x_test = np.nan_to_num(x_test)
    x_test_tensor = torch.from_numpy(x_test).float()

    # Add channel dimension: (batch, 1, length, channel)
    x_test_tensor_all = x_test_tensor.unsqueeze(1)
    print('ここまでOK')
    #print(x_test_tensor_all.shape)
    # Normalize each (length, channel) column for each sample in the batch
    max_values_per_column = torch.max(x_test_tensor_all, dim=2, keepdim=True)[0]
    #print(f"max_values_per_column.shape: {max_values_per_column.shape}")
    max_values_per_column[max_values_per_column == 0] = 1.0  # Prevent division by zero
    x_test_tensor_all = x_test_tensor_all / max_values_per_column
    #print(f"max: {torch.max(x_test_tensor_all)}")

    # Use only the first channel for CNN input
    x_test_tensor_cnn = x_test_tensor_all[:, :, :]
    x_test_tensor_cnn = x_test_tensor_cnn.to(device)
    if if_log1p:
        x_test_tensor_cnn = torch.log1p(x_test_tensor_cnn)
    #x_test_tensor_cnn = torch.log1p(x_test_tensor_cnn)
    #print(x_test_tensor.shape)
    #print(x_test_tensor_cnn.shape)
    #print(f"max: {torch.max(x_test_tensor_cnn)}")
    #print(x_test_tensor_cnn)
    # Plot a sample signal
    plt.figure(figsize=(10, 4))
    plt.plot(x_test_tensor_cnn[5, 0,:].cpu().numpy())
    plt.title("x_test_tensor_cnn Signal")
    plt.xlabel("sample Index")
    plt.ylabel("Value")
    plt.grid(True)
    #plt.show()
    plt.close()
    #print(x_test_tensor_cnn[plot_index,0,:].shape)
    # Model prediction
    model.eval()
    with torch.no_grad():
        x_test_tensor_cnn = x_test_tensor_cnn.to(device)
        predictions = model(x_test_tensor_cnn)
        mean, var = torch.mean(predictions), torch.var(predictions)
        #print(f"predictions.shape: {predictions.shape}")
        #print(predictions)
        print(f"mean: {mean}, var: {var}")
        # Release memory after computation
        del predictions
        torch.cuda.empty_cache()
    return mean, var