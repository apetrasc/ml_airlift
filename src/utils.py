import numpy as np
import torch
from scipy.signal import hilbert
from scipy import signal
import matplotlib.pyplot as plt
import yaml
import os


def preprocess(x_raw, fs=None):
    """
    CPU-only preprocessing of raw input data for model prediction.

    This includes:
    - Optional bandpass filtering (when fs is provided)
    - Hilbert transform to obtain amplitude envelope
    - Morphological erosion (moving minimum) via erode_signals
    - NaN handling

    Args:
        x_raw (np.ndarray): Raw input data of shape (batch, length)
        fs (float | None): Sampling frequency. If None, bandpass is skipped.

    Returns:
        np.ndarray: Preprocessed data on CPU with shape (batch, length)
    """

    # バンドパスフィルタ（サンプリング周波数 fs が与えられている場合のみ適用）
    # SciPy の butter に fs を渡す場合、Wn は [0, fs/2) の範囲である必要がある
    if fs is not None:
        # 1–10 MHz の帯域を想定したフィルタ
        sos = signal.butter(
            N=16,
            Wn=[1e6, 10e6],
            btype="bandpass",
            output="sos",
            fs=fs,
        )
        x_raw = signal.sosfiltfilt(sos, x_raw, axis=1)
        print(f"bandpass exec@max: {np.max(x_raw)}")
    else:
        # fs が渡されていない場合はフィルタをスキップ（古い npz との互換のため）
        print("WARNING: fs is None in preprocess; skipping bandpass filter.")

    # Hilbert 変換で包絡線を計算
    x_env = np.abs(signal.hilbert(x_raw, axis=1))

    # モルフォロジー的な侵食（移動最小値フィルタ）
    x_env = erode_signals(x_env, window_size=30)

    # NaN を安全な値に置き換え
    if np.isnan(x_env).any():
        print("nan")
        x_env = np.nan_to_num(x_env)

    return x_env


def prepare_cnn_input(x_processed, device):
    """
    Convert preprocessed CPU numpy array into CNN input tensor.

    Steps:
    - Convert to torch.float32 tensor on CPU
    - Add channel dimension: (batch, 1, length)
    - Per-sample max normalization
    - log1p transformation
    - Move to the specified device

    Args:
        x_processed (np.ndarray): Preprocessed data of shape (batch, length)
        device (str | torch.device): Target device for the tensor

    Returns:
        torch.Tensor: Tensor of shape (batch, 1, length) on the given device
    """
    # CPU 上でテンソル化
    x_tensor = torch.from_numpy(x_processed).float()  # (batch, length)

    # Add channel dimension: (batch, 1, length)
    x_tensor = x_tensor.unsqueeze(1)

    # Normalize each (length) column for each sample in the batch
    c=1.2
    max_values_per_column = torch.max(x_tensor, dim=2, keepdim=True)[0]
    max_values_per_column[max_values_per_column == 0] = 1.0  # Prevent division by zero
    x_tensor = x_tensor / (max_values_per_column * c)

    # log1p transformation
    x_tensor = torch.log1p(x_tensor)
    print(f"max: {torch.max(x_tensor)}")

    # 指定されたデバイスへ転送
    return x_tensor.to(device)


def hilbert_cuda(img_data_torch, device):
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
            # Apply the Hilbert transformer
            Xf = Xf * h
            # IFFT to get the analytic signal
            analytic_signal = torch.fft.ifft(Xf, dim=1)
            # Take the amplitude envelope
            img_data_torch_abs = torch.abs(analytic_signal)
            # Move to CPU and convert to numpy array
            return img_data_torch_abs.cpu().numpy()
def preprocess_and_predict(path, model, device, plot_index=80):
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

    # Load and preprocess data
    data = np.load(path)
    x_raw = data["processed_data"][:, :, 0]
    # npz 内にサンプリング周波数 fs が含まれていれば取得してプリプロセスに渡す
    fs = None
    if "fs" in data:
        try:
            fs = data["fs"].item() if hasattr(data["fs"], "item") else float(data["fs"])
        except Exception:
            fs = None
    
    import os
    filename = os.path.basename(path)
    print(f"loading successful and processing {filename}..")

    # CPU-only preprocessing
    x_processed = preprocess(x_raw, fs=fs)

    # CNN 向けテンソルへ変換し、デバイスに載せる
    x_test_tensor_cnn = prepare_cnn_input(x_processed, device)

    # Model prediction
    model.eval()
    with torch.no_grad():
        predictions = model(x_test_tensor_cnn)
        mean, var = torch.mean(predictions), torch.var(predictions)
        print(f"mean: {mean}, var: {var}")
        # Release memory after computation
        del predictions
        torch.cuda.empty_cache()
    return mean, var

def npz2png(file_path, save_path, channel_index=0, start_time=0.0, end_time=5.0, full=True, pulse_index=0, vmin=0, vmax=1, max_pulses=None):    
    """
    Convert processed .npz signal data to PNG image.
    
    Parameters
    ----------
    file_path : str
        Path to the .npz file containing processed data.
    save_path : str
        Path to save the output PNG image.
    channel_index : int, optional
        Index of the channel to visualize (default is 0).
    start_time : float, optional
        Start time in seconds for visualization (default is 0.0).
    end_time : float or None, optional
        End time in seconds for visualization (default is None, meaning till the end).
    full : bool, optional
        If True, visualize all pulses as an image. If False, visualize only one pulse waveform (default is True).
    pulse_index : int, optional
        Index of the pulse to visualize when full=False (default is 0).
    max_pulses : int or None, optional
        If set, use at most this many pulses (rows) in the image when full=True (default is None = no limit).
    
    Returns
    -------
    None
    """
    # .npzファイルからデータを読み込む
    data = np.load(file_path)
    processed_data = data["processed_data"]
    fs = data["fs"].item() if hasattr(data["fs"], "item") else float(data["fs"])
    #print(f"processed_data.shape:{processed_data.shape}")
    # full=Trueの場合は全パルスを画像化
    if full:
        # processed_dataのshape: (n_pulses, n_samples, n_channels)
        # 指定チャンネルの全パルスを抽出
        if processed_data.ndim == 3:
            # Check if the channel_index is within the valid range
            if channel_index < 0 or channel_index >= processed_data.shape[2]:
                raise IndexError(f"channel_index {channel_index} is out of bounds for axis 2 with size {processed_data.shape[2]}")
            img_data = processed_data[:, :, channel_index]
        elif processed_data.ndim == 2:
            img_data = processed_data  # (n_pulses, n_samples)
        # If processed_data has other dimensions, raise an error
        else:
            raise ValueError("processed_data shape is not supported.")
        
        n_samples = img_data.shape[1]
        n_samples = int(n_samples)
        t = np.arange(n_samples) / fs
        start_idx = int(start_time * fs)
        end_idx = int(end_time * fs)
        if end_idx > n_samples:
            end_idx = n_samples
        h,w = img_data.shape
        print(f"h: {h}, w: {w}")
        interval = 1/3*1e-3
        total_time = 15000*interval
        endid = int(h *end_time/total_time)
        print(f"endid: {endid}")
        img_data = img_data[:endid, start_idx:end_idx]
        # 縦方向に max_pulses 本だけ使う（指定時のみ）
        if max_pulses is not None:
            img_data = img_data[:max_pulses, :]
            print(f"using first {img_data.shape[0]} pulses (max_pulses={max_pulses})")
        # Apply Hilbert transform to each pulse in img_data along the time axis
        # The analytic signal is computed for each pulse (row) individually
        # ヒルベルト変換をtorchで実装する
        import torch

        neglegible_time = 3e-6
        zero_samples = int(neglegible_time * fs)

        # img_data: (n_pulses, n_samples)
        # GPUにデータを転送
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_data_torch = torch.from_numpy(img_data).float().to(device)
        #print(f"device: {device}")
        # 初期部分を0にする
        if zero_samples > 0:
            img_data_torch[:, :zero_samples] = 0

        # ヒルベルト変換のためのハイライザー（周波数領域での乗数）を作成
        n_samples = img_data_torch.shape[1]

        img_data = hilbert_cuda(img_data_torch, device)
        #print(img_data.shape)
        t = t[start_idx:end_idx]
        #print(f"end_idx: {end_idx}, start_idx: {start_idx}")
        #print(t.shape)
        #print(np.max(img_data),np.min(img_data))
        print(f"img_data.shape: {img_data.shape}")
        # 縦パルス数に応じて図の高さを調整（30パルスなら高さ10程度）
        n_rows = img_data.shape[0]
        fig_h = max(6, min(100, n_rows * 0.35))
        plt.figure(figsize=(10, fig_h))
        #plt.imshow(img_data, aspect='auto', cmap='viridis', extent=[t[0]*1e6, t[-1]*1e6, img_data.shape[0]-0.5, -0.5],vmin=0,vmax=1)
        plt.imshow(img_data, aspect='auto', cmap='viridis', extent=[t[0]*1e6, t[-1]*1e6, img_data.shape[0]-0.5, -0.5],vmin=vmin,vmax=vmax)
        #plt.imshow(img_data, aspect='auto', cmap='viridis', extent=[t[0], t[-1], img_data.shape[0]-0.5, -0.5])
        cbar = plt.colorbar(label='Amplitude')
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label('Amplitude', fontsize=18)
        plt.xlabel('Time (μs)', fontsize=18)
        plt.ylabel('Pulse Number', fontsize=18)
        plt.title('All Pulses (Channel {})'.format(channel_index), fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()
        import os
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        new_save_path = os.path.join(save_path, f"{base_name}_{channel_index}img.png")
        print(new_save_path)
        plt.savefig(new_save_path)
        plt.close()
    else:
        # full=Falseの場合は指定パルスのみをプロット
        if processed_data.ndim == 3:
            pulse = processed_data[pulse_index, :, channel_index]
           
        elif processed_data.ndim == 2:
            pulse = processed_data[pulse_index, :]
        else:
            raise ValueError("processed_data shape is not supported.")
        n_samples = len(pulse)
        t = np.arange(n_samples) / fs
        if end_time is None:
            end_time = t[-1]
        start_idx = int(start_time * fs)
        end_idx = int(end_time * fs)
        if end_idx > n_samples:
            end_idx = n_samples
        t = t[start_idx:end_idx]
        pulse = pulse[start_idx:end_idx]
        # Apply Hilbert transform to the pulse to obtain its analytic signal
        # The absolute value of the analytic signal gives the envelope of the pulse
        from scipy.signal import hilbert
        neglegible_time = 3e-6 # 3μs
        zero_samples = int(neglegible_time * fs)
        pulse[:zero_samples] = 0
        analytic_pulse = np.abs(hilbert(pulse))
        #analytic_pulse = np.log1p(analytic_pulse)
        #print(pulse) 
        plt.figure(figsize=(10, 4))
        plt.plot(t*1e6, analytic_pulse, color='red', label='Envelope')
        plt.plot(t*1e6, pulse, color='blue', label='Original Pulse')
        plt.legend(fontsize=16)
        plt.xlabel('Time (μs)', fontsize=18)
        plt.ylabel('Amplitude', fontsize=18)
        plt.title('Pulse {} (Channel {})'.format(pulse_index, channel_index), fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()
        import os
        #base = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        channel=channel_index
        new_save_path = os.path.join(save_path, f"{base_name}_{channel}pulse.png")
        print(new_save_path)
        plt.savefig(new_save_path)
        plt.close()

def debug_pipeline(base_dir, config, file_path):
    """
    Load processed data from a .npz file, preprocess it, plot a specific slice, and save the plot as an image.
    Args:
        base_dir (str): Base output directory.
        config (dict): Resolved config with 'evaluation' key (e.g. config['evaluation']['device']).
        file_path (str): Path to the processed .npz file.
    """
    # Check if the file exists
    print("DEBUG: File exists:", os.path.exists(file_path))

    # Load the processed data
    data = np.load(file_path)
    x_raw_bug = data["processed_data"][:, :, 0]
    # サンプリング周波数 fs が保存されていれば取得
    fs = None
    if "fs" in data:
        try:
            fs = data["fs"].item() if hasattr(data["fs"], "item") else float(data["fs"])
        except Exception:
            fs = None
    filename = os.path.basename(file_path)
    print(f"DEBUG: loading successful and processing {filename}..")

    # CPU-only preprocessing
    x_processed = preprocess(x_raw_bug, fs=fs)

    # CNN 向けテンソルへ変換し、デバイスに載せる
    x_debug_tensor = prepare_cnn_input(x_processed, config['evaluation']['device'])
    x_debug = x_debug_tensor.detach().cpu().numpy()
    print(x_debug.shape)
    print(x_debug[100, :, :].shape)

    # Plot and save the 100th slice of the first channel
    plt.figure(figsize=(10, 4))
    plt.ylim(0, 1)
    plt.plot(x_debug[100, 0, :])

    save_debug = os.path.join(base_dir, "logs", f"x_debug_{filename}.png")
    plt.savefig(save_debug)
    plt.close()
    print(f"DEBUG: saved x_debug_{filename}.png")


def get_valid_data(x, y, yerr):
    """
    Remove NaN values from x, y, and yerr, and return only valid data.
    """
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(yerr)
    x_valid = x[mask]
    y_valid = y[mask]
    yerr_valid = yerr[mask]
    return x_valid, y_valid, yerr_valid

def erode_signals(x, window_size=30):
    import polars as pl
    s=pl.from_numpy(np.transpose(x))

    s = s.select(pl.all().rolling_min(
        window_size=window_size,
        min_periods=1
    ))
    x = s.to_numpy().transpose()
    return x