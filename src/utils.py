import numpy as np
import torch
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import yaml
import os


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

def preprocess_and_predict(path, model, plot_index=80, device='cuda:0',
                           filter_freq=[0, 1.0e9],
                           rolling_window=False, window_size=20,
                           window_stride=10, if_log1p=True, if_hilbert=True,
                           if_reduce=False, x_liquid_only=None, 
                           if_sigmoid=False,
                           if_drawsignal=False, png_save_dir=None,
                           png_name=None):
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
    import matplotlib.pyplot as plt
    import polars as pl

    torch.cuda.empty_cache()

    # Load and preprocess data
    x_raw = np.load(path)["processed_data"][:,:3000,0]
    fs = np.load(path)["fs"]
    print(x_raw.shape)
    print(f'fs:{fs}[Hz]')
    x_test = x_raw.copy()
    
    import os
    filename = os.path.basename(path)
    print(f"loading successful and processing {filename}..")
    #npz2png(file_path=path,save_path=output_folder_path,full=False,pulse_index=1)
    #npz2png(file_path=path,save_path=output_folder_path,full=True,pulse_index=2)
    #print(f"max: {np.max(x_raw)}")
    #x_test = np.abs(hilbert(x_raw))
    x_test_tensor_cnn = preprocess(path=path, device=device, 
                                   filter_freq=filter_freq, 
                                   rolling_window=rolling_window, 
                                   window_size=window_size, 
                                   window_stride=window_stride, 
                                   if_log1p=if_log1p,if_hilbert=if_hilbert,
                                   if_reduce=if_reduce,x_liquid_only=x_liquid_only,
                                   if_sigmoid=if_sigmoid,
                                   if_drawsignal=if_drawsignal,png_save_dir=png_save_dir,
                                   png_name=png_name,fs=fs,x_test=x_test)
    #print(x_test_tensor_cnn[plot_index,0,:].shape)
    # Model prediction
    model.eval()
    with torch.no_grad():
        x_test_tensor_cnn = x_test_tensor_cnn.to(device)
        print(f'shape of x_test_tensor_cnn: {x_test_tensor_cnn.shape}')
        predictions = model(x_test_tensor_cnn)
        predictions_numpy = predictions.cpu().numpy()
        mean, var = torch.mean(predictions), torch.var(predictions)
        print(f"mean: {mean}, var: {var}")
        # Release memory after computation
        del predictions
        torch.cuda.empty_cache()
    return mean, var

def preprocess(path, device="cuda:0", filter_freq=[0, 1.0e9], 
               rolling_window=False, window_size=20,
                window_stride=10, if_log1p=True, if_hilbert=True,
                if_reduce=False, x_liquid_only=None, 
                if_sigmoid=False,
                if_drawsignal=False, png_save_dir=None,
                png_name=None, plot_idx=1000,
               fs=None, x_test=None):
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import polars as pl
    import os
    from scipy.optimize import curve_fit
    from scipy import signal
    x_tmp = x_test.copy()
    x_tmp_for_fft = x_test.copy()
    x_test=filter_signal(filter_freq, x_tmp, fs)
    if if_hilbert:
        print(f'hilbert started')
        x_raw_torch = torch.from_numpy(x_test).float()
        x_raw_torch = x_raw_torch.to(device)
        x_test = hilbert_cuda(x_raw_torch,device)
    x_tmp2 = x_test.copy()
    x_test = rolling_window_signal(rolling_window, window_size, window_stride, np, pl, x_tmp2)
    #print(f"max: {np.max(x_test[10])}")
    print(f'xtest shap: {x_test.shape}')
    if np.isnan(x_test).any():
        print("nan")
        x_test = np.nan_to_num(x_test)
    print(f'x_test shape: {x_test.shape}')
    pulse_num = x_test.shape[0]
    argmax_pipe_ref2 = np.argmax(x_test[1000,:])
    x_test = x_test[:,argmax_pipe_ref2-200:argmax_pipe_ref2+2300]
    print(f'argmax {argmax_pipe_ref2}')
    x_test_tensor = torch.from_numpy(x_test).float()

    # Add channel dimension: (batch, 1, length, channel)
    x_test_tensor_all = x_test_tensor.unsqueeze(1)
    print(f'x_test_tensor_all shape: {x_test_tensor_all.shape}')
    # Normalize each (length, channel) column for each sample in the batch
    c=0.4
    max_values_per_column = torch.max(x_test_tensor_all, dim=2, keepdim=True)[0]+c
    #print(f"max_values_per_column.shape: {max_values_per_column.shape}")
    max_values_per_column[max_values_per_column == 0] = 1.0  # Prevent division by zero
    x_test_tensor_all = x_test_tensor_all / (max_values_per_column)
    #print(f"max: {torch.max(x_test_tensor_all)}")

    # Use only the first channel for CNN input
    x_test_tensor_cnn = x_test_tensor_all[:, :, :]
    x_test_tensor_cnn = x_test_tensor_cnn.to(device)
    if if_sigmoid:
        #x_test_tensor_cnn *= 0.928
        x_test_nd = x_test_tensor_cnn.cpu().numpy()
        x_test_sigmoid = sigmoid_fitting(x_test_nd, fs)
        print('OK1')
        x_test_sigmoid_plot = x_test_sigmoid[plot_idx,:]
        x_test_sigmoid = torch.from_numpy(x_test_sigmoid).float()
        x_test_sigmoid = x_test_sigmoid.to(device)
        print('OK2')
        x_test_sigmoid = x_test_sigmoid.unsqueeze(1)
        max_per_col = torch.max(x_test_sigmoid, dim=2, keepdims=True)[0]
        print(max_per_col[1000])
        print('OK3')
        x_test_tensor_cnn = torch.from_numpy(x_test_nd).float()
        x_test_tensor_cnn = x_test_tensor_cnn.to(device)
        x_test_tensor_cnn = x_test_tensor_cnn/max_per_col
        print('OK4')
    if if_log1p:
        x_test_tensor_cnn = torch.log1p(x_test_tensor_cnn)
        #x_test_sigmoid_plot = np.log1p(x_test_sigmoid_plot)
    if if_reduce:
        x_liquid_only_tensor = torch.from_numpy(x_liquid_only).float()
        #print(f'x_liquid_only_tensor shape: {x_liquid_only_tensor.shape}')
        x_liquid_only_tensor = x_liquid_only_tensor.unsqueeze(0)
        #print(f'x_liquid_only_tensor shape: {x_liquid_only_tensor.shape}')
        x_liquid_only_tensor = x_liquid_only_tensor.expand(pulse_num,-1)
        #print(f'x_liquid_only_tensor shape: {x_liquid_only_tensor.shape}')
        x_liquid_only_tensor = x_liquid_only_tensor.unsqueeze(1)
        #print(f'x_liquid_only_tensor shape: {x_liquid_only_tensor.shape}')
        x_liquid_only_tensor_cnn = x_liquid_only_tensor.to(device)
        x_test_tensor_cnn = x_test_tensor_cnn-x_liquid_only_tensor_cnn
    #print(x_test_tensor.shape)
    #print(x_test_tensor_cnn.shape)
    print(f"max: {torch.max(x_test_tensor_cnn)}")
    #print(x_test_tensor_cnn)



    # Plot a sample signal
    if if_drawsignal:
        n_samples = x_test_tensor_cnn[plot_idx, 0,:].cpu().numpy().shape[0]
        #print(f'n_samples: {n_samples}')
        t = np.arange(n_samples)/fs
        plt.figure(figsize=(10, 4))
        plt.rcParams["font.size"] = 18
        plt.plot(t*1e6,x_test_tensor_cnn[plot_idx, 0,:].cpu().numpy(),
                 color='blue',label='Processed Signal')
        #print(f'first plot done')
        # if not (if_hilbert or rolling_window):
        #     x_test_envelope = hilbert_cuda(x_test_tensor_cnn[:,0,:],device)
        #     plt.plot(t*1e6,x_test_envelope[1000,:],
        #              color='red',label='Envelope')
            #print(f'second plot done')
        if if_sigmoid:
            plt.plot(t*1e6,x_test_sigmoid_plot,
                     color='red',label='Sigmoid')
        plt.xlabel("Time (µs)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        if png_save_dir!=None:
            base_filename = os.path.splitext(os.path.basename(path))[0]
            plt.savefig(os.path.join(png_save_dir,f'{base_filename}_{png_name}.png'))
        #plt.show()
        plt.close()
        plt.figure(figsize=(10, 4))
        freq = np.arange(n_samples//2)*fs/n_samples
        x_test_for_fft = torch.from_numpy(x_tmp_for_fft[plot_idx,:]).cpu()
        x_test_fft = torch.abs(torch.fft.fft(x_test_for_fft))
        x_test_fft = torch.pow(x_test_fft, 2)
        plt.rcParams["font.size"] = 18
        plt.plot(freq*1e-6,x_test_fft.cpu().numpy()[:n_samples//2],
                 color='blue',label='FFT')
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude")
        plt.ylim(-0.1*np.max(x_test_fft.cpu().numpy()[40:n_samples//2]), 1.1*np.max(x_test_fft.cpu().numpy()[40:n_samples//2]))
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        if png_save_dir!=None:
            base_filename = os.path.splitext(os.path.basename(path))[0]
            plt.savefig(os.path.join(png_save_dir,f'{base_filename}_{png_name}_fft.png'))
    return x_test_tensor_cnn

def func_sigmoid(x, a, b, c, cx ,k):
    import numpy as np
    if x.shape == ():
        if int(c*np.abs(x-cx)-k) > 10:
            return a
        elif int(c*np.abs(x-cx)-k) < -10:
            return a+b
        else:
            return a+b/(1+np.exp(c*np.abs(x-cx)-k))
    else:
        return [func_sigmoid(x0,a,b,c,cx,k) for x0 in x]

def sigmoid_fitting(x_test_all, fs):
    from scipy.optimize import curve_fit
    from scipy import signal
    #x_test: 14000,1,2500
    n_samples = x_test_all.shape[2]
    x_test_sigmoid = []
    x_array=np.arange(n_samples)
    end_idx=0
    start_idx = 6e-6*fs
    count=0
    for x_test in x_test_all[:,0,:]:
        argmax_signal = np.argmax(x_test)
        argrelmin_array = signal.argrelmin(x_test[argmax_signal:])
        for argrelmin in argrelmin_array[0]:
            if x_test[argmax_signal+argrelmin]<0.2:
                end_idx = argrelmin+argmax_signal
                break
        x_test_fitting = x_test[:end_idx]
        x_array_tmp = np.arange(end_idx)
        popt, pcov = curve_fit(func_sigmoid,x_array_tmp,x_test_fitting,
                               p0=[0,1,10,300,500])
        x_sigmoid = func_sigmoid(x_array,popt[0],popt[1],popt[2],popt[3],popt[4])
        x_test_sigmoid.append(x_sigmoid)
        count += 1
        if count%100 == 0:
            print(f'{count}回目終了')
    return np.array(x_test_sigmoid)

def rolling_window_signal(rolling_window, window_size, window_stride, np, pl, x_test):
    if rolling_window:
        x_test_tmp = []
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
    return x_test

def filter_signal(filter_freq, x_raw, fs, device='cuda:0'):
    min_freq = filter_freq[0]
    max_freq = filter_freq[1]
    x_size = x_raw.shape[1]
    #print(f'x_raw shape: {x_raw.shape}')
    x_tensor = torch.from_numpy(x_raw).float()
    x_tensor = x_tensor.to(device)
    Xf = torch.fft.fft(x_tensor,dim=1)
    #print(f'Xf shape: {Xf.shape}')
    min_freq_idx = int(x_size*min_freq/fs)
    #print(f'min freq index: {min_freq_idx}')
    #print(f'Xf shape: {Xf.shape}')
    max_freq_idx = np.min([x_size*max_freq/fs, x_size//2-1])
    max_freq_idx = int(max_freq_idx)
    if min_freq_idx!=0 or max_freq_idx<x_size//2:
        #print(f'max freq index: {max_freq_idx}')
        Xf = torch.fft.fft(x_tensor,dim=1)
        Xf[:,0:min_freq_idx]=0
        Xf[:,max_freq_idx:x_size//2]=0
        #print(f'Xf shape: {Xf.shape}')
        x_tensor_new = torch.fft.ifft(Xf,dim=1)
        #print(f'x_tensor_new shape: {x_tensor_new.shape}')
        x_tensor_new = torch.real(x_tensor_new)
        return x_tensor_new.cpu().numpy()
    else:
        return x_raw

def preprocess_liquidonly(path, plot_index=80, device='cuda:0',
                           filter_freq=[0, 1.0e9],
                           rolling_window=False, window_size=20,
                           window_stride=10, if_log1p=False, 
                           if_hilbert=True):
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
    import matplotlib.pyplot as plt
    import polars as pl

    # Load and preprocess data
    x_raw = np.load(path)["processed_data"][:,:,0]
    fs = np.load(path)["fs"]
    print(x_raw.shape)
    print(f'fs:{fs}[Hz]')
    x_test=x_raw.copy()
    
    import os
    filename = os.path.basename(path)
    print(f"loading successful and processing {filename}..")
    #npz2png(file_path=path,save_path=output_folder_path,full=False,pulse_index=1)
    #npz2png(file_path=path,save_path=output_folder_path,full=True,pulse_index=2)
    #print(f"max: {np.max(x_raw)}")
    #x_test = np.abs(hilbert(x_raw))
    filter_signal(filter_freq, x_raw, fs)
    if if_hilbert:
        x_raw_torch = torch.from_numpy(x_raw).float()
        x_raw_torch = x_raw_torch.to(device)
        x_test = hilbert_cuda(x_raw_torch,device)
    x_test = rolling_window_signal(rolling_window, window_size, window_stride, np, pl, x_test)
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
    x_test_tensor_1d = torch.mean(x_test_tensor_cnn, dim=0)
    x_test_tensor_1d = torch.mean(x_test_tensor_1d, dim=0)
    x_test = x_test_tensor_1d.cpu().numpy()
    print(f'liquid only xtest shape: {x_test.shape}')
    return x_test

def npz2png(file_path, save_path, channel_index=0, start_time=0.0, end_time=None, full=True, pulse_index=0):    
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
        
        # Determine the time axis range
        n_samples = img_data.shape[1]
        t = np.arange(n_samples) / fs
        if end_time is None:
            end_time = t[-1]
        start_idx = int(start_time * fs)
        end_idx = int(end_time * fs)
        if end_idx > n_samples:
            end_idx = n_samples
        img_data = img_data[:, start_idx:end_idx]
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
        #print(t.shape)
        #print(np.max(img_data),np.min(img_data))
        
        
        plt.figure(figsize=(10, 4))
        #plt.imshow(img_data, aspect='auto', cmap='viridis', extent=[t[0]*1e6, t[-1]*1e6, img_data.shape[0]-0.5, -0.5],vmin=0,vmax=1)
        plt.imshow(img_data, aspect='auto', cmap='viridis', extent=[t[0]*1e6, t[-1]*1e6, img_data.shape[0]-0.5, -0.5])
        #plt.imshow(img_data, aspect='auto', cmap='viridis', extent=[t[0], t[-1], img_data.shape[0]-0.5, -0.5])
        plt.colorbar(label='Amplitude')
        plt.xlabel('Time (μs)')
        plt.ylabel('Pulse Number')
        plt.title('All Pulses (Channel {})'.format(channel_index))
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
        plt.legend()
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude')
        plt.title('Pulse {} (Channel {})'.format(pulse_index, channel_index))
        plt.tight_layout()
        import os
        #base = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        channel=channel_index
        new_save_path = os.path.join(save_path, f"{base_name}_{channel}pulse.png")
        print(new_save_path)
        plt.savefig(new_save_path)
        plt.close()

def debug_pipeline(base_dir, config_path, file_path):
    """
    Load processed data from a .npz file, preprocess it, plot a specific slice, and save the plot as an image.
    Args:
        config_path (str): Path to the YAML configuration file.
        file_path (str): Path to the processed .npz file.
    """
    # Load configuration from YAML file
    config = yaml.safe_load(open(config_path))

    # Check if the file exists
    print("DEBUG: File exists:", os.path.exists(file_path))

    # Load the processed data
    x_raw_bug = np.load(file_path)["processed_data"][:,:,0]
    filename = os.path.basename(file_path)
    print(f"DEBUG: loading successful and processing {filename}..")

    # Preprocess the data
    x_debug = preprocess(x_raw_bug, config['evaluation']['device'])
    x_debug = x_debug.cpu().numpy()
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
