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

    # Load and preprocess data
    x_raw = np.load(path)["processed_data"][:,:,0]
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
                                   if_drawsignal=if_drawsignal,png_save_dir=png_save_dir,
                                   png_name=png_name,fs=fs,x_test=x_test)
    #print(x_test_tensor_cnn[plot_index,0,:].shape)
    # Model prediction
    model.eval()
    with torch.no_grad():
        x_test_tensor_cnn = x_test_tensor_cnn.to(device)
        predictions = model(x_test_tensor_cnn)
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
                if_drawsignal=False, png_save_dir=None,
                png_name=None,
               fs=None, x_test=None):
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import polars as pl
    import os
    x_tmp = x_test.copy()
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
    x_test_tensor = torch.from_numpy(x_test).float()

    # Add channel dimension: (batch, 1, length, channel)
    x_test_tensor_all = x_test_tensor.unsqueeze(1)
    print(f'x_test_tensor_all shape: {x_test_tensor_all.shape}')
    # Normalize each (length, channel) column for each sample in the batch
    max_values_per_column = torch.max(torch.abs(x_test_tensor_all), dim=2, keepdim=True)[0]
    #print(f"max_values_per_column.shape: {max_values_per_column.shape}")
    max_values_per_column[max_values_per_column == 0] = 1.0  # Prevent division by zero
    x_test_tensor_all = x_test_tensor_all / max_values_per_column
    #print(f"max: {torch.max(x_test_tensor_all)}")

    # Use only the first channel for CNN input
    x_test_tensor_cnn = x_test_tensor_all[:, :, :]
    x_test_tensor_cnn = x_test_tensor_cnn.to(device)
    if if_log1p:
        x_test_tensor_cnn = torch.log1p(x_test_tensor_cnn)
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
        n_samples = x_test_tensor_cnn[1000, 0,:].cpu().numpy().shape[0]
        #print(f'n_samples: {n_samples}')
        t = np.arange(n_samples)/fs
        plt.figure(figsize=(10, 4))
        plt.rcParams["font.size"] = 18
        plt.plot(t*1e6,x_test_tensor_cnn[1000, 0,:].cpu().numpy(),
                 color='blue',label='Processed Signal')
        #print(f'first plot done')
        if not (if_hilbert or rolling_window):
            x_test_envelope = hilbert_cuda(x_test_tensor_cnn[:,0,:],device)
            plt.plot(t*1e6,x_test_envelope[1000,:],
                     color='red',label='Envelope')
            #print(f'second plot done')
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
    return x_test_tensor_cnn

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

def fft_analysis_and_png(path=None, device='cuda:0', 
                         png_save_dir=None, 
                        fs=None, x_test=None,
                        window_type='Blackman'):
    n_samples = x_test.shape[1]
    x_torch = torch.from_numpy(x_test).float()
    x_torch = x_torch.to(device)
    
