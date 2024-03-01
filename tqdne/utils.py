import logging
from pathlib import Path
from typing import Type

import PIL
import pytorch_lightning as pl
from scipy import signal
import torch
import matplotlib.pyplot as plt

from tqdne.representations import *
#from tqdne.metric import *
from tqdne.conf import Config

from tqdne.diffusion import LightningDiffusion

from torchsummary import summary


import torch

def to_numpy(x):
    return x.numpy(force=True) if isinstance(x, torch.Tensor) else x


def load_model_by_name(model_name: str, path: Path = None, **kwargs):
    """Load the model from the outputs directory given the name of the model. 
    If a path is specified, return the specific model from that path.

    Parameters
    ----------
    model_name : str
        The name of the model to load. Only available options are "diffusion_1d" and "GAN".
    path : Path
        The checkpoint path.
    **kwargs : dict
        The keyword arguments to pass to the load_from_checkpoint method.
        Instead one can add `self.save_hyperparameters()` to the init method
        of the model.

    Returns
    -------
    model : pl.LightningModule
        The trained model.
    data_representation : tqdne.representations.Representation
        The data representation used by the model.

    """

    available_models = ["diffusion_1d", "GAN"]
    if model_name not in available_models:
        raise ValueError(f"Invalid model name. Available options are: {', '.join(available_models)}")
    
    if model_name == "diffusion_1d":
        if path is None:
            path = get_last_checkpoint("/users/abosisio/scratch/tqdne/outputs/COND-1D-UNET-DDPM-envelope") # TODO: channge to CSCS path
            # TODO: use on_save_checkpoint
            # to save and load the representation (thte model itself it's not safe to store)
        data_representation = SignalWithEnvelope(Config()) # TODO: not sure if it is the best design choice
        model = load_model(LightningDiffusion, path, **kwargs)
    
    elif model_name == "GAN":
        #if path is None:
        #    path = get_last_checkpoint("/users/abosisio/scratch/tqdne/outputs/GAN") # TODO: ask Francisco (or Robins(?))
        data_representation = None
        model = None

    else:
        raise ValueError(f"Invalid model name. Available options are: {', '.join(available_models)}")        


    #model = getattr(pl.LightningModule, f"load_from_checkpoint")(path, **kwargs)
    
    return model, data_representation


def load_model(type: Type[pl.LightningModule], path: Path, **kwargs):
    """Load the model from the outputs directory.

    Parameters
    ----------
    type : Type[pl.LightningModule]
        The type of the model to load.
    path : Path
        The checkpoint path.
    **kwargs : dict
        The keyword arguments to pass to the load_from_checkpoint method.
        Instead one can add `self.save_hyperparameters()` to the init method
        of the model.

    Returns
    -------
    model : pl.LightningModule
        The trained model.

    """
    if not path.exists():
        logging.info("Model not found. Returning None.")
        return None
    model = type.load_from_checkpoint(path, **kwargs)
    return model


def fig2PIL(fig):
    """Convert a matplotlib figure to a PIL Image.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure.

    Returns
    -------
    PIL.Image
        The PIL Image.

    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.tostring_rgb()
    return PIL.Image.frombytes(mode="RGB", size=(w, h), data=buf)



def get_last_checkpoint(dirpath):
    checkpoints = sorted(list(Path(dirpath).glob("*.ckpt")))
    if len(checkpoints) == 0:
        logging.info("No checkpoint found. Returning None.")
        return None
    # Load the checkpoint with the latest epoch
    checkpoint = checkpoints[-1]
    logging.info(f"Last checkpoint is : {checkpoint}")
    return checkpoint


# def get_model_summary(model: Type[pl.LightningModule], input_size=(1, 1, 64), batch_size=-1, device="cuda":
#     """Get the model summary.

#     Parameters
#     ----------
#     model : Type[pl.LightningModule]
#         The model.

#     Returns
#     -------
#     dict
#         The model summary.

#     """
#     return summary(model, input_size=input_size, batch_size=batch_size, device=device)


def get_cond_input_tensor(*conditioning_params: dict):
    """Get the conditional input tensor.

    Parameters
    ----------
    conditioning_params : dict
        The conditioning parameters.

    Returns
    -------
    torch.Tensor
        The conditional input tensor.

    """ 
    cond_params = []
    for params in conditioning_params:
        cond_params.append([param for param in params.values() if param is not None])
    return torch.tensor(cond_params, dtype=torch.float32)


def generate_data(model: Type[pl.LightningModule], model_data_representation: Type[Representation], batch_size: int, cond_input_params: dict[str, list] = None, cond_input: torch.Tensor = None, device: str = 'cuda') -> np.ndarray:
    """
    Generates synthetic data using a given model and data representation.

    Args:
        model (Type[pl.LightningModule]): The model used to generate the data.
        model_data_representation (Type[Representation]): The data representation used by the model.
        batch_size (int): The number of samples to generate.
        cond_input_params (dict[str, list], optional): The parameters for generating conditional inputs. Defaults to None.
        cond_input (torch.Tensor, optional): The conditional inputs. Defaults to None.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.

    Returns:
        np.ndarray: A dictionary containing the generated waveforms and the conditional inputs ("waveforms" and "cond").
    """
    assert cond_input_params is not None or cond_input is not None, "Either cond_input_params or cond_input must be provided."
    config = Config()
    conditional_params_range = config.conditional_params_range
    signal_len = config.signal_length
    num_channels = config.num_channels
    if cond_input is None:
        cond_input = generate_cond_inputs(batch_size, cond_input_params)
        max_batch_size = 64
        num_samples = batch_size
        generated_waveforms = []
        with torch.no_grad():
            for i in range(0, num_samples, max_batch_size):
                batch_cond_input = cond_input[i:i+max_batch_size]
                batch_size = batch_cond_input.shape[0]
                model_output = model.sample(shape=model_data_representation._get_input_shape((batch_size, num_channels, signal_len)), cond=torch.from_numpy(batch_cond_input).to(device))
                generated_waveforms.append(model_data_representation.invert_representation(model_output))
        generated_waveforms = np.concatenate(generated_waveforms, axis=0)
    return {"waveforms": model_data_representation.invert_representation(generated_waveforms), "cond": cond_input} # TODO: maybe refactor with  "cond": _get_cond_params_dict(cond_input)

def get_samples(data: dict[str, np.ndarray], num_samples = None, indexes: list = None) -> dict[str, np.ndarray]:
    assert num_samples is not None or indexes is not None, "Either num_samples or indexes must be provided."
    if indexes is None:
        indexes = np.random.choice(data['waveforms'].shape[0], num_samples, replace=False)
    return {key: value[indexes] for key, value in data.items()}



def generate_cond_inputs(batch_size: int, cond_input_params: dict[str, list]) -> np.ndarray:
    """
    Generate conditional inputs for a given batch size.

    Args:
        batch_size (int): The size of the batch.
        cond_input_params (dict[str, list]): A dictionary containing the conditional input parameters. Dict values are lists of possible values for each parameter, which will be used to uniformly draw conditional inputs. If a value is None, the values will be drawn from a uniform distribution between the corresponding min and max values in cond_params_range.

    Returns:
        np.ndarray: An array of shape [batch_size, len(cond_input_params.keys())] containing the generated conditional inputs.
    """
    cond_inputs = []
    for param, values in cond_input_params.items():
        print(param, values)
        if values is not None:
            cond_inputs.append(np.random.choice(values, size=batch_size))
        else:
            if param == "is_shallow_crustal":
                cond_inputs.append(np.random.randint(0, 2, (batch_size,)))
            else:
                cond_params_range = Config().conditional_params_range
                cond_inputs.append(np.random.uniform(cond_params_range[param][0], cond_params_range[param][1], size=batch_size))
    return np.stack(cond_inputs, axis=1)

# TODO: maybe not the best design choice to use Metrics. Simply define some plot functions (and rollback changes to metric.py that i made for the title)
#def plot_metrics(data: np.ndarray, metrics: list[Metric], target=None) -> None:
#    """
#    Plots the metrics for the given data.

    # Args:
    #     data (np.ndarray): The input data.
    #     metrics (list[Metric]): A list of Metric objects.
    #     target: The target data (optional).

    # Returns:
    #     None
    # """
    # # if data is a single sample, 
    # for metric in metrics:
    #     metric.reset()
    #     metric.update(data, target)
    #     metric.plot().show()

def _get_cond_params_dict(cond_input: np.ndarray) -> dict[str, float]:
    return {key: cond_input[i] for i, key in enumerate(Config().features_keys)}    


def plot_waveform_and_psd(data: dict[str, np.ndarray]) -> None:
    """
    Plots the waveform and power spectral density (PSD) of the given data.

    Args:
        data (dict[str, np.ndarray]): A dictionary containing the waveform and condition input data.

    Returns:
        None
    """
    config = Config()
    fs = config.fs
    signal_length = config.signal_length

    waveform = data['waveforms'][0]
    cond_input = data['cond'][0]

    # Plotting the three channels over time
    fig, axs = plt.subplots(waveform.shape[0], 2, figsize=(18, 22), constrained_layout=True)

    channels = ['channel1', 'channel2', 'channel3']  # TODO: Replace with actual channel names
    time_ax = np.arange(0, signal_length) / fs
    freq_ax = np.fft.rfftfreq(signal_length, 1 / fs)
    for i, channel in enumerate(channels):
        axs[i, 0].plot(time_ax, waveform[i], label=channel)
        axs[i, 0].set_xlabel('Time (s)')
        axs[i, 0].set_ylabel('Amplitude') # TODO: insert correct unit of measurement
        axs[i, 0].legend()

        psd = np.abs(np.fft.rfft(waveform[i], axis=-1)) ** 2
        axs[i, 1].semilogx(freq_ax, 10 * np.log10(psd),  label=channel)
        axs[i, 1].set_xlabel('Frequency (Hz)')
        axs[i, 1].set_ylabel('Power Spectral Density (dB/Hz)')
        axs[i, 1].legend()
    
    fig.suptitle(f"Cond. params: {_get_cond_params_dict(cond_input)}")
    #plt.tight_layout()
    plt.show()
        

def plot_waveforms(data: dict[str, np.ndarray], channel_index: int = 0, plot_envelope: bool = True, plot_log_envelope: bool = True):
    """
    Plot waveforms for multiple signals.

    Args:
        data (dict[str, np.ndarray]): A dictionary containing the waveform and condition input data.
        channel_index (int, optional): The index of the channel to plot. Defaults to 0.
        plot_envelope (bool, optional): Whether to plot the envelope of the signal. Defaults to True.
        plot_log_envelope (bool, optional): Whether to plot the logarithm of the envelope. Defaults to True.

    Returns:
        None
    """
    config = Config()
    fs = config.fs
    signal_length = config.signal_length

    waveforms = data['waveforms']
    cond_input = data['cond']

    n = waveforms.shape[0]
    m = 1 if plot_log_envelope == False else 2
    time_ax = np.arange(0, signal_length) / fs

    fig = plt.figure(constrained_layout=True, figsize=(15, 5*n))
    fig.suptitle(f'Plot of {n} different signals - channel {channel_index}', fontsize=16)

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=n, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f'Sample {row} - Cond. Input: {_get_cond_params_dict(cond_input[row])}')

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=m)
        if m == 1:
            axs = [axs]
        axs[0].plot(time_ax, waveforms[row, channel_index], label=f"Gen. Signal")
        if plot_envelope:
            env = _get_moving_avg_envelope(waveforms[row, channel_index].reshape(1, -1), window_len=int(1*fs))[0]
            axs[0].plot(time_ax, env, label=f"Envelope")
        axs[0].set_xlabel('Time (s)')  
        axs[0].set_ylabel('Amplitude')  # TODO: insert correct unit of measurement  
        axs[0].legend()
        if plot_log_envelope:
            assert(plot_envelope), "If plot_log_envelope is True, plot_envelope must be True as well."
            axs[1].plot(time_ax, np.log(env + 1e-10), label=f"Log Envelope")    
            axs[1].set_xlabel('Time (s)')  
            axs[1].legend()

    plt.show()

# def _get_moving_avg_envelope(a, window_len=50, step_length=50):
#     def _rolling_window(x, window_len, step_length):
#         pos = 0
#         while pos < len(x) - window_len:
#             yield x[pos : pos+window_len]
#             pos += step_length

#     a_pad = np.pad(a, (window_len//2, window_len//2), mode='edge')
#     rw = _rolling_window(a_pad, window_len, step_length)
#     rolling_mean = []
#     for i, window in enumerate(rw):
#         rolling_mean.append(window.mean())
#     return np.array(rolling_mean)
    
def _get_moving_avg_envelope(a, window_len=50):
    return signal.convolve(np.abs(a), np.ones((a.shape[0], window_len)), mode='same') / window_len
          
def plot_by_bins(data: dict[str, np.ndarray], num_magnitude_bins:int, num_distance_bins: int, channel_index: int = 0, plot: str = 'waveform'):                  
    """
    Plot the data by dividing it into bins based on conditional parameters.

    Args:
        data (dict[str, np.ndarray]): A dictionary containing the waveform and condition input data.
        num_magnitude_bins (int): The number of magnitude bins.
        num_distance_bins (int): The number of distance bins.
        channel_index (int, optional): The index of the channel to plot. Defaults to 0.
        plot (str, optional): The type of plot. Can be 'waveform' or 'log_envelope'. Defaults to 'waveform'.

    Returns:
        None
    """
    config = Config()
    fs = config.fs
    signal_length = config.signal_length
    conditional_params_range = config.conditional_params_range

    waveforms = data['waveforms']
    cond_input = data['cond']

    plots = [[list() for _ in range(num_distance_bins)] for _ in range(num_magnitude_bins)]  

    min_mag, max_mag = conditional_params_range['magnitude']
    min_dist, max_dist = conditional_params_range['hypocentral_distance']
    
    count_mag_values = len(np.unique(cond_input[:, 3]))
    count_dist_values = len(np.unique(cond_input[:, 0]))
    if num_magnitude_bins > count_mag_values:
        num_magnitude_bins = count_mag_values
    if num_distance_bins > count_dist_values:
        num_distance_bins = count_dist_values
    
    for i, cond_sample in enumerate(cond_input):
        mag = cond_sample[3]
        dist = cond_sample[0]
        mag_bin = int(
            (mag - min_mag) / (max_mag - min_mag) * num_magnitude_bins
        )
        dist_bin = int(
            (dist - min_dist) / (max_dist - min_dist) * num_distance_bins
        )
        to_plot = waveforms[i, channel_index] if plot == 'waveform' else np.log(_get_moving_avg_envelope(waveforms[i, channel_index].reshape(1, -1))[0] + 1e-10)
        plots[mag_bin][dist_bin].append(to_plot)

    fig, axs = plt.subplots(num_distance_bins, num_magnitude_bins, figsize=(18, 22), constrained_layout=True) 
    time_ax = np.arange(0, signal_length) / fs
    for i in range(num_distance_bins):
        for j in range(num_magnitude_bins):
            plots_tensor = np.stack(plots[j][i]) if len(plots[j][i])>0 else np.zeros((1, signal_length))
            #mean_signal = np.mean(plots_tensor, axis=0) # TODO: or median?
            median_signal = np.median(plots_tensor, axis=0)
            #std_signal = np.std(plots_tensor, axis=0)
            p25 = np.percentile(plots_tensor, 25, axis=0)
            p75 = np.percentile(plots_tensor, 75, axis=0)
            #axs[i, j].plot(time_ax, mean_signal, label='mean signal')
            axs[i, j].plot(time_ax, median_signal, label='median signal')
            #axs[i, j].fill_between(time_ax, mean_signal - std_signal, mean_signal + std_signal, alpha=0.3, label='+- std dev')
            axs[i, j].fill_between(time_ax, p75, p25, alpha=0.5, label='IQR (25-75%)')
            axs[i, j].set_title(f"Mag. Bin: [{min_mag + (max_mag - min_mag) / num_magnitude_bins * i:.2f}, {min_mag + (max_mag - min_mag) / num_magnitude_bins * (i+1):.2f}] - Dist Bin: [{min_dist + (max_dist - min_dist) / num_distance_bins * j:.2f}, {min_dist + (max_dist - min_dist) / num_distance_bins * (j+1):.2f}] - Num. Samples: {len(plots[j][i])}")
            axs[i, j].set_xlabel('Time (s)')
            ylabel = 'Amplitude' if plot == 'waveform' else 'Log Envelope' 
            axs[i, j].set_ylabel(ylabel) 
            axs[i, j].legend()

    plt.show()

def divide_data_by_bins(data: dict[str, np.ndarray], magnitude_bins: list[tuple], distance_bins: list[tuple]) -> dict[(tuple, tuple), dict[str, np.ndarray]]:
    """
    Divides the given data into bins based on magnitude and distance.

    Args:
        data (dict[str, np.ndarray]): The input data dictionary containing arrays.
        magnitude_bins (list[tuple]): The list of magnitude bins as tuples.
        distance_bins (list[tuple]): The list of distance bins as tuples.

    Returns:
        dict[str, dict[str, np.ndarray]]: A dictionary containing the divided data.

    """
    divided_data = {}
    cond_input = data['cond']
    for i, dist_bin in enumerate(distance_bins):
        for j, mag_bin in enumerate(magnitude_bins):
            bins_indexes = (cond_input[:, 0] >= dist_bin[0]) & (cond_input[:, 0] < dist_bin[1]) & (cond_input[:, 3] >= mag_bin[0]) & (cond_input[:, 3] < mag_bin[1])
            if np.any(bins_indexes):
                divided_data[f"({dist_bin}, {mag_bin})"] = {"waveforms": data['waveforms'][bins_indexes], "cond": cond_input[bins_indexes]}
    return divided_data


def plot_log_envelope_bins(distance_bins: list[tuple], magnitude_bins: list[tuple], test_data: dict[str, np.ndarray], data: dict[str, np.ndarray] = None, model: Type[pl.LightningModule] = None, model_data_representation: Type[Representation] = None, channel_index: int = 0) -> None:
    assert (data is not None) or (model is not None and model_data_representation is not None), "Either data or model and model_data_representation must be provided."
    test_data_by_bins = divide_data_by_bins(test_data, magnitude_bins, distance_bins)
    # keep only one sample per bin
    test_sample_by_bins = {key: get_samples(test_data_by_bins[key], num_samples=1) for key in test_data_by_bins.keys()}
    cond_inputs = np.array([test_sample_by_bins[key]['cond'][0] for key in test_sample_by_bins.keys()])
    if data is None:
        data = generate_data(model, model_data_representation, len(test_sample_by_bins), cond_input=cond_inputs)
    data_by_bins = divide_data_by_bins(data, magnitude_bins, distance_bins)

    config = Config()
    signal_length = config.signal_length
    fs = config.fs


    time_ax = np.arange(0, signal_length) / fs
    fig, axs = plt.subplots(len(distance_bins), 2, figsize=(18, 22))
    for i, dist_bin in enumerate(distance_bins):
        for j, mag_bin in enumerate(magnitude_bins):
            axs[i, 0].plot(np.log(_get_moving_avg_envelope(data_by_bins[f"({dist_bin}, {mag_bin})"]['waveforms'][0] + 1e-10))[channel_index, :], label=f"Mag. Bin: {mag_bin} - Dist Bin: {dist_bin}")
            axs[i, 1].plot(np.log(_get_moving_avg_envelope(test_sample_by_bins[f"({dist_bin}, {mag_bin})"]['waveforms'][0] + 1e-10))[channel_index, :], label=f"Mag. Bin: {mag_bin} - Dist Bin: {dist_bin}")
            axs[i, 0].set_title('Generated')
            axs[i, 1].set_title('Real')
            axs[i, 0].set_xlabel('Time (s)')
            axs[i, 0].set_ylabel('Log Envelope') 
            axs[i, 0].legend()
            axs[i, 1].set_xlabel('Time (s)')
            axs[i, 1].set_ylabel('Log Envelope') 
            axs[i, 1].legend()
     
    plt.show()

    
