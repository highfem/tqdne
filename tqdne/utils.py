import logging
from pathlib import Path
from typing import Type

import PIL
import pytorch_lightning as pl
import torch

from tqdne.representations import *
from tqdne.metric import *
from tqdne.conf import Config

from tqdne.diffusion import LightningDDMP

from torchsummary import summary


import torch


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
        model = load_model(LightningDDMP, path, **kwargs)
    
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


def generate_data(model: Type[pl.LightningModule], model_data_representation: Type[Representation], batch_size: int, num_channels: int, signal_len: int, cond_params_range: dict[str, tuple] = None, cond_input_params: dict[str, list] = None, cond_input: torch.Tensor = None, device: str = 'cuda') -> np.ndarray:
    """
    Generates synthetic data using the provided model.

    Args:
        model (Type[pl.LightningModule]): The model used for generating data.
        model_data_representation (Type[Representation]): The data representation class.
        batch_size (int): The number of samples to generate.
        num_channels (int): The number of channels in the generated data.
        signal_len (int): The length of the generated signal.
        cond_params_range (dict[str, tuple], optional): The range of conditional parameters. Defaults to None.
        cond_input_params (dict[str, list], optional): The parameters for generating conditional inputs. Defaults to None.
        cond_input (torch.Tensor, optional): The conditional input tensor. Defaults to None.
        device (str, optional): The device to use for generating data. Defaults to 'cuda'.

    Returns:
        np.ndarray: The generated synthetic data.
        np.ndarray: The relative conditional input tensor.
    """
    assert (cond_params_range is not None and cond_input_params is not None) or (cond_input is not None), "Either cond_params_range and cond_input_params or cond_input must be provided."
    if cond_input is None:
        cond_input = generate_cond_inputs(batch_size, cond_input_params, cond_params_range)     
    with torch.no_grad():
        model_output = model.sample(shape=model_data_representation._get_input_shape((batch_size, num_channels, signal_len)), cond=torch.from_numpy(cond_input).to(device))
    return model_data_representation._invert_representation(model_output.to('cpu')), cond_input


def generate_cond_inputs(batch_size: int, cond_input_params: dict[str, list], cond_params_range: dict[str, tuple]) -> np.ndarray:
    """
    Generate conditional inputs for a given batch size.

    Args:
        batch_size (int): The size of the batch.
        cond_input_params (dict[str, list]): A dictionary containing the conditional input parameters. Dict values are lists of possible values for each parameter, which will be used to uniformly draw conditional inputs. If a value is None, the values will be drawn from a uniform distribution between the corresponding min and max values in cond_params_range.
        cond_params_range (dict[str, tuple]): A dictionary containing the range of values for each conditional parameter.

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
                cond_inputs.append(np.random.uniform(cond_params_range[param][0], cond_params_range[param][1], size=batch_size))
    return np.stack(cond_inputs, axis=1)

# TODO: maybe not the best design choice to use Metrics. Simply define some plot functions (and rollback changes to metric.py that i made for the title)
def plot_metrics(data: np.ndarray, metrics: list[Metric], target=None) -> None:
    """
    Plots the metrics for the given data.

    Args:
        data (np.ndarray): The input data.
        metrics (list[Metric]): A list of Metric objects.
        target: The target data (optional).

    Returns:
        None
    """
    # if data is a single sample, 
    for metric in metrics:
        metric.reset()
        metric.update(data, target)
        metric.plot().show()


# TODO: function that plots waveform and psd of all the 3 channels for a single signal
def plot_waveform_and_psd(data: np.ndarray, cond_input: np.ndarray, signal_length: int, fs: int):
    """
    Plots the waveform and power spectral density (PSD) of the given data.

    Args:
        data (np.ndarray): The input data array with shape (num_channels, signal_length).
        cond_input (np.ndarray): The conditional input array with shape (num_features,).
        signal_length (int): The length of the signal.
        fs (int): The sampling frequency.

    Returns:
        None
    """
    # Plotting the three channels over time
    fig, axs = plt.subplots(data.shape[0], 2, figsize=(18, 22), constrained_layout=True)

    channels = ['channel1', 'channel2', 'channel3']  # TODO: Replace with actual channel names
    time_ax = np.arange(0, signal_length) / fs
    freq_ax = np.fft.rfftfreq(signal_length, 1 / fs)
    for i, channel in enumerate(channels):
        axs[i, 0].plot(time_ax, data[i], label=channel)
        axs[i, 0].set_xlabel('Time (s)')
        axs[i, 0].set_ylabel('Amplitude') # TODO: insert correct unit of measurement
        axs[i, 0].legend()

        psd = np.abs(np.fft.rfft(data[i], axis=-1)) ** 2
        axs[i, 1].semilogx(freq_ax, 10 * np.log10(psd),  label=channel)
        axs[i, 1].set_xlabel('Frequency (Hz)')
        axs[i, 1].set_ylabel('Power Spectral Density (dB/Hz)')
        axs[i, 1].legend()
    
    cond_input_dict = {key: cond_input[i] for i, key in enumerate(Config().features_keys)}
    fig.suptitle(f"Cond. params: {cond_input_dict}")
    #plt.tight_layout()
    plt.show()
        
# TODO: function that plots only 1 channel of the waveform but for multiple signals (optionally with the log envelope on the same plot)
def plot_waveforms(data: np.ndarray, signal_length: int, fs: int, channel_index: int = 0, plot_log_envelope: bool = False):
    # TODO ADD cond_input: np.ndarray for the title
    """
    Plots the waveform of a given channel for multiple signals.

    Args:
        data (np.ndarray): The input data.
        signal_length (int): The length of the signals.
        fs (int): The sampling frequency.
        channel_index (int): The index of the channel to plot. Defaults to 0.
        plot_log_envelope (bool): Whether to plot the log envelope (optional).

    Returns:
        None
    """
    n = data.shape[0]
    fig, axs = plt.subplots(n, 1, figsize=(10, 6*n))

    time_ax = np.arange(0, signal_length) / fs

    for i in range(n):
        axs[i].plot(time_ax, data[i, channel_index], label=f"Signal {i+1}")
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Amplitude')  # TODO: insert correct unit of measurement
        axs[i].legend()

        if plot_log_envelope:
            log_envelope = np.log(_get_envelope(data[i, channel_index]) + 1e-10) #TODO: check if it is the same envelope used by Robin
            axs[i].plot(time_ax, log_envelope, label='Log Envelope')
            axs[i].legend()

    plt.show()

def _get_envelope(signal, type='mean', window_size=100):
    """
    Compute the envelope of the given signal.

    Args:
        signal (np.ndarray): The input signal.
        type (str): The type of the envelope to compute. Available options are 'mean', 'max', and 'hilbert'. Defaults to 'mean'.
        window_size (int): The size of the window for the median filter. Defaults to 100.

    Returns:
        np.ndarray: The envelope of the signal.
    """
    if type == 'mean':
        return np.convolve(np.abs(signal), np.ones(window_size)/window_size, mode='same') # TODO: check if it is the correct way to compute the mean envelope
    elif type == 'hilbert':
        return np.abs(hilbert(signal))
    else:
        raise ValueError(f"Invalid envelope type. Available options are 'mean', 'max', and 'hilbert'.")

     

# TODO: function that plots the waveform (or log envelope) of a given channel for multiple signals dividen in bins by distance and magnitude    
def plot_by_bins(data: np.ndarray, cond_input: np.ndarray, conditional_params_range: dict[str, tuple], num_magnitude_bins:int, num_distance_bins: int, signal_length: int, fs: int, channel_index: int = 0, plot: str = 'waveform'):                  
    plots = [[list() for _ in range(num_distance_bins)] for _ in range(num_magnitude_bins)]  

    min_mag, max_mag = conditional_params_range['magnitude']
    min_dist, max_dist = conditional_params_range['hypocentral_distance']
    
    # TODO: check enough samples for bin
    #        count_mag_values = len(random_cond_inputs_batch[:, 3].unique())
#        count_dist_values = len(random_cond_inputs_batch[:, 0].unique())
#        
#        if num_magnitude_bins > count_mag_values:
#            num_magnitude_bins = count_mag_values
#        
#        if num_distance_bins > count_dist_values:
#            num_distance_bins = count_dist_values
    
    for i, cond_sample in enumerate(cond_input):
        mag = cond_sample[3]
        dist = cond_sample[0]
        mag_bin = int(
            (mag - min_mag) / (max_mag - min_mag) * num_magnitude_bins
        )
        dist_bin = int(
            (dist - min_dist) / (max_dist - min_dist) * num_distance_bins
        )
        to_plot = data[i, channel_index] if plot == 'waveform' else np.log(_get_envelope(data[i, channel_index]) + 1e-10)
        plots[mag_bin][dist_bin].append(to_plot)

    fig, axs = plt.subplots(num_distance_bins, num_magnitude_bins, figsize=(18, 22), constrained_layout=True) 
    time_ax = np.arange(0, signal_length) / fs
    for i in range(num_distance_bins):
        for j in range(num_magnitude_bins):
            plots_tensor = np.stack(plots[j][i]) if len(plots[j][i])>0 else np.zeros(signal_length)
            mean_signal = np.mean(plots_tensor, axis=0) # TODO: or median?
            std_signal = np.std(plots_tensor, axis=0)
            #p25 = np.percentile(plots_tensor, 0.25, axis=0)
            #p75 = np.percentile(plots_tensor, 0.75, axis=0)
            axs[i, j].plot(time_ax, mean_signal, label='mean signal')
            axs[i, j].fill_between(time_ax, mean_signal - std_signal, mean_signal + std_signal, color='grey', alpha=0.5, label='+- std dev')
            #axs[i, j].fill_between(time_ax, p25, p75, color='red', alpha=0.3, label='25th-75th percentile') # TODO: maybe fix with mean - p25 amd meam+p75
            axs[i, j].set_title(f"Mag. Bin: [{min_mag + (max_mag - min_mag) / num_magnitude_bins * i:.2f}, {min_mag + (max_mag - min_mag) / num_magnitude_bins * (i+1):.2f}] - Dist Bin: [{min_dist + (max_dist - min_dist) / num_distance_bins * j:.2f}, {min_dist + (max_dist - min_dist) / num_distance_bins * (j+1):.2f}] - Num. Samples: {plots_tensor.shape[0]}")
            axs[i, j].set_xlabel('Time (s)')
            ylabel = 'Amplitude' if plot == 'waveform' else 'Log Envelope' 
            axs[i, j].set_ylabel(ylabel) 
            axs[i, j].legend()
