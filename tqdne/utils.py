import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Type
from pathlib import Path

import h5py
import PIL
import os
import pytorch_lightning as pl
from scipy import signal
import torch
import matplotlib.pyplot as plt


from tqdne.conf import Config

from tqdne.consistency_model import LightningConsistencyModel
from tqdne.diffusion import LightningDiffusion
from tqdne.classification import LightningClassifier
from tqdne.representations import *
from diffusers import DDPMScheduler, DDIMScheduler

general_config = Config()

def get_data_representation(configs):
    """
    Get the data representation based on the given configurations.

    Args:
        configs (object): The configurations object.

    Returns:
        object: The data representation object.

    Raises:
        ValueError: If the representation name is unknown.
    """
    repr_name = configs.data_repr.name if hasattr(configs, "data_repr") else "Signal"
    repr_params = configs.data_repr.params if hasattr(configs, "data_repr") else {}
    if repr_name == "SignalWithEnvelope":
        configs.model.net_params.dims = 1
        return SignalWithEnvelope(**repr_params)
    elif repr_name == "LogSpectrogram":
        configs.model.net_params.dims = 2
        return LogSpectrogram(**repr_params)
    elif repr_name == "Signal":
        configs.model.net_params.dims = 1
        return Signal(**repr_params)
    else:
        raise ValueError(f"Unknown representation name: {repr_name}")


def load_model(model_ckpt_path: Path, use_ddim: bool = True, **kwargs):
    """
    Loads a model from a given checkpoint directory path.

    Args:
        model_ckpt_path (Path): The path of the directory containing the model checkpoints or a checkpoint file.
        use_ddim (bool, optional): Whether to use the DDIM scheduler. Defaults to True.
        **kwargs: Additional keyword arguments to be passed to the model's `load_from_checkpoint` method.

    Returns:
        Tuple: A tuple containing the loaded model, the data representation used by the model, and the checkpoint path.

    Raises:
        FileNotFoundError: If the model checkpoint file is not found.
        ValueError: If the model name in the checkpoint path is unknown.
    """
    if not model_ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_ckpt_path}.")

    model_ckpt_path = str(model_ckpt_path)
    if model_ckpt_path.split(".")[-1] == "ckpt":
        ckpt = model_ckpt_path
        model_dir_name = model_ckpt_path.split("/")[-2]
    else:
        ckpt = get_last_checkpoint(model_ckpt_path)
        model_dir_name = model_ckpt_path.split("/")[-1]

    if "ddpm" in model_dir_name:
        model = LightningDiffusion.load_from_checkpoint(ckpt, **kwargs)
        assert model.hparams.ml_config is not None, "The model must have a ml_config attribute."
        ml_configs = model.hparams.ml_config
        if use_ddim:
            noise_scheduler = DDIMScheduler(**ml_configs.model.scheduler_params)
            noise_scheduler.set_timesteps(num_inference_steps=ml_configs.model.scheduler_params.num_train_timesteps // 10)
        else:
            noise_scheduler = DDPMScheduler(**ml_configs.model.scheduler_params)
        model.noise_scheduler = noise_scheduler
    elif "ddim" in model_dir_name:
        model = LightningDiffusion.load_from_checkpoint(ckpt, **kwargs)
        assert model.hparams.ml_config is not None, "The model must have a ml_config attribute."
        ml_configs = model.hparams.ml_config
    elif "consistency-model" in model_dir_name:
        model = LightningConsistencyModel.load_from_checkpoint(ckpt, **kwargs)
        assert model.hparams.ml_config is not None, "The model must have a ml_config attribute."
        ml_configs = model.hparams.ml_config
    elif "classifier" in model_dir_name:
        model = LightningClassifier.load_from_checkpoint(ckpt, **kwargs)
        assert model.hparams.ml_config is not None, "The model must have a ml_config attribute."
        ml_configs = model.hparams.ml_config
    else:
        raise ValueError(f"Unknown model name: {model_dir_name}")

    data_repr = get_data_representation(ml_configs)
    model.eval()
    return model, data_repr, ckpt

@staticmethod


def adjust_signal_length(original_signal_length: int, unet: pl.LightningModule, data_repr: Type[Representation], downsample_factor: int = 1):

    def closest_divisible_number(n, div):
        return round(n / div) * div

    unet_max_divisor = 2 * len(unet.channel_mult) # count the number of down/up blocks in the UNet, which is the number of times the signal is downsampled (i.e., divided by 2)
    data_repr_divisor = data_repr.adjust_length_params(unet_max_divisor) * downsample_factor

    return closest_divisible_number(original_signal_length, data_repr_divisor)


def plot_envelope(signal, envelope_function, title=None, **envelope_params):
    envelope = envelope_function(signal, **envelope_params)
    fig = plt.figure()
    
    ax1 = fig.add_subplot(211)
    ax1.plot(signal, alpha=0.6, label='signal')
    ax1.plot(envelope, linewidth=1,  label='envelope')
    title = title if title is not None else envelope_function.__name__
    ax1.set_title(title)
    ax1.legend()

    ax2 = fig.add_subplot(212)
    sig_scaled_by_envelope = np.nan_to_num(signal/envelope, nan=0., posinf=0., neginf=0.) # since if the envelope is 0, the signal is also small. 

    ax2.plot(sig_scaled_by_envelope)
    ax2.set_title("Signal scaled by its envelope")
    
    plt.tight_layout()
    plt.show()

    return envelope

def print_model_info(model, model_data_repr, ckpt):
    model_data_repr.plot_representation()
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):}")
    print(f"Model size: {ckpt.stat().st_size / 1e6:.2f} MB")
    print(f"UNet scheme: \n base num. channels: {model.hparams.ml_config.model.net_params.model_channels} \n channel multipliers (down/up blocks): {model.hparams.ml_config.model.net_params.channel_mult} \n num. ResBlocks per down/up block: {model.hparams.ml_config.model.net_params.num_res_blocks} \n use Attention: {model.hparams.ml_config.model.net_params.num_heads is not None} \n conv. kernel size: {model.hparams.ml_config.model.net_params.conv_kernel_size} ")
    print(f"Diffusion prediction type: {model.hparams.ml_config.model.scheduler_params.prediction_type}")
    print(f"Learning rate schedule: \n start: {model.hparams.ml_config.optimizer_params.learning_rate} \n scheduler: {model.hparams.ml_config.optimizer_params.scheduler_name} \n warmup steps: {model.hparams.ml_config.optimizer_params.lr_warmup_steps}")
    print(f"Batch size: {model.hparams.ml_config.optimizer_params.batch_size}")
    downsampling_factor = int(str(ckpt).split("downsampling:")[-1][0])
    if downsampling_factor > 1:
        print(f"Downsampling factor: {downsampling_factor}. The model was trained on signals with length {general_config.signal_length // downsampling_factor}, as the sampling rate used was {general_config.fs // downsampling_factor} instead of {general_config.fs}" )
    else: 
        print(f"The model was trained on signals with length {general_config.signal_length}, as the sampling rate used was {general_config.fs}, whihc is the original sampling rate." )
    print(f"Data representation shape: {model_data_repr.get_shape((1, general_config.num_channels, general_config.signal_length))} (batch_size, channels, signal_length)")
    print(f"Data representation name: {model_data_repr.__class__.__name__}")
    if hasattr(model.hparams.ml_config.data_repr, "env_function"):
        print(f"Data representation envelope function: {model.hparams.ml_config.data_repr.params.env_function}") 
    print("ckpt file:", ckpt)    


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
    checkpoints = sorted(list(Path(dirpath).glob("*.ckpt")), key=lambda x: os.path.getmtime(x))
    if len(checkpoints) == 0:
        logging.info("No checkpoint found. Returning None.")
        return None
    # Load the checkpoint with the latest epoch
    checkpoint = checkpoints[-1]
    logging.info(f"Last checkpoint is : {checkpoint}")
    return checkpoint


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


def generate_data(model: Type[pl.LightningModule], model_data_representation: Type[Representation], raw_output: bool, num_samples: int, cond_input_params: dict[str, list] = None, cond_input: torch.Tensor = None, device: str = 'cuda', save_path: Path = None) -> np.ndarray:
    """
    Generates synthetic data using a given model and data representation.

    Args:
        model (Type[pl.LightningModule]): The model used to generate the data.
        model_data_representation (Type[Representation]): The data representation used by the model.
        num_samples (int): The number of samples to generate.
        cond_input_params (dict[str, list], optional): The parameters for generating conditional inputs. Defaults to None.
        cond_input (torch.Tensor, optional): The conditional inputs. Defaults to None.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.
        save_path (Path, optional): The directory path to save the generated inverted data. Defaults to None.

    Returns:
        np.ndarray: A dictionary containing the generated waveforms and the conditional inputs ("waveforms" and "cond").
    """
    assert not(cond_input_params is not None and cond_input is not None), "Either cond_input_params or cond_input must be provided."
    if cond_input_params is None and cond_input is None:
        logging.warning("No conditional input provided. Using the range values for each parameter.")
        cond_input_params = {k: None for k,v in general_config.conditional_params_range.items()}
    signal_len = general_config.signal_length
    num_channels = general_config.num_channels
    if cond_input is None:
        cond_input = generate_cond_inputs(num_samples, cond_input_params)
    else:
        cond_input = np.resize(cond_input, (num_samples, cond_input.shape[1]))
    generated_waveforms = []
    with torch.no_grad():
        batch_size = model.hparams.ml_config.optimizer_params.batch_size
        if num_samples < batch_size:
            batch_size = num_samples
        model_input_shape = model_data_representation.get_shape((batch_size, num_channels, signal_len))
        for i in range(0, num_samples, batch_size):
            print(f"Batch {i//batch_size + 1}/{num_samples//batch_size}")
            batch_cond_input = cond_input[i:i+batch_size]
            shape = (batch_cond_input.shape[0], *model_input_shape[1:])
            model_output = model.sample(shape=shape, cond=torch.from_numpy(batch_cond_input).to(device, dtype=torch.float32))
            if raw_output:
                generated_waveforms.append(to_numpy(model_output))
                if save_path is not None:
                    batch_waveforms = model_data_representation.invert_representation(model_output)
                    save_data({"waveforms": batch_waveforms, "cond": batch_cond_input}, save_path)
            else:    
                generated_waveforms.append(model_data_representation.invert_representation(model_output))
                if save_path is not None:
                    save_data({"waveforms": generated_waveforms[-1], "cond": batch_cond_input}, save_path)
    generated_waveforms = np.concatenate(generated_waveforms, axis=0)
    return {"waveforms": generated_waveforms, "cond": cond_input} # TODO: maybe refactor with  "cond": _get_cond_params_dict(cond_input)

def save_data(data: dict[str, np.ndarray], save_path: Path) -> None:
    """
    Save the given data to a directory.

    Args:
        data (dict[str, np.ndarray]): The data to save.
        save_path (Path): The directory path to save the data.

    Returns:
        None
    """
    with h5py.File(save_path, 'a') as f:
        for key, value in data.items():
            if key in f:
                dataset = f[key]
                dataset.resize((dataset.shape[0] + value.shape[0], *dataset.shape[1:]))
                dataset[-value.shape[0]:] = value
            else:
                dataset = f.create_dataset(key, data=value, maxshape=(None, *value.shape[1:]))


def get_samples(data: dict[str, np.ndarray], num_samples = None, indexes: list = None) -> dict[str, np.ndarray]:
    """
    Get a subset of samples from the given data.

    Args:
        data (dict[str, np.ndarray]): A dictionary containing arrays of samples.
        num_samples (int, optional): The number of samples to retrieve. Defaults to None.
        indexes (list, optional): The specific indexes of samples to retrieve. Defaults to None.

    Returns:
        dict[str, np.ndarray]: A dictionary containing the subset of samples.

    Raises:
        AssertionError: If neither num_samples nor indexes are provided.
    """
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
        if values is not None:
            cond_inputs.append(np.random.choice(values, size=batch_size))
        else:
            if param == "is_shallow_crustal":
                cond_inputs.append(np.random.randint(0, 2, (batch_size,)))
            else:
                cond_params_range = general_config.conditional_params_range
                cond_inputs.append(np.random.uniform(cond_params_range[param][0], cond_params_range[param][1], size=batch_size))
    return np.stack(cond_inputs, axis=1)


def get_cond_params_dict(cond_input: np.ndarray) -> dict[str, float]:
    return {key: cond_input[i] for i, key in enumerate(general_config.features_keys)}    


def plot_waveform_and_psd(data: dict[str, np.ndarray]) -> None:
    """
    Plots the waveform and power spectral density (PSD) of the given data.

    Args:
        data (dict[str, np.ndarray]): A dictionary containing the waveform and condition input data.

    Returns:
        None
    """
    fs = general_config.fs
    signal_length = general_config.signal_length
    num_channels = general_config.num_channels

    waveform = data['waveforms'][0]
    cond_input = data['cond'][0]

    # Plotting the three channels over time
    fig, axs = plt.subplots(waveform.shape[0], 2, figsize=(18, 12), constrained_layout=True)

    # TODO: Replace with actual channel names of channels
    time_ax = np.arange(0, signal_length) / fs
    freq_ax = np.fft.rfftfreq(signal_length, 1 / fs)
    for i in range(num_channels):
        axs[i, 0].plot(time_ax, waveform[i], label=f'Channel {i}')
        axs[i, 0].set_xlabel('Time (s)')
        axs[i, 0].set_ylabel('Amplitude') # TODO: insert correct unit of measurement
        axs[i, 0].legend()

        psd = np.abs(np.fft.rfft(waveform[i], axis=-1)) ** 2
        axs[i, 1].semilogx(freq_ax, 10 * np.log10(psd),  label=f'Channel {i}')
        axs[i, 1].set_xlabel('Frequency (Hz)')
        axs[i, 1].set_ylabel('Power Spectral Density (dB/Hz)')
        axs[i, 1].legend()
    
    fig.suptitle(f"Cond. params: {get_cond_params_dict(cond_input)}")
    #plt.tight_layout()
    plt.show()
        

def plot_waveforms(data: dict[str, np.ndarray], test_waveforms: np.ndarray = None, channel_index: int = 0, plot_envelope: bool = True, plot_log_envelope: bool = True):
    fs = general_config.fs
    signal_length = general_config.signal_length

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
        subfig.suptitle(f'Sample {row} - Cond. params: {get_cond_params_dict(cond_input[row])}')

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=m)
        if m == 1:
            axs = [axs]
        axs[0].plot(time_ax, waveforms[row, channel_index], label=f"Gen. Signal")
        if plot_envelope:
            env = _get_moving_avg_envelope(waveforms[row, channel_index].reshape(1, -1))[0]
            axs[0].plot(time_ax, env, label=f"Envelope")
            if test_waveforms is not None:
                test_env = _get_moving_avg_envelope(test_waveforms[:, channel_index])
                test_env_median = np.median(test_env, axis=0)
                test_env_p25 = np.percentile(test_env, 25, axis=0)
                test_env_p75 = np.percentile(test_env, 75, axis=0)
                axs[0].plot(time_ax, test_env_median, alpha=0.5, color='r', label=f"Real Distribution Env. - Median")
                axs[0].fill_between(time_ax, test_env_p25, test_env_p75, alpha=0.3, color='r', label=f'IQR (25-75%) - Num. Samples: {test_waveforms.shape[0]}')
        axs[0].set_xlabel('Time (s)')  
        axs[0].set_ylabel('Amplitude')  # TODO: insert correct unit of measurement  
        axs[0].legend()
        if plot_log_envelope:
            assert(plot_envelope), "If plot_log_envelope is True, plot_envelope must be True as well."
            axs[1].plot(time_ax, get_log_envelope(waveforms[row, channel_index].reshape(1, -1), env_function=_get_moving_avg_envelope)[0], label=f"Log Envelope")   
            if test_waveforms is not None:
                test_log_env = get_log_envelope(test_waveforms[:, channel_index], env_function=_get_moving_avg_envelope)
                test_log_env_median = np.median(test_log_env, axis=0)
                test_log_env_p25 = np.percentile(test_log_env, 25, axis=0)
                test_log_env_p75 = np.percentile(test_log_env, 75, axis=0)
                axs[1].plot(time_ax, test_log_env_median, alpha=0.5, color='r', label=f"Real Distribution Log. Env. - Median")
                axs[1].fill_between(time_ax, test_log_env_p25, test_log_env_p75, alpha=0.3, color='r', label=f'IQR (25-75%) - Num. Samples: {test_waveforms.shape[0]}')
            axs[1].set_xlabel('Time (s)')  
            axs[1].legend()

    plt.show()


def _get_moving_avg_envelope(x, window_len=100):
    return SignalWithEnvelope._moving_average_env(x, window_size=window_len)    

def get_log_envelope(data: np.ndarray, env_function=_get_moving_avg_envelope, env_function_params={}, eps=1e-7):
    """
    Get the log envelope of the given data.

    Args:
        data (np.ndarray): The input data.
        env_function (str): The envelope function to use.
        env_function_params (dict, optional): The parameters for the envelope function. Defaults to None.
        eps (float, optional): A small value to add to the envelope before taking the logarithm. Defaults to 1e-7.

    Returns:
        np.ndarray: The log envelope of the input data.
    """
    return np.log(env_function(data, **env_function_params) + eps)    


def get_power_spectral_density(data: np.ndarray, log_scale: bool = True, eps=1e-7):
    """
    Calculate the power spectral density of the input data.

    Parameters:
        data (np.ndarray): The input data.
        log_scale (bool, optional): Whether to return the result in log scale (decibel). Default is True.
        eps (float, optional): A small value added to avoid division by zero. Default is 1e-7.

    Returns:
        np.ndarray: The power spectral density of the input data.
    """
    if log_scale:
        return 10 * np.log(np.abs(np.fft.rfft(data, axis=-1)) ** 2 + eps) # dB
    return np.abs(np.fft.rfft(data, axis=-1)) ** 2

#####
    
          
def plot_by_bins(data: dict[str, np.ndarray], num_magnitude_bins:int, num_distance_bins: int, channel_index: int = 0, plot_type: str = 'waveform'):                  
    """
    Plot the data by dividing it into bins based on conditional parameters.

    Args:
        data (dict[str, np.ndarray]): A dictionary containing the waveform and condition input data.
        num_magnitude_bins (int): The number of magnitude bins.
        num_distance_bins (int): The number of distance bins.
        channel_index (int, optional): The index of the channel to plot. Defaults to 0.
        plot_type (str, optional): The type of plot. Can be 'waveform' or 'log_envelope'. Defaults to 'waveform'.

    Returns:
        None
    """
    #TODO: keep axis fixed for all plots (to that in all the plotting functions)
    fs = general_config.fs
    signal_length = general_config.signal_length
    conditional_params_range = general_config.conditional_params_range

    waveforms = data['waveforms'][:, channel_index, :] if channel_index is not None else data['waveforms']  
    cond_input = data['cond'] 

    min_mag, max_mag = conditional_params_range['magnitude']
    min_dist, max_dist = conditional_params_range['hypocentral_distance']
    
    count_mag_values = len(np.unique(cond_input[:, 2]))
    count_dist_values = len(np.unique(cond_input[:, 0]))
    if num_magnitude_bins > count_mag_values:
        num_magnitude_bins = count_mag_values
    if num_distance_bins > count_dist_values:
        num_distance_bins = count_dist_values

    plots = [[list() for _ in range(num_magnitude_bins)] for _ in range(num_distance_bins)] 
    for i, cond_sample in enumerate(cond_input):
        mag = cond_sample[2]
        dist = cond_sample[0]
        # TODO: modify the binning in other places as well (use round instead of int)
        mag_bin = round(
            (mag - min_mag) / (max_mag - min_mag) * (num_magnitude_bins-1)
        )
        dist_bin = round(
            (dist - min_dist) / (max_dist - min_dist) * (num_distance_bins-1)
        )
        if plot_type == 'waveform':
            to_plot = waveforms[i]
            x_axis = np.arange(0, signal_length) / fs
            xlabel = 'Time (s)'
            ylabel = 'Amplitude'
        elif plot_type == 'log_envelope':
            to_plot = get_log_envelope(waveforms[i].reshape(1, -1), env_function=_get_moving_avg_envelope)[0]
            x_axis = np.arange(0, signal_length) / fs
            xlabel = 'Time (s)'
            ylabel = 'Log Envelope'
        elif plot_type == 'power_spectral_density':
            to_plot = get_power_spectral_density(waveforms[i].reshape(1, -1), log_scale=True)[0]
            x_axis = np.fft.rfftfreq(signal_length, 1 / fs)
            xlabel = 'Frequency (Hz)'
            ylabel = 'Power Spectral Density (dB)'
        else:
            raise ValueError(f"Unknown plot type: {plot_type}. Available options are 'waveform', 'log_envelope' and 'power_spectral_density'.")    
        plots[dist_bin][mag_bin].append(to_plot)

    fig, axs = plt.subplots(num_distance_bins, num_magnitude_bins, figsize=(12, 5*num_distance_bins), constrained_layout=True) 
    axs = np.atleast_2d(axs)  # Adjust dimensions to ensure axs can be indexed with axs[i, j]
    for i in range(num_distance_bins):
        for j in range(num_magnitude_bins):
            plots_tensor = np.stack(plots[i][j]) if len(plots[i][j])>0 else np.zeros((1, signal_length))
            #mean_signal = np.mean(plots_tensor, axis=0) # TODO: or median?
            median_signal = np.median(plots_tensor, axis=0)
            #std_signal = np.std(plots_tensor, axis=0)
            p25 = np.percentile(plots_tensor, 25, axis=0)
            p75 = np.percentile(plots_tensor, 75, axis=0)
            #axs[i, j].plot(time_ax, mean_signal, label='mean signal')
            axs[i, j].plot(x_axis, median_signal, label='median signal')
            #axs[i, j].fill_between(time_ax, mean_signal - std_signal, mean_signal + std_signal, alpha=0.3, label='+- std dev')
            axs[i, j].fill_between(x_axis, p75, p25, alpha=0.5, label=f'IQR (25-75%) - Num. Samples: {len(plots[i][j])}')
            axs[i, j].set_title(f"Mag. Bin: [{min_mag + (max_mag - min_mag) / (num_magnitude_bins) * j:.1f}, {min_mag + (max_mag - min_mag) / (num_magnitude_bins) * (j+1):.1f}] - Dist Bin: [{min_dist + (max_dist - min_dist) / (num_distance_bins) * i:.1f}, {min_dist + (max_dist - min_dist) / (num_distance_bins) * (i+1):.1f}]")
            axs[i, j].set_xlabel(xlabel)
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
        dict[str, dict[str, np.ndarray]]: A dictionary containing the divided data (key: str((dist_bin, mag_bin))).

    """
    divided_data = {}
    cond_input = data['cond']
    data_waveforms = data['waveforms'] if 'waveforms' in data.keys() else data['repr']
    for i, dist_bin in enumerate(distance_bins):
        for j, mag_bin in enumerate(magnitude_bins):
            bins_indexes = (cond_input[:, 0] >= dist_bin[0]) & (cond_input[:, 0] < dist_bin[1]) & (cond_input[:, 2] >= mag_bin[0]) & (cond_input[:, 2] < mag_bin[1])
            if np.any(bins_indexes):
                divided_data[f"({dist_bin}, {mag_bin})"] = {"waveforms": data_waveforms[bins_indexes], "cond": cond_input[bins_indexes]}
    return divided_data


def plot_bins(plot_type: str, distance_bins: list[tuple], magnitude_bins: list[tuple], test_data: dict[str, np.ndarray], data: dict[str, np.ndarray] = None, model: Type[pl.LightningModule] = None, model_data_representation: Type[Representation] = None, channel_index: int = 0) -> None:
    assert (data is not None) or (model is not None and model_data_representation is not None), "Either data or model and model_data_representation must be provided."
    test_data_by_bins = divide_data_by_bins(test_data, magnitude_bins, distance_bins)

    if data is None:
        data = generate_data(model, model_data_representation, raw_output=False, num_samples=test_data["cond"].shape[0], cond_input=test_data["cond"])
    gen_data_by_bins = divide_data_by_bins(data, magnitude_bins, distance_bins)

    signal_length = general_config.signal_length
    fs = general_config.fs

    if plot_type == 'log_envelope':
        x_axis = np.arange(0, signal_length) / fs
        plot_fun = lambda x: get_log_envelope(x, env_function=_get_moving_avg_envelope)
        x_label = 'Time (s)'
        y_label = 'Log Envelope'
        y_limit = [-10, 4]          
    elif plot_type == 'power_spectral_density':
        x_axis = np.fft.rfftfreq(signal_length, 1 / fs)
        plot_fun = lambda x: get_power_spectral_density(x, log_scale=True)
        x_label = 'Frequency (Hz)'
        y_label = 'Power Spectral Density (dB)'
        y_limit = [-150, 150]
    else:
        raise ValueError(f"Unknown plot type: {plot_type}. Available options are 'log_envelope' and 'power_spectral_density'.")    

    
    fig, axs = plt.subplots(len(distance_bins), 2, figsize=(14, 5*len(distance_bins)))
    for i, dist_bin in enumerate(distance_bins):
        for j, mag_bin in enumerate(magnitude_bins):
            if f"({dist_bin}, {mag_bin})" in gen_data_by_bins.keys():
                gen_data_bin = plot_fun(gen_data_by_bins[f"({dist_bin}, {mag_bin})"]['waveforms'][:, channel_index, :])
                gen_data_median = np.median(gen_data_bin, axis=0)
                gen_data_p25, gen_data_p75 = np.percentile(gen_data_bin, 25, axis=0), np.percentile(gen_data_bin, 75, axis=0)
                axs[i, 0].plot(x_axis, gen_data_median, label=f"Mag: {mag_bin}, Dist: {dist_bin} - {len(gen_data_bin)} samples")
                axs[i, 0].fill_between(x_axis, gen_data_p25, gen_data_p75, alpha=0.5)

                test_data_bin = plot_fun(test_data_by_bins[f"({dist_bin}, {mag_bin})"]['waveforms'][:, channel_index, :])
                test_data_median = np.median(test_data_bin, axis=0)
                test_data_p25, test_data_p75 = np.percentile(test_data_bin, 25, axis=0), np.percentile(test_data_bin, 75, axis=0)
                axs[i, 1].plot(x_axis, test_data_median, label=f"Mag: {mag_bin}, Dist: {dist_bin} - {len(test_data_bin)} samples")
                axs[i, 1].fill_between(x_axis, test_data_p25, test_data_p75, alpha=0.5)
                
                axs[i, 0].set_title('Generated')
                axs[i, 1].set_title('Real')
                axs[i, 0].set_xlabel(x_label)
                axs[i, 0].set_ylabel(y_label) 
                axs[i, 0].set_ylim(y_limit)
                axs[i, 0].legend()
                axs[i, 0].grid()
                axs[i, 1].set_xlabel(x_label)
                axs[i, 1].set_ylabel(y_label) 
                axs[i, 1].set_ylim(y_limit)
                axs[i, 1].legend()
                axs[i, 1].grid()
            else:
                axs[i, 0].plot(x_axis, np.zeros_like(x_axis), alpha=0, label=f"Mag. Bin: {mag_bin} - Dist Bin: {dist_bin} -- No Data")
                axs[i, 1].plot(x_axis, np.zeros_like(x_axis), alpha=0, label=f"Mag. Bin: {mag_bin} - Dist Bin: {dist_bin} -- No Data")    
     
    plt.show()



def plot_log_envelope_bins(distance_bins: list[tuple], magnitude_bins: list[tuple], test_data: dict[str, np.ndarray], data: dict[str, np.ndarray] = None, model: Type[pl.LightningModule] = None, model_data_representation: Type[Representation] = None, channel_index: int = 0) -> None:
    assert (data is not None) or (model is not None and model_data_representation is not None), "Either data or model and model_data_representation must be provided."
    test_data_by_bins = divide_data_by_bins(test_data, magnitude_bins, distance_bins)

    if data is None:
        data = generate_data(model, model_data_representation, raw_output=False,  num_samples=len(test_data['cond']), cond_input=test_data['cond'])
    gen_data_by_bins = divide_data_by_bins(data, magnitude_bins, distance_bins)

    signal_length = general_config.signal_length
    fs = general_config.fs


    time_ax = np.arange(0, signal_length) / fs
    fig, axs = plt.subplots(len(distance_bins), 2, figsize=(14, 5*len(distance_bins)))
    for i, dist_bin in enumerate(distance_bins):
        for j, mag_bin in enumerate(magnitude_bins):
            if f"({dist_bin}, {mag_bin})" in gen_data_by_bins.keys():
                gen_data_bin = get_log_envelope(gen_data_by_bins[f"({dist_bin}, {mag_bin})"]['waveforms'][:, channel_index, :], env_function=_get_moving_avg_envelope)
                gen_data_median = np.median(gen_data_bin, axis=0)
                gen_data_p25, gen_data_p75 = np.percentile(gen_data_bin, 25, axis=0), np.percentile(gen_data_bin, 75, axis=0)
                axs[i, 0].plot(time_ax, gen_data_median, label=f"Mag. Bin: {mag_bin} - Dist Bin: {dist_bin}")
                axs[i, 0].fill_between(time_ax, gen_data_p75, gen_data_p25, alpha=0.5, label=f'IQR (25-75%) - n. samples: {len(gen_data_bin)}')

                test_data_bin = get_log_envelope(test_data_by_bins[f"({dist_bin}, {mag_bin})"]['waveforms'][:, channel_index, :], env_function=_get_moving_avg_envelope)
                test_data_median = np.median(test_data_bin, axis=0)
                test_data_p25, test_data_p75 = np.percentile(test_data_bin, 25, axis=0), np.percentile(test_data_bin, 75, axis=0)
                axs[i, 1].plot(time_ax, test_data_median, label=f"Mag. Bin: {mag_bin} - Dist Bin: {dist_bin}")
                axs[i, 1].fill_between(time_ax, test_data_p75, test_data_p25, alpha=0.5, label=f'IQR (25-75%) - n. samples: {len(test_data_bin)}')
                
                axs[i, 0].set_title('Generated')
                axs[i, 1].set_title('Real')
                axs[i, 0].set_xlabel('Time (s)')
                axs[i, 0].set_ylabel('Log Envelope') 
                axs[i, 0].legend()
                axs[i, 1].set_xlabel('Time (s)')
                axs[i, 1].set_ylabel('Log Envelope') 
                axs[i, 1].legend()
            else:
                axs[i, 0].plot(time_ax, np.zeros_like(time_ax), alpha=0, label=f"Mag. Bin: {mag_bin} - Dist Bin: {dist_bin} -- No Data")
                axs[i, 1].plot(time_ax, np.zeros_like(time_ax), alpha=0, label=f"Mag. Bin: {mag_bin} - Dist Bin: {dist_bin} -- No Data")    
     
    plt.show()


def plot_raw_waveform(raw_waveform, cond, data_representation, inverted_waveform=None):
    data_representation.plot(raw_waveform, inverted_waveform=inverted_waveform, title=str(get_cond_params_dict(cond)))    

def plot_raw_output_distribution(pred_raw_waveforms, test_raw_waveforms, model_data_repr):
    model_data_repr.plot_distribution(pred_raw_waveforms, test_raw_waveforms)

