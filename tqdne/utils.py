import logging
from pathlib import Path
from typing import Type
from pathlib import Path

import h5py
import PIL
import os
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt

from tqdne.conf import Config

from tqdne.consistency_model import LightningConsistencyModel
from tqdne.diffusion import LightningDiffusion
from tqdne.classification import LightningClassifier
from tqdne.representations import *
from tqdne.metric import PowerSpectralDensity, LogEnvelope
from diffusers import DDPMScheduler, DDIMScheduler
import datetime

general_config = Config()

def get_data_representation(configs, signal_length=None):
    """
    Get the data representation based on the given configurations.

    Args:
        configs (object): The configurations object.
        signal_length (int, optional): The length of the signal. Defaults to None.

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
        return LogSpectrogram(**repr_params, output_signal_length=signal_length)
    elif repr_name == "Signal":
        configs.model.net_params.dims = 1
        return Signal(**repr_params)
    else:
        raise ValueError(f"Unknown representation name: {repr_name}")


def load_model(model_ckpt_path: Path, use_ddim: bool = True, print_info = True, device='cpu', signal_length=None,  **kwargs):
    """
    Loads a model from a given checkpoint directory path.

    Args:
        model_ckpt_path (Path): The path of the directory containing the model checkpoints or a checkpoint file.
        use_ddim (bool, optional): Whether to use the DDIM scheduler. Defaults to True.
        print_info (bool, optional): Whether to print the model information. Defaults to True.
        device (str, optional): The device to use for computation. Defaults to 'cpu'.
        signal_length (int, optional): The length of the signal. Defaults to None.
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
        ckpt = Path(model_ckpt_path)
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
        print_info = False
    else:
        raise ValueError(f"Unknown model name: {model_dir_name}")

    data_repr = get_data_representation(ml_configs, signal_length)
    model = model.to(device)
    model.eval()
    if print_info:
        print_model_info(model, data_repr, ckpt)
    return model, data_repr, ckpt

def plot_envelope(
        signal: np.ndarray,
        envelope_function: callable, 
        title: str = None, 
        **envelope_params: dict
    ) -> np.ndarray:
    """
    Plots the signal and its envelope.

    Args:
        signal (np.ndarray): The input signal.
        envelope_function (callable): The function used to calculate the envelope of the signal.
        title (str, optional): The title of the plot. Defaults to None.
        **envelope_params (dict): Additional parameters to be passed to the envelope function.

    Returns:
        np.ndarray: The envelope of the signal.
    """
    
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
    """
    Prints information about the model, model data representation, and checkpoint.

    Args:
        model (torch.nn.Module): The model to print information about.
        model_data_repr (ModelDataRepresentation): The model data representation object.
        ckpt (Path): The path to the checkpoint file.

    Returns:
        None
    """
    downsampling_factor = int(str(ckpt).split("downsampling:")[-1][0])
    model_data_repr.plot_representation(downsampling_factor)
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):}")
    print(f"Model size: {ckpt.stat().st_size / 1e6:.2f} MB")
    print(f"UNet scheme: \n base num. channels: {model.hparams.ml_config.model.net_params.model_channels} \n channel multipliers (down/up blocks): {model.hparams.ml_config.model.net_params.channel_mult} \n num. ResBlocks per down/up block: {model.hparams.ml_config.model.net_params.num_res_blocks} \n use Attention: {model.hparams.ml_config.model.net_params.num_heads is not None} \n conv. kernel size: {model.hparams.ml_config.model.net_params.conv_kernel_size} ")
    print(f"Diffusion prediction type: {model.hparams.ml_config.model.scheduler_params.prediction_type}")
    print(f"Learning rate schedule: \n start: {model.hparams.ml_config.optimizer_params.learning_rate} \n scheduler: {model.hparams.ml_config.optimizer_params.scheduler_name} \n warmup steps: {model.hparams.ml_config.optimizer_params.lr_warmup_steps}")
    print(f"Batch size: {model.hparams.ml_config.optimizer_params.batch_size}")
    if downsampling_factor > 1:
        print(f"Downsampling factor: {downsampling_factor}. The model was trained on signals with length {general_config.signal_length}, as the sampling rate used was {general_config.fs } instead of {general_config.original_fs}" )
    else: 
        print(f"The model was trained on signals with length {general_config.signal_length}, as the sampling rate used was {general_config.fs}, which is the original sampling rate." )
    data_repr_shape = model_data_repr.get_shape((1, general_config.num_channels, general_config.signal_length))
    data_repr_dims_names = "(batch_size, channels, signal_length)" if len(data_repr_shape) == 3 else "(batch_size, channels, freq_bins, time_frames)"
    print(f"Data representation shape: {data_repr_shape} - {data_repr_dims_names}")
    print(f"Data representation name: {model_data_repr.__class__.__name__}")
    if hasattr(model.hparams.ml_config.data_repr, "env_function"):
        print(f"Data representation envelope function: {model.hparams.ml_config.data_repr.params.env_function}") 
    print(f"Checkpoint: {ckpt} - Last modified: {datetime.datetime.fromtimestamp(ckpt.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}")


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


def adjust_signal_length(original_signal_length: int, unet: pl.LightningModule, data_repr: Type[Representation], downsample_factor: int = 1) -> int:
    def closest_divisible_number(n, div):
        return round(n / div) * div

    unet_max_divisor = 2 * len(unet.channel_mult) # count the number of down/up blocks in the UNet, which is the number of times the signal is downsampled (i.e., divided by 2)
    data_repr_divisor = data_repr.get_length_divisor(unet_max_divisor) * downsample_factor
    adjusted_signal_length = closest_divisible_number(original_signal_length, data_repr_divisor)
    new_signal_length = int(adjusted_signal_length / downsample_factor)
    data_repr.adjust_length_params(unet_max_divisor=unet_max_divisor, new_signal_length=new_signal_length)
    return adjusted_signal_length


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
    data_repr_shape = model_data_repr.get_shape((1, general_config.num_channels, general_config.signal_length))
    data_repr_dims_names = "(batch_size, channels, signal_length)" if len(data_repr_shape) == 3 else "(batch_size, channels, freq_bins, time_frames)"
    print(f"Data representation shape: {data_repr_shape} - {data_repr_dims_names}")
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
    """
    Get the path of the last checkpoint file in the specified directory.

    Args:
        dirpath (str): The directory path where the checkpoint files are located.

    Returns:
        str or None: The path of the last checkpoint file, or None if no checkpoint is found.
    """
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


def generate_data(
        model: Type[pl.LightningModule],
        model_data_representation: Type[Representation],
        raw_output: bool = False, 
        num_samples: int = None,
        cond_input_params: dict[str, list] = None, 
        cond_input: np.ndarray = None,
        device: str = 'cuda', 
        batch_size: int = None, 
        save_path: Path = None, 
        restart : bool = False
    ) -> dict[str, np.ndarray]:
    """
    Generates synthetic data using a given model and data representation.

    Args:
        model (Type[pl.LightningModule]): The model used to generate the data.
        model_data_representation (Type[Representation]): The data representation used by the model.
        raw_output (bool, optional): Whether to return the raw output of the model. Defaults to False.
        num_samples (int): The number of samples to generate. Defaults to None (all samples in the conditional input).
        cond_input_params (dict[str, list], optional): The parameters for generating conditional inputs. Defaults to None.
        cond_input (np.ndarray, optional): The conditional inputs. Defaults to None.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.
        batch_size (int, optional): The batch size to use for generating the data. Defaults to None, in which case the batch size from the model configuration is used.
        save_path (Path, optional): The directory path to save the generated inverted data. Defaults to None.
        restart (bool, optional): Whether to restart generating the data from the last index in the save file. Defaults to False.

    Returns:
        dict[str, np.ndarray]: A dictionary containing the generated waveforms and the conditional inputs ("waveforms" and "cond").
    """
    assert not(cond_input_params is not None and cond_input is not None), "Either cond_input_params or cond_input must be provided."
    batch_size = model.hparams.ml_config.optimizer_params.batch_size if batch_size is None else batch_size
    if cond_input_params is None and cond_input is None:
        logging.warning("No conditional input provided. Using the range values for each parameter.")
        cond_input_params = {k: None for k,v in general_config.conditional_params_range.items()}
    signal_len = general_config.signal_length
    num_channels = general_config.num_channels
    if num_samples is None:
        if cond_input is not None:
            num_samples = cond_input.shape[0]
        else:
            num_samples = batch_size
            print("No number of samples provided. Using the batch size.")
    if cond_input is None:
        cond_input = generate_cond_inputs(num_samples, cond_input_params)
    else:
        cond_input = np.resize(cond_input, (num_samples, cond_input.shape[1]))
        if restart:
            if save_path is not None:
                with h5py.File(save_path, 'r') as f:
                    start_index = f['waveforms'].shape[0]
                    print(f"Restarting from index {start_index} of {num_samples}")
                    cond_input = cond_input[start_index:]
                    num_samples = cond_input.shape[0]     
    generated_waveforms = []
    with torch.no_grad():
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


def get_cond_params_str(cond_input: np.ndarray) -> str:
    """
    Converts the given conditional input array into a formatted string representation.

    Args:
        cond_input (np.ndarray): The input array containing the conditional parameters.

    Returns:
        str: The formatted string representation of the conditional parameters.

    Raises:
        ValueError: If the input array has more than one sample or a different length than the number of conditional parameters.
    """
    if cond_input.ndim > 1:
        if cond_input.shape[0] > 1:
            raise ValueError("The input array must have only one sample.")
        cond_input = cond_input[0]
    
    if len(cond_input) != len(general_config.features_keys):
        raise ValueError(f"The input array must have the same length as the number of conditional parameters. Expected {len(general_config.features_keys)} but got {len(cond_input)}.")
    
    dist_str = f"Hyp. dist: {cond_input[0]:.2f} Km"
    mag_str = f"Mag. ($M_w$): {cond_input[2]:.3f}"
    is_shallow_str = "Shallow crustal: " + ("Yes" if bool(cond_input[1]) else "No")
    vs30_str = f"Vs30: {cond_input[3]:.2f} m/s"
    return f"{dist_str}, {mag_str}, {is_shallow_str}, {vs30_str}"


def plot_waveform_and_psd(
        data: dict[str, np.ndarray], 
        save_path: Path = None 
    ) -> None:
    """
    Plots the waveform and power spectral density (PSD) of the given sample. Each channel is plotted separately in a row.

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
        axs[i, 0].set_xlabel('Time [s]')
        axs[i, 0].set_ylabel("$m/s^2$")
        axs[i, 0].legend()
        axs[i, 1].plot(freq_ax, PowerSpectralDensity.get_power_spectral_density(waveform[i], log_scale=True), label=f'Channel {i}')
        axs[i, 1].set_xlabel('Frequency [Hz]')
        axs[i, 1].set_ylabel('Power Spectral Density[dB]')
        axs[i, 1].legend()
    
    fig.suptitle(f"Cond. params: {get_cond_params_str(cond_input)}")
    #plt.tight_layout()
    plt.show()

    if save_path is not None:
        # Create the directory if it does not exist
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f"waveform_psd_mag{cond_input[2]}_dist{cond_input[0]}_vs30{cond_input[3]}_isShallowCrustal{bool(cond_input[1])}.png")

def plot_generated_against_real_waveform(
        real_waveform: np.ndarray,
        real_params: np.ndarray,
        model: Type[pl.LightningModule],
        model_data_representation: Type[Representation],
        num_samples: int = 5,
        channel_index: int = -1,
        device: str = 'cuda'
    ) -> None:
    """
    Plots the generated waveform against the real waveform.

    Args:
        real_waveform (np.ndarray): The real waveform.
        real_params (np.ndarray): The parameters of the real waveform.
        model (Type[pl.LightningModule]): The model used to generate the data.
        model_data_representation (Type[Representation]): The data representation used by the model.
        num_samples (int, optional): The number of samples to generate. Defaults to 5.
        channel_index (int, optional): The channel index to plot. Defaults to -1 (all channels).

    Returns:
        None
    """
    fs = general_config.fs
    signal_length = general_config.signal_length
    n_channels = general_config.num_channels
    
    gen_waveforms = generate_data(model, model_data_representation, num_samples=num_samples, cond_input=real_params, device=device)['waveforms']

    time_ax = np.arange(0, signal_length) / fs
    fig, axs = plt.subplots(num_samples+1, n_channels, figsize=(20, 3*num_samples), constrained_layout=True) if channel_index == -1 else plt.subplots(num_samples+1, 1, figsize=(20, 3*num_samples), constrained_layout=True)
    if channel_index != -1:
        axs = axs[:, np.newaxis]

    fig.suptitle(f"{get_cond_params_str(real_params[0])}")
    # fix ylim based on the extremal values of both real and generated waveforms
    y_range = (
        min(min(gen_waveforms.flatten()), min(real_waveform.flatten())),
        max(max(gen_waveforms.flatten()), max(real_waveform.flatten()))
    )
    
    channels = range(n_channels) if channel_index == -1 else [channel_index]
    for ch in channels:
        axs[0, ch].plot(time_ax, real_waveform[0, ch], label=f'Real Waveform - ch. {ch}', color='r')
        axs[0, 0].set_ylabel('Amplitude [$m/s^2$]')
        axs[0, ch].legend()
        axs[0, ch].set_ylim(y_range)
        for i in range(num_samples):
            axs[i+1, ch].plot(time_ax, gen_waveforms[i, ch], label=f'Generated Waveform - ch. {ch}', color='b')
            if ch == 0:
                axs[i+1, ch].set_ylabel('Amplitude [$m/s^2$]')
            axs[i+1, ch].legend()
            axs[i+1, ch].set_ylim(y_range)
        axs[i+1, ch].set_xlabel('Time [s]')

    plt.show()



def plot_generated_vs_real_waveforms(
        generated_data: dict[str, np.ndarray],
        real_data: dict[str, np.ndarray],
        num_samples: int = None,
        channel_index: int = 0,
        fs: int = general_config.fs,
    ) -> None:
    """
    Plots the generated waveforms against the real waveforms.

    Args:
        generated_data (dict[str, np.ndarray]): The generated data.
        real_data (dict[str, np.ndarray]): The real data.
        num_samples (int, optional): The number of samples to plot. Defaults to None (all samples).
        channel_index (int, optional): The channel index to plot (-1 to plot all channels in the same plot). Defaults to 0.
        fs (int, optional): The sampling frequency. Defaults to general_config.fs.

    Returns:
        None
    """
    num_channels = general_config.num_channels
    signal_length = general_config.signal_length
    time_ax = np.arange(0, signal_length) / fs

    if num_samples is None:
        num_samples = generated_data['waveforms'].shape[0]

    fig = plt.figure(figsize=(20, 3*num_samples), constrained_layout=True)
    subfigs = fig.subfigures(nrows=num_samples, ncols=1)
    for i, subfig in enumerate(subfigs):
        subfig.suptitle(f"Sample {i} - Cond. params: {get_cond_params_str(generated_data['cond'][i])}")
        axs = subfig.subplots(nrows=1, ncols=2)
        gen_wf = generated_data['waveforms'] if 'waveforms' in generated_data else generated_data['waveform']
        real_wf = real_data['waveforms'] if 'waveforms' in real_data else real_data['waveform']
        if channel_index == -1:
            for ch in range(num_channels):
                axs[0].plot(time_ax, gen_wf[i, ch], label=f'Gen. - Ch. {ch}', alpha=1 - ch/num_channels)
                axs[1].plot(time_ax, real_wf[i, ch], label=f'Real - Ch. {ch}', alpha=1 - ch/num_channels)
        else:
            axs[0].plot(time_ax, gen_wf[i, channel_index], label=f'Gen. - Ch. {channel_index}')
            axs[1].plot(time_ax, real_wf[i, channel_index], label=f'Real - Ch. {channel_index}')        
        
        axs[0].set_ylabel('Amplitude [$m/s^2$]')
        #axs[1].set_ylabel('Amplitude [$m/s^2$]')

        legend_loc = 'lower right'
        axs[0].legend(loc=legend_loc)
        axs[1].legend(loc=legend_loc)

    
        # Fix axis limits based on the extremal values of both real and generated waveforms
        if channel_index != -1:
            y_range = (
                min(min(gen_wf[i, channel_index]), min(real_wf[i, channel_index])),
                max(max(gen_wf[i, channel_index]), max(real_wf[i, channel_index]))
            )
            axs[0].set_ylim(y_range)
            axs[1].set_ylim(y_range)
        else:
            y_range = (
                min(min(gen_wf[i].flatten()), min(real_wf[i].flatten())),
                max(max(gen_wf[i].flatten()), max(real_wf[i].flatten()))
            )
            axs[0].set_ylim(y_range)
            axs[1].set_ylim(y_range)


    axs[0].set_xlabel('Time [s]')
    axs[1].set_xlabel('Time [s]')

    #plt.tight_layout()
    plt.show()


def plot_waveforms(data: dict[str, np.ndarray], test_waveforms: np.ndarray = None, channel_index: int = 0, plot_envelope: bool = True, plot_log_envelope: bool = True):
    """
    Plot generated waveforms and their envelopes (optional) against the real distribution of the test waveforms (if provided).

    Args:
        data (dict[str, np.ndarray]): Dictionary containing waveforms and condition input.
        test_waveforms (np.ndarray, optional): Test waveforms. Defaults to None.
        channel_index (int, optional): Channel index. Defaults to 0.
        plot_envelope (bool, optional): Whether to plot the envelope on the generated waveforms. Defaults to True.
        plot_log_envelope (bool, optional): Whether to plot the log envelope on a separate subplot. Defaults to True.
    """
    fs = general_config.fs
    signal_length = general_config.signal_length

    waveforms = data['waveforms']
    cond_input = data['cond']

    n = waveforms.shape[0]
    m = 1 if plot_log_envelope == False else 2
    time_ax = np.arange(0, signal_length) / fs

    fig = plt.figure(constrained_layout=True, figsize=(20, 5*n))
    fig.suptitle(f'Plot of {n} different signals - channel {channel_index}', fontsize=16)

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=n, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f'Sample {row} - Cond. params: {get_cond_params_str(cond_input[row])}')

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=m)
        if m == 1:
            axs = [axs]
        axs[0].plot(time_ax, waveforms[row, channel_index], label=f"Gen. Signal", zorder=0)
        if plot_envelope:
            env = _get_moving_avg_envelope(waveforms[row, channel_index].reshape(1, -1))[0]
            axs[0].plot(time_ax, env, label=f"Envelope")
            if test_waveforms is not None:
                test_env = _get_moving_avg_envelope(test_waveforms[:, channel_index])
                test_env_median = np.median(test_env, axis=0)
                test_env_p15 = np.percentile(test_env, 15, axis=0)
                test_env_p85 = np.percentile(test_env, 85, axis=0)
                axs[0].plot(time_ax, test_env_median, alpha=0.5, color='r', label=f"Real Distrib. Env. - Median")
                axs[0].fill_between(time_ax, test_env_p15, test_env_p85, alpha=0.3, color='r', label=f'IQR (15-85%) - N: {test_waveforms.shape[0]}')
        axs[0].set_xlabel('Time [$s$]')  
        axs[0].set_ylabel('Amplitude [$m/s^2$]')  
        axs[0].legend()
        if plot_log_envelope:
            assert(plot_envelope), "If plot_log_envelope is True, plot_envelope must be True as well."
            axs[1].plot(time_ax, LogEnvelope.get_log_envelope(waveforms[row, channel_index].reshape(1, -1), env_function=_get_moving_avg_envelope)[0], label=f"Log Env.")   
            if test_waveforms is not None:
                test_log_env = LogEnvelope.get_log_envelope(test_waveforms[:, channel_index], env_function=_get_moving_avg_envelope)
                test_log_env_median = np.median(test_log_env, axis=0)
                test_log_env_p15 = np.percentile(test_log_env, 15, axis=0)
                test_log_env_p85 = np.percentile(test_log_env, 85, axis=0)
                axs[1].plot(time_ax, test_log_env_median, alpha=0.5, color='r', label=f"Real Distrib. Log. Env. - Median")
                axs[1].fill_between(time_ax, test_log_env_p15, test_log_env_p85, alpha=0.3, color='r', label=f'IQR (15-85%) - N: {test_waveforms.shape[0]}')
            axs[1].set_xlabel('Time [$s$]')  
            axs[1].legend()

    plt.show()


def _get_moving_avg_envelope(x, window_len=100):
    """
    Calculate the moving average envelope of a signal.

    Args:
        x (array-like): The input signal.
        window_len (int, optional): The length of the moving average window. Defaults to 100.

    Returns:
        array-like: The moving average envelope of the input signal.
    """
    return SignalWithEnvelope._moving_average_env(x, window_size=window_len)
          
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
        mag_bin = round(
            (mag - min_mag) / (max_mag - min_mag) * (num_magnitude_bins-1)
        )
        dist_bin = round(
            (dist - min_dist) / (max_dist - min_dist) * (num_distance_bins-1)
        )
        if plot_type == 'waveform':
            to_plot = waveforms[i]
            x_axis = np.arange(0, signal_length) / fs
            xlabel = 'Time [s]'
            ylabel = 'Amplitude [$m/s^2$]'
            y_range = None
        elif plot_type == 'log_envelope':
            to_plot = LogEnvelope.get_log_envelope(waveforms[i].reshape(1, -1), env_function=_get_moving_avg_envelope)[0]
            x_axis = np.arange(0, signal_length) / fs
            xlabel = 'Time [s]'
            ylabel = 'Log Envelope'
            y_range = (-10, 5)
        elif plot_type == 'power_spectral_density':
            to_plot = PowerSpectralDensity.get_power_spectral_density(waveforms[i].reshape(1, -1), log_scale=True)[0]
            x_axis = np.fft.rfftfreq(signal_length, 1 / fs)
            xlabel = 'Frequency [Hz]'
            ylabel = 'Power Spectral Density [dB]'
            y_range = (-10, 10)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}. Available options are 'waveform', 'log_envelope' and 'power_spectral_density'.")    
        plots[dist_bin][mag_bin].append(to_plot)

    fig, axs = plt.subplots(num_distance_bins, num_magnitude_bins, figsize=(7*num_magnitude_bins, 5*num_distance_bins)) 
    axs = np.atleast_2d(axs)  # Adjust dimensions to ensure axs can be indexed with axs[i, j]
    for i in range(num_distance_bins):
        for j in range(num_magnitude_bins):
            if len(plots[i][j])>0: 
                plots_tensor = np.stack(plots[i][j])
                median_signal = np.median(plots_tensor, axis=0)
                p15 = np.percentile(plots_tensor, 15, axis=0)
                p85 = np.percentile(plots_tensor, 85, axis=0)
                axs[i, j].plot(x_axis, median_signal, label='Median')
                axs[i, j].fill_between(x_axis, p85, p15, alpha=0.5, label=f'IQR (15-85%) - N. Samples: {len(plots[i][j])}')
                axs[i, j].legend()
            axs[i, j].set_title(f"Mag.: [{min_mag + (max_mag - min_mag) / (num_magnitude_bins) * j:.1f}, {min_mag + (max_mag - min_mag) / (num_magnitude_bins) * (j+1):.1f}] - Dist.: [{min_dist + (max_dist - min_dist) / (num_distance_bins) * i:.1f}, {min_dist + (max_dist - min_dist) / (num_distance_bins) * (i+1):.1f}]")
            if y_range:
                axs[i, j].set_ylim(*y_range)
        axs[i, 0].set_ylabel(ylabel)
    for j in range(num_magnitude_bins):
        axs[-1, j].set_xlabel(xlabel)
    plt.tight_layout()
    plt.show()

def divide_data_by_bins(
        data: dict[str, np.ndarray], 
        magnitude_bins: list[tuple], 
        distance_bins: list[tuple],
    ) -> dict[(tuple, tuple), dict[str, np.ndarray]]:
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


def plot_bins(
        plot_type: str, 
        distance_bins: list[tuple], 
        magnitude_bins: list[tuple], 
        test_data: dict[str, np.ndarray], 
        data: dict[str, np.ndarray] = None, 
        model: Type[pl.LightningModule] = None, 
        model_data_representation: Type[Representation] = None, 
        channel_index: int = 0,
        save_path: Path = None
    ) -> None:
    """
    Plots the waveforms of generated and real data in different bins based on distance and magnitude.

    Args:
        plot_type (str): The type of plot to generate. Available options are 'log_envelope' and 'power_spectral_density'.
        distance_bins (list[tuple]): The bins for distance.
        magnitude_bins (list[tuple]): The bins for magnitude.
        test_data (dict[str, np.ndarray]): The test data to plot.
        data (dict[str, np.ndarray], optional): The generated data to plot. Defaults to None.
        model (Type[pl.LightningModule], optional): The model used to generate the data. Required if `data` is None. Defaults to None.
        model_data_representation (Type[Representation], optional): The data representation used by the model. Required if `data` is None. Defaults to None.
        channel_index (int, optional): The index of the channel to plot. Defaults to 0.
        save_path (Path, optional): The path to save the plot. Defaults to None.

    Raises:
        ValueError: If an unknown plot type is provided.

    Returns:
        None
    """
    assert (data is not None) or (model is not None and model_data_representation is not None), "Either data or model and model_data_representation must be provided."
    test_data_by_bins = divide_data_by_bins(test_data, magnitude_bins, distance_bins)

    if data is None:
        data = generate_data(model, model_data_representation, raw_output=False, num_samples=test_data["cond"].shape[0], cond_input=test_data["cond"])
    gen_data_by_bins = divide_data_by_bins(data, magnitude_bins, distance_bins)

    signal_length = general_config.signal_length
    fs = general_config.fs

    if plot_type == 'log_envelope':
        x_axis = np.arange(0, signal_length) / fs
        plot_fun = lambda x: LogEnvelope.get_log_envelope(x, env_function=_get_moving_avg_envelope)
        x_label = 'Time [s]'
        y_label = 'Log Envelope'
        y_limit = [-8, 1]       
        legend_loc = 'lower right'   
    elif plot_type == 'power_spectral_density':
        x_axis = np.fft.rfftfreq(signal_length, 1 / fs)
        plot_fun = lambda x: PowerSpectralDensity.get_power_spectral_density(x, log_scale=True)
        x_label = 'Frequency [Hz]'
        y_label = 'Power Spectral Density [dB]'
        y_limit = [-15, 10]
        legend_loc = 'best'
    else:
        raise ValueError(f"Unknown plot type: {plot_type}. Available options are 'log_envelope' and 'power_spectral_density'.")    
  
    fig, axs = plt.subplots(len(distance_bins), 2, figsize=(20, 5*len(distance_bins)))
    for i, dist_bin in enumerate(distance_bins):
        for j, mag_bin in enumerate(magnitude_bins):
            if f"({dist_bin}, {mag_bin})" in gen_data_by_bins.keys():
                gen_data_bin = plot_fun(gen_data_by_bins[f"({dist_bin}, {mag_bin})"]['waveforms'][:, channel_index, :])
                gen_data_median = np.median(gen_data_bin, axis=0)
                gen_data_p15, gen_data_p85 = np.percentile(gen_data_bin, 15, axis=0), np.percentile(gen_data_bin, 85, axis=0)
                axs[i, 0].plot(x_axis, gen_data_median, label=f"$M_w$: [{mag_bin[0]}, {mag_bin[1]}] - N={len(gen_data_bin)}")
                axs[i, 0].fill_between(x_axis, gen_data_p15, gen_data_p85, alpha=0.5)
                test_data_bin = plot_fun(test_data_by_bins[f"({dist_bin}, {mag_bin})"]['waveforms'][:, channel_index, :])
                test_data_median = np.median(test_data_bin, axis=0)
                test_data_p15, test_data_p85 = np.percentile(test_data_bin, 15, axis=0), np.percentile(test_data_bin, 85, axis=0)
                axs[i, 1].plot(x_axis, test_data_median, label=f"$M_w$: [{mag_bin[0]}, {mag_bin[1]}] - N={len(test_data_bin)}")
                axs[i, 1].fill_between(x_axis, test_data_p15, test_data_p85, alpha=0.5)
                #axs[i, 0].set_xlabel(x_label)
                axs[i, 0].set_ylabel(y_label) 
                axs[i, 0].set_ylim(y_limit)
                #axs[i, 1].set_xlabel(x_label)
                #axs[i, 1].set_ylabel(y_label)
                axs[i, 1].set_ylim(y_limit)
                axs[i, 0].legend(loc=legend_loc)
                axs[i, 1].legend(loc=legend_loc)
                axs[i, 0].grid(True)
                axs[i, 1].grid(True)
            else:
                axs[i, 0].plot(x_axis, np.zeros_like(x_axis), alpha=0, label=f"No Data")
                axs[i, 1].plot(x_axis, np.zeros_like(x_axis), alpha=0, label=f"No Data")   
    
    # set x-axis label only for the last row
    axs[-1, 0].set_xlabel(x_label) 
    axs[-1, 1].set_xlabel(x_label)   

    # legend
    # handles, labels = axs[-1, 0].get_legend_handles_labels()
    # labels = [f'$M_w$ [{mag_bin[0]}, {mag_bin[1]}]' for mag_bin in magnitude_bins]
    # fig.legend(
    #     handles,
    #     labels,
    #     loc="lower center",
    #     ncol=len(magnitude_bins),
    #     title="Magnitude bins",
    #     bbox_to_anchor=(0.5, -0.15 / (len(distance_bins))),
    # )

    # column titles
    for ax, title in zip(axs[0], ["Generated - IQR (15-85%)", "Real - IQR (15-85%)"]):    
        ax.annotate(
            title,
            xy=(0.5, 1.05),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="baseline",
            fontsize=20,
            xycoords="axes fraction",
        )

    # row titles
    for i, ax in enumerate(axs):
        ax[0].annotate(
            f"[{distance_bins[i][0]}, {distance_bins[i][1]}] Km",
            xy=(-0.15, 0.5),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            rotation=90,
            fontsize=20,
            xycoords="axes fraction",
        )

    plt.tight_layout()
    plt.show()

    if save_path is not None:
        # Create the directory if it does not exist
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f"{plot_type}_bins_ch{channel_index}.png")


def plot_raw_waveform(
        raw_waveform: np.ndarray,
        cond: np.ndarray, 
        data_representation: Type[Representation], 
        inverted_waveform: np.ndarray = None
    ) -> None:
    """
    Plot the raw waveform.

    Args:
        raw_waveform (np.ndarray): The raw waveform data.
        cond (np.ndarray): The condition data.
        data_representation (Type[Representation]): The data representation class.
        inverted_waveform (np.ndarray, optional): The inverted waveform data. Defaults to None (computed from the raw waveform through the data representation).
    """
    if inverted_waveform is None:
        inverted_waveform = data_representation.invert_representation(raw_waveform[None, ...])[0] 
    data_representation.plot(raw_waveform, inverted_waveform=inverted_waveform, title=str(get_cond_params_str(cond)))

def plot_raw_output_distribution(
        pred_raw_waveforms: np.ndarray, 
        test_raw_waveforms: np.ndarray, 
        model_data_repr: Type[Representation]
    ) -> None:
    """
    Plots the distribution of raw output waveforms.

    Args:
        pred_raw_waveforms (np.ndarray): The predicted raw waveforms.
        test_raw_waveforms (np.ndarray): The test raw waveforms.
        model_data_repr (Type[Representation]): The representation of the model data.

    Returns:
        None
    """
    model_data_repr.plot_distribution(pred_raw_waveforms, test_raw_waveforms)

