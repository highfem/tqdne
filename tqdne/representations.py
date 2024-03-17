
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import numpy as np

from scipy.signal import hilbert
import torch

from tqdne.conf import Config

# class NumpyArgMixin:
#     """Mixin for automatic conversion of method arguments to numpy arrays."""

#     def __getattribute__(self, name):
#         """Return a function wrapper that converts method arguments to numpy arrays."""
#         attr = super().__getattribute__(name)
#         if not callable(attr):
#             return attr

#         def wrapper(*args, **kwargs):
#             def to_numpy(x):
#                 if isinstance(x, Sequence):
#                     return x.__class__(to_numpy(v) for v in x)
#                 elif isinstance(x, Mapping):
#                     return x.__class__((k, to_numpy(v)) for k, v in x.items())
#                 else:
#                     return x.numpy(force=True) if isinstance(x, torch.Tensor) else x
#             args = to_numpy(args)
#             kwargs = to_numpy(kwargs)
#             return attr(*args, **kwargs)

#         return wrapper

@staticmethod
def to_numpy(x):
    if isinstance(x, Sequence):
        return x.__class__(to_numpy(v) for v in x)
    elif isinstance(x, Mapping):
        return x.__class__((k, to_numpy(v)) for k, v in x.items())
    else:
        return x.numpy(force=True) if isinstance(x, torch.Tensor) else x



class Representation(ABC):

    def __init__(self, config=Config()):
        self.config = config

    def get_representation(self, signal):
        return self._get_representation(np.nan_to_num(to_numpy(signal), nan=0))

    @abstractmethod
    def _get_representation(self, signal):
        pass

    def invert_representation(self, representation):
        return self._invert_representation(np.nan_to_num(to_numpy(representation), nan=0))

    @abstractmethod
    def _invert_representation(self, representation):
        pass

    @abstractmethod
    def _get_shape(self, signal_input_shape):
        pass

    def get_shape(self, signal_input_shape):
        return self._get_shape(signal_input_shape)
    
    def update_stats(self, config: Config):
        return self._update_stats(config)

    @abstractmethod
    def _update_stats(self, config: Config):
        pass



class SignalWithEnvelope(Representation):

    def __init__(self, repr_params, config: Config):
        self.env_function = self._get_env_function_by_name(repr_params.env_function)
        self.env_function_params = repr_params.env_function_params

        self.trans_function, self.inv_trans_function = self._get_trans_function_by_name(repr_params.env_transform)
        self.trans_function_params = repr_params.env_transform_params

        self.scaling = repr_params.scaling

        if self.scaling.type == "normalize":
            if self.scaling.scalar:
                # Signal statistics for normalization
                # Assuming that maximum of the envelope coincides with the maximum of the signal and that the minimum of the envelope is 0.
                dataset_stats_dict = config.signal_statistics 
                max_env_peak_per_channel = np.array([dataset_stats_dict[ch]['max'] for ch in dataset_stats_dict.keys()])[:, np.newaxis]
                min_env_peak_per_channel = np.array([0 for _ in max_env_peak_per_channel])[:, np.newaxis]
                max_trans_env_peak_per_channel = self.trans_function(max_env_peak_per_channel, **self.trans_function_params)
                min_trans_env_peak_per_channel = self.trans_function(min_env_peak_per_channel, **self.trans_function_params)
                #Normalize the transformed envelope to the range [-1, 1]
                self.scaling_function = lambda trans_env: 2 * (trans_env - min_trans_env_peak_per_channel) / (max_trans_env_peak_per_channel - min_trans_env_peak_per_channel) - 1
                self.inv_scaling_function = lambda norm_trans_env: (norm_trans_env + 1) * (max_trans_env_peak_per_channel - min_trans_env_peak_per_channel) / 2 + min_trans_env_peak_per_channel
            else:
                raise NotImplementedError("Channel-wise normalization is not implemented yet -- missing statistics")

        elif self.scaling_type == "standardize":
            if self.scaling.scalar:
                raise NotImplementedError("Scalar standardization is not implemented yet -- missing statistics")
            else:
                # Statistics of the transformed envelope (computer over a subset of the training dataset)
                env_stats_per_channel = [dataset_stats_dict[ch] for ch in dataset_stats_dict.keys()]
                trans_env_mean_per_channel = np.array([env_stats['mean'] for env_stats in env_stats_per_channel]) # shape: (num_channels, signal_length)
                trans_env_std_per_channel = np.array([env_stats['std_dev'] for env_stats in env_stats_per_channel]) # shape: (num_channels, signal_length)
                # Standardize the transformed envelope 
                self.scaling_function = lambda trans_env: (trans_env - self.trans_env_mean[:, : trans_env.shape[-1]]) / self.trans_env_std[:, : trans_env.shape[-1]] 
                self.inv_scaling_function = lambda std_trans_env: std_trans_env * self.trans_env_std[:, : std_trans_env.shape[-1]] + self.trans_env_mean[:, : std_trans_env.shape[-1]]


    def _get_trans_function_by_name(self, name):
        if name == "log":
            return self._log_transform, self._inverse_log_transform
        else:
            raise ValueError(f"Unknown transformation function: {name}")
        
    def _get_env_function_by_name(self, name):
        if name == "hilbert":
            return self._hilbert_env
        elif name == "moving_average":
            return self._moving_average_env
        else:
            raise ValueError(f"Unknown envelope function: {name}")
    
    def _get_representation(self, signal):
        envelope = self.env_function(signal)
        scaled_signal = np.divide(signal, envelope, out=np.zeros_like(signal), where=envelope!=0) # when envelope is 0, the signal is also 0. Hence, the scaled signal should also be 0.
        
        trans_envelope = self.trans_function(envelope, **self.trans_function_params) 
        scaled_envelope = self.scaling_function(trans_envelope)

        return np.concatenate([scaled_envelope, scaled_signal], axis=0) # The model will learn to associated channels of the envelope with the corresponding channels of the signal

    
    
    def _invert_representation(self, representation):
        num_channels = representation.shape[1] // 2
        scaled_signal = representation[:, num_channels:, :]
        scaled_trans_envelope = representation[:, :num_channels, :]

        trans_envelope = self.inv_scaling_function(scaled_trans_envelope)
        
        signal = scaled_signal * self.inv_trans_function(trans_envelope, **self.trans_function_params)
        return signal
    
    def _get_shape(self, signal_input_shape):
        return (signal_input_shape[0], 2 * signal_input_shape[1], signal_input_shape[2])

    def _update_stats(self, config: Config):
        # TODO
        # Compute the statistics of the transformed envelope (mean, std, min, max) over a subset of the training dataset
        # Save the statistics in a pickle file
        pass    
    
    # ENVELOPES
    @staticmethod
    def _hilbert_env(signal):
        return np.abs(hilbert(signal))
    
    @staticmethod
    def _moving_average_env(signal, window_size=100):
        return np.convolve(np.abs(signal), np.ones(window_size)/window_size, mode='same')
    
    # TRANSFORMATION FUNCTIONS
    @staticmethod
    def _log_transform(x, log_offset):
        return np.log10(x + log_offset)
    
    @staticmethod
    def _inverse_log_transform(x, log_offset):
        return 10 ** x - log_offset
    

class Upsample(Representation):
    def __init__(self, config=Config()):
        super().__init__(config)

    def _get_representation(self, signal):
        return {"high_res": torch.tensor(signal[0], dtype=torch.float32), "low_res": torch.tensor(signal[1], dtype=torch.float32)} #wrong
        
    def _invert_representation(self, representation):
        return representation["high_res"], representation["low_res"] 