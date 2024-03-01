
from abc import ABC, abstractmethod

import numpy as np

from scipy.signal import hilbert
import torch

from tqdne.conf import Config


class Representation(ABC):

    def __init__(self):
        pass

    def get_representation(self, signal):
        return self._get_representation(np.nan_to_num(self._to_numpy(signal), nan=0)) # TODO: fix. maybe put nan_to_num when loading the data (?)

    @abstractmethod
    def _get_representation(self, signal):
        pass

    def invert_representation(self, representation):
        return self._invert_representation(np.nan_to_num(self._to_numpy(representation), nan=0)) #TODO: np.nan_to_num should not be needed 

    @abstractmethod
    def _invert_representation(self, representation):
        pass

    @abstractmethod
    def _get_input_shape(self, signal_input_shape):
        pass

    def get_input_shape(self, signal_input_shape):
        return self._get_input_shape(signal_input_shape)
    
    def _to_numpy(self, x):
        return x.numpy(force=True) if isinstance(x, torch.Tensor) else x
    
    def update_stats(self, config: Config):
        return self._update_stats(config)

    @abstractmethod
    def _update_stats(self, config: Config):
        pass



class SignalWithEnvelope(Representation):

    def __init__(self, repr_params, dataset_stats_dict: dict):

        # Signal statistics for normalization
        # Assuming that maximum of the envelope coincides with the maximum of the signal and that the minimum of the envelope is 0. 
        self.max_env_peak_per_channel = [self.config.signal_statistics[ch]['max'] for ch in self.config.signal_statistics.keys()]
        self.min_env_peak_per_channel = [0 for _ in self.max_env_peak_per_channel]
        

        self.env_function = self._get_env_function_by_name(repr_params["env_function"])
        self.env_function_params = repr_params["env_function_params"]

        self.trans_function, self.inv_trans_function = self._get_trans_function_by_name(repr_params["env_function"])
        self.trans_function_params = repr_params["env_function_params"]

        self.max_trans_env, self.min_trans_env, self.mean_trans_env, self.std_trans_env = self._get_trams_env_stats()


        # Statistics of the transformed envelope (computer over a subset of the training dataset)
        # env_stats_per_channel = [self.config.transformed_env_statistics[f'ch{i+1}'] for i in range(3)]
        # self.trans_env_mean = np.array([env_stats_per_channel[i]['mean'] for i in range(3)]) # shape: (num_channels, signal_length)
        # self.trans_env_std = np.array([env_stats_per_channel[i]['std_dev'] for i in range(3)]) # shape: (num_channels, signal_length)

        # Transform function used 
        # self.env_transform_function = self.config.transformed_env_statistics["trans_function"]
        # self.inverse_env_transform_function = self.config.transformed_env_statistics["inv_trans_function"]
        #self.log_offset = self.config.transformed_env_statistics["trans_log_function_offset"] # TODO: this should be FLAGS.log_env_trans_offset
        #self.env_window_size = FLAGS.env_window_size
        # TODO:
        # self.trans_function = _get_trans_function_by_name(self.FLAG.trans_function)
        # self.

        # TODO: compute just the min and max of the transformed envelope of the signal across all the timestep and samples (i.e, one per channel)
        # Do only the normalization with no scaling. Maybe also compute the std and mean to see if they're close to 1 and 0

    def _get_trans_functions_by_name(self, name):
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

        # Normalize NOT NEEDED BECAUSE ALREADY SCALED BY THE ENVELOPE
        #signal_mean = self.config.signal_mean
        #signal_std = self.config.signal_std
        #scaled_signal = (scaled_signal - signal_mean) / signal_std
        # ....

        # Normalize the transformed envelope to the range [-1, 1]
        trans_envelope = self.trans_function(envelope, **self.trans_function_params)
        norm_trans_envelope = 2 * (trans_envelope - self.min_env_peak_per_channel) / (self.max_env_peak_per_channel - self.min_env_peak_per_channel) - 1
        #scaled_envelope = (trans_envelope - self.trans_env_mean[:, : envelope.shape[-1]]) / self.trans_env_std[:, : envelope.shape[-1]] # shape: (num_channels, signal_length) (?)

        return np.concatenate([norm_trans_envelope, scaled_signal], axis=0) # The model will learn to associated channels of the envelope with the corresponding channels of the signal
    
    def _invert_representation(self, representation):
        num_channels = representation.shape[1] // 2
        trans_scaled_envelope = representation[:, :num_channels, :]
        scaled_signal = representation[:, num_channels:, :]
        
        trans_envelope = trans_scaled_envelope * self.trans_env_std[:, : trans_scaled_envelope.shape[-1]] + self.trans_env_mean[:, : trans_scaled_envelope.shape[-1]]
        signal = scaled_signal * self.inv_trans_function(trans_envelope, **self.trans_function_params)

        return signal  
    
    def _get_input_shape(self, signal_input_shape):
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