
from abc import ABC, abstractmethod

import numpy as np

from scipy.signal import hilbert
import torch

from tqdne.conf import Config


class Representation(ABC):

    def __init__(self, config=Config()):
        self.config = config

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
    
    def _to_numpy(x):
        return x.numpy(force=True) if isinstance(x, torch.Tensor) else x


class SignalWithEnvelope(Representation):

    def __init__(self, config=Config()):
        super().__init__(config)

        # Statistics of the transformed envelope (computer over a subset of the training dataset)
        env_stats_per_channel = [self.config.transformed_env_statistics[f'ch{i+1}'] for i in range(3)]
        self.trans_env_mean = np.array([env_stats_per_channel[i]['mean'] for i in range(3)]) # shape: (num_channels, signal_length)
        self.trans_env_std = np.array([env_stats_per_channel[i]['std_dev'] for i in range(3)]) # shape: (num_channels, signal_length)

        # Transform fucntion used 
        # self.env_transform_function = self.config.transformed_env_statistics["trans_function"]
        # self.inverse_env_transform_function = self.config.transformed_env_statistics["inv_trans_function"]
        self.log_offset = self.config.transformed_env_statistics["trans_log_function_offset"]
 
    
    def _get_representation(self, signal):

        def compute_envelope(signal):
            return np.abs(hilbert(signal))

        def log_transform(x, offset=1e-5):
            return np.log10(x + offset)
    
        envelope = compute_envelope(signal)

        scaled_signal = np.divide(signal, envelope, out=np.zeros_like(signal), where=envelope!=0) # when envelope is 0, the signal is also 0. Hence, the scaled signal should also be 0.

        # Normalize NOT NEEDED BECAUSE ALREADY SCALED BY THE ENVELOPE
        #signal_mean = self.config.signal_mean
        #signal_std = self.config.signal_std
        #scaled_signal = (scaled_signal - signal_mean) / signal_std
        # ....

        # Standardize the transformed envelope to get mean=0 and std=1
        scaled_envelope = (log_transform(envelope, offset=self.log_offset) - self.trans_env_mean[:, : envelope.shape[-1]]) / self.trans_env_std[:, : envelope.shape[-1]] # shape: (num_channels, signal_length) (?)

        return np.concatenate([scaled_envelope, scaled_signal], axis=0) # The model will learn to associated channels of the envelope with the corresponding channels of the signal
    
    def _invert_representation(self, representation):
        num_channels = representation.shape[1] // 2
        trans_scaled_envelope = representation[:, :num_channels, :]
        scaled_signal = representation[:, num_channels:, :]

        def inverse_log_transform(x, offset=1e-5):
            return 10 ** x - offset
        
        trans_envelope = trans_scaled_envelope * self.trans_env_std[:, : trans_scaled_envelope.shape[-1]] + self.trans_env_mean[:, : trans_scaled_envelope.shape[-1]]
        signal = scaled_signal * inverse_log_transform(trans_envelope, offset=self.log_offset)

        return signal  
    
    def _get_input_shape(self, signal_input_shape):
        return (signal_input_shape[0], 2 * signal_input_shape[1], signal_input_shape[2])
    

class Upsample(Representation):
    def __init__(self, config=Config()):
        super().__init__(config)

    def _get_representation(self, signal):
        return {"high_res": torch.tensor(signal[0], dtype=torch.float32), "low_res": torch.tensor(signal[1], dtype=torch.float32)} #wrong
        
    def _invert_representation(self, representation):
        return representation["high_res"], representation["low_res"] 