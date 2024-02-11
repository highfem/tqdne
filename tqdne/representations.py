
from abc import ABC, abstractmethod

import numpy as np

from scipy.signal import hilbert

from tqdne.conf import Config
from tqdne.utils import to_numpy


class Representation(ABC):

    def __init__(self, config=Config()):
        self.config = config

    def get_representation(self, signal):
        return self._get_representation(to_numpy(signal))

    @abstractmethod
    def _get_representation(self, signal):
        pass

    def invert_representation(self, representation):
        return self._invert_representation(to_numpy(representation))

    @abstractmethod
    def _invert_representation(self, representation):
        pass


class SignalWithEnvelope(Representation):

    def __init__(self, config=Config()):
        super().__init__(config)

        # Statistics of the transformed envelope (computer over a subset of the training dataset)
        env_stats_per_channel = [self.config.transformed_env_statistics[f'ch{i+1}'] for i in range(3)]
        self.trans_env_mean = np.array([env_stats_per_channel[i]['mean'] for i in range(3)]) # shape: (num_channels, signal_length)
        self.trans_env_std = np.array([env_stats_per_channel[i]['std'] for i in range(3)]) # shape: (num_channels, signal_length)

        # Transform fucntion used 
        self.env_transform_function = self.config.transformed_env_statistics["trans_function"]
        self.inverse_env_transform_function = self.config.transformed_env_statistics["inv_trans_function"]

        
    def _compute_envelope(self, signal):
        return np.abs(hilbert(signal)) 
    
    def _get_representation(self, signal):
        signal = to_numpy(signal)
        envelope = self._compute_envelope(signal)

        scaled_signal = signal / envelope

        # Normalize NOT NEEDED BECAUSE ALREADY SCALED BY THE ENVELOPE
        #signal_mean = self.config.signal_mean
        #signal_std = self.config.signal_std
        #scaled_signal = (scaled_signal - signal_mean) / signal_std
        # ....

        # Standardize the transformed envelope to get mean=0 and std=1
        scaled_envelope = (self.env_transform_function(envelope) - self.trans_env_mean) / self.trans_env_std # shape: (num_channels, signal_length) (?)

        # QUESTIIONS: 
        # train/val split or also train/val/test split?
        # envelope_mean should be the mean of the envelopes generated for the training samples, right? 

        return np.concatenate([scaled_envelope, scaled_signal], axis=0) # The model will learn to associated channels of the envelope with the corresponding channels of the signal
    
    def _invert_representation(self, representation):
        num_channels = representation.shape[0] // 2
        trans_scaled_envelope = representation[:num_channels]
        scaled_signal = representation[num_channels:]
        
        trans_envelope = trans_scaled_envelope * self.trans_env_std + self.trans_env_mean
        signal = scaled_signal * self.inverse_env_transform_function(trans_envelope)

        return signal  