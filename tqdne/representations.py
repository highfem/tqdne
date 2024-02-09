
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

        # where and when are they computed? offline?
        envelope_mean = self.config.envelope_mean # dimension: (num_channels, signal_length) (?)
        envelope_std = self.config.envelope_std # dimension: (num_channels, signal_length) (?)
        scaled_envelope = (envelope - envelope_mean) / envelope_std # dimension: (num_channels, signal_length) (?)

        # QUESTIIONS: 
        # train/val split or also train/val/test split?
        # envelope_mean should be the mean of the envelopes generated for the training samples, right? 

        return np.concatenate([scaled_envelope, scaled_signal], axis=0) # aren't we mixing envelope with different channels?
    
    def _invert_representation(self, representation):
        num_channels = representation.shape[0] // 2
        envelope = representation[:num_channels]
        scaled_signal = representation[num_channels:]
        
        envelope = envelope * self.config.envelope_std + self.config.envelope_mean
        signal = scaled_signal * envelope

        return signal  