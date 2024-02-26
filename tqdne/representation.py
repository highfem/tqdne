from abc import abstractmethod

import numpy as np

from tqdne.conf import Config
from tqdne.utils import NumpyArgMixin


class Representation(NumpyArgMixin):
    def __init__(self, config=Config()):
        self.config = config

    @abstractmethod
    def get_representation(self, signal):
        pass

    @abstractmethod
    def invert_representation(self, representation):
        pass


class Envelope(Representation):
    def get_representation(self, signal):
        envelope = np.zeros_like(signal)  # TODO: just placeholder

        scaled_signal = signal / envelope

        # TODO: normalize
        # signal_mean = self.config.signal_mean
        # signal_std = self.config.signal_std
        # scaled_signal = (scaled_signal - signal_mean) / signal_std
        # ....

        return np.concatenate([envelope, scaled_signal], axis=0)

    def invert_representation(self, representation):
        num_channels = representation.shape[0] // 2
        envelope = representation[:num_channels]
        scaled_signal = representation[num_channels:]
        # ...
