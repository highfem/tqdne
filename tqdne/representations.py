
from abc import ABC, abstractmethod

import numpy as np

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

class LogMaxEnvelope(Representation):

    def _get_representation(self, signal):
        norm = np.max(np.abs(signal), axis=-1, keepdims=True)
        norm = np.repeat(norm, signal.shape[-1], axis=-1)
        scaled_signal = (signal / norm)
        envelope = np.log10(norm + 1e-7)
        # self.min_norm, self.max_norm = np.min(envelope, axis=0, keepdims=True), np.max(envelope, axis=0, keepdims=True)
        # envelope = 2.0 * (envelope - self.min_norm) / (self.max_norm - self.min_norm) - 1.0
        return np.concatenate([envelope, scaled_signal], axis=-2)
    
    def _invert_representation(self, representation):
        # envelope = (self.max_norm - self.max_norm) * (representation[:, 0, :] + 1.0) / 2.0 + self.min_norm
        num_channels = representation.shape[0] // 2
        envelope = representation[:num_channels]
        norm = 10 ** envelope
        return norm * representation[num_channels:]

def _centered_window(x, window_len):
    assert window_len % 2, "Centered Window has to have odd length"
    mid = window_len // 2
    pos = 0
    while pos < x.shape[-1]:
        yield x[..., max(pos - mid, 0) : min(pos + mid + 1, len(x))]
        pos += 1

def centered_max(x, window_len):
    out = np.concatenate(
        [
            np.max(window, axis=-1, keepdims=True) if window.shape[-1] > 0 else np.zeros((*x.shape[:-1], 1))
            for window in _centered_window(x, window_len)
        ],
        axis=-1,
    )
    return out

class CenteredMaxEnvelope(Representation):
    
    def _get_representation(self, signal):
        envelope = centered_max(signal, 7)
        envelope = np.maximum(envelope, 1e-10)
        scaled_signal = signal / envelope
        envelope = np.log10(envelope)
        return np.concatenate([envelope, scaled_signal], axis=0)

    def _invert_representation(self, representation):
        num_channels = representation.shape[0] // 2
        norm = 10 ** representation[:num_channels]
        scaled_signal = representation[num_channels:]
        return norm * scaled_signal
    
class Envelope(Representation):
    
    def _get_representation(self, signal):
        envelope = np.zeros_like(signal) # TODO: just placeholder

        scaled_signal = signal / envelope

        # TODO: normalize
        # signal_mean = self.config.signal_mean
        # signal_std = self.config.signal_std
        # scaled_signal = (scaled_signal - signal_mean) / signal_std
        # ....


        return np.concatenate([envelope, scaled_signal], axis=0)
    
    def _invert_representation(self, representation):
        num_channels = representation.shape[0] // 2
        envelope = representation[:num_channels]
        scaled_signal = representation[num_channels:]
        # ...