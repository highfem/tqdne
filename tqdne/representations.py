
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

class GlobalMaxEnvelope(Representation):
    def __str__(self):
        return "GlobalMaxEnvelope"

    def _get_representation(self, signal):
        norm = np.max(np.abs(signal), axis=-1, keepdims=True) + 1e-5
        norm = np.repeat(norm, signal.shape[-1], axis=-1)
        scaled_signal = (signal / norm)
        envelope = np.log10(norm)
        # self.min_norm, self.max_norm = np.min(envelope, axis=0, keepdims=True), np.max(envelope, axis=0, keepdims=True)
        # envelope = 2.0 * (envelope - self.min_norm) / (self.max_norm - self.min_norm) - 1.0
        return np.concatenate([envelope, scaled_signal], axis=-2)
    
    def _invert_representation(self, representation):
        # envelope = (self.max_norm - self.max_norm) * (representation[:, 0, :] + 1.0) / 2.0 + self.min_norm
        num_channels = representation.shape[-2] // 2
        norm = 10 ** np.take(representation, np.arange(0, num_channels), axis=-2)
        scaled_signal = np.take(representation, np.arange(num_channels, 2 * num_channels), axis=-2)
        ans = norm * scaled_signal
        return ans

def _centered_window(x, window_len):
    assert window_len % 2, "Centered Window has to have odd length"
    mid = window_len // 2
    pos = 0
    while pos < x.shape[-1]:
        yield x[..., max(pos - mid, 0) : min(pos + mid + 1, len(x))]
        pos += 1

def centered_p_mean(x, window_len, p):
    out = np.concatenate(
        [
            np.mean(np.abs(window) ** p, axis=-1, keepdims=True) ** (1 / p) if window.shape[-1] > 0 else np.zeros((*x.shape[:-1], 1))
            for window in _centered_window(x, window_len)
        ],
        axis=-1,
    )
    return out

class CenteredPMeanEnvelope(Representation):
    def __init__(self, config=Config(), window_length=15, p=3):
        super().__init__(config)
        self.p = p
        self.window_length = window_length

    def __str__(self):
        return f"CenteredPMeanEnvelope-{self.p}"

    def _get_representation(self, signal):
        signal = signal.reshape(-1)
        envelope = centered_p_mean(signal, self.window_length, self.p)
        envelope = np.maximum(envelope, 1e-10)
        scaled_signal = signal / envelope
        envelope = np.log10(envelope).reshape(1, -1)
        signal = scaled_signal.reshape(1, -1)
        return np.concatenate([envelope, signal], axis=0)

    def _invert_representation(self, representation):
        num_channels = representation.shape[-2] // 2
        norm = 10 ** np.take(representation, np.arange(0, num_channels), axis=-2)
        scaled_signal = np.take(representation, np.arange(num_channels, 2 * num_channels), axis=-2)
        return norm * scaled_signal

def centered_max(x, window_len):
    out = np.concatenate(
        [
            np.max(np.abs(window), axis=-1, keepdims=True) if window.shape[-1] > 0 else np.zeros((*x.shape[:-1], 1))
            for window in _centered_window(x, window_len)
        ],
        axis=-1,
    )
    return out

class CenteredMaxEnvelope(Representation):
    def __init__(self, config=Config(), window_length=15):
        super().__init__(config)
        self.window_length = window_length

    def __str__(self):
        return "CenteredMaxEnvelope"

    def _get_representation(self, signal):
        signal = signal.reshape(-1)
        envelope = centered_max(signal, 15)
        envelope = np.maximum(envelope, 1e-10)
        scaled_signal = signal / envelope
        envelope = np.log10(envelope).reshape(1, -1)
        signal = scaled_signal.reshape(1, -1)
        return np.concatenate([envelope, signal], axis=0)

    def _invert_representation(self, representation):
        num_channels = representation.shape[-2] // 2
        norm = 10 ** np.take(representation, np.arange(0, num_channels), axis=-2)
        scaled_signal = np.take(representation, np.arange(num_channels, 2 * num_channels), axis=-2)
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