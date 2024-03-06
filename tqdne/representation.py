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


class LogSpectrogram(Representation):
    """Represents a signal as a log-spectrogram.

    Parameters
    ----------
    stft_channels : int, default=512
        Number of channels to use in the Short-Time Fourier Transform (STFT).
    hop_size : int, default=16
        Hop size in the STFT.
    clip : float, default=1e-8
        Clip value for spectrogram before taking the logarithm.
    log_max : float, default=3
        Empirical maximum value for the log-spectrogram. Used to normalize the log-spectrogram.
    """

    def __init__(self, stft_channels=512, hop_size=16, clip=1e-8, log_max=3):
        from tifresi.stft import GaussTruncTF

        self.stft_system = GaussTruncTF(hop_size=hop_size, stft_channels=stft_channels)
        self.clip = clip
        self.log_clip = np.log(clip)
        self.log_max = log_max

    def get_spectrogram(self, signal):
        shape = signal.shape
        signal = signal.reshape(-1, shape[-1])  # flatten trailing dimensions
        spec = [self.stft_system.spectrogram(x) for x in signal]
        spec = np.array(spec)[:, :-1, :]  # remove nyquist frequency
        spec = spec.reshape(shape[:-1] + spec.shape[1:])  # restore trailing dimensions
        return spec

    def invert_spectrogram(self, spec):
        shape = spec.shape
        spec = spec.reshape(-1, shape[-2], shape[-1])  # flatten trailing dimensions
        spec = np.pad(spec, ((0, 0), (0, 1), (0, 0)))  # add nyquist frequency
        signal = np.array([self.stft_system.invert_spectrogram(x) for x in spec])
        signal = signal.reshape(
            shape[:-2] + signal.shape[1:]
        )  # restore trailing dimensions
        return signal

    def get_representation(self, signal):
        spec = self.get_spectrogram(signal)
        log_spec = np.log(np.clip(spec, self.clip, None))  # [log_clip, log_max]
        norm_log_spec = (log_spec - self.log_clip) / (
            self.log_max - self.log_clip
        )  # [0, 1]
        norm_log_spec = norm_log_spec * 2 - 1  # [-1, 1]
        return norm_log_spec

    def invert_representation(self, representation):
        norm_log_spec = (representation + 1) / 2
        log_spec = norm_log_spec * (self.log_max - self.log_clip) + self.log_clip
        spec = np.exp(log_spec)
        return self.invert_spectrogram(spec)


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
