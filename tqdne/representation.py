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
    output_shape : tuple, default=None
        Shape of the returned spectrogram.
        If given, the spectrogram will be padded or truncated to this shape.
    internal_shape : tuple, default=None
        Shape of the spectrogram passed to the inverse transform.
        If not given, it will be inferred during the first call to `get_representation`.
    library : str, default="librosa"
        Library to use for the STFT. Currently `librosa` and `tifresi` are supported.
    """

    def __init__(
        self,
        stft_channels=512,
        hop_size=16,
        clip=1e-8,
        log_max=3,
        output_shape=None,
        internal_shape=None,
        library="librosa",
    ):
        self.clip = clip
        self.log_clip = np.log(clip)
        self.log_max = log_max
        self.output_shape = output_shape
        self.internal_shape = internal_shape

        if library == "librosa":
            from librosa import griffinlim
            from librosa.core import stft

            self.stft = lambda x: stft(x, n_fft=stft_channels, hop_length=hop_size)
            self.istft = lambda x: griffinlim(
                x, hop_length=hop_size, n_fft=stft_channels, n_iter=128, random_state=0
            )

        elif library == "tifresi":
            from tifresi.stft import GaussTruncTF

            stft_system = GaussTruncTF(hop_size=hop_size, stft_channels=stft_channels)
            self.stft = stft_system.spectrogram
            self.istft = stft_system.invert_spectrogram

    def adjust_to_shape(self, x, shape=None):
        if not shape:
            return x
        if x.shape[1] < shape[0]:
            x = np.pad(x, ((0, 0), (0, shape[0] - x.shape[0]), (0, 0)), mode="constant")
        if x.shape[2] < shape[1]:
            x = np.pad(x, ((0, 0), (0, 0), (0, shape[1] - x.shape[1])), mode="constant")

        return x[:, : shape[0], : shape[1]]

    def get_spectrogram(self, signal):
        shape = signal.shape
        signal = signal.reshape(-1, shape[-1])  # flatten trailing dimensions
        spec = np.array([self.stft(x) for x in signal])
        if not self.internal_shape:
            self.internal_shape = spec.shape[1:]
        spec = self.adjust_to_shape(spec, self.output_shape)
        spec = spec.reshape(shape[:-1] + spec.shape[1:])  # restore trailing dimensions
        return spec

    def invert_spectrogram(self, spec):
        shape = spec.shape
        spec = spec.reshape(-1, shape[-2], shape[-1])  # flatten trailing dimensions
        spec = self.adjust_to_shape(spec, self.internal_shape)
        signal = np.array([self.istft(x) for x in spec])
        signal = signal.reshape(shape[:-2] + signal.shape[1:])  # restore trailing dimensions
        return signal

    def get_representation(self, signal):
        spec = self.get_spectrogram(signal)
        spec = np.abs(spec)
        log_spec = np.log(np.clip(spec, self.clip, None))  # [log_clip, log_max]
        norm_log_spec = (log_spec - self.log_clip) / (self.log_max - self.log_clip)  # [0, 1]
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
