from abc import abstractmethod

import numpy as np

from tqdne.utils import NumpyArgMixin


class Representation(NumpyArgMixin):
    """Abstract representation class."""

    @abstractmethod
    def get_representation(self, signal):
        pass

    @abstractmethod
    def invert_representation(self, representation):
        pass


class Identity(Representation):
    def get_representation(self, signal):
        return signal

    def invert_representation(self, representation):
        return representation


class Normalization(Representation):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def get_representation(self, signal):
        return (signal - self.mean) / self.std

    def invert_representation(self, representation):
        return representation * self.std + self.mean


class MovingAverageEnvelope(Representation):
    def __init__(self, window_size=128, log_eps=1e-6, eps=1e-6):
        self.window_size = window_size
        self.log_eps = log_eps
        self.eps = eps

    def get_representation(self, signal):
        env = np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(self.window_size) / self.window_size, mode="same"),
            axis=-1,
            arr=np.abs(signal),
        )
        scaled_signal = signal / (env + self.eps)
        log_env = np.log(env + self.log_eps) - np.log(self.log_eps) / 2
        return np.concatenate([scaled_signal, log_env], axis=-2)

    def invert_representation(self, representation):
        scaled_signal, log_env = np.split(representation, 2, axis=-2)
        env = np.exp(log_env + np.log(self.log_eps) / 2)
        return scaled_signal * (env + self.eps)


class LogSpectrogram(Representation):
    """Represents a signal as a log-spectrogram.

    Parameters
    ----------
    stft_channels : int, default=512
        Number of channels to use in the Short-Time Fourier Transform (STFT).
    hop_size : int, default=None
        How many samples to move the window for each STFT frame.
        If `None`, defaults to `stft_channels // 4`.
    clip : float, default=1e-8
        Clip value for spectrogram before taking the logarithm.
    log_max : float, default=3
        Empirical maximum value for the log-spectrogram. Used to normalize the log-spectrogram.
    library : str, default="librosa"
        Library to use for the STFT. Currently `librosa` and `tifresi` are supported.
        `tifresi` is faster and more accurate, but installing the package is cumbersome.
        Reconstruction accuracy in terms of spectral content is similar for both libraries.
    """

    def __init__(self, stft_channels=512, hop_size=None, clip=1e-8, log_max=3, library="librosa"):
        self.clip = clip
        self.log_clip = np.log(clip)
        self.log_max = log_max

        if hop_size is None:
            hop_size = stft_channels // 4

        if library == "librosa":
            from librosa import griffinlim, stft

            self.stft = lambda x: stft(x, n_fft=stft_channels, hop_length=hop_size)
            self.istft = lambda x: griffinlim(
                x, hop_length=hop_size, n_fft=stft_channels, n_iter=128, random_state=0
            )

        elif library == "tifresi":
            from tifresi.stft import GaussTruncTF

            stft_system = GaussTruncTF(hop_size=hop_size, stft_channels=stft_channels)
            self.stft = stft_system.spectrogram
            self.istft = stft_system.invert_spectrogram

    def get_spectrogram(self, signal):
        shape = signal.shape
        signal = signal.reshape(-1, shape[-1])  # flatten trailing dimensions
        spec = np.array([self.stft(x) for x in signal])
        spec = spec[:, :-1]  # remove nquist frequency
        assert spec.shape[1] % 2 == 0
        spec = spec.reshape(shape[:-1] + spec.shape[1:])  # restore trailing dimensions
        return spec

    def invert_spectrogram(self, spec):
        shape = spec.shape
        spec = spec.reshape(-1, shape[-2], shape[-1])  # flatten trailing dimensions
        spec = np.concatenate([spec, np.zeros_like(spec[:, :1])], axis=1)  # add nquist frequency
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
