from abc import abstractmethod

import numpy as np
from pathos.multiprocessing import Pool

from tqdne.utils import NumpyArgMixin


class Representation(NumpyArgMixin):
    """Abstract representation class."""

    @abstractmethod
    def get_representation(self, waveform):
        pass

    @abstractmethod
    def invert_representation(self, representation):
        pass


class Identity(Representation):
    def get_representation(self, waveform):
        return waveform

    def invert_representation(self, representation):
        return representation


class Normalization(Representation):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def get_representation(self, waveform):
        return (waveform - self.mean) / self.std

    def invert_representation(self, representation):
        return representation * self.std + self.mean


class MovingAverageEnvelope(Representation):
    def __init__(self, window_size=128, log_eps=1e-6, eps=1e-6):
        self.window_size = window_size
        self.log_eps = log_eps
        self.eps = eps

    def get_representation(self, waveform):
        env = np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(self.window_size) / self.window_size, mode="same"),
            axis=-1,
            arr=np.abs(waveform),
        )
        scaled_waveform = waveform / (env + self.eps)
        log_env = np.log(env + self.log_eps) - np.log(self.log_eps) / 2
        return np.concatenate([scaled_waveform, log_env], axis=-2)

    def invert_representation(self, representation):
        scaled_waveform, log_env = np.split(representation, 2, axis=-2)
        env = np.exp(log_env + np.log(self.log_eps) / 2)
        return scaled_waveform * (env + self.eps)


class LogSpectrogram(Representation):
    """Represents a waveform as a log-spectrogram.

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
    multiprocessing : bool, default=True
        Whether to use multiprocessing for computing the STFT and iSTFT.
    """

    def __init__(
        self,
        stft_channels=256,
        hop_size=None,
        clip=1e-8,
        log_max=3,
        library="librosa",
        multiprocessing=True,
    ):
        self.clip = clip
        self.log_clip = np.log(clip)
        self.log_max = log_max
        self.library = library

        if hop_size is None:
            hop_size = stft_channels // 4

        if library == "librosa":
            from librosa import griffinlim, stft

            self.stft = lambda x: stft(x, n_fft=stft_channels, hop_length=hop_size)
            self.istft = lambda x: griffinlim(
                x, hop_length=hop_size, n_fft=stft_channels, n_iter=128, random_state=0
            )

        elif library == "torchaudio":
            from torchaudio.transforms import Spectrogram

            spec = Spectrogram(
                n_fft=stft_channels, hop_length=hop_size, power=1, center=True, pad_mode="constant"
            )
            self.stft = lambda x: spec(x, n_fft=stft_channels, hop_length=hop_size)
            self.istft = lambda x: griffinlim(
                x, hop_length=hop_size, n_fft=stft_channels, n_iter=128, random_state=0
            )

        elif library == "tifresi":
            from tifresi.stft import GaussTruncTF

            stft_system = GaussTruncTF(hop_size=hop_size, stft_channels=stft_channels)
            self.stft = stft_system.spectrogram
            self.istft = stft_system.invert_spectrogram

        if multiprocessing:
            self.pool = Pool()

    def __del__(self):
        if hasattr(self, "pool"):
            self.pool.close()

    def disable_multiprocessing(self):
        if hasattr(self, "pool"):
            self.pool.close()
            del self.pool

    def get_spectrogram(self, waveform):
        shape = waveform.shape
        waveform = waveform.reshape(-1, shape[-1])  # flatten trailing dimensions
        if hasattr(self, "pool"):
            spec = np.array(self.pool.map(self.stft, waveform))
        else:
            spec = np.array([self.stft(x) for x in waveform])
        spec = spec[:, :-1]  # remove nquist frequency
        assert spec.shape[1] % 2 == 0
        spec = spec.reshape(shape[:-1] + spec.shape[1:])  # restore trailing dimensions
        return spec

    def invert_spectrogram(self, spec):
        shape = spec.shape
        spec = spec.reshape(-1, shape[-2], shape[-1])  # flatten trailing dimensions
        spec = np.concatenate([spec, np.zeros_like(spec[:, :1])], axis=1)  # add nquist frequency
        if hasattr(self, "pool"):
            waveform = np.array(self.pool.map(self.istft, spec))
        else:
            waveform = np.array([self.istft(x) for x in spec])
        waveform = waveform.reshape(shape[:-2] + waveform.shape[1:])  # restore trailing dimensions
        return waveform

    def get_representation(self, waveform):
        spec = self.get_spectrogram(waveform)
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
