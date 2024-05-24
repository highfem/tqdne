from abc import ABC, abstractmethod

import numpy as np

from tqdne.utils import to_numpy


class Metric(ABC):
    """Abstract metric class.

    All metrics should inherit from this class.
    """

    def __init__(self, channel=0):
        self.channel = channel

    @property
    def name(self):
        name = self.__class__.__name__
        return f"{name} - Channel {self.channel}"

    def __call__(self, pred, target):
        pred = to_numpy(pred)
        target = to_numpy(target)
        if self.channel is not None:
            pred = pred[:, self.channel]
            target = target[:, self.channel]
        return self.compute(pred, target)

    @abstractmethod
    def compute(self, pred, target):
        pass


class MeanSquaredError(Metric):
    def compute(self, pred, target):
        return ((pred - target) ** 2).mean()


class AmplitudeSpectralDensity(Metric, ABC):
    def __init__(self, fs, channel=0, log_eps=1e-8):
        super().__init__(channel)
        self.fs = fs
        self.log_eps = log_eps

    def spectral_density(self, signal):
        sd = np.abs(np.fft.rfft(signal, axis=-1))
        log_sd = np.log(np.clip(sd, self.log_eps, None))
        return log_sd

    def compute(self, pred, target):
        pred_sd = self.spectral_density(pred)
        target_sd = self.spectral_density(target)

        # Compute mean and std of SD in log scale
        pred_mean = pred_sd.mean(axis=0)
        target_mean = target_sd.mean(axis=0)
        pred_std = pred_sd.std(axis=0)
        target_std = target_sd.std(axis=0)

        # Frechét distance between isotropic Gaussians (Wasserstein-2)
        fid = np.sum((pred_mean - target_mean) ** 2, axis=-1) + np.sum(
            pred_std**2 + target_std**2 - 2 * pred_std * target_std, axis=-1
        )

        return fid
