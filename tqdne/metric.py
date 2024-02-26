from abc import ABC, abstractmethod

import numpy as np

from tqdne.utils import to_numpy


class Metric(ABC):
    """Abstract metric class.

    All metrics should inherit from this class.
    """

    def __init__(self, channel=None):
        self.channel = channel

    @property
    def name(self):
        name = self.__class__.__name__
        return f"{name} - Channel {self.channel}" if self.channel else name

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


class PowerSpectralDensity(Metric):
    def __init__(self, fs, channel=0):
        super().__init__(channel)
        self.fs = fs

    def compute(self, pred, target):
        pred_psd = np.abs(np.fft.rfft(pred, axis=-1)) ** 2
        target_psd = np.abs(np.fft.rfft(target, axis=-1)) ** 2

        # Compute mean and std of PSD in log scale
        pred_mean = np.log(pred_psd).mean(axis=0)
        target_mean = np.log(target_psd).mean(axis=0)
        pred_std = np.log(pred_psd).std(axis=0)
        target_std = np.log(target_psd).std(axis=0)

        # Frechét distance between isotropic Gaussians (Wasserstein-2)
        fid = np.sum((pred_mean - target_mean) ** 2, axis=-1) + np.sum(
            pred_std**2 + target_std**2 - 2 * pred_std * target_std, axis=-1
        )

        return fid
