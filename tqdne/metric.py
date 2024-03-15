from abc import ABC, abstractmethod

import numpy as np



class Metric(ABC):
    """Abstract metric class.

    All metrics should inherit from this class.
    """

    def __init__(self, channel=None, data_representation=None):
        self.channel = channel
        self.data_representation = data_representation

    @property
    def name(self):
        # TODO: fix the bug that for channel 0 is like self.channel is None (no channel name is displayed)
        name = self.__class__.__name__
        return f"{name} - Channel {self.channel}" if self.channel else name

    def __call__(self, pred, target):
        #pred = to_numpy(pred)
        #target = to_numpy(target)
        if self.data_representation is not None:
            pred = self.data_representation.invert_representation(pred)
            target = self.data_representation.invert_representation(target)
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
    def __init__(self, fs, channel=0, data_representation=None):
        super().__init__(channel, data_representation)
        self.fs = fs

    def compute(self, pred, target):
        pred_psd = np.abs(np.fft.rfft(pred, axis=-1)) ** 2
        target_psd = np.abs(np.fft.rfft(target, axis=-1)) ** 2

        # Compute mean and std of PSD in log scale
        eps = 1e-7
        pred_mean = np.log(pred_psd + eps).mean(axis=0)
        target_mean = np.log(target_psd + eps).mean(axis=0)
        pred_std = np.log(pred_psd + eps).std(axis=0)
        target_std = np.log(target_psd + eps).std(axis=0)

        # Frechét distance between isotropic Gaussians (Wasserstein-2)
        fid = np.sum((pred_mean - target_mean) ** 2, axis=-1) + np.sum(
            pred_std**2 + target_std**2 - 2 * pred_std * target_std, axis=-1
        )

        return fid
