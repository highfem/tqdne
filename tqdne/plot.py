from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tqdne.metric import Metric
from tqdne.utils import to_numpy

# TODO: maybe rename into plots.py
class Plot(ABC):
    """Abstract plot class.

    All plots should inherit from this class.
    """

    def __init__(self, channel=None):
        self.channel = channel

    @property
    def name(self):
        name = self.__class__.__name__
        return f"{name} - Channel {self.channel}" if self.channel else name

    def __call__(self, pred, target=None, cond_signal=None, cond=None):
        pred = to_numpy(pred)
        target = to_numpy(target)
        cond_signal = to_numpy(cond_signal)
        cond = to_numpy(cond)
        if self.channel is not None:
            pred = pred[:, self.channel]
            target = target[:, self.channel]
            cond_signal = cond_signal[:, self.channel] if cond_signal is not None else None
        return self.plot(pred, target, cond_signal, cond)

    @abstractmethod
    def plot(self, pred, target=None, cond_signal=None, cond=None):
        pass


class SamplePlot(Plot):
    """Plot a sample of the predicted signal."""

    def __init__(self, fs=100, channel=0):
        super().__init__(channel)
        self.fs = fs

    def plot(self, pred, target=None, cond_signal=None, cond=None):
        time = np.arange(0, pred.shape[-1]) / self.fs
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(time, pred[0], "g", label="Reconstructed")
        ax.set_title(self.name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        fig.tight_layout()
        return fig


class UpsamplingSamplePlot(Plot):
    """Plot a sample of the input, target, and reconstructed signals."""

    def __init__(self, fs=100, channel=0):
        super().__init__(channel)
        self.fs = fs

    def plot(self, pred, target, cond_signal, cond=None):
        time = np.arange(0, pred.shape[-1]) / self.fs
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(time, cond_signal[0], "b", label="Input")
        ax.plot(time, target[0], "r", label="Target")
        ax.plot(time, pred[0], "g", label="Reconstructed")
        ax.set_title(self.name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        fig.tight_layout()
        return fig


class PowerSpectralDensity(Plot):
    def __init__(self, fs=100, channel=0):
        super().__init__(channel)
        self.fs = fs

    def plot(self, pred, target, cond_signal=None, cond=None):
        pred_psd = np.abs(np.fft.rfft(pred, axis=-1)) ** 2
        target_psd = np.abs(np.fft.rfft(target, axis=-1)) ** 2

        # Compute mean and std of PSD in log scale
        pred_mean = np.log(pred_psd).mean(axis=0)
        target_mean = np.log(target_psd).mean(axis=0)
        pred_std = np.log(pred_psd).std(axis=0)
        target_std = np.log(target_psd).std(axis=0)

        # Plot
        freq = np.fft.rfftfreq(pred.shape[-1], d=1 / self.fs)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(freq, pred_mean, "g", label="Predicted")
        ax.fill_between(freq, pred_mean - pred_std, pred_mean + pred_std, color="g", alpha=0.2)
        ax.plot(freq, target_mean, "r", label="Target")
        ax.fill_between(
            freq, target_mean - target_std, target_mean + target_std, color="r", alpha=0.2
        )
        ax.set_title(self.name)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Log Power")
        ax.legend()
        fig.tight_layout()
        return fig


class BinPlot(Plot):
    """Creates a bin plot for a given metric."""

    def __init__(
        self,
        metric: Metric,
        num_mag_bins=10,
        num_dist_bins=10,
        min_mag=4.5,
        max_mag=9.5,
        min_dist=0,
        max_dist=180,
    ):
        super().__init__()
        self.metric = metric
        self.num_mag_bins = num_mag_bins
        self.num_dist_bins = num_dist_bins
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.min_dist = min_dist
        self.max_dist = max_dist

    @property
    def name(self):
        return f"Bin {self.metric.name}"

    def plot(self, pred, target, cond_signal, cond):
        # extract the magnitude and distance (this is specific to the dataset)
        mags = cond[:, 3]
        dists = cond[:, 0]

        # create the bins
        mag_bins = np.linspace(self.min_mag, self.max_mag, self.num_mag_bins + 1)
        dist_bins = np.linspace(self.min_dist, self.max_dist, self.num_dist_bins + 1)
        results = np.zeros((self.num_dist_bins, self.num_mag_bins))

        # fill the bins
        for i in range(self.num_mag_bins):
            for j in range(self.num_dist_bins):
                mask = (mags >= mag_bins[i]) & (mags < mag_bins[i + 1])
                mask &= (dists >= dist_bins[j]) & (dists < dist_bins[j + 1])
                results[i, j] = self.metric(pred[mask], target[mask])

        # Plotting the heatmap using seaborn
        mag_bins_center = (mag_bins[1:] + mag_bins[:-1]) / 2
        dist_bins_center = (dist_bins[1:] + dist_bins[:-1]) / 2
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(
            results,
            annot=True,
            fmt=".1f",
            cmap="viridis",
            xticklabels=[f"{mag:.1f}" for mag in mag_bins_center],
            yticklabels=[f"{dist:.0f}" for dist in dist_bins_center],
        )

        ax.set_xlabel("Magnitude Bin")
        ax.set_ylabel("Distance Bin (in km)")
        ax.set_title(self.name)
        fig.tight_layout()
        return fig