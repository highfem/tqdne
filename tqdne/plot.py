from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tqdne.metric import Metric
from tqdne.utils import to_numpy


class Plot(ABC):
    """Abstract plot class.

    All plots should inherit from this class.

    Parameters
    ----------
    channel : int, optional
        The channel number. Default is 0.
    """

    def __init__(self, channel=None):
        self.channel = channel

    @property
    def name(self):
        name = self.__class__.__name__
        if self.channel is None:
            return name
        return f"{name} - Channel {self.channel}"

    def __call__(self, pred, target=None, cond_signal=None, **kwargs):
        """Call the plot.

        Parameters
        ----------
        pred : numpy.ndarray
            The predicted waveform.
        target : numpy.ndarray, optional
            The target waveform. Default is None.
        cond_signal : numpy.ndarray, optional
            The conditional waveform. Default is None.

        Returns
        -------
        pyplot.Figure
            The figure object.
        """
        pred = to_numpy(pred)
        target = to_numpy(target)
        cond_signal = to_numpy(cond_signal)
        if self.channel is not None:
            pred = pred[:, self.channel]
            target = target[:, self.channel]
            cond_signal = cond_signal[:, self.channel] if cond_signal is not None else None
        kwargs = {k: to_numpy(v) for k, v in kwargs.items()}
        return self.plot(pred, target, cond_signal, **kwargs)

    @abstractmethod
    def plot(self, pred, target=None, cond_signal=None, cond=None):
        pass


class SamplePlot(Plot):
    """Plot a sample of the predicted signal."""

    def __init__(self, plot_target=False, fs=100, channel=0):
        super().__init__(channel)
        self.plot_target = plot_target
        self.fs = fs

    def plot(self, pred, target=None, *args, **kwargs):
        time = np.arange(0, pred.shape[-1]) / self.fs
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.plot(time, pred[0], "b", label="Predicted")
        if self.plot_target:
            ax.plot(time, target[0], "orange", label="Target")
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

    def plot(self, pred, target, cond_signal, *args, **kwargs):
        time = np.arange(0, pred.shape[-1]) / self.fs
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.plot(time, cond_signal[0], "g", label="Input")
        ax.plot(time, target[0], "orange", label="Target")
        ax.plot(time, pred[0], "b", label="Predicted")
        ax.set_title(self.name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        fig.tight_layout()
        return fig


class AmplitudeSpectralDensity(Plot, ABC):
    def __init__(self, fs, channel=0, log_eps=1e-8):
        super().__init__(channel)
        self.fs = fs
        self.log_eps = log_eps

    def spectral_density(self, signal):
        sd = np.abs(np.fft.rfft(signal, axis=-1))
        log_sd = np.log(np.clip(sd, self.log_eps, None))
        return log_sd

    def plot(self, pred, target, *args, **kwargs):
        pred_sd = self.spectral_density(pred)
        target_sd = self.spectral_density(target)

        # Compute mean and std of SD in log scale
        pred_mean = pred_sd.mean(axis=0)
        target_mean = target_sd.mean(axis=0)
        pred_std = pred_sd.std(axis=0)
        target_std = target_sd.std(axis=0)

        # Plot
        freq = np.fft.rfftfreq(pred.shape[-1], d=1 / self.fs)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(freq, pred_mean, "b", label="Predicted")
        ax.fill_between(freq, pred_mean - pred_std, pred_mean + pred_std, color="b", alpha=0.2)
        ax.plot(freq, target_mean, "orange", label="Target")
        ax.fill_between(
            freq, target_mean - target_std, target_mean + target_std, color="orange", alpha=0.2
        )
        ax.set_title("Log-Amplitude Spectral Density")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Log Fourier Amplitude Spectral Density")
        ax.legend()
        fig.tight_layout()
        return fig


class BinPlot(Plot):
    """Creates a bin plot for a given metric."""

    def __init__(self, metric: Metric, mag_bins, dist_bins, fmt=".2f"):
        super().__init__()
        self.metric = metric
        self.mag_bins = mag_bins
        self.dist_bins = dist_bins
        self.fmt = fmt

    @property
    def name(self):
        return f"Bin {self.metric.name}"

    def plot(self, pred, target, cond_signal, mag, dist):
        # compute metrics for each bin
        results = []
        for i in range(len(self.dist_bins) - 1):
            results.append([])
            for j in range(len(self.mag_bins) - 1):
                mask = (dist >= self.dist_bins[i]) & (dist < self.dist_bins[i + 1])
                mask &= (mag >= self.mag_bins[j]) & (mag < self.mag_bins[j + 1])
                results[i].append(self.metric(pred[mask], target[mask]))

        # Plotting the heatmap using seaborn
        plot = sns.heatmap(np.array(results), annot=True, fmt=self.fmt, cmap="viridis")
        plot.set_xticks(np.arange(len(self.mag_bins)))
        plot.set_xticklabels(self.mag_bins)
        plot.set_yticks(np.arange(len(self.dist_bins)))
        plot.set_yticklabels(self.dist_bins)
        plot.invert_yaxis()
        plot.set_xlabel("Magnitude bin")
        plot.set_ylabel("Distance bin [km]")
        fig = plot.get_figure()
        fig.tight_layout()
        return fig
