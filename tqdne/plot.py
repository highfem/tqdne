from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tqdne import utils
from tqdne.metric import Metric
from tqdne.representations import to_numpy

# TODO: maybe rename into plots.py
class Plot(ABC):
    """Abstract plot class.

    All plots should inherit from this class.
    """

    def __init__(self, channel=None, data_representation=None, invert_representation=True):
        self.channel = channel
        self.data_representation = data_representation
        self.invert_representation = invert_representation
        self.invert_representation_fun = data_representation.invert_representation if invert_representation else to_numpy  

    @property
    def name(self):
        name = self.__class__.__name__
        name = f"{name} - Raw Output" if not self.invert_representation else name
        name = f"{name} - Channel {self.channel}" if self.channel is not None else name
        return name

    def __call__(self, pred, target=None, cond_signal=None, cond=None):
        if self.data_representation is not None:
            pred = self.invert_representation_fun(pred)
            target = self.invert_representation_fun(target) 
            cond = to_numpy(cond)
        if self.channel is not None:
            pred = pred[:, self.channel]
            target = target[:, self.channel]
            cond_signal = cond_signal[:, self.channel] if cond_signal is not None else None

        ##### DEBUG
        #figuretoplot = target[0,:]
        #fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        #ax.plot(figuretoplot)
        #fig.savefig("target_inv_is_working.png")   # save the figure to file
        #plt.close(fig)    # close the figure window


        return self.plot(pred, target, cond_signal, cond)

    @abstractmethod
    def plot(self, pred, target=None, cond_signal=None, cond=None):
        pass


class SamplePlot(Plot):
    """Plot a sample of the predicted signal."""

    def __init__(self, fs=100, channel=0, data_representation=None, invert_representation=True):
        super().__init__(channel, data_representation, invert_representation)
        self.fs = fs

    def plot(self, pred, target=None, cond_signal=None, cond=None):
        time = np.arange(0, pred.shape[-1]) / self.fs
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(time, pred[0], "g", label="Generated")
        if not self.invert_representation:
            ax.plot(time, target[0], "r", alpha=0.5, label="Target")
        title = f"{self.name}\n{', '.join([f'{key}: {value:.1f}' for key, value in utils.get_cond_params_dict(cond[0]).items()])}" if cond is not None else self.name
        #return {key: f"{value:.2f}" for key, value in cond.items()}
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        fig.tight_layout()
        return fig


class UpsamplingSamplePlot(Plot):
    """Plot a sample of the input, target, and reconstructed signals."""

    def __init__(self, fs=100, channel=0, data_representation=None, invert_representation=True):
        super().__init__(channel, data_representation)
        self.fs = fs

    def plot(self, pred, target, cond_signal, cond=None):
        time = np.arange(0, pred.shape[-1]) / self.fs
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(time, cond_signal[0], "b", label="Input")
        ax.plot(time, target[0], "r", label="Target")
        ax.plot(time, pred[0], "g", label="Reconstructed")
        title = f"{self.name}\n{', '.join([f'{key}: {value:.1f}' for key, value in utils.get_cond_params_dict(cond[0]).items()])}" if cond is not None else self.name
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        fig.tight_layout()
        return fig

class LogEnvelopePlot(Plot):
    def __init__(self, fs=100, channel=0, data_representation=None, invert_representation=True):
        super().__init__(channel, data_representation, invert_representation)
        self.fs = fs

    def plot(self, pred, target=None, cond_signal=None, cond=None):

        pred_logenv = utils.get_log_envelope(pred)
        target_logenv = utils.get_log_envelope(target)

        # TODO: mean or median?

        pred_logenv_median = np.median(pred_logenv, axis=0)
        pred_logenv_p25 = np.percentile(pred_logenv, 25, axis=0)
        pred_logenv_p75 = np.percentile(pred_logenv, 75, axis=0)
        target_logenv_median = np.median(target_logenv, axis=0)
        target_logenv_p25 = np.percentile(target_logenv, 25, axis=0)
        target_logenv_p75 = np.percentile(target_logenv, 75, axis=0)

        time_ax = np.arange(0, len(pred_logenv_median)) / self.fs
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(time_ax, pred_logenv_median, "g", label="Generated - median")
        ax.fill_between(time_ax, pred_logenv_p25, pred_logenv_p75, color="g", alpha=0.2, label="Generated - IQR (25-75%)")
        ax.plot(time_ax, target_logenv_median, "r", label="Target - median")
        ax.fill_between(time_ax, target_logenv_p25, target_logenv_p75, color="r", alpha=0.2, label="Target - IQR (25-75%)")

        ax.set_title(self.name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Log Envelope")
        ax.legend()
        fig.tight_layout()
        return fig
        

class PowerSpectralDensityPlot(Plot):
    def __init__(self, fs=100, channel=0, data_representation=None, invert_representation=True):
        super().__init__(channel, data_representation, invert_representation)
        self.fs = fs

    def plot(self, pred, target, cond_signal=None, cond=None):
        # TODO: bad design choice: code is duplicated from metric.py
        pred_psd = np.abs(np.fft.rfft(pred, axis=-1)) ** 2
        target_psd = np.abs(np.fft.rfft(target, axis=-1)) ** 2

        # Compute mean and std of PSD in log scale
        eps = 1e-7
        pred_mean = np.log(pred_psd + eps).mean(axis=0)
        target_mean = np.log(target_psd + eps).mean(axis=0)
        pred_std = np.log(pred_psd + eps).std(axis=0)
        target_std = np.log(target_psd + eps).std(axis=0)

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
        

# TODO: it doesn't work like this since it needs to be updated, not just evaluated with a single batch 
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
        max_dist=180
    ):
        super().__init__(data_representation=None, invert_representation=False)
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
