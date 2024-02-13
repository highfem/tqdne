from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch import Tensor
from torchmetrics import Metric

from tqdne.representations import Representation
from tqdne.utils import to_numpy


class AbstractMetric(Metric, ABC):
    """Abstract metric class.

    All metrics should inherit from this class.
    `name` and `update` must be implemented. `compute` and `plot` are optional.
    """

    @property
    @abstractmethod
    def name(self):
        pass

    def update(self, pred, target):
        """Update the metric with the predicted and target signals.

        Args:
            pred (dict): The predicted signals.
            target (dict): The target signals.

        """
        pred = {k: to_numpy(v) for k, v in pred.items()}
        target = {k: to_numpy(v) for k, v in target.items()}
        self._update(pred, target)

    @abstractmethod
    def _update(self, pred, target):
        pass

    def compute(self):
        pass

    def plot(self):
        pass


class SamplePlot(AbstractMetric):
    """Plot a sample of the predicted signal."""

    def __init__(self, fs, channel):
        super().__init__()
        self.fs = fs
        self.channel = channel
        self.add_state("Generated", default=[], dist_reduce_fx=None)

    @property
    def name(self):
        return f"Sample Plot - Channel {self.channel}"

    def _update(self, pred, target):
        self.generated = pred["generated"][0, self.channel] 

    def compute(self):
        return None

    def plot(self):
        time = np.arange(0, self.generated.shape[-1]) / self.fs
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(time, self.generated, "g", label="Generated")
        ax.set_title(self.name)
        ax.set_xlabel("Time (s)") 
        ax.set_ylabel("Amplitude") #TODO: unit of measure?
        ax.legend()
        fig.tight_layout()
        return fig


class UpsamplingSamplePlot(AbstractMetric):
    """Plot a sample of the input, target, and reconstructed signals."""

    def __init__(self, fs, channel):
        super().__init__()
        self.fs = fs
        self.channel = channel
        self.add_state("low_res", default=[], dist_reduce_fx=None)
        self.add_state("high_res", default=[], dist_reduce_fx=None)
        self.add_state("reconstructed", default=[], dist_reduce_fx=None)

    @property
    def name(self):
        return f"Upsampling Sample Plot - Channel {self.channel}"

    def _update(self, pred, target):
        self.low_res = target["low_res"][0, self.channel]
        self.high_res = target["high_res"][0, self.channel]
        self.reconstructed = pred["high_res"][0, self.channel]

    def compute(self):
        return None

    def plot(self):
        time = np.arange(0, self.low_res.shape[-1]) / self.fs
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(time, self.low_res, "b", label="Input")
        ax.plot(time, self.high_res, "r", label="Target")
        ax.plot(time, self.reconstructed, "g", label="Reconstructed")
        ax.set_title(self.name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        fig.tight_layout()
        return fig


class MeanSquaredError(AbstractMetric):
    """Compute the mean squared error between the predicted and target signals."""

    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.add_state("pred", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

    @property
    def name(self):
        return f"Mean Squared Error - Channel {self.channel}"

    def _update(self, pred, target):
        self.pred.append(pred["high_res"][:, self.channel])
        self.target.append(target["high_res"][:, self.channel])

    def compute(self):
        pred = np.concatenate(self.pred)
        target = np.concatenate(self.target)
        return ((pred - target) ** 2).mean()


class PowerSpectralDensity(AbstractMetric):
    """Compute the Frechét Inception Distance between the power spectral density distributions of the predicted and target signals.

    Args:
        fs (int): The sampling frequency of the signals.
    """

    def __init__(self, fs, channel):
        super().__init__()
        self.fs = fs
        self.channel = channel
        self.add_state("pred_psd", default=[], dist_reduce_fx=None)
        self.add_state("target_psd", default=[], dist_reduce_fx=None)

    @property
    def name(self):
        return f"Power Spectral Density - Channel {self.channel}"

    def _update(self, pred, target):
        pred = pred["generated"][:, self.channel]
        target = target["representation"][:, self.channel]
        pred_psd = np.abs(np.fft.rfft(pred, axis=-1)) ** 2
        target_psd = np.abs(np.fft.rfft(target, axis=-1)) ** 2
        self.pred_psd.append(pred_psd)
        self.target_psd.append(target_psd)
        self.sig_len = pred.shape[-1]

    def compute(self):
        pred_psd = np.concatenate(self.pred_psd)
        target_psd = np.concatenate(self.target_psd)

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

    def plot(self):
        pred_psd = np.concatenate(self.pred_psd)
        target_psd = np.concatenate(self.target_psd)

        # Compute mean and std of PSD in log scale
        pred_mean = np.log(pred_psd).mean(axis=0)
        target_mean = np.log(target_psd).mean(axis=0)
        pred_std = np.log(pred_psd).std(axis=0)
        target_std = np.log(target_psd).std(axis=0)

        # Frequency axis
        freqs = np.fft.rfftfreq(self.sig_len, 1 / self.fs)

        # Plot
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(freqs, pred_mean, "g", label="Reconstructed")
        ax.fill_between(
            freqs, 
            pred_mean - pred_std, 
            pred_mean + pred_std, 
            color="g", 
            alpha=0.2,
        )
        ax.plot(freqs, target_mean, "r", label="Target")
        ax.fill_between(
            freqs,
            target_mean - target_std,
            target_mean + target_std,
            color="r",
            alpha=0.2,
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Log PSD")
        ax.set_title(self.name)
        ax.legend()
        fig.tight_layout()
        return fig


class BinMetric(AbstractMetric):
    def __init__(
        self,
        metric,
        num_mag_bins=10,
        num_dist_bins=10,
        min_mag=4.5,
        max_mag=9.5,
        min_dist=0,
        max_dist=180,
    ):
        super().__init__()
        self.metric = metric
        self.metrics = [
            [metric.clone() for _ in range(num_mag_bins)] for _ in range(num_dist_bins)
        ]
        self.num_mag_bins = num_mag_bins
        self.num_dist_bins = num_dist_bins
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.min_dist = min_dist
        self.max_dist = max_dist

    @property
    def name(self):
        return f"Bin {self.metric.name}"

    def _update(self, pred, target):
        # extract magnitude and distance (this is specific to the dataset)
        mags = target["cond"][:, 3]
        dists = target["cond"][:, 0]

        # update the corresponding metric
        for mag, dist in zip(mags, dists):
            assert (
                self.min_mag <= mag <= self.max_mag
            ), f"{mag} not in range {self.min_mag} to {self.max_mag}"
            assert (
                self.min_dist <= dist <= self.max_dist
            ), f"{dist} not in range {self.min_dist} to {self.max_dist}"
            mag_bin = int(
                (mag - self.min_mag) / (self.max_mag - self.min_mag) * self.num_mag_bins
            )
            dist_bin = int(
                (dist - self.min_dist)
                / (self.max_dist - self.min_dist)
                * self.num_dist_bins
            )
            self.metrics[dist_bin][mag_bin].update(pred, target)

    def plot(self):
        values = []
        for row in self.metrics:
            row_values = []
            for metric in row:
                if metric.update_called:
                    row_values.append(metric.compute())
                else:
                    row_values.append(np.nan)
            values.append(row_values)

        # Plotting the heatmap using seaborn
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(
            values,
            annot=True,
            fmt=".1f",
            cmap="viridis",
            # mean of the bin
            xticklabels=[
                f"{(i + 0.5) * (self.max_mag - self.min_mag) / self.num_mag_bins + self.min_mag:.1f}"
                for i in range(self.num_mag_bins)
            ],
            yticklabels=[
                f"{(i + 0.5) * (self.max_dist - self.min_dist) / self.num_dist_bins + self.min_dist:.0f}"
                for i in range(self.num_dist_bins)
            ],
            ax=ax,
        )

        ax.set_xlabel("Magnitude Bin")
        ax.set_ylabel("Distance Bin (in km)")
        ax.set_title(self.name)
        fig.tight_layout()
        return fig

    def reset(self):
        for row in self.metrics:
            for metric in row:
                metric.reset()


class RepresentationInversion(AbstractMetric):
    """Wrapper to invert the representation and compute a metric on the resulting signal."""

    def __init__(self, metric: AbstractMetric, representation: Representation):
        super().__init__()
        self.metric = metric
        self.representation = representation

    def _update(self, pred, target):
        pred["generated"] = self.representation.invert_representation(pred["generated"])

        target["representation"] = self.representation.invert_representation(target["representation"])

        self.metric.update(pred, target)

    @property
    def name(self):
        return self.metric.name

    def compute(self):
        return self.metric.compute()

    def plot(self):
        return self.metric.plot()

    def reset(self):
        self.metric.reset()
