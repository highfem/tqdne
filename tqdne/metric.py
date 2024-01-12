from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torchmetrics import Metric


class SamplePlot(Metric):
    """Plot a sample of the predicted signal."""

    def __init__(self, fs):
        super().__init__()
        self.fs = fs
        self.add_state("reconstructed", default=[], dist_reduce_fx=None)

    def update(self, pred, target):
        self.reconstructed = pred["high_res"][0]

    def compute(self):
        return None

    def plot(self):
        time = np.arange(0, self.reconstructed.shape[-1]) / self.fs
        fig, axs = plt.subplots(3, 1, figsize=(9, 6))
        for i, ax in enumerate(axs):
            ax.plot(
                time,
                self.reconstructed[i].numpy(force=True),
                "g",
                label="Reconstructed",
            )
            ax.set_title(f"Channel {i}")

        # common legend for all subplots
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3)
        plt.tight_layout()
        return fig


class UpsamplingSamplePlot(Metric):
    """Plot a sample of the input, target, and reconstructed signals."""

    def __init__(self, fs):
        super().__init__()
        self.fs = fs
        self.add_state("low_res", default=[], dist_reduce_fx=None)
        self.add_state("high_res", default=[], dist_reduce_fx=None)
        self.add_state("reconstructed", default=[], dist_reduce_fx=None)

    def update(self, pred, target):
        self.low_res = target["low_res"][0]
        self.high_res = target["high_res"][0]
        self.reconstructed = pred["high_res"][0]

    def compute(self):
        return None

    def plot(self):
        time = np.arange(0, self.low_res.shape[-1]) / self.fs
        fig, axs = plt.subplots(3, 1, figsize=(9, 6))
        for i, ax in enumerate(axs):
            ax.plot(time, self.low_res[i].numpy(force=True), "b", label="Input")
            ax.plot(time, self.high_res[i].numpy(force=True), "r", label="Target")
            ax.plot(
                time,
                self.reconstructed[i].numpy(force=True),
                "g",
                label="Reconstructed",
            )
            ax.set_title(f"Channel {i}")

        # common legend for all subplots
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3)
        plt.tight_layout()
        return fig


class MeanSquaredError(Metric):
    """Compute the mean squared error between the predicted and target signals."""

    def __init__(self):
        super().__init__()
        self.add_state("pred", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

    def update(self, pred, target):
        self.pred.append(pred["high_res"])
        self.target.append(target["high_res"])

    def compute(self):
        pred = th.cat(self.pred)
        target = th.cat(self.target)
        return {
            f"MSE - Channel {i}": ((pred[:, i] - target[:, i]) ** 2).mean()
            for i in range(pred.shape[1])
        }


class PowerSpectralDensityFID(Metric):
    """Compute the Frechét Inception Distance between the power spectral density distributions of the predicted and target signals.

    Args:
        fs (int): The sampling frequency of the signals.
    """

    def __init__(self, fs):
        super().__init__()
        self.fs = fs
        self.add_state("pred_psd", default=[], dist_reduce_fx=None)
        self.add_state("target_psd", default=[], dist_reduce_fx=None)

    def update(self, pred, target):
        pred_psd = th.fft.rfft(pred["high_res"], dim=-1).abs() ** 2
        target_psd = th.fft.rfft(target["high_res"], dim=-1).abs() ** 2
        self.pred_psd.append(pred_psd)
        self.target_psd.append(target_psd)
        self.sig_len = pred["high_res"].shape[-1]

    def compute(self):
        pred_psd = th.cat(self.pred_psd)
        target_psd = th.cat(self.target_psd)

        # Compute mean and std of PSD in log scale
        pred_mean = pred_psd.log().mean(dim=0)
        target_mean = target_psd.log().mean(dim=0)
        pred_std = pred_psd.log().std(dim=0)
        target_std = target_psd.log().std(dim=0)

        # Frechét distance between isotropic Gaussians (Wasserstein-2)
        distance = th.sum((pred_mean - target_mean) ** 2, dim=0) + th.sum(
            pred_std**2 + target_std**2 - 2 * pred_std * target_std, dim=0
        )

        return {f"PSD FID - Channel {i}": d for i, d in enumerate(distance)}

    def plot(self):
        pred_psd = th.cat(self.pred_psd)
        target_psd = th.cat(self.target_psd)

        # Compute mean and std of PSD in log scale
        pred_mean = pred_psd.log().mean(dim=0).numpy(force=True)
        target_mean = target_psd.log().mean(dim=0).numpy(force=True)
        pred_std = pred_psd.log().std(dim=0).numpy(force=True)
        target_std = target_psd.log().std(dim=0).numpy(force=True)

        # Frequency axis
        freqs = np.fft.rfftfreq(self.sig_len, 1 / self.fs)

        # Plot
        fig, axs = plt.subplots(3, 1, figsize=(9, 6))
        for i, ax in enumerate(axs):
            ax.plot(freqs, pred_mean[i], "g", label="Reconstructed")
            ax.fill_between(
                freqs,
                pred_mean[i] - pred_std[i],
                pred_mean[i] + pred_std[i],
                color="g",
                alpha=0.2,
            )
            ax.plot(freqs, target_mean[i], "r", label="Target")
            ax.fill_between(
                freqs,
                target_mean[i] - target_std[i],
                target_mean[i] + target_std[i],
                color="r",
                alpha=0.2,
            )
            ax.set_title(f"Channel {i}")

        # common legend for all subplots
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3)
        plt.tight_layout()
        return fig
