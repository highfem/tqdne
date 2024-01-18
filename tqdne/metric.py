import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from torchmetrics import Metric


def to_numpy(x):
    return x.numpy(force=True) if isinstance(x, Tensor) else x


class SamplePlot(Metric):
    """Plot a sample of the predicted signal."""

    def __init__(self, fs):
        super().__init__()
        self.fs = fs
        self.add_state("reconstructed", default=[], dist_reduce_fx=None)

    def update(self, pred, target):
        self.reconstructed = to_numpy(pred["high_res"][0])

    def compute(self):
        return None

    def plot(self):
        time = np.arange(0, self.reconstructed.shape[-1]) / self.fs
        fig, axs = plt.subplots(3, 1, figsize=(9, 6))
        for i, ax in enumerate(axs):
            ax.plot(
                time,
                self.reconstructed[i],
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
        self.low_res = to_numpy(target["low_res"][0])
        self.high_res = to_numpy(target["high_res"][0])
        self.reconstructed = to_numpy(pred["high_res"][0])

    def compute(self):
        return None

    def plot(self):
        time = np.arange(0, self.low_res.shape[-1]) / self.fs
        fig, axs = plt.subplots(3, 1, figsize=(9, 6))
        for i, ax in enumerate(axs):
            ax.plot(time, self.low_res[i], "b", label="Input")
            ax.plot(time, self.high_res[i], "r", label="Target")
            ax.plot(
                time,
                self.reconstructed[i],
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

    def __init__(self, per_channel=False):
        super().__init__()
        self.per_channel = per_channel
        self.add_state("pred", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

    def update(self, pred, target):
        self.pred.append(to_numpy(pred["high_res"]))
        self.target.append(to_numpy(target["high_res"]))

    def compute(self):
        pred = np.concatenate(self.pred)
        target = np.concatenate(self.target)
        if self.per_channel:
            return {
                f"MSE - Channel {i}": ((pred[:, i] - target[:, i]) ** 2).mean()
                for i in range(pred.shape[1])
            }
        else:
            return {"MSE": ((pred - target) ** 2).mean()}


class PowerSpectralDensityFID(Metric):
    """Compute the Frechét Inception Distance between the power spectral density distributions of the predicted and target signals.

    Args:
        fs (int): The sampling frequency of the signals.
    """

    def __init__(self, fs, per_channel=False):
        super().__init__()
        self.fs = fs
        self.per_channel = per_channel
        self.add_state("pred_psd", default=[], dist_reduce_fx=None)
        self.add_state("target_psd", default=[], dist_reduce_fx=None)

    def update(self, pred, target):
        pred = to_numpy(pred["high_res"])
        target = to_numpy(target["high_res"])
        pred_psd = np.abs(np.fft.rfft(pred, axis=-1)) ** 2
        target_psd = np.abs(np.fft.rfft(target, axis=-1)) ** 2
        self.pred_psd.append(pred_psd)
        self.target_psd.append(target_psd)
        self.sig_len = pred.shape[-1]

    def compute(self):
        pred_psd = np.concatenate(self.pred_psd)
        target_psd = np.concatenate(self.target_psd)

        # Compute mean and std of PSD in log scale
        mean_dim = 0 if self.per_channel else (0, 1)
        pred_mean = np.log(pred_psd).mean(axis=mean_dim)
        target_mean = np.log(target_psd).mean(axis=mean_dim)
        pred_std = np.log(pred_psd).std(axis=mean_dim)
        target_std = np.log(target_psd).std(axis=mean_dim)

        # Frechét distance between isotropic Gaussians (Wasserstein-2)
        distance = np.sum((pred_mean - target_mean) ** 2, axis=-1) + np.sum(
            pred_std**2 + target_std**2 - 2 * pred_std * target_std, axis=-1
        )

        if self.per_channel:
            return {f"PSD FID - Channel {i}": d for i, d in enumerate(distance)}
        else:
            return {"PSD FID": distance}

    def plot(self):
        pred_psd = np.concatenate(self.pred_psd)
        target_psd = np.concatenate(self.target_psd)

        # Compute mean and std of PSD in log scale
        mean_dim = 0 if self.per_channel else (0, 1)
        pred_mean = np.log(pred_psd).mean(axis=mean_dim)
        target_mean = np.log(target_psd).mean(axis=mean_dim)
        pred_std = np.log(pred_psd).std(axis=mean_dim)
        target_std = np.log(target_psd).std(axis=mean_dim)

        # Frequency axis
        freqs = np.fft.rfftfreq(self.sig_len, 1 / self.fs)

        # Plot
        if self.per_channel:
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
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Log PSD")
                ax.set_title(f"Power Spectral Density - Channel {i}")

            # common legend for all subplots
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=3)

        else:
            fig, ax = plt.subplots(1, 1, figsize=(9, 6))
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
            ax.set_title("Power Spectral Density")
            ax.legend()

        fig.tight_layout()
        return fig
