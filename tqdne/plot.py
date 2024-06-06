from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tqdne import utils
from tqdne.metric import Metric
from tqdne.representations import to_numpy

# TODO: MERGE ALL WITH METRIC.PY

def get_plots_list(plots_config, metrics, general_config, data_representation=None):
    plots = []
    for plot, v_plot in plots_config.items():
        if plot == "bin":
            if v_plot.metrics == "all":
                for m in metrics:
                    plots.append(BinPlot(metric=m, num_mag_bins=v_plot.num_mag_bins, num_dist_bins=v_plot.num_dist_bins))
            elif v_plot.metrics == "channels-avg":
                #for i in range(0, len(metrics), general_config.num_channels):
                #    metric_avg = np.mean(metrics[i:i+general_config.num_channels])
                #    plots.append(BinPlot())
                # TODO: it should be handled by BinPlot itself 
                raise NotImplementedError("channels-avg not implemented yet")
        else:
            if v_plot == -1:
                channels = [c for c in range(general_config.num_channels)]
            else:
                channels = [v_plot]    
            if plot == "psd":
                for c in channels:
                    plots.append(PowerSpectralDensityPlot(fs=general_config.fs, channel=c, data_representation=data_representation))
            elif plot == "sample":
                for c in channels:
                    plots.append(SamplePlot(fs=general_config.fs, channel=c, data_representation=data_representation))
            elif plot == "logenv":
                for c in channels:
                    plots.append(LogEnvelopePlot(fs=general_config.fs, channel=c, data_representation=data_representation)) 
            elif plot == "debug":
                data_repr_channels = data_representation.get_shape((1, general_config.num_channels, general_config.signal_length))[1]
                channels = [c for c in range(data_repr_channels)]
                for c in channels:
                    plots.append(SamplePlot(fs=general_config.fs, channel=c, data_representation=data_representation, invert_representation=False))              
            else:
                raise ValueError(f"Unknown metric name: {plot}")
    return plots    


# TODO: maybe rename into plots.py
class Plot(ABC):
    """Abstract plot class.

    All plots should inherit from this class.
    """

    def __init__(self, channel=None, data_representation=None, invert_representation=True, metric=None):
        if metric is not None:
            # TODO: rethink this, because metric.compute return a scalar (FID)
            self.metric = metric
        else:    
            self.metric = None
            self.channel = channel
            assert not invert_representation or data_representation is not None, "Data representation must be provided if invert_representation is True"
            self.data_representation = data_representation
            self.invert_representation = invert_representation
            self.invert_representation_fun = data_representation.invert_representation if invert_representation else to_numpy  

    @property
    def name(self):
        name = self.__class__.__name__
        name = f"{name} - Raw Output" if not self.invert_representation else name
        name = f"{name} - Channel {self.channel}" if self.channel is not None else name
        return name

    def __call__(self, preds, target=None, cond_signal=None, cond=None):
        preds = self.invert_representation_fun(preds)
        target = self.invert_representation_fun(target) 
        cond = to_numpy(cond)
        if self.channel is not None:
            preds = preds[:, self.channel]
            target = target[:, self.channel]
            cond_signal = cond_signal[:, self.channel] if cond_signal is not None else None

        return self.plot(preds, target, cond_signal, cond)

    @abstractmethod
    def plot(self, preds, target=None, cond_signal=None, cond=None):
        pass


class SamplePlot(Plot):
    """Plot a sample of the predicted signal."""

    def __init__(self, fs=100, channel=0, data_representation=None, invert_representation=True):
        super().__init__(channel, data_representation, invert_representation)
        self.fs = fs

    def plot(self, preds, target=None, cond_signal=None, cond=None):
        time = np.arange(0, preds.shape[-1]) / self.fs
        fig, ax = plt.subplots(figsize=(9, 6))

        if preds[0].ndim == 1:
            ax.plot(time, preds[0], label="Generated")
            if not self.invert_representation:
                assert target[0].ndim == 1, "Target must be 1D too"
                ax.plot(time, target[0], alpha=0.5, label="Target")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
        elif preds[0].ndim == 2:
            ax.imshow(preds[0], aspect="auto", origin="lower")
            if not self.invert_representation:
                assert target[0].ndim == 2, "Target must be 2D too"
                ax.imshow(target[0], aspect="auto", origin="lower", alpha=0.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")    
        else:
            raise ValueError("Invalid shape for preds (and target): must be 1D or 2D but got {preds.ndim}D")

        title = f"{self.name}\n{', '.join([f'{key}: {value:.1f}' for key, value in utils.get_cond_params_dict(cond[0]).items()])}" if cond is not None else self.name
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        return fig


class UpsamplingSamplePlot(Plot):
    """Plot a sample of the input, target, and reconstructed signals."""

    def __init__(self, fs=100, channel=0, data_representation=None, invert_representation=True):
        super().__init__(channel, data_representation)
        self.fs = fs

    def plot(self, preds, target, cond_signal, cond=None):
        time = np.arange(0, preds.shape[-1]) / self.fs
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(time, cond_signal[0], "b", label="Input")
        ax.plot(time, target[0], label="Target")
        ax.plot(time, preds[0], label="Reconstructed")
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

    def plot(self, preds, target=None, cond_signal=None, cond=None):

        preds_logenv = utils.get_log_envelope(preds)
        target_logenv = utils.get_log_envelope(target)

        # TODO: mean or median?

        preds_logenv_median = np.median(preds_logenv, axis=0)
        preds_logenv_p25 = np.percentile(preds_logenv, 25, axis=0)
        preds_logenv_p75 = np.percentile(preds_logenv, 75, axis=0)
        target_logenv_median = np.median(target_logenv, axis=0)
        target_logenv_p25 = np.percentile(target_logenv, 25, axis=0)
        target_logenv_p75 = np.percentile(target_logenv, 75, axis=0)

        time_ax = np.arange(0, len(preds_logenv_median)) / self.fs
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(time_ax, preds_logenv_median, label="Generated - median")
        ax.fill_between(time_ax, preds_logenv_p25, preds_logenv_p75, alpha=0.2, label="Generated - IQR (25-75%)")
        ax.plot(time_ax, target_logenv_median, label="Target - median")
        ax.fill_between(time_ax, target_logenv_p25, target_logenv_p75, alpha=0.2, label="Target - IQR (25-75%)")

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

    def plot(self, preds, target, cond_signal=None, cond=None, eps=1e-7):
        fig, ax = plt.subplots(figsize=(9, 6))
        if self.metric is not None:
            preds_mean, preds_std, target_mean, target_std = self.metric.get_preds_and_target_stats()
        else:  
            target = preds if target is None else target
            target_psd = np.abs(np.fft.rfft(target, axis=-1)) ** 2
            target_mean = np.log(target_psd + eps).mean(axis=0)
            target_std = np.log(target_psd + eps).std(axis=0)

            freq = np.fft.rfftfreq(target.shape[-1], d=1 / self.fs)

            if preds is not None and target is not None:
                preds_psd = np.abs(np.fft.rfft(preds, axis=-1)) ** 2
                preds_mean = np.log(preds_psd + eps).mean(axis=0)
                preds_std = np.log(preds_psd + eps).std(axis=0)
                ax.plot(freq, preds_mean, label="Predicted")
                ax.fill_between(freq, preds_mean - preds_std, preds_mean + preds_std, alpha=0.2)


        ax.plot(freq, target_mean, label="Target")
        ax.fill_between(freq, target_mean - target_std, target_mean + target_std, alpha=0.2)
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

    def plot(self, preds, target, cond_signal, cond):
        # create the bins
        mag_bins = np.linspace(self.min_mag, self.max_mag, self.num_mag_bins + 1)
        dist_bins = np.linspace(self.min_dist, self.max_dist, self.num_dist_bins + 1)
        results = np.zeros((self.num_dist_bins, self.num_mag_bins))

        def _fill_bins(preds_wf, preds_cond, target_wf, target_cond):
            preds_mag = preds_cond[:, 2]
            preds_dist = preds_cond[:, 0]
            target_mag = target_cond[:, 2]
            target_dist = target_cond[:, 0]

            for i in range(self.num_mag_bins):
                for j in range(self.num_dist_bins):
                    preds_mask = (preds_mag >= mag_bins[i]) & (preds_mag < mag_bins[i + 1])
                    preds_mask &= (preds_dist >= dist_bins[j]) & (preds_dist < dist_bins[j + 1])
                    target_mask = (target_mag >= mag_bins[i]) & (target_mag < mag_bins[i + 1])
                    target_mask &= (target_dist >= dist_bins[j]) & (target_dist < dist_bins[j + 1])
                    results[j, i] = self.metric(preds_wf[preds_mask], target_wf[target_mask]) if preds_mask.any() else np.nan

        if isinstance(preds, dict) and isinstance(target, dict):
            preds_wf = preds["waveforms"]
            preds_cond = preds["cond"]
            target_wf = target["waveforms"]
            target_cond = target["cond"]
            _fill_bins(preds_wf, preds_cond, target_wf, target_cond)
        elif isinstance(preds, np.ndarray) and isinstance(target, np.ndarray):
            assert cond is not None, "Conditioning inputs must be provided if preds and target are numpy arrays"
            _fill_bins(preds, cond, target, cond) 
        else:
            raise ValueError("Invalid input format")     

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
