# TODO: MERGE ALL WITH METRIC.PY


from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tqdne import utils
from tqdne.metric import Metric, LogEnvelope, PowerSpectralDensity


from tqdne.representations import to_numpy

def get_plots_list(plots_config, metrics, general_config, data_representation=None):
    """
    Generate a list of plots based on the provided configuration.

    Args:
        plots_config (dict): A dictionary containing the plot configurations.
        metrics (list): A list of metrics to be used in the plots.
        general_config (object): An object containing general configuration settings.
        data_representation (object, optional): An object representing the data representation. Defaults to None.

    Returns:
        list: A list of plot objects.

    Raises:
        NotImplementedError: If the 'channels-avg' metric is not implemented yet.
        ValueError: If an unknown metric name is encountered.

    """
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
                # TODO: it should be handled by BinPlot itself 
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

    @property
    def name(self):
        name = self.__class__.__name__
        name = f"{name} - Raw Output" if not self.invert_representation else name
        name = f"{name} - Channel {self.channel}" if self.channel is not None else name
        return name

    def __call__(self, preds, target=None, cond=None):
        if self.invert_representation:
            preds = self.data_representation.invert_representation(preds)
            target = to_numpy(target["waveform"]) if isinstance(target, dict) else self.data_representation.invert_representation(target)
        else:
            preds = to_numpy(preds)
            try:
                target = to_numpy(target["repr"]) if isinstance(target, dict) else to_numpy(target)
            except:
                target = to_numpy(target)
        cond = to_numpy(cond)
        if self.channel is not None:
            preds = preds[:, self.channel]
            target = target[:, self.channel]
            #cond_signal = cond_signal[:, self.channel] if cond_signal is not None else None

        return self.plot(preds, target, cond)

    @abstractmethod
    def plot(self, preds, target=None, cond=None):
        pass


class SamplePlot(Plot):
    """Plot a sample of the predicted signal."""

    def __init__(self, fs=100, channel=0, data_representation=None, invert_representation=True):
        super().__init__(channel, data_representation, invert_representation)
        self.fs = fs

    def plot(self, preds, target=None, cond=None):
        title = f"{self.name}\n{', '.join([f'{key}: {value:.1f}' for key, value in utils.get_cond_params_str(cond[0]).items()])}" if cond is not None else self.name
        if preds[0].ndim == 1:
            fig, ax = plt.subplots(figsize=(9, 6))
            time = np.arange(0, preds.shape[-1]) / self.fs
            ax.plot(time, preds[0], label="Generated")
            assert target[0].ndim == 1, "Target must be 1D too"
            ax.plot(time, target[0], alpha=0.5, label="Target")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Amplitude")
            ax.set_title(title)
            ax.legend()
        elif preds[0].ndim == 2:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(preds[0], aspect="auto", origin="lower", cmap="viridis")
            axs[0].set_title("Generated")
            assert target[0].ndim == 2, "Target must be 2D too"
            axs[1].imshow(target[0], aspect="auto", origin="lower", cmap="viridis")
            axs[1].set_title("Target")
            for ax in axs:
                ax.set_xlabel("Time bins")
                ax.set_ylabel("Frequency bins")
            fig.suptitle(title)    
        else:
            raise ValueError("Invalid shape for preds (and target): must be 1D or 2D but got {preds.ndim}D")

        fig.tight_layout()
        return fig


class LogEnvelopePlot(Plot):
    def __init__(self, fs=100, channel=0, data_representation=None, invert_representation=True):
        super().__init__(channel, data_representation, invert_representation)
        self.fs = fs

    def plot(self, preds, target=None, cond=None):

        preds_logenv = LogEnvelope.get_log_envelope(preds)
        target_logenv = LogEnvelope.get_log_envelope(target)

        preds_logenv_median = np.median(preds_logenv, axis=0)
        preds_logenv_p15 = np.percentile(preds_logenv, 15, axis=0)
        preds_logenv_p85 = np.percentile(preds_logenv, 85, axis=0)
        target_logenv_median = np.median(target_logenv, axis=0)
        target_logenv_p15 = np.percentile(target_logenv, 15, axis=0)
        target_logenv_p85 = np.percentile(target_logenv, 85, axis=0)

        time_ax = np.arange(0, len(preds_logenv_median)) / self.fs
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(time_ax, preds_logenv_median, label="Generated - median")
        ax.fill_between(time_ax, preds_logenv_p15, preds_logenv_p85, alpha=0.2, label="Generated - IQR (15-85%)")
        ax.plot(time_ax, target_logenv_median, label="Target - median")
        ax.fill_between(time_ax, target_logenv_p15, target_logenv_p85, alpha=0.2, label="Target - IQR (15-85%)")

        ax.set_title(self.name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Log Envelope")
        ax.legend()
        fig.tight_layout()
        return fig
        

class PowerSpectralDensityPlot(Plot):
    def __init__(self, fs=100, channel=0, data_representation=None, invert_representation=True):
        super().__init__(channel, data_representation, invert_representation)
        self.fs = fs

    def plot(self, preds, target, cond=None, eps=1e-7):
        fig, ax = plt.subplots(figsize=(9, 6))
        if self.metric is not None:
            preds_mean, preds_std, target_mean, target_std = self.metric.get_preds_and_target_stats()
        else:  
            target = preds if target is None else target
            target_psd = PowerSpectralDensity.get_power_spectral_density(target, log_scale=True)
            target_mean = target_psd.mean(axis=0)
            target_std = target_mean.std(axis=0)

            freq = np.fft.rfftfreq(target.shape[-1], d=1 / self.fs)

            if preds is not None and target is not None:
                preds_psd = PowerSpectralDensity.get_power_spectral_density(preds, log_scale=True)
                preds_mean = preds_psd.mean(axis=0)
                preds_std = preds_psd.std(axis=0)
                ax.plot(freq, preds_mean, label="Predicted")
                ax.fill_between(freq, preds_mean - preds_std, preds_mean + preds_std, alpha=0.2)


        ax.plot(freq, target_mean, label="Target")
        ax.fill_between(freq, target_mean - target_std, target_mean + target_std, alpha=0.2)
        ax.set_title(self.name)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power Spectral Density [dB]")
        ax.legend()
        fig.tight_layout()
        return fig
        

class BinPlot(Plot):
    """Creates a bin plot for a given metric."""

    def __init__(
        self,
        metric: Metric,
        mag_bins: list[tuple],
        dist_bins: list[tuple],
        fmt=".2f",
        title=None,
    ):
        super().__init__(data_representation=None, invert_representation=False)
        self.metric = metric
        self.mag_bins = mag_bins
        self.dist_bins = dist_bins
        self.fmt = fmt
        self.title = title

    @property
    def name(self):
        if self.title:
            return self.title
        return f"Bin {self.metric.name}"

    def plot(self, preds, target, cond=None):

        results = np.zeros((len(self.dist_bins), len(self.mag_bins)))

        def _fill_bins(preds_wf, preds_cond, target_wf, target_cond):
            preds_mag = preds_cond[:, 2]
            preds_dist = preds_cond[:, 0]
            target_mag = target_cond[:, 2]
            target_dist = target_cond[:, 0]

            for i in range(len(self.mag_bins)):
                for j in range(len(self.dist_bins)):
                    preds_mask = (preds_mag >= self.mag_bins[i][0]) & (preds_mag < self.mag_bins[i][1])
                    preds_mask &= (preds_dist >= self.dist_bins[j][0]) & (preds_dist < self.dist_bins[j][1])
                    target_mask = (target_mag >= self.mag_bins[i][0]) & (target_mag < self.mag_bins[i][1])
                    target_mask &= (target_dist >= self.dist_bins[j][0]) & (target_dist < self.dist_bins[j][1])
                    results[j, i] = self.metric(preds_wf[preds_mask], target_wf[target_mask]) if preds_mask.sum() > 1 else np.nan

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
            raise ValueError(f"Invalid input format: preds and target must be both dictionaries or both numpy arrays, got {type(preds)} for preds and {type(target)} for target.")   

        # Plotting the heatmap using seaborn
        plot = sns.heatmap(results, annot=True, fmt=self.fmt, cmap="viridis")
        plot.set_xticks(np.arange(len(self.mag_bins) + 1))
        plot.set_xticklabels([m[0] for m in self.mag_bins] + [self.mag_bins[-1][1]])
        plot.set_yticks(np.arange(len(self.dist_bins) + 1))
        plot.set_yticklabels([d[0] for d in self.dist_bins] + [self.dist_bins[-1][1]])
        plot.invert_yaxis()
        plot.set_xlabel("Magnitude bin")
        plot.set_ylabel("Distance bin [km]")
        plot.set_title(self.name)
        fig = plot.get_figure()
        fig.tight_layout()
        return fig
