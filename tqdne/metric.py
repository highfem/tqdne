from abc import ABC, abstractmethod

import numpy as np

from scipy import signal

from tqdne.conf import Config
from tqdne.representations import to_numpy


def get_metrics_list(metrics_config, general_config, data_representation=None):
    metrics = []
    for metric, v in metrics_config.items():
        if v == -1:
            channels = [c for c in range(general_config.num_channels)]
        else:
            channels = [v]    
        if metric == "psd":
            for c in channels:
                metrics.append(PowerSpectralDensity(fs=general_config.fs, channel=c, data_representation=data_representation))
        elif metric == "logenv":
            for c in channels:
                metrics.append(LogEnvelope(channel=c, data_representation=data_representation))        
        elif metric == "mse":
            for c in channels:
                metrics.append(MeanSquaredError(channel=c, data_representation=data_representation))
        else:
            raise ValueError(f"Unknown metric name: {metric}")
    return metrics    




class Metric(ABC):
    """Abstract metric class.

    All metrics should inherit from this class.
    """

    def __init__(self, channel=None, data_representation=None, invert_representation=True):
        self.channel = channel
        self.data_representation = data_representation
        self.invert_representation = invert_representation
        assert data_representation is not None or invert_representation is False, "invert_representation can only be True if data_representation is not None"
        self.invert_representation_fun = data_representation.invert_representation if invert_representation else to_numpy  
        self.pred_mean = None, 
        self.pred_std = None,
        self.target_mean = None,
        self.target_std = None

    @property
    def name(self):
        name = self.__class__.__name__
        return f"{name} - Channel {self.channel}" if self.channel is not None else name

    def __call__(self, pred, target):
        if self.data_representation is not None:
            pred = self.invert_representation_fun(pred)
            target = self.invert_representation_fun(target) 
        if self.channel is not None:
            pred = pred[:, self.channel]
            target = target[:, self.channel]
        return self.compute(pred, target)

    @abstractmethod
    def compute(self, pred, target):
        pass

    def get_pred_and_target_stats(self):
        return self.pred_mean, self.pred_std, self.target_mean, self.target_std


class MeanSquaredError(Metric):
    def compute(self, pred, target):
        return ((pred - target) ** 2).mean()


class PowerSpectralDensity(Metric):
    def __init__(self, fs, channel=0, data_representation=None, invert_representation=True):
        super().__init__(channel, data_representation, invert_representation)
        self.fs = fs

    def compute(self, pred, target):
        pred_psd = np.abs(np.fft.rfft(pred, axis=-1)) ** 2
        target_psd = np.abs(np.fft.rfft(target, axis=-1)) ** 2

        # Compute mean and std of PSD in log scale
        eps = 1e-7
        self.pred_mean = np.log(pred_psd + eps).mean(axis=0)
        self.target_mean = np.log(target_psd + eps).mean(axis=0)
        self.pred_std = np.log(pred_psd + eps).std(axis=0)
        self.target_std = np.log(target_psd + eps).std(axis=0)

        # Frechét distance between isotropic Gaussians (Wasserstein-2)
        fid = np.sum((self.pred_mean - self.target_mean) ** 2, axis=-1) + np.sum(
            self.pred_std**2 + self.target_std**2 - 2 * self.pred_std * self.target_std, axis=-1
        )

        return fid
    

class LogEnvelope(Metric):
    def __init__(self, channel=0, data_representation=None, invert_representation=True):
        super().__init__(channel, data_representation, invert_representation)

    @staticmethod
    def _get_moving_avg_envelope(x, window_len=50):
        return signal.convolve(np.abs(x), np.ones((x.shape[0], window_len)), mode='same') / window_len    
    
    @staticmethod
    def get_log_envelope(data: np.ndarray, env_function=_get_moving_avg_envelope, env_function_params={}, eps=1e-7):
        """
        Get the log envelope of the given data.

        Args:
            data (np.ndarray): The input data.
            env_function (str): The envelope function to use.
            env_function_params (dict, optional): The parameters for the envelope function. Defaults to None.
            eps (float, optional): A small value to add to the envelope before taking the logarithm. Defaults to 1e-7.

        Returns:
            np.ndarray: The log envelope of the input data.
        """
        return np.log(env_function(data, **env_function_params) + eps)        

    def compute(self, pred, target):
        
        pred_logenv = self.get_log_envelope(pred)
        target_logenv = self.get_log_envelope(target)

        self.pred_mean = pred_logenv.mean(axis=0)
        self.target_mean = target_logenv.mean(axis=0)
        self.pred_std = pred_logenv.std(axis=0)
        self.target_std = target_logenv.std(axis=0)

        # Frechét distance between isotropic Gaussians (Wasserstein-2)
        fid = np.sum((self.pred_mean - self.target_mean) ** 2, axis=-1) + np.sum(
            self.pred_std**2 + self.target_std**2 - 2 * self.pred_std * self.target_std, axis=-1
        )

        return fid
