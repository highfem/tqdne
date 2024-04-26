from abc import ABC, abstractmethod

import numpy as np

from scipy import signal

from tqdne.conf import Config
from tqdne.representations import to_numpy


@staticmethod
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


@staticmethod
def compute_fid(preds, target):
    """
    Compute the Fréchet Inception Distance (FID) between two sets of samples.

    Args:
        preds (numpy.ndarray): Predicted samples.
        target (numpy.ndarray): Target samples.

    Returns:
        float: The Fréchet Inception Distance (FID) between the two sets of samples.
    """
    preds_mean = np.mean(preds, axis=0)
    target_mean = np.mean(target, axis=0)
    preds_std = np.std(preds, axis=0)
    target_std = np.std(target, axis=0)
    # Frechét distance between isotropic Gaussians (Wasserstein-2)
    fid = np.sum((preds_mean - target_mean) ** 2, axis=-1) + np.sum(
        preds_std**2 + target_std**2 - 2 * preds_std * target_std, axis=-1
    )
    return fid

@staticmethod
def compute_inception_score(preds, preds_2=None):
    """
    Computes the Inception Score for a given set of predictions.

    Args:
        preds (numpy.ndarray): Array of predictions.
        preds_2 (numpy.ndarray, optional): Array of additional predictions. Defaults to None.

    Returns:
        float: The computed Inception Score.
    """
    if preds_2 is None:
        preds_1 = preds[: preds.shape[0] // 2]
        preds_2 = preds[preds.shape[0] // 2:]
    else:
        preds_1 = preds
    p_hat = np.sum(preds_1, axis=0) / preds_1.shape[0]
    kl_divergences = compute_kl_divergence(preds_2, p_hat)
    inception_score = np.exp(np.mean(kl_divergences))
    return inception_score


@staticmethod
def compute_kl_divergence(prob_dist_1, prob_dist_2, eps=1e-7):
    """
    Compute the Kullback-Leibler (KL) divergence between two probability distributions.

    Parameters:
        prob_dist_1 (numpy.ndarray): The first probability distribution.
        prob_dist_2 (numpy.ndarray): The second probability distribution.
        eps (float, optional): A small value to add to the denominator. Defaults to 1e-7.

    Returns:
        numpy.ndarray: The KL divergence between the two probability distributions.
    """
    return np.sum(prob_dist_1 * np.log((prob_dist_1 / (prob_dist_2 + eps)) + eps), axis=-1)


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
        self.preds_mean = None, 
        self.preds_std = None,
        self.target_mean = None,
        self.target_std = None

    @property
    def name(self):
        name = self.__class__.__name__
        return f"{name} - Channel {self.channel}" if self.channel is not None else name

    def __call__(self, preds, target):
        if self.data_representation is not None:
            preds = self.invert_representation_fun(preds)
            target = self.invert_representation_fun(target) 
        if self.channel is not None:
            preds = preds[:, self.channel]
            target = target[:, self.channel]
        return self.compute(preds, target)

    @abstractmethod
    def compute(self, preds, target):
        pass

    def get_preds_and_target_stats(self):
        return self.preds_mean, self.preds_std, self.target_mean, self.target_std


class MeanSquaredError(Metric):
    def compute(self, preds, target):
        return ((preds - target) ** 2).mean()


class PowerSpectralDensity(Metric):
    def __init__(self, fs, channel=0, data_representation=None, invert_representation=True):
        super().__init__(channel, data_representation, invert_representation)
        self.fs = fs

    def compute(self, preds, target):
        preds_psd = np.abs(np.fft.rfft(preds, axis=-1)) ** 2
        target_psd = np.abs(np.fft.rfft(target, axis=-1)) ** 2

        # Compute mean and std of PSD in log scale
        eps = 1e-7
        self.preds_mean = np.log(preds_psd + eps).mean(axis=0)
        self.target_mean = np.log(target_psd + eps).mean(axis=0)
        self.preds_std = np.log(preds_psd + eps).std(axis=0)
        self.target_std = np.log(target_psd + eps).std(axis=0)

        # Frechét distance between isotropic Gaussians (Wasserstein-2)
        fid = np.sum((self.preds_mean - self.target_mean) ** 2, axis=-1) + np.sum(
            self.preds_std**2 + self.target_std**2 - 2 * self.preds_std * self.target_std, axis=-1
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

    def compute(self, preds, target):
        
        pred_logenv = self.get_log_envelope(preds)
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

        
