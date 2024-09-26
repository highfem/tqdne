from abc import ABC, abstractmethod

import numpy as np

from scipy import signal
from scipy.linalg import sqrtm

from tqdne.representations import to_numpy


@staticmethod
def get_metrics_list(metrics_config, general_config, data_representation=None):
    """
    Get a list of metric objects based on the given metrics configuration.

    Args:
        metrics_config (dict): A dictionary containing the metrics configuration.
        general_config (object): An object containing general configuration settings.
        data_representation (object, optional): An object representing the data representation. Defaults to None.

    Returns:
        list: A list of metric objects.

    Raises:
        ValueError: If an unknown metric name is encountered in the metrics configuration.
    """
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
        elif metric == "mean":
            for c in channels:
                metrics.append(SignalMeanMSE(channel=c, data_representation=data_representation))        
        else:
            raise ValueError(f"Unknown metric name: {metric}")
    return metrics


# TODO: remove
@staticmethod
def _calculate_frechet_distance(
    mu1,
    sigma1,
    mu2,
    sigma2,
    eps=1e-6,
):
    """
    Calculate the Frechet Distance between two multivariate Gaussian distributions.

    Args:
        mu1 (Tensor): The mean of the first distribution.
        sigma1 (Tensor): The covariance matrix of the first distribution.
        mu2 (Tensor): The mean of the second distribution.
        sigma2 (Tensor): The covariance matrix of the second distribution.
        eps (float, optional): A small value to add to the diagonal of the covariance matrices. Defaults to 1e-6.

    Returns:
        tensor: The Fréchet Distance between the two distributions.
    """
    import torch
    # Compute the squared distance between the means
    mean_diff = mu1 - mu2
    mean_diff_squared = mean_diff.square().sum(dim=-1)

    # Compute the eigenvalues of the matrix product of the real and fake covariance matrices
    sigma_mm = torch.matmul(sigma1, sigma2)
    eigenvals = torch.linalg.eigvals(sigma_mm)

    # Check if there are large negative eigenvalues
    for i in range(5):
        if not torch.allclose(eigenvals.imag, torch.tensor([0.0], dtype=eigenvals.imag.dtype), atol=1e-3):
            m = torch.max(torch.abs(eigenvals.imag))
            print(f"Imaginary component in eigenvalues: {m}")

            # Add a small value to the diagonal of the covariance matrices
            print(f"FID calculation produces singular product ({i}); adding {eps} to diagonal of cov estimates")
            offset = torch.eye(sigma1.shape[0]) * eps
            sigma1 = sigma1 + offset
            sigma2 = sigma2 + offset
            sigma_mm = torch.matmul(sigma1, sigma2)
            eigenvals = torch.linalg.eigvals(sigma_mm)
        else:
            break    

    # Take the square root of each eigenvalue and take its sum
    sqrt_eigenvals_sum = eigenvals.sqrt().real.sum(dim=-1)   

    # Calculate the sum of the traces of both covariance matrices
    trace_sum = sigma1.trace() + sigma2.trace() 

    # Calculate the Frechet Distance using the squared distance between the means,
    # the sum of the traces of the covariance matrices, and the sum of the square roots of the eigenvalues
    return mean_diff_squared + trace_sum - 2 * sqrt_eigenvals_sum

@staticmethod
def frechet_distance(x, y, eps=1e-6, torch_fun=True):
    """
    Compute the Frechet Distance between two sets of samples.

    Parameters:
        x (ndarray or tuple): The first set of samples. If a tuple, the first element should be the mean and the second element should be the covariance matrix.
        y (ndarray or tuple): The second set of samples. If a tuple, the first element should be the mean and the second element should be the covariance matrix.
        eps (float, optional): A small value to add to the covariance matrix. Defaults to 1e-9.

    Returns:
        float: The computed Frechet Distance.
    """        

    # TODO: remove or rearrange (it works in the sense it doesn't raise an error. Still have to test whether it works as expected.)
    if torch_fun:
        import torch
        if isinstance(x, tuple):
            mu1 = torch.tensor(x[0]) 
            sigma1 = torch.tensor(x[1])
        else:
            mu1 = torch.tensor(x.mean(axis=0))
            sigma1 = torch.tensor(np.cov(x, rowvar=False))
        if isinstance(y, tuple):
            mu2 = torch.tensor(y[0])
            sigma2 = torch.tensor(y[1])    
        else:
            mu2 = torch.tensor(y.mean(axis=0))
            sigma2 = torch.tensor(np.cov(y, rowvar=False))

        if sigma1.dim() < 2 or sigma2.dim() < 2:
            raise ValueError("Covariance matrices must have at least two dimensions")

        return _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    if isinstance(x, tuple):
        mu_x = x[0]
        cov_x = x[1]
    else:
        mu_x = x.mean(axis=0)
        cov_x = np.cov(x, rowvar=False)

    if isinstance(y, tuple):
        mu_y = y[0]
        cov_y = y[1]
    else:        
        mu_y = y.mean(axis=0)
        cov_y = np.cov(y, rowvar=False)

    # Product might be almost singular
    covmean, _ = sqrtm(cov_x @ cov_y, disp=False)
    if np.any(np.abs(covmean) > 1e10):
        msg = (
            "fid calculation produces singular product; " "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(cov_x.shape[0]) * eps
        covmean = sqrtm((cov_x + offset) @ (cov_y + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component in covmean: %s" % m)
        covmean = covmean.real

    return np.sum((mu_x - mu_y) ** 2) + np.trace(cov_x) + np.trace(cov_y) - 2 * np.trace(covmean)

@staticmethod
def compute_inception_score(preds, preds_2=None):
    """
    Computes the Inception Score for a given set of predictions.

    Args:
        preds (numpy.ndarray): Array of classification predictions.
        preds_2 (numpy.ndarray, optional): Array of additional predictions. Defaults to None.

    Returns:
        float: The computed Inception Score.
    """
    print(preds.shape)
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
    assert not np.any(prob_dist_1 < 0) and not np.any(prob_dist_2 < 0), "Probabilities must be non-negative"
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
        self.preds_mean = None, 
        self.preds_std = None,
        self.target_mean = None,
        self.target_std = None

    @property
    def name(self):
        name = self.__class__.__name__
        return f"{name} - Channel {self.channel}" if self.channel is not None else name

    def __call__(self, preds, target):
        if self.invert_representation:
            preds = self.data_representation.invert_representation(preds)
            target = to_numpy(target["waveform"]) if isinstance(target, dict) else self.data_representation.invert_representation(target)
        else:
            preds = to_numpy(preds)
            target = to_numpy(target["repr"]) if isinstance(target, dict) else to_numpy(target)
        if self.channel is not None:
            preds = preds[:, self.channel]
            target = target[:, self.channel]
        return self.compute(preds, target)

    @abstractmethod
    def compute(self, preds, target):
        pass

    def get_preds_and_target_stats(self):
        return self.preds_mean, self.preds_std, self.target_mean, self.target_std

class SignalMeanMSE(Metric):
    ''' Mean Squared Error between the mean of the predictions and the mean of the targets (sample-wise).'''
    def compute(self, preds, target):
        return ((preds.mean(axis=-1) - target.mean(axis=-1)) ** 2).mean()

class MeanSquaredError(Metric):
    ''' Mean Squared Error between the predictions and the targets (sample-wise).'''
    def compute(self, preds, target):
        return ((preds - target) ** 2).mean()

class PowerSpectralDensity(Metric):
    ''' Fréchet distance between the PSD of the predictions and the PSD of the targets.'''
    def __init__(self, fs, channel=0, data_representation=None, invert_representation=True):
        super().__init__(channel, data_representation, invert_representation)
        self.fs = fs

    @staticmethod
    def get_power_spectral_density(data: np.ndarray, log_scale=False, eps=1e-7):
        """
        Get the power spectral density of the given data.

        Args:
            data (np.ndarray): The input data.
            log_scale (bool, optional): Whether to return the PSD in log scale. Defaults to False.
            eps (float, optional): A small value to add to the PSD before taking the logarithm. Defaults to 1e-7.

        Returns:
            np.ndarray: The power spectral density of the input data.
        """
        psd = np.abs(np.fft.rfft(data, axis=-1)) ** 2
        return np.log(psd + eps) if log_scale else psd  

    def compute(self, preds, target):
        log_preds_psd = self.get_power_spectral_density(preds, log_scale=True)
        log_target_psd = self.get_power_spectral_density(target, log_scale=True)
        self.preds_mean = log_preds_psd.mean(axis=0)
        self.target_mean = log_target_psd.mean(axis=0)
        self.preds_cov = np.cov(log_preds_psd, rowvar=False)
        self.target_cov = np.cov(log_target_psd, rowvar=False)

        return frechet_distance((self.preds_mean, self.preds_cov), (self.target_mean, self.target_cov))
    

class LogEnvelope(Metric):
    ''' Fréchet distance between the log envelope of the predictions and the log envelope of the targets.'''
    def __init__(self, channel=0, data_representation=None, invert_representation=True):
        super().__init__(channel, data_representation, invert_representation)

    @staticmethod
    def _get_moving_avg_envelope(x, window_len=50):
        return signal.convolve(np.abs(x), np.ones((x.shape[0], window_len)), mode='same') / window_len    
    
    @staticmethod
    def get_log_envelope(data: np.ndarray, env_function=_get_moving_avg_envelope, env_function_params={}, log_scale=True, eps=1e-7):
        env = env_function(data, **env_function_params)
        return np.log10(env + eps) if log_scale else env   

    def compute(self, preds, target):
        preds_logenv = self.get_log_envelope(preds)
        targets_logenv = self.get_log_envelope(target)

        self.preds_mean = preds_logenv.mean(axis=0)
        self.targets_mean = targets_logenv.mean(axis=0)
        self.preds_cov = np.cov(preds_logenv, rowvar=False)
        self.targets_cov = np.cov(targets_logenv, rowvar=False)

        return frechet_distance((self.preds_mean, self.preds_cov), (self.targets_mean, self.targets_cov))

        
