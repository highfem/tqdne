from abc import ABC, abstractmethod

import numpy as np
import torch as th
from scipy import linalg

from tqdne.representation import Representation

from .classifier import LithningClassifier
from .utils import to_numpy


def frechet_distance(x: np.ndarray, y: np.ndarray, isotropic=False, eps=1e-6):
    """Compute the Frechet Distance between two sets of samples."""

    mu_x = x.mean(0)
    mu_y = y.mean(0)

    if isotropic:
        std_x = x.std(0)
        std_y = y.std(0)
        return np.sum((mu_x - mu_y) ** 2) + np.sum((std_x - std_y) ** 2)

    cov_x = np.cov(x, rowvar=False)
    cov_y = np.cov(y, rowvar=False)

    # # Product might be almost singular
    covmean, _ = linalg.sqrtm(cov_x @ cov_y, disp=False)
    if not np.isfinite(covmean).all():
        print(
            f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
        )
        offset = np.eye(cov_x.shape[0]) * eps
        covmean = linalg.sqrtm((cov_x + offset) @ (cov_y + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    return np.sum((mu_x - mu_y) ** 2) + np.trace(cov_x) + np.trace(cov_y) - 2 * np.trace(covmean)


class Metric(ABC):
    """Abstract metric class."""

    def __init__(self, channel=0):
        self.channel = channel

    @property
    def name(self):
        name = self.__class__.__name__
        return f"{name} - Channel {self.channel}"

    def __call__(self, pred, target):
        pred = to_numpy(pred)
        target = to_numpy(target)
        if self.channel is not None:
            pred = pred[:, self.channel]
            target = target[:, self.channel]
        return self.compute(pred, target)

    @abstractmethod
    def compute(self, pred, target):
        pass


class MeanSquaredError(Metric):
    def compute(self, pred, target):
        return ((pred - target) ** 2).mean()


class AmplitudeSpectralDensity(Metric):
    """Amplitude Spectral Density (ASD) metric.

    Computes the Frechet Distance between the amplitude spectral densities of two signals.
    """

    def __init__(self, fs, channel=0, log_eps=1e-8, isotropic=False):
        super().__init__(channel)
        self.fs = fs
        self.log_eps = log_eps
        self.isotropic = isotropic

    def spectral_density(self, signal):
        sd = np.abs(np.fft.rfft(signal, axis=-1))
        log_sd = np.log(np.clip(sd, self.log_eps, None))
        return log_sd

    def compute(self, pred, target):
        pred_sd = self.spectral_density(pred)
        target_sd = self.spectral_density(target)

        return frechet_distance(pred_sd, target_sd, isotropic=self.isotropic)


class NeuralMetric(Metric):
    """Abstract neural metric class.

    Used to compute metrics based on the output of a pre-trained classifier.

    Parameters
    ----------
    classifier : LithningClassifier
        Pre-trained classifier.
    representation : Representation, optional
        Representation to convert the input to before passing it to the classifier.
        Needs to be the same representation used to train the classifier.
    batch_size : int, optional
        Batch size used by the classifier.
        If None, the entire input is passed to the classifier at once.
    """

    def __init__(
        self,
        classifier: LithningClassifier,
        representation: Representation,
        batch_size: None | int = None,
    ):
        self.classifier = classifier.eval()
        self.representation = representation
        self.batch_size = batch_size

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, pred, target=None):
        pred = self.representation.get_representation(pred)
        pred = th.tensor(pred, device=self.classifier.device)
        if target is not None:
            target = self.representation.get_representation(target)
            target = th.tensor(target, device=self.classifier.device)
        return self.compute(pred, target)


class FrechetInceptionDistance(NeuralMetric):
    """Frechet Inception Distance (FID) metric."""

    @th.no_grad()
    def compute(self, pred, target):
        pred_emb = np.concatenate(
            [
                self.classifier.embed(pred[i : i + self.batch_size]).numpy(force=True)
                for i in range(0, len(pred), self.batch_size)
            ]
        )
        target_emb = np.concatenate(
            [
                self.classifier.embed(target[i : i + self.batch_size]).numpy(force=True)
                for i in range(0, len(target), self.batch_size)
            ]
        )

        return frechet_distance(pred_emb, target_emb)


class InceptionScore(NeuralMetric):
    """Inception Score (IS) metric."""

    @th.no_grad()
    def compute(self, pred, target=None):
        prob = np.concatenate(
            [
                th.softmax(self.classifier(pred[i : i + self.batch_size]), dim=-1).numpy(force=True)
                for i in range(0, len(pred), self.batch_size)
            ]
        )
        marginal = prob.mean(axis=0)
        kl = np.sum(prob * (np.log(prob) - np.log(marginal)), axis=-1)
        return np.exp(kl.mean())
