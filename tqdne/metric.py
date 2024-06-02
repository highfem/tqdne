from abc import ABC, abstractmethod

import numpy as np
import torch as th
from scipy import linalg

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
    """Abstract metric class.

    All metrics should inherit from this class.
    """

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


class NeuralMetric(ABC):
    """Abstract neural metric class.

    Used to compute metrics based on the output of a pre-trained classifier.
    """

    def __init__(self, classifier: LithningClassifier, batch_size):
        self.classifier = classifier.eval()
        self.batch_size = batch_size

    @th.no_grad()
    def __call__(self, pred, target=None):
        pred = th.tensor(pred, dtype=self.classifier.dtype, device=self.classifier.device)
        target = (
            th.tensor(target, dtype=self.classifier.dtype, device=self.classifier.device)
            if target is not None
            else None
        )

        emb = self.classifier.embed(pred)
        logits = self.classifier.output_layer(emb)
        prob = th.softmax(logits, dim=-1)

        return self.compute(prob.numpy(force=True), emb.numpy(force=True))

    @abstractmethod
    def compute(self, prob, emb):
        pass


class FrechetInceptionDistance(NeuralMetric):
    """Frechet Inception Distance (FID) metric."""

    def __call__(self, pred, target):
        pred = th.tensor(pred, dtype=self.classifier.dtype, device=self.classifier.device)
        target = th.tensor(target, dtype=self.classifier.dtype, device=self.classifier.device)
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

    def __call__(self, pred, target=None):
        pred = th.tensor(pred, dtype=self.classifier.dtype, device=self.classifier.device)
        prob = np.concatenate(
            [
                th.softmax(
                    self.classifier.output_layer(
                        self.classifier.embed(pred[i : i + self.batch_size])
                    ),
                    dim=-1,
                ).numpy(force=True)
                for i in range(0, len(pred), self.batch_size)
            ]
        )
        marginal = prob.mean(axis=0)
        kl = np.sum(prob * (np.log(prob) - np.log(marginal)), axis=-1)
        return np.exp(kl.mean())
