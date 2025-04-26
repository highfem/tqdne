import torch as th
import numpy as np


def asd_loss(pred, target, log_eps=1e-8, weight=1):
    def spectral_density(signal):
        sd = np.abs(np.fft.rfft(signal, axis=-1))
        log_sd = np.log(np.clip(sd, log_eps, None))
        return log_sd

    pred_sd = spectral_density(pred.detach().cpu())
    target_sd = spectral_density(target.detach().cpu())

    loss = weight * th.square(pred_sd - target_sd)
    return loss