#!/usr/bin/env python3

import numpy as np
import torch

from tqdne.ganutils.random_fields import rand_noise

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

def get_synthetic_data(G, n_waveforms, sdat_train, dist, mag, args):
    """get_synthetic_data returns n=n_waveforms number of synthetic waveforms for the corresponding conditional variables.

    _extended_summary_

    :param G: Generator object to create waveforms
    :type G: Generator
    :param n_waveforms: number of waveforms to create
    :type n_waveforms: int
    :param n_cond_bins: number of distance bins to use
    :type n_cond_bins: int
    :param dist: conditional variable for distance
    :type dist: float
    :param dataset: _description_
    :type dataset: WaveDatasetDist
    :param noise_dim: _description_
    :type noise_dim: int
    :param lt: _description_
    :type lt: int
    :param dt: _description_
    :type dt: float
    :return: list of len n_waveforms of synthetic waveforms
    :rtype: list
    """
    # Create some extra waveforms to filter out mode collapse
    samples = 2 * n_waveforms

    dist_max = sdat_train.vc_max["dist"]
    mag_max = sdat_train.vc_max["mag"]

    vc_list = [
        dist / dist_max * torch.ones(samples, 1).cuda(),
        mag / mag_max * torch.ones(samples, 1).cuda(),
    ]

    grf = rand_noise(1, args.noise_dim, device=device)
    z = grf.sample(samples)

    G.eval()
    x_g, x_scaler = G(z, *vc_list)

    x_g = x_g.squeeze().detach().cpu()
    x_scaler = x_scaler.squeeze().detach().cpu()

    good_samples = []
    for wf, scaler in zip(x_g, x_scaler):
        tv = np.sum(np.abs(np.diff(wf)))
        # If generator sample fails to generate a seismic signal, skip it.
        # threshold value is emperically chosen
        """
        if tv < 40:
            continue
        """
        
        good_samples.append(wf * scaler)
    return good_samples[:n_waveforms]

