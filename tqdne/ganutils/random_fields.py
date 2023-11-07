#!/usr/bin/env python3

import numpy as np
import torch


class rand_noise(object):
    def __init__(self, nchan, size, device=None):
        self.nchan = nchan
        self.size = size
        self.device = device
    def sample(self, Nb, ):
        ur = torch.randn(Nb, self.nchan, self.size, device=self.device)
        return ur

def uniform_noise(Nbatch, dim):
    # Generate noise from a uniform distribution
    m = 1
    return np.random.normal(size=[Nbatch, m, dim]).astype(
        dtype=np.float32)
