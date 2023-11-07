#!/usr/bin/env python3

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from thisquakedoesnotexist.utils.preprocessor import Preprocessor
sns.set()


def main():
    input_file = '/data/projects/japan/wforms_GAN_input_v20220805.h5'
    outfile = 'thisquakedoesnotexist/data/japan/downsampled.h5'

    # initial time of recording
    time_ini_sec = 0.05

    # length of recording
    tt_wave_sec = 50.0
    sample_rate = 20.0

    # downssampling factors
    factor = 5
    lower_bound = 4.0
    upper_bound = 7.5
    channels = 1

    preprocessor = Preprocessor(input_file, outfile, time_ini_sec, tt_wave_sec, sample_rate, channels)

    preprocessor.downsample(factor, lower_bound, upper_bound)
    preprocessor.compute_pga()
    preprocessor.select_shallow_crustal()
    preprocessor.save_to_files()


if __name__ == '__main__':
    main()