from pathlib import Path
import h5py
from scipy import signal
import numpy as np
from tqdne.conf import DATASETDIR
import tqdm

# Some global variables
datapath = DATASETDIR / Path("wforms_GAN_input_v20220805.h5")
features_keys = ['hypocentral_distance', 'is_shallow_crustal', 'log10snr', 'magnitude', 'vs30',]


def compute_mean_std(array):
    """Compute mean and std of an array.
    
    This function remove all nan, inf and -inf values before computing mean and std.

    Parameters
    ----------
    array : np.array

    Returns
    -------
    mean : float
        Mean of the array.
    std : float
        Standard deviation of the array.
    """
    array = array[np.isfinite(array)]
    return np.mean(array), np.std(array)

def compute_mean_std_features(datapath, features_keys):
    """ Compute mean and std of features in a dataset. """
    with h5py.File(datapath, "r") as f:
        stds = []
        means = []
        for key in features_keys:
            mean, std = compute_mean_std(f[key][0])
            means.append(mean)
            stds.append(std)

    return np.array(means), np.array(stds)

def extract_sample_from_h5file(f, idx):
    """ """
    # time = f["time_vector"][:]
    waveform = f["waveforms"][:, :, idx]
    # replace nan with 0
    waveform = np.nan_to_num(waveform)
    features = [np.nan_to_num(f[key][0, idx]) for key in features_keys]
    return waveform, np.array(features)

def build_sample(waveform, features, means, stds, sos):

    # Filter the waveform
    datafilt = []
    for channel in waveform:
        datafilt.append(signal.sosfilt(sos, channel))
    datafilt = np.array(datafilt)

    # Normalize the waveform with respect of the datafilt
    scale = np.abs(datafilt).max() 
    datafilt = datafilt / scale * 2
    waveform = waveform / scale / 5

    # normalize the features
    features = (features - means) / stds

    return features, datafilt, waveform, scale

def build_dataset(output_path=DATASETDIR):
    fs = 100
    sos = signal.butter(2, 1, 'lp', fs=fs, output='sos')
    with h5py.File(datapath, "r") as f:
        time = f["time_vector"][:]
        t = len(time)
        nf = len(features_keys)
        n = f["waveforms"].shape[2]
        n_train = 1024*(128+64)
        n_test = n - n_train
        # reset the random state
        np.random.seed(42)
        permutation = np.random.permutation(n)
        train_indices = permutation[:n_train]
        test_indices = permutation[n_train:]

        processed_path = output_path / Path("data_train.h5")
        with h5py.File(processed_path, "w") as fout:
            fout.create_dataset("time", data=time)
            waveforms = fout.create_dataset("waveform", (n_train, 3, t))
            filtereds = fout.create_dataset("filtered", (n_train, 3, t))
            featuress = fout.create_dataset("features", (n_train, nf))
            scales = fout.create_dataset("scale", (n_train, ))
            for i, idx in tqdm.tqdm(enumerate(train_indices), total=n_train):
                waveform, features = extract_sample_from_h5file(f, idx)
                features, datafilt, waveform, scale = build_sample(waveform, features, means, stds, sos)
                waveforms[i] = waveform
                filtereds[i] = datafilt
                featuress[i] = features
                scales[i] = scale

        processed_path = output_path / Path("data_test.h5")
        with h5py.File(processed_path, "w") as fout:
            fout.create_dataset("time", data=time)
            waveforms = fout.create_dataset("waveform", (n_test, 3, t))
            filtereds = fout.create_dataset("filtered", (n_test, 3, t))
            featuress = fout.create_dataset("features", (n_test, nf))
            scales = fout.create_dataset("scale", (n_test, ))
            for i, idx in tqdm.tqdm(enumerate(test_indices), total=n_test):
                waveform, features = extract_sample_from_h5file(f, idx)
                features, datafilt, waveform, scale = build_sample(waveform, features, means, stds, sos)
                waveforms[i] = waveform
                filtereds[i] = datafilt
                featuress[i] = features
                scales[i] = scale