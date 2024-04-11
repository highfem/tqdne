from pathlib import Path

import h5py
import numpy as np
import scipy
import torch
import tqdm
from scipy import signal
from typing import Type

from tqdne.conf import Config

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
    """Compute mean and std of features in a dataset."""
    with h5py.File(datapath, "r") as f:
        stds = []
        means = []
        for key in features_keys:
            mean, std = compute_mean_std(f[key][0])
            means.append(mean)
            stds.append(std)

    return np.array(means), np.array(stds)


def extract_sample_from_h5file(f, idx, config=Config()):
    """Extract a sample from a h5 file.

    Args:
        f: h5 file
        idx: index of the sample to extract

    """
    # time = f["time_vector"][:]
    waveform = f["waveforms"][:, :, idx]
    # replace nan with 0
    waveform = np.nan_to_num(waveform)
    features = [np.nan_to_num(f[key][0, idx]) for key in config.features_keys]
    return waveform, np.array(features)


def build_dataset(config=Config()):
    """Build the dataset."""

    # extract the config information
    #output_path = config.datasetdir
    output_path = Path("/store/sdsc/sd28/data/GM0-dataset-split/")
    datapath = Path("/store/sdsc/sd28/wforms_GAN_input_v20220805.h5")
    features_keys = config.features_keys

    # Create the filter
    #sos = signal.butter(**config.params_filter, fs=config.fs, output="sos")

    with h5py.File(datapath, "r") as f:
        time = f["time_vector"][:]
        t = len(time)
        nf = len(features_keys)
        n = f["waveforms"].shape[2]
        n_train = 1024 * (128 + 64) # TO ASK: why these numbers?
        # reset the random state
        np.random.seed(42)
        permutation = np.random.permutation(n)
        train_indices = permutation[:n_train]
        test_indices = permutation[n_train:]
        means, stds = compute_mean_std_features(datapath, features_keys)

        def create_dataset(name, indices):
            processed_path = output_path / Path(name)
            with h5py.File(processed_path, "w") as fout:
                fout.create_dataset("time", data=time)
                fout.create_dataset("feature_means", data=means)
                fout.create_dataset("feature_stds", data=stds)
                waveforms = fout.create_dataset("waveform", (len(indices), 3, t))
                #filtered = fout.create_dataset("filtered", (len(indices), 3, t))
                featuress = fout.create_dataset("features", (len(indices), nf))
                for i, idx in tqdm.tqdm(enumerate(indices), total=len(indices)):
                    waveform, features = extract_sample_from_h5file(f, idx)
                    #filtered[i] = np.array(
                    #    [signal.sosfilt(sos, channel) for channel in waveform]
                    #)
                    waveforms[i] = waveform
                    featuress[i] = features

        create_dataset(config.data_train, train_indices) 
        create_dataset(config.data_test, test_indices) 
        

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, representation, n = 1024 * 8, t = 5472):
        super().__init__()
        self.n = n
        self.t = t
        self.representation = representation
        self.lp = signal.butter(10, 1, "hp", fs=100, output="sos")
        self.bp = signal.butter(2, [0.25, 10], "bp", fs=100, output="sos")

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        noise = np.random.randn(self.t)
        x = signal.sosfilt(self.bp, noise)
        lowpass = signal.sosfilt(self.lp, x)  # + 0.1 * x

        return {
            "signal": torch.tensor(x.reshape(1, -1), dtype=torch.float32),
            "cond_signal": torch.tensor(lowpass.reshape(1, -1), dtype=torch.float32),
        }


class UpsamplingDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, cut=None, cond=False, config=Config()):
        super().__init__()
        self.h5_path = h5_path
        self.cut = cut
        self.cond = cond
        self.sigma_in = config.sigma_in

        self.file = h5py.File(h5_path, "r")
        self.waveform = self.file["waveform"]
        self.filtered = self.file["filtered"]
        if cond:
            self.features = self.file["features"][:]
            self.features_means = self.file["feature_means"][:]
            self.features_stds = self.file["feature_stds"][:]

    def __del__(self):
        if not self.in_memory:
            self.file.close()

    def __len__(self):
        return len(self.waveform)

    def __getitem__(self, index):
        waveform = self.waveform[index]
        filtered = self.filtered[index]

        # normalize
        scale = np.abs(filtered).max()
        waveform = waveform / scale / 5
        filtered = filtered / scale * 2

        # add noise to filtered
        filtered += np.random.randn(*filtered.shape) * self.sigma_in

        # features
        features = self.features[index]
        # features = (features - self.features_means) / self.features_stds

        if self.cut:
            signal = waveform[:, : self.cut]
            cond_signal = filtered[:, : self.cut]
        else:
            signal = waveform
            cond_signal = filtered

        return {
            "signal": torch.tensor(signal, dtype=torch.float32),
            "cond_signal": torch.tensor(cond_signal, dtype=torch.float32),
            "cond": torch.tensor(features, dtype=torch.float32),
        }


class EnvelopeDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, data_repr, cut=None, downsample=1):
        super().__init__()
        self.h5_path = h5_path
        self.representation = data_repr

        self.file = h5py.File(h5_path, "r", locking=False)
        self.features = self.file["features"][:]
        self.waveforms = self.file["waveform"]
        #self.time = self.file["time"][:]
        self.features_means = self.file["feature_means"][:] #  Not really needed. Scaling is meaningless since embedding are sin/cos so periodic
        self.features_stds = self.file["feature_stds"][:] 
        
        # Remove the third feature (log10snr)
        self.features = self.features[:, [0, 1, 3, 4]] 

        self.n = len(self.features)
        assert self.n == len(self.waveforms)
        self.cut = cut
        self.downsample = downsample

    def __del__(self):
        pass
        #if not self.in_memory:
        #    self.file.close()

    def __len__(self):
        return self.n

    def __getitem__(self, index):

        signal = self.waveforms[index]
        features = self.features[index]
        
        # cannot be scaled because BinMetric uses non-scaled features
        # features = (features - self.features_means) / (self.features_stds + 1e-6) 


        if self.cut:
            signal = signal[:, : self.cut]

        if self.downsample > 1:
            signal = scipy.signal.decimate(signal, self.downsample, axis=1, zero_phase=False)   

        repr = self.representation.get_representation(signal)    

        return {
            "representation": torch.tensor(repr, dtype=torch.float32),
            "cond": torch.tensor(features, dtype=torch.float32),
        }
    
    def get_waveforms_by_cond_input(self, cond_input):
        idxs = np.where(np.all(self.features[:, None] == cond_input, axis=2))[1]
        return self.waveforms[np.sort(idxs), :, :self.cut]
    
    
    def get_data_by_bins(self, magnitude_bin: tuple, distance_bin: tuple, is_shallow_crustal: bool = None):
        bins_indexes = (self.features[:, 0] >= distance_bin[0]) & (self.features[:, 0] < distance_bin[1]) & (self.features[:, 2] >= magnitude_bin[0]) & (self.features[:, 2] < magnitude_bin[1])
        if is_shallow_crustal is not None:
            bins_indexes = bins_indexes & (self.features[:, 1] == is_shallow_crustal)
        if np.any(bins_indexes):
            return {"waveforms": self.waveforms[bins_indexes, :, :self.cut], "cond": self.features[bins_indexes]}
        raise ValueError("No data in the given bins")
        