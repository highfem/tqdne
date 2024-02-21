from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm
from scipy import signal

from tqdne.conf import Config
from tqdne.representations import Representation


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
    output_path = config.datasetdir
    datapath = config.datapath
    features_keys = config.features_keys

    # Create the filter
    sos = signal.butter(**config.params_filter, fs=config.fs, output="sos")

    with h5py.File(datapath, "r") as f:
        time = f["time_vector"][:]
        t = len(time)
        nf = len(features_keys)
        n = f["waveforms"].shape[2]
        n_train = 1024 * (128 + 64)
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
                filtered = fout.create_dataset("filtered", (len(indices), 3, t))
                featuress = fout.create_dataset("features", (len(indices), nf))
                for i, idx in tqdm.tqdm(enumerate(indices), total=len(indices)):
                    waveform, features = extract_sample_from_h5file(f, idx)
                    filtered[i] = np.array(
                        [signal.sosfilt(sos, channel) for channel in waveform]
                    )
                    waveforms[i] = waveform
                    featuress[i] = features

        create_dataset(config.data_upsample_train, train_indices)
        create_dataset(config.data_upsample_test, test_indices)


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, n=1024 * 8, t=5472):
        super().__init__()
        self.n = n
        self.t = t
        self.lp = signal.butter(10, 1, "hp", fs=100, output="sos")
        self.bp = signal.butter(2, [0.25, 10], "bp", fs=100, output="sos")

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        noise = np.random.randn(self.t)
        x = signal.sosfilt(self.bp, noise)
        lowpass = signal.sosfilt(self.lp, x)  # + 0.1 * x

        return {
            "high_res": torch.tensor(x.reshape(1, -1), dtype=torch.float32),
            "low_res": torch.tensor(lowpass.reshape(1, -1), dtype=torch.float32),
        }


class WaveformDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, representation: Representation=None, cut=None, reduced=None):
        super().__init__()
        self.h5_path = h5_path
        self.representation = representation
        with h5py.File(h5_path, "r") as file:
            self.waveform = file["waveform"][:]
            self.features = file["features"][:]
            self.features_means = file["feature_means"][:]
            self.features_stds = file["feature_stds"][:]
            
        if reduced: # reduce the waveform
            # Pick Distance and Magnitude Features Only
            self.features = self.features[:, [0, 3]]
            self.features_means = self.features_means[[0, 3]]
            self.features_stds = self.features_stds[[0, 3]]
            filter = np.any(np.isinf(self.features), axis=1)
            self.waveform = self.waveform[~filter]
            self.features = self.features[~filter]
            skip = self.waveform.shape[-1] // reduced
            self.waveform = self.waveform[:, 0:1, 0: skip * reduced: skip]
            assert self.waveform.shape[-1] == reduced
            
        self.cut = cut

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        waveform = self.waveform[index]
        if self.cut:
            waveform = waveform[:, : self.cut]

        features = self.features[index]
        features = (features - self.features_means) / (self.features_stds + 1e-6)

        if self.representation:
            waveform = self.representation.get_representation(waveform)
        return {
            "high_res": torch.tensor(waveform, dtype=torch.float32),
            "cond": torch.tensor(features, dtype=torch.float32),
        }


class UpsamplingDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, cut=None, in_memory=False, config=Config()):
        super().__init__()
        self.h5_path = h5_path
        self.in_memory = in_memory
        self.sigma_in = config.sigma_in
        if in_memory:
            with h5py.File(h5_path, "r") as file:
                self.features = file["features"][:]
                self.waveform = file["waveform"][:]
                self.filtered = file["filtered"][:]
                self.time = file["time"][:]
                self.features_means = file["feature_means"][:]
                self.features_stds = file["feature_stds"][:]

        else:
            self.file = h5py.File(h5_path, "r")
            self.features = self.file["features"]
            self.waveform = self.file["waveform"]
            self.filtered = self.file["filtered"]
            self.time = self.file["time"][:]
            self.features_means = self.file["feature_means"][:]
            self.features_stds = self.file["feature_stds"][:]

        self.n = len(self.features)
        assert self.n == len(self.waveform)
        self.cut = cut

    def __del__(self):
        if not self.in_memory:
            self.file.close()

    def __len__(self):
        return self.n

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
            high_res = waveform[:, : self.cut]
            low_res = filtered[:, : self.cut]
        else:
            high_res = waveform
            low_res = filtered

        return {
            "high_res": torch.tensor(high_res, dtype=torch.float32),
            "low_res": torch.tensor(low_res, dtype=torch.float32),
            "cond": torch.tensor(features, dtype=torch.float32),
        }



class StationarySignalDataset(torch.utils.data.Dataset):
    """
    A simple dataset for testing purposes.
    """
    def __init__(self, data_size, representation: Representation=None, wave_size=1024):
        super().__init__()
        self.representation = representation
        self.wave_size = wave_size
        self.p = []
        self.waveform = []

        def random_stationary_signal(n = 1024, sigma = 1):
            def bandpass_filter(n, p):
                f = np.fft.fftfreq(n)
                # f = np.fft.fftshift(f)
                filt = np.exp(- (np.abs(f) - p)**2 / 0.0001)
                return filt
            noise = sigma * np.random.randn(n)
            p = np.random.rand(1) / 16
            # apply a bandpass filter
            noise_hat = np.fft.fft(noise)
            filt = bandpass_filter(n, p)
            sig_hat = noise_hat * filt
            sig = np.fft.ifft(sig_hat)
            sig = sig.real
            return p*16, sig

        for _ in range(data_size):
            p, sig = random_stationary_signal(wave_size)
            self.p.append(p)
            self.waveform.append(sig)
        self.p = np.array(self.p)
        self.p_mean = np.mean(self.p)
        self.p_std = np.std(self.p)
        self.waveform = np.array(self.waveform)[: , np.newaxis, :]

    def __len__(self):
        return len(self.waveform)
    
    def __getitem__(self, index):
        waveform = self.waveform[index]
        p = (self.p[index] - self.p_mean) / (self.p_std + 1e-6)
        if self.representation:
            waveform = self.representation.get_representation(waveform)
        return {
            "high_res": torch.tensor(waveform, dtype=torch.float32),
            "cond": torch.tensor(p, dtype=torch.float32),
        }