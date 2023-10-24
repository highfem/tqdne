from pathlib import Path
import h5py
from scipy import signal
import numpy as np
from tqdne.conf import DATASETDIR
import tqdm
import torch

# Some global variables
datapath = DATASETDIR / Path("wforms_GAN_input_v20220805.h5")
features_keys = [
    "hypocentral_distance",
    "is_shallow_crustal",
    "log10snr",
    "magnitude",
    "vs30",
]


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


def extract_sample_from_h5file(f, idx):
    """Extract a sample from a h5 file.

    Args:
        f: h5 file
        idx: index of the sample to extract

    """
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
    sos = signal.butter(2, 1, "lp", fs=fs, output="sos")
    with h5py.File(datapath, "r") as f:
        time = f["time_vector"][:]
        t = len(time)
        nf = len(features_keys)
        n = f["waveforms"].shape[2]
        n_train = 1024 * (128 + 64)
        n_test = n - n_train
        # reset the random state
        np.random.seed(42)
        permutation = np.random.permutation(n)
        train_indices = permutation[:n_train]
        test_indices = permutation[n_train:]
        means, stds = compute_mean_std_features(datapath, features_keys)

        processed_path = output_path / Path("data_train.h5")
        with h5py.File(processed_path, "w") as fout:
            fout.create_dataset("time", data=time)
            waveforms = fout.create_dataset("waveform", (n_train, 3, t))
            filtereds = fout.create_dataset("filtered", (n_train, 3, t))
            featuress = fout.create_dataset("features", (n_train, nf))
            scales = fout.create_dataset("scale", (n_train,))
            for i, idx in tqdm.tqdm(enumerate(train_indices), total=n_train):
                waveform, features = extract_sample_from_h5file(f, idx)
                features, datafilt, waveform, scale = build_sample(
                    waveform, features, means, stds, sos
                )
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
            scales = fout.create_dataset("scale", (n_test,))
            for i, idx in tqdm.tqdm(enumerate(test_indices), total=n_test):
                waveform, features = extract_sample_from_h5file(f, idx)
                features, datafilt, waveform, scale = build_sample(
                    waveform, features, means, stds, sos
                )
                waveforms[i] = waveform
                filtereds[i] = datafilt
                featuress[i] = features
                scales[i] = scale


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, n=1024 * 8, t=5488):
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
        lowpass = signal.sosfilt(self.lp, x) + 0.1 * x
        return torch.tensor(lowpass.reshape(1, -1), dtype=torch.float32), torch.tensor(
            x.reshape(1, -1), dtype=torch.float32
        )


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, cut=None, in_memory=False):
        super(H5Dataset, self).__init__()
        self.h5_path = h5_path
        self.in_memory = in_memory
        if in_memory:
            with h5py.File(h5_path, "r") as file:
                self.features = file["features"][:]
                self.filtered = file["filtered"][:]
                self.scale = file["scale"][:]
                self.waveform = file["waveform"][:]
                self.time = file["time"][:]

        else:
            self.file = h5py.File(h5_path, "r")
            self.features = self.file["features"]
            self.filtered = self.file["filtered"]
            self.scale = self.file["scale"]
            self.waveform = self.file["waveform"]
            self.time = self.file["time"][:]

        self.n = len(self.features)
        assert self.n == len(self.waveform)
        assert self.n == len(self.scale)
        assert self.n == len(self.filtered)
        self.cut = cut

    def __del__(self):
        if not self.in_memory:
            self.file.close()

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if self.cut:
            return torch.tensor(self.filtered[index, :, : self.cut]), torch.tensor(
                self.waveform[index, :, : self.cut]
            )
        else:
            return torch.tensor(self.filtered[index]), torch.tensor(
                self.waveform[index]
            )
